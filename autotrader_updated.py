# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
from collections import deque
import time
from typing import List

import numpy as np

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

LOT_SIZE = 10
POSITION_LIMIT = 100 - LOT_SIZE
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS
                        ) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class Price():

    def __init__(self):
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.estimate = 0.0
        self.updated = False


class Position():

    def __init__(self):
        self.ids = []
        self.volume = 0


def get_instrument_price(ask_price, ask_volume, bid_price, bid_volume):
    bid_weighted_price = np.dot(ask_price, ask_volume)
    ask_weighted_price = np.dot(bid_price, bid_volume)

    total_volume = np.sum(ask_volume) + np.sum(bid_volume)

    return (bid_weighted_price + ask_weighted_price) / total_volume


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str,
                 secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

        self._count = 0

        self._etf = Price()
        self._future = Price()

        self._short_window = 5
        self._long_window = 50

        self._small_short_que = deque(maxlen=self._short_window)
        self._big_short_que = deque(maxlen=self._long_window)

        self._small_long_que = deque(maxlen=self._short_window)
        self._big_long_que = deque(maxlen=self._long_window)

        self._short_zscore = 0.
        self._long_zscore = 0.0

        self._open_thresholds = [-1.0, 1.0]
        self._close_thresholds = [-0.25, 0.25]

        self._open_short = Position()
        self._open_long = Position()

    def on_error_message(self, client_order_id: int,
                         error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id,
                            error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids
                                     or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int,
                                volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(
            "received hedge filled for order %d with average price %d and volume %d",
            client_order_id, price, volume)

    def on_order_book_update_message(self, instrument: int,
                                     sequence_number: int,
                                     ask_prices: List[int],
                                     ask_volumes: List[int],
                                     bid_prices: List[int],
                                     bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info(
            "received order book for instrument %d with sequence number %d",
            instrument, sequence_number)

        if instrument == Instrument.FUTURE:
            self._future.estimate = get_instrument_price(
                ask_prices, ask_volumes, bid_prices, bid_volumes)

            self._future.best_bid = bid_prices[0]
            self._future.best_ask = ask_prices[0]

            self._future.updated = True

        if instrument == Instrument.ETF:
            self._etf.estimate = get_instrument_price(ask_prices, ask_volumes,
                                                      bid_prices, bid_volumes)

            self._etf.best_bid = bid_prices[0]
            self._etf.best_ask = ask_prices[0]

            self._etf.updated = True

        if self._etf.updated and self._future.updated:
            self._count += 1

            spread_short = self._etf.best_bid / self._future.estimate
            spread_long = self._etf.best_ask / self._future.estimate

            if spread_long != spread_long:
                self._etf.updated = False
                self._future.updated = False
                return

            self._small_short_que.append(spread_short)
            self._big_short_que.append(spread_short)

            self._small_long_que.append(spread_long)
            self._big_long_que.append(spread_long)

            ssq = np.array(self._small_short_que, dtype=np.float32)
            bsq = np.array(self._big_short_que, dtype=np.float32)

            slq = np.array(self._small_long_que, dtype=np.float32)
            blq = np.array(self._big_long_que, dtype=np.float32)

            ssq_mean = np.mean(ssq)
            bsq_mean = np.mean(bsq)
            bsq_std = np.std(bsq)

            slq_mean = np.mean(slq)
            blq_mean = np.mean(blq)
            blq_std = np.std(blq)

            self._short_zscore = (ssq_mean - bsq_mean) / bsq_std
            self._long_zscore = (slq_mean - blq_mean) / blq_std

            if self._short_zscore > self._open_thresholds[1]:
                """
                Short ETF, long Future
                """
                if self.position > -POSITION_LIMIT:
                    self.ask_id = next(self.order_ids)
                    self.ask_price = self._etf.best_bid
                    self.send_insert_order(self.ask_id, Side.SELL,
                                           self.ask_price, LOT_SIZE,
                                           Lifespan.FILL_AND_KILL)
                    self.asks.add(self.ask_id)

                    self._open_short.ids.append(self.ask_id)

                    print(f"SHORT ETF - {self.ask_id} at {self._etf.best_bid}")

            if self._long_zscore < self._open_thresholds[0]:
                """
                Long ETF, short Future
                """
                if self.position < POSITION_LIMIT:
                    self.bid_id = next(self.order_ids)
                    self.bid_price = self._etf.best_ask
                    self.send_insert_order(self.bid_id, Side.BUY,
                                           self.bid_price, LOT_SIZE,
                                           Lifespan.FILL_AND_KILL)
                    self.bids.add(self.bid_id)

                    self._open_long.ids.append(self.bid_id)

                    print(f"Long ETF - {self.bid_id} at {self._etf.best_ask}")

            # Close open short positions
            if self._long_zscore < self._close_thresholds[
                    1] and self._open_short.volume != 0:

                self.bid_id = next(self.order_ids)
                self.bid_price = self._etf.best_ask
                self.send_insert_order(self.bid_id, Side.BUY, self.bid_price,
                                       self._open_short.volume,
                                       Lifespan.FILL_AND_KILL)
                self.bids.add(self.bid_id)
                print(f"Close short - {self.bid_id} at {self.bid_price}")

            # Close open long position
            if self._short_zscore > self._close_thresholds[
                    0] and self._open_long.volume != 0:
                self.ask_id = next(self.order_ids)
                self.ask_price = self._etf.best_bid
                self.send_insert_order(self.ask_id, Side.SELL, self.ask_price,
                                       self._open_long.volume,
                                       Lifespan.FILL_AND_KILL)
                self.asks.add(self.ask_id)

                print(f"Close long - {self.ask_id} at {self.ask_price}")

            self._etf.updated = False
            self._future.updated = False

    def on_order_filled_message(self, client_order_id: int, price: int,
                                volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(
            "received order filled for order %d with price %d and volume %d",
            client_order_id, price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK,
                                  MIN_BID_NEAREST_TICK, volume)

            if client_order_id in self._open_long.ids:
                self._open_long.volume += volume
                print(f"Hedge long - {client_order_id}, {volume}")

            else:
                self._open_short.volume -= volume
                print(f"Close short - {client_order_id}, {volume}")

        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID,
                                  MAX_ASK_NEAREST_TICK, volume)

            if client_order_id in self._open_short.ids:
                self._open_short.volume += volume
                print(f"Hedge short - {client_order_id}, {volume}")

            else:
                self._open_long.volume -= volume
                print(f"Close long - {client_order_id}, {volume}")

    def on_order_status_message(self, client_order_id: int, fill_volume: int,
                                remaining_volume: int, fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info(
            "received order status for order %d with fill volume %d remaining %d and fees %d",
            client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int,
                               ask_prices: List[int], ask_volumes: List[int],
                               bid_prices: List[int],
                               bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info(
            "received trade ticks for instrument %d with sequence number %d",
            instrument, sequence_number)
