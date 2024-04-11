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
from typing import List

import numpy as np
import pandas as pd

from ready_trader_go import (MAXIMUM_ASK, MINIMUM_BID, BaseAutoTrader,
                             Instrument, Lifespan, Side)

LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS
                        ) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class InstrumentPricing():

    def __init__(self, record: bool = False):
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.estimate = 0.0

        self.best_ask_volume = None
        self.best_bid_volume = None

        self.__updated = False
        self.__record = record

        if self.__record:
            self.__best_bids = []
            self.__best_asks = []
            self.__estimates = []

    def update(self, ask_prices: List[int], ask_volumes: List[int],
               bid_prices: List[int], bid_volumes: List[int]):

        self.best_bid = bid_prices[0]
        self.best_ask = ask_prices[0]
        self.estimate = self.estimate_price(ask_prices, ask_volumes,
                                            bid_prices, bid_volumes)

        self.best_ask_volume = ask_volumes[0]
        self.best_bid_volume = bid_volumes[0]

        self.__updated = True

        if self.__record:
            self.__best_bids.append(self.best_bid)
            self.__best_asks.append(self.best_ask)
            self.__estimates.append(self.estimate)

    def is_updated(self):
        return self.__updated

    def reset(self):
        self.__updated = False

    @staticmethod
    def estimate_price(ask_prices: List[int], ask_volumes: List[int],
                       bid_prices: List[int], bid_volumes: List[int]):

        bid_weighted_price = np.dot(ask_prices, ask_volumes)
        ask_weighted_price = np.dot(bid_prices, bid_volumes)

        max_bid = bid_prices[0]
        min_ask = ask_prices[0]

        total_volume = np.sum(ask_volumes) + np.sum(bid_volumes)

        # weighted_avg_estimate = (bid_weighted_price + ask_weighted_price) / total_volume

        if sum(bid_prices) == 0:
            max_bid = min_ask
        
        if sum(ask_prices) == 0:
            max_bid = min_ask

        midpoint_estimate = (max_bid + min_ask)/2

        return midpoint_estimate

    def save(self, filename: str):
        __frame = pd.DataFrame({
            "Bids": self.__best_bids,
            "Asks": self.__best_asks,
            "Est": self.__estimates
        })

        __frame.to_csv(filename)


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
        self.   bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

        self.__record = True
        self.etf = InstrumentPricing(record=self.__record)
        self.future = InstrumentPricing(record=self.__record)

        self.__long_window = 20
        self.long_que = deque(maxlen=self.__long_window)

        self.__open_thresholds = (-1.5, 1.5)
        self.__close_thresholds = (1, -1)

        self.__open_short_ids = []
        self.__close_short_ids = []
        self.__open_long_ids = []
        self.__close_long_ids = []
        self.__short_volume = 0
        self.__long_volume = 0

        self.__open_short_count = []
        self.__open_long_count = []
        self.__close_short_count = []
        self.__close_long_count = []

        self.__count = 0

        self.__data = {
            "count": [],
            "spread": [],
            "long_mean": [],
            "z_score": [],
        }

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
        print(f"FILLED FUTURE {client_order_id}, PRICE - {price}, VOLUME - {volume}")
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

        # Usually the first received update is invalid so skip that
        if sequence_number < 2:
            return

        if instrument == Instrument.FUTURE:
            self.future.update(ask_prices, ask_volumes, bid_prices,
                               bid_volumes)

        if instrument == Instrument.ETF:
            self.etf.update(ask_prices, ask_volumes, bid_prices, bid_volumes)

        # Only do calculations if both the ETF and the Future have been updated
        if self.etf.is_updated() and self.future.is_updated():
            # reset the prices so that the new ones can be retrieved after
            self.etf.reset()
            self.future.reset()

            self.__count += 1
            self.__data["count"].append(self.__count)
            print('-__________________________________________________-')
            print('-- ETF -- Max bid:', self.etf.best_bid, 'Min ask:', self.etf.best_ask)
            print('-- FUTURE -- Max bid:', self.future.best_bid, 'Min ask:', self.future.best_ask)
            
            ########## 

            spread = self.etf.estimate / self.future.estimate

            spread_short = self.etf.best_bid / self.future.estimate
            spread_long = self.etf.best_ask / self.future.estimate
            
            self.long_que.append(spread)
            long_mean = np.mean(self.long_que)
            long_std = np.std(self.long_que)

            z_score = (spread - long_mean) / long_std

            par = 1.0
            z_score_short = (spread_short - long_mean) / long_std * par + z_score * (1 - par) 
            z_score_long = (spread_long - long_mean) / long_std * par + z_score * (1 - par) 
            # print('+++ z_short:', z_score_short, '+++ z_long:', z_score_long)
            if len(self.long_que) < self.__long_window:
                z_score = 0.

            # print(z_score)

            ### SHORT ETF
            if z_score_short > self.__open_thresholds[1]:

                if self.ask_id == 0 and self.position > -POSITION_LIMIT + LOT_SIZE:
                    self.ask_id = next(self.order_ids)
                    self.ask_price = self.etf.best_bid
                    self.send_insert_order(self.ask_id, Side.SELL,
                                           self.ask_price, LOT_SIZE,
                                           Lifespan.FILL_AND_KILL)
                    self.asks.add(self.ask_id)

                    self.__open_short_ids.append(self.ask_id)
                    
                    print(f"SELL ORDER ETF, ID - {self.ask_id}, PRICE - {self.ask_price}")

            ### LONG ETF
            if z_score_long < self.__open_thresholds[0]:
            
                if self.bid_id == 0 and self.position < POSITION_LIMIT - LOT_SIZE:

                    self.bid_id = next(self.order_ids)
                    self.bid_price = self.etf.best_ask
                    self.send_insert_order(self.bid_id, Side.BUY,
                                           self.bid_price, LOT_SIZE,
                                           Lifespan.FILL_AND_KILL)
                    self.bids.add(self.bid_id)

                    self.__open_long_ids.append(self.bid_id)

                    print(f"BUY ORDER ETF, ID - {self.bid_id}, PRICE - {self.bid_price}")

            ### Close short position
            if z_score_long < self.__close_thresholds[1] and self.__short_volume > 0:
            
                if self.bid_id == 0 and self.position < POSITION_LIMIT - LOT_SIZE:

                    self.bid_id = next(self.order_ids)
                    self.bid_price = self.etf.best_ask
                    self.send_insert_order(self.bid_id, Side.BUY,
                                           self.bid_price, self.__short_volume,
                                           Lifespan.FILL_AND_KILL)
                    self.bids.add(self.bid_id)

                    self.__close_short_ids.append(self.bid_id)
                    # print("Close short position")
                    print(
                        f"Close: BUY ETF, ID - {self.bid_id}, PRICE - {self.bid_price}, Vol - {self.__short_volume}"
                    )

            ### Close long position
            if z_score_short > self.__close_thresholds[0] and self.__long_volume > 0:
            
                if self.ask_id == 0 and self.position > -POSITION_LIMIT + LOT_SIZE:

                    self.ask_id = next(self.order_ids)
                    self.ask_price = self.etf.best_bid
                    self.send_insert_order(self.ask_id, Side.SELL,
                                           self.ask_price, self.__long_volume,
                                           Lifespan.FILL_AND_KILL)
                    self.asks.add(self.ask_id)

                    self.__close_long_ids.append(self.ask_id)
                    # print("Close long position")
                    print(
                        f"Close SELL ETF, ID - {self.ask_id}, PRICE - {self.ask_price}, Vol - {self.__long_volume}"
                    )

            self.__data["spread"].append(spread)
            self.__data["long_mean"].append(long_mean)
            self.__data["z_score"].append(z_score)

            self.__open_short_count.append(0)
            self.__open_long_count.append(0)
            self.__close_short_count.append(0)
            self.__close_long_count.append(0)

        # Save the data every 10 tickz
        if sequence_number % 10 == 0 and self.__record:
            self.etf.save("ETF.csv")
            self.future.save("Future.csv")

            frame = pd.DataFrame.from_dict(self.__data)
            frame.to_csv("data.csv")

            positions = pd.DataFrame.from_dict({
                "open_short":
                self.__open_short_count,
                "open_long":
                self.__open_long_count,
                "close_short":
                self.__close_short_count,
                "close_long":
                self.__close_long_count
            })
            positions.to_csv("positions.csv")

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

        print(f"FILLED ETF {client_order_id}, PRICE - {price}, VOLUME - {volume}")
        if client_order_id in self.bids:

            self.position += volume
            self.future.best_bid = MIN_BID_NEAREST_TICK
            self.send_hedge_order(next(self.order_ids), Side.ASK,
                                  self.future.best_bid, volume)
            print(f"SELL ORDER FUTURE, {self.future.best_bid }, VOLUME - {volume}")

            if client_order_id in self.__close_short_ids:
                # print("Removing Short volume")
                self.__short_volume -= volume

                self.__close_short_count[-1] = 1

            if client_order_id in self.__open_long_ids:
                # print("Adding Long volume")
                self.__long_volume += volume

                self.__open_long_count[-1] = 1

            print()

        elif client_order_id in self.asks:
            self.position -= volume
            self.future.best_ask = MAX_ASK_NEAREST_TICK
            self.send_hedge_order(next(self.order_ids), Side.BID,
                                  self.future.best_ask, volume)
            print(f"BUY ORDER FUTURE, {self.future.best_ask}, VOLUME - {volume}")

            if client_order_id in self.__open_short_ids:
                # print("Adding short volume")
                self.__short_volume += volume

                self.__open_short_count[-1] = 1

            if client_order_id in self.__close_long_ids:
                # print("Removing Long volume")
                self.__long_volume -= volume

                self.__close_long_count[-1] = 1

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
