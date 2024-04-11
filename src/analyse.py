import pandas
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ETF = "/Users/ashokolarov/Documents/Projects/pyready_trader_go/ETF.csv"
    Future = "/Users/ashokolarov/Documents/Projects/pyready_trader_go/Future.csv"
    data = "/Users/ashokolarov/Documents/Projects/pyready_trader_go/data.csv"
    positions = "/Users/ashokolarov/Documents/Projects/pyready_trader_go/positions.csv"

    ETF = pandas.read_csv(ETF)
    Future = pandas.read_csv(Future)
    data = pandas.read_csv(data)
    positions = pandas.read_csv(positions)

    open_short = positions['open_short']
    open_long = positions['open_long']
    close_short = positions['close_short']
    close_long = positions['close_long']

    ### Get indices
    osi = np.where(open_short == 1)[0]
    oli = np.where(open_long == 1)[0]
    csi = np.where(close_short == 1)[0]
    cli = np.where(close_long == 1)[0]

    etf_open_short = ETF['Bids'][osi]
    future_open_short = Future['Asks'][osi]

    etf_close_short = ETF['Asks'][csi]
    future_close_short = Future['Bids'][csi]

    etf_open_long = ETF['Asks'][oli]
    future_open_long = Future['Bids'][oli]

    etf_close_long = ETF['Bids'][cli]
    future_close_long = Future['Asks'][cli]

    fig, axs = plt.subplots(4, 1, sharex=True)

    axs[0].plot(ETF['Bids'])
    axs[0].plot(ETF['Asks'])
    axs[0].plot(osi,
                etf_open_short,
                linestyle="None",
                marker="^",
                label="Open short",
                color='red')
    axs[0].plot(csi,
                etf_close_short,
                linestyle="None",
                marker="*",
                label="Close short",
                color='red')

    axs[0].plot(oli,
                etf_open_long,
                linestyle="None",
                marker="^",
                label="Open long",
                color='green')
    axs[0].plot(cli,
                etf_close_long,
                linestyle="None",
                marker="*",
                label="Close long",
                color='green')

    axs[0].grid(True)
    axs[0].set_title("ETF")
    axs[0].legend()

    axs[1].plot(Future['Bids'])
    axs[1].plot(Future['Asks'])
    axs[1].plot(osi,
                future_open_short,
                label="Open short",
                linestyle="None",
                marker="^")
    axs[1].plot(csi,
                future_close_short,
                label="Close short",
                linestyle="None",
                marker="*")

    axs[1].plot(oli,
                future_open_long,
                linestyle="None",
                marker="^",
                label="Open long")
    axs[1].plot(cli,
                future_close_long,
                linestyle="None",
                marker="*",
                label="Close long")

    axs[1].grid(True)
    axs[1].set_title("Future")
    axs[1].legend()

    axs[2].plot(data['z_score'])
    axs[2].axhline(1.0, label="Open short threshold", linestyle="--", c="red")
    axs[2].axhline(-1.0,
                   label="Open long threshold",
                   linestyle="--",
                   c="green")
    axs[2].axhline(0.25,
                   label="Close short threshold",
                   linestyle="--",
                   c="red")
    axs[2].axhline(-0.25,
                   label="Close long threshold",
                   linestyle="--",
                   c="green")
    axs[2].grid(True)
    axs[2].set_title("Z Score")

    axs[3].plot(data['spread'])

    plt.tight_layout()
    plt.show()