from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "financial_prices.csv"
)

RETURNS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "financial_returns.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures"


def load_data():

    prices = pd.read_csv(
        PRICES_PATH,
        parse_dates=["Data"]
    )

    returns = pd.read_csv(
        RETURNS_PATH,
        parse_dates=["Data"]
    )

    return prices, returns


def plot_prices(prices):

    fig, ax1 = plt.subplots(figsize=(12, 6))

    line1 = ax1.plot(
        prices["Data"],
        prices["LREN3"],
        label="LREN3",
        color="blue"
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("LREN3 price (R$)")
    ax1.set_title("Price Evolution - LREN3 and IBOV")

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        prices["Data"],
        prices["IBOV"],
        label="IBOV",
        color="orange"
    )

    ax2.set_ylabel("IBOV points")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]

    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH / "price_evolution.png",
        dpi=300
    )

    plt.close(fig)


def plot_returns(returns):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        returns["Data"],
        returns["Ret_LREN3_pct"],
        label="LREN3",
        alpha=0.7
    )

    ax.plot(
        returns["Data"],
        returns["Ret_IBOV_pct"],
        label="IBOV",
        alpha=0.7
    )

    ax.set_title("Daily Log Returns - LREN3 and IBOV")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return (%)")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH / "daily_returns.png",
        dpi=300
    )

    plt.close(fig)


def plot_return_distributions(returns):

    assets = {
        "LREN3": "Ret_LREN3_pct",
        "IBOV": "Ret_IBOV_pct"
    }

    for asset, column in assets.items():

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            returns[column],
            bins=40
        )

        ax.set_title(f"Daily Return Distribution - {asset}")
        ax.set_xlabel("Log return (%)")
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        fig.savefig(
            OUTPUT_PATH / f"{asset.lower()}_return_distribution.png",
            dpi=300
        )

        plt.close(fig)


def run_visualizations():

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    prices, returns = load_data()

    plot_prices(prices)
    plot_returns(returns)
    plot_return_distributions(returns)

    print("visualizations completed successfully.")
    print(f"figures saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_visualizations()
