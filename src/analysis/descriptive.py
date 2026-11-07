
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "financial_returns.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables"


def load_returns():

    if not RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"returns file not found: {RETURNS_PATH}"
        )

    returns = pd.read_csv(RETURNS_PATH, parse_dates=["Data"])

    required_columns = [
        "Data",
        "Ret_LREN3_pct",
        "Ret_IBOV_pct",
    ]

    missing = [
        column
        for column in required_columns
        if column not in returns.columns
    ]

    if missing:
        raise ValueError(
            f"missing columns in returns dataset: {missing}"
        )

    returns = returns[required_columns].copy()

    returns["Ret_LREN3_pct"] = pd.to_numeric(
        returns["Ret_LREN3_pct"], errors="coerce"
    )

    returns["Ret_IBOV_pct"] = pd.to_numeric(
        returns["Ret_IBOV_pct"], errors="coerce"
    )

    returns = returns.dropna().sort_values("Data")

    return returns.reset_index(drop=True)


def calculate_daily_statistics(returns):

    assets = {
        "LREN3": "Ret_LREN3_pct",
        "IBOV": "Ret_IBOV_pct",
    }

    statistics = {}

    for asset, column in assets.items():
        series = returns[column]

        statistics[asset] = {
            "observations": series.count(),
            "mean (%)": series.mean(),
            "median (%)": series.median(),
            "standard deviation (%)": series.std(),
            "minimum (%)": series.min(),
            "maximum (%)": series.max(),
            "annualized volatility (%)": (
                series.std() * np.sqrt(252)
            ),
            "cumulative return (%)": (
                np.exp(series.sum() / 100) - 1
            ) * 100
        }

    return pd.DataFrame(statistics)


def calculate_monthly_returns(returns):

    data = returns.copy()

    data["Month"] = data["Data"].dt.to_period("M")

    monthly = data.groupby("Month")[
        ["Ret_LREN3_pct", "Ret_IBOV_pct"]
    ].sum()

    monthly = (np.exp(monthly / 100) - 1) * 100

    monthly.index = monthly.index.to_timestamp()
    monthly.index.name = "Data"

    return monthly.reset_index()


def calculate_monthly_statistics(monthly_returns):

    assets = {
        "LREN3": "Ret_LREN3_pct",
        "IBOV": "Ret_IBOV_pct",
    }

    statistics = {}

    for asset, column in assets.items():
        series = monthly_returns[column]

        statistics[asset] = {
            "observations": series.count(),
            "mean (%)": series.mean(),
            "median (%)": series.median(),
            "standard deviation (%)": series.std(),
            "minimum (%)": series.min(),
            "maximum (%)": series.max(),
            "annualized volatility (%)": (
                series.std() * np.sqrt(12)
            ),
        }

    return pd.DataFrame(statistics)


def calculate_correlation(returns):

    return returns[
        ["Ret_LREN3_pct", "Ret_IBOV_pct"]
    ].corr()


def run_descriptive_analysis():

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    returns = load_returns()

    daily_statistics = calculate_daily_statistics(returns)

    monthly_returns = calculate_monthly_returns(returns)

    monthly_statistics = calculate_monthly_statistics(
        monthly_returns
    )

    correlation = calculate_correlation(returns)

    daily_statistics.to_csv(
        OUTPUT_PATH / "daily_statistics.csv"
    )

    monthly_returns.to_csv(
        OUTPUT_PATH / "monthly_returns.csv",
        index=False,
    )

    monthly_statistics.to_csv(
        OUTPUT_PATH / "monthly_statistics.csv"
    )

    correlation.to_csv(
        OUTPUT_PATH / "return_correlation.csv"
    )

    print("descriptive analysis completed successfully")
    print(f"results saved to: {OUTPUT_PATH}")

    print("\ndaily statistics:")
    print(daily_statistics.round(4))

    print("\nmonthly statistics:")
    print(monthly_statistics.round(4))

    print("\ndaily return correlation:")
    print(correlation.round(4))


if __name__ == "__main__":
    run_descriptive_analysis()
