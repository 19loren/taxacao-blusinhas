from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "financial_prices.csv"
)

RETURNS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "financial_returns.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables"


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


def run_adf_test(series, series_name, regression):

    series = series.dropna()

    result = adfuller(
        series,
        regression=regression,
        autolag="AIC",
        result_object=False
    )

    statistic = result[0]
    p_value = result[1]
    used_lag = result[2]
    observations = result[3]
    critical_values = result[4]

    return {
        "series": series_name,
        "ADF statistic": statistic,
        "p-value": p_value,
        "lags": used_lag,
        "observations": observations,
        "critical value (1%)": critical_values["1%"],
        "critical value (5%)": critical_values["5%"],
        "critical value (10%)": critical_values["10%"],
        "stationary (5%)": p_value < 0.05
    }


def calculate_stationarity_tests(prices, returns):

    results = []

    series_to_test = [
        (
            np.log(prices["LREN3"]),
            "Log price LREN3",
            "ct"
        ),
        (
            np.log(prices["IBOV"]),
            "Log price IBOV",
            "ct"
        ),
        (
            returns["Ret_LREN3_pct"],
            "LREN3 daily log returns",
            "c"
        ),
        (
            returns["Ret_IBOV_pct"],
            "IBOV daily log returns",
            "c"
        )
    ]

    for series, name, regression in series_to_test:

        result = run_adf_test(
            series,
            name,
            regression
        )

        results.append(result)

    return pd.DataFrame(results)


def run_stationarity_analysis():

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    prices, returns = load_data()

    results = calculate_stationarity_tests(
        prices,
        returns
    )

    results.to_csv(
        OUTPUT_PATH / "stationarity_tests.csv",
        index=False
    )

    print("stationarity analysis completed successfully.")
    print(f"results saved to: {OUTPUT_PATH}")

    print("\nADF test results:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    run_stationarity_analysis()
