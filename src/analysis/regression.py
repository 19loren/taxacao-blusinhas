from pathlib import Path

import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "financial_returns.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables"


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        parse_dates=["Data"]
    )

    return returns


def estimate_market_model(returns):

    data = returns[
        ["Data", "Ret_LREN3_pct", "Ret_IBOV_pct"]
    ].dropna().copy()

    y = data["Ret_LREN3_pct"]
    X = data[["Ret_IBOV_pct"]]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5}
    )

    return model, data


def save_results(model):

    results = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "standard_error": model.bse.values,
        "t_statistic": model.tvalues.values,
        "p_value": model.pvalues.values
    })

    results.to_csv(
        OUTPUT_PATH / "market_model_results.csv",
        index=False
    )

    with open(
        OUTPUT_PATH / "market_model_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(model.summary().as_text())


def run_regression_analysis():

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    returns = load_data()

    model, data = estimate_market_model(returns)

    save_results(model)

    print("market regression completed successfully.")
    print(f"observations: {int(model.nobs)}")
    print(f"results saved to: {OUTPUT_PATH}")

    print("\nregression results:")
    print(model.summary())


if __name__ == "__main__":
    run_regression_analysis()
