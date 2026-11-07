from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    acorr_breusch_godfrey,
    linear_reset
)

from statsmodels.stats.stattools import jarque_bera


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
        ["Ret_LREN3_pct", "Ret_IBOV_pct"]
    ].dropna()

    y = data["Ret_LREN3_pct"]
    X = sm.add_constant(data["Ret_IBOV_pct"])

    model = sm.OLS(y, X).fit()

    return model


def run_diagnostics(model):

    results = []

    # Breusch-Pagan: heterocedasticity
    bp_test = het_breuschpagan(
        model.resid,
        model.model.exog
    )

    results.append({
        "test": "Breusch-Pagan",
        "statistic": bp_test[0],
        "p-value": bp_test[1],
        "conclusion": (
            "Heterocedasticity detected"
            if bp_test[1] < 0.05
            else "No evidence of heterocedasticity"
        )
    })

    # Breusch-Godfrey: autocorrelation
    bg_test = acorr_breusch_godfrey(
        model,
        nlags=5
    )

    results.append({
        "test": "Breusch-Godfrey",
        "statistic": bg_test[0],
        "p-value": bg_test[1],
        "conclusion": (
            "Autocorrelation detected"
            if bg_test[1] < 0.05
            else "No evidence of autocorrelation"
        )
    })

    # Jarque-Bera: normality
    jb_test = jarque_bera(model.resid)

    results.append({
        "test": "Jarque-Bera",
        "statistic": jb_test[0],
        "p-value": jb_test[1],
        "conclusion": (
            "Residuals are not normally distributed"
            if jb_test[1] < 0.05
            else "No evidence against normality"
        )
    })

    # Ramsey RESET: functional form
    reset_test = linear_reset(
        model,
        power=2,
        use_f=True
    )

    results.append({
        "test": "Ramsey RESET",
        "statistic": reset_test.fvalue,
        "p-value": reset_test.pvalue,
        "conclusion": (
            "Possible specification problem"
            if reset_test.pvalue < 0.05
            else "No evidence of specification problems"
        )
    })

    return pd.DataFrame(results)


def run_diagnostics_analysis():

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    returns = load_data()

    model = estimate_market_model(returns)

    results = run_diagnostics(model)

    results.to_csv(
        OUTPUT_PATH / "regression_diagnostics.csv",
        index=False
    )

    print("regression diagnostics completed successfully.")
    print(f"results saved to: {OUTPUT_PATH}")

    print("\ndiagnostic test results:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    run_diagnostics_analysis()
