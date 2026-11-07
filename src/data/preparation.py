from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

def load_financial_data():

    file_path = RAW_DATA_PATH / "base_econometrica_renner_ibov.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dpythonataset not found: {file_path}"
            )
    prices = pd.read_excel(
        file_path,
        sheet_name="Precos_alinhados"
    )

    returns = pd.read_excel(
        file_path,
        sheet_name="Retornos_diarios"
    )

    return prices, returns


def validate_data(prices, returns):
    if prices.empty or returns.empty:
        raise ValueError("DataFrame is empty.")

    if "Data" not in prices.columns or "Data" not in returns.columns:
        raise ValueError("missing 'Data' column in one of the DataFrames.")

    if prices["Data"].duplicated().any() or returns["Data"].duplicated().any():
        raise ValueError("duplicate dates found in one of the DataFrames.")

    prices["Data"] = pd.to_datetime(prices["Data"])
    returns["Data"] = pd.to_datetime(returns["Data"])

    prices = prices.sort_values("Data").reset_index(drop=True)
    returns = returns.sort_values("Data").reset_index(drop=True)

    return prices, returns

def save_processed_data(prices, returns):

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    prices.to_csv(PROCESSED_DATA_PATH / "financial_prices.csv", index=False)
    returns.to_csv(PROCESSED_DATA_PATH / "financial_returns.csv", index=False)

    print(f"processed data saved to {PROCESSED_DATA_PATH}")