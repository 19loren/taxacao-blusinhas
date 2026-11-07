from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "pmc_tabela_8882.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pmc_vestuario.csv"

with pd.ExcelFile(INPUT_PATH) as excel:
    SHEET_NAME = next(
        sheet for sheet in excel.sheet_names
        if "Número-índice com ajus" in sheet
    )

MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def prepare_pmc_data():
    raw = pd.read_excel(
        INPUT_PATH,
        sheet_name=SHEET_NAME,
        header=None
    )

    activity = raw.iloc[5, 1]

    if activity != "Tecidos, vestuário e calçados":
        raise ValueError(
            f"atividade inesperada na planilha: {activity}"
        )

    observations = []

    for month_column in range(2, raw.shape[1], 2):
        month_label = raw.iloc[3, month_column]
        volume_column = month_column + 1

        if pd.isna(month_label):
            continue

        month_name, year = month_label.lower().split()
        month_number = MONTHS[month_name]

        date = pd.Timestamp(
            year=int(year),
            month=month_number,
            day=1
        )

        value = raw.iloc[5, volume_column]

        observations.append({
            "Data": date,
            "PMC_vestuario": value
        })

    data = pd.DataFrame(observations)

    data = data.sort_values("Data").reset_index(drop=True)

    if data["Data"].duplicated().any():
        raise ValueError("existem meses duplicados na serie")

    if data["PMC_vestuario"].isna().any():
        raise ValueError("existem valores ausentes na serie")

    expected_dates = pd.date_range(
        start="2021-01-01",
        end=data["Data"].max(),
        freq="MS"
    )

    if not data["Data"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="Data")
    ):
        raise ValueError("a serie apresenta meses ausentes ou fora de ordem")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("PMC data preparation completed successfully.")
    print(f"observations: {len(data)}")
    print(f"period: {data['Data'].min():%Y-%m} to {data['Data'].max():%Y-%m}")
    print(f"output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_pmc_data()