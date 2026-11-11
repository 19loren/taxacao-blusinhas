from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PMC_PATH = PROJECT_ROOT / "data" / "processed" / "pmc_vestuario.csv"
IMPORTS_PATH = PROJECT_ROOT / "data" / "processed" / "imports_vestuario.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "econometric_base.csv"


def main():
    pmc = pd.read_csv(PMC_PATH, parse_dates=["Data"])
    imports = pd.read_csv(IMPORTS_PATH, parse_dates=["Data"])

    # integra as bases pelos meses em comum
    base = pd.merge(
        pmc,
        imports,
        on="Data",
        how="inner",
        validate="one_to_one"
    )

    base = base.sort_values("Data").reset_index(drop=True)

    # verificaçoes
    print("Observações:", len(base))
    print("Período:", base["Data"].min(), "a", base["Data"].max())
    print("Valores ausentes:", base.isna().sum().sum())
    print("Duplicados:", base["Data"].duplicated().sum())

    # salva a base consolidada
    base.to_csv(OUTPUT_PATH, index=False)

    print("\nBase econométrica integrada com sucesso!")
    print("Arquivo salvo em:", OUTPUT_PATH)
    print("\nPrimeiras observações:")
    print(base.head())


if __name__ == "__main__":
    main()
