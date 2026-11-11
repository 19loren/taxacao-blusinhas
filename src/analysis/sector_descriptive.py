from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "econometric_base.csv"
TABLES_PATH = PROJECT_ROOT / "outputs" / "tables"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"


def main():
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["Data"])
    df = df.sort_values("Data").reset_index(drop=True)

    variables = [
        "PMC_vestuario",
        "Importacoes_FOB_USD",
        "Importacoes_kg"
    ]

    # estatisticas descritivas dos niveis
    descriptive = df[variables].describe().T
    descriptive = descriptive[["mean", "std", "min", "max"]]
    descriptive.columns = ["Media", "Desvio_padrao", "Minimo", "Maximo"]

    descriptive.to_csv(
        TABLES_PATH / "sector_descriptive_levels.csv",
        encoding="utf-8-sig"
    )

    # variaçoes mensais em percentual usando diferenças logaritmicas
    changes = df[["Data"]].copy()

    for variable in variables:
        changes[f"Var_{variable}_pct"] = (
            np.log(df[variable]).diff() * 100
        )

    changes = changes.dropna().reset_index(drop=True)

    changes.to_csv(
        TABLES_PATH / "sector_monthly_changes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # estatisticas das variaçoes mensais
    change_variables = [
        f"Var_{variable}_pct" for variable in variables
    ]

    change_stats = changes[change_variables].describe().T
    change_stats = change_stats[["mean", "std", "min", "max"]]
    change_stats.columns = ["Media", "Desvio_padrao", "Minimo", "Maximo"]

    change_stats.to_csv(
        TABLES_PATH / "sector_descriptive_changes.csv",
        encoding="utf-8-sig"
    )

    # correlaçao entre as variaçoes mensais
    correlation = changes[change_variables].corr()

    correlation.to_csv(
        TABLES_PATH / "sector_correlation.csv",
        encoding="utf-8-sig"
    )

    # grafico 1: evoluçao dos indicadores, todos com base 100
    indexed = df.set_index("Data")[variables].copy()
    indexed = indexed.div(indexed.iloc[0]).mul(100)

    indexed.columns = [
        "PMC vestuario",
        "Importacoes FOB",
        "Importacoes kg"
    ]

    plt.figure(figsize=(11, 6))
    indexed.plot(ax=plt.gca(), linewidth=1.8)

    plt.title("Evolucao dos indicadores do setor de vestuario")
    plt.xlabel("Data")
    plt.ylabel("Indice (jan/2021 = 100)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "sector_indexed_evolution.png",
        dpi=300
    )
    plt.close()

    # grafico 2: variaçoes mensais
    plt.figure(figsize=(11, 6))

    for variable in change_variables:
        plt.plot(
            changes["Data"],
            changes[variable],
            label=variable.replace("Var_", "").replace("_pct", ""),
            linewidth=1.2
        )

    plt.axhline(0, linewidth=0.8, color="black")
    plt.title("Variacoes mensais dos indicadores")
    plt.xlabel("Data")
    plt.ylabel("Variacao logaritmica (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "sector_monthly_changes.png",
        dpi=300
    )
    plt.close()

    # resultados no terminal
    print("\nESTATISTICAS DESCRITIVAS - NIVEIS")
    print(descriptive.round(4))

    print("\nESTATISTICAS DESCRITIVAS - VARIACOES MENSAIS (%)")
    print(change_stats.round(4))

    print("\nCORRELACAO ENTRE VARIACOES MENSAIS")
    print(correlation.round(4))

    print("\nAnalise descritiva concluida.")
    print("Tabelas salvas em:", TABLES_PATH)
    print("Graficos salvos em:", FIGURES_PATH)


if __name__ == "__main__":
    main()
