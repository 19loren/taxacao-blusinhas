from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PMC_PATH = PROJECT_ROOT / "data" / "processed" / "pmc_vestuario.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "financial_prices.csv"

TABLE_PATH = PROJECT_ROOT / "outputs" / "tables"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"


def load_data():
    pmc = pd.read_csv(PMC_PATH, parse_dates=["Data"])
    prices = pd.read_csv(PRICES_PATH, parse_dates=["Data"])

    return pmc, prices


def prepare_comparison(pmc, prices):
    # PMC mensal
    pmc["Mes"] = pmc["Data"].dt.to_period("M")

    pmc_monthly = (
        pmc.groupby("Mes")["PMC_vestuario"]
        .last()
    )

    # preços financeiros no ultimo pregao de cada mes
    prices["Mes"] = prices["Data"].dt.to_period("M")

    prices_monthly = (
        prices.groupby("Mes")[["LREN3", "IBOV"]]
        .last()
    )

    # combinar as tres series
    comparison = pd.concat(
        [pmc_monthly, prices_monthly],
        axis=1,
        join="inner"
    )

    # periodo comum da comparacao
    comparison = comparison.loc["2021-01":"2025-12"].copy()

    if len(comparison) != 60:
        raise ValueError(
            f"Esperadas 60 observações, encontradas {len(comparison)}."
        )

    # transformar as series em indices com base 100
    comparison["PMC_vestuario_100"] = (
        comparison["PMC_vestuario"]
        / comparison["PMC_vestuario"].iloc[0]
        * 100
    )

    comparison["LREN3_100"] = (
        comparison["LREN3"]
        / comparison["LREN3"].iloc[0]
        * 100
    )

    comparison["IBOV_100"] = (
        comparison["IBOV"]
        / comparison["IBOV"].iloc[0]
        * 100
    )

    comparison.index = comparison.index.to_timestamp()
    comparison.index.name = "Data"

    return comparison


def save_results(comparison):
    TABLE_PATH.mkdir(parents=True, exist_ok=True)

    output = comparison[
        [
            "PMC_vestuario_100",
            "LREN3_100",
            "IBOV_100"
        ]
    ].copy()

    output.to_csv(
        TABLE_PATH / "sector_comparison_2021_2025.csv",
        encoding="utf-8"
    )

    print("\nComparison results:")
    print(output.tail())

    print("\nCumulative variation in the period (%):")
    variation = output.iloc[-1] - 100
    print(variation.round(2))


def create_plot(comparison):
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison.index,
        comparison["PMC_vestuario_100"],
        label="PMC - Vestuário",
        linewidth=2
    )

    plt.plot(
        comparison.index,
        comparison["LREN3_100"],
        label="LREN3",
        linewidth=2
    )

    plt.plot(
        comparison.index,
        comparison["IBOV_100"],
        label="IBOV",
        linewidth=2
    )

    plt.axhline(
        y=100,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    plt.title("Evolução do setor de vestuário e do mercado financeiro")
    plt.xlabel("Período")
    plt.ylabel("Índice (janeiro de 2021 = 100)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "sector_comparison_2021_2025.png",
        dpi=300
    )

    plt.close()


def run_comparison():
    pmc, prices = load_data()

    comparison = prepare_comparison(pmc, prices)

    save_results(comparison)
    create_plot(comparison)

    print("\nSector comparison completed successfully.")
    print(f"Observations: {len(comparison)}")
    print(f"Period: {comparison.index.min():%Y-%m} to "
          f"{comparison.index.max():%Y-%m}")


if __name__ == "__main__":
    run_comparison()
