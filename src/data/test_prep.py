from src.data.preparation import (
    load_financial_data,
    validate_data,
    save_processed_data
)


def main():
    prices, returns = load_financial_data()

    prices, returns = validate_data(prices, returns)

    save_processed_data(prices, returns)

    print("Data preparation completed successfully!")
    print(f"Price observations: {len(prices)}")
    print(f"Return observations: {len(returns)}")


if __name__ == "__main__":
    main()