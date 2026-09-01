import re
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/Tamilnadu.csv")
OUTPUT_FILE = Path("data/processed/market_prices_clean.csv")


def extract_crop(source_name: str) -> str | None:
    match = re.search(
        r"Prices-of-(.+?)-as-on-",
        source_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).strip().title()


def extract_date(source_name: str) -> str | None:
    match = re.search(
        r"as-on-(\d{2}-\d{2}-\d{4})",
        source_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return pd.to_datetime(
        match.group(1),
        format="%d-%m-%Y",
    ).strftime("%Y-%m-%d")


def clean_market_name(name: str) -> str:
    name = str(name)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # Normalize spacing before closing parentheses
    name = re.sub(r"\s+\)", ")", name)

    return name


def main():
    print("=" * 60)
    print("PREPARING MARKET DATA")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")

    required_columns = [
        "Source.Name",
        "StateName",
        "Market",
        "Mandi Arrival Quantity",
        "Mandi WholeSale Price",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Extract crop and date
    # ---------------------------------------------------------

    df["crop"] = df["Source.Name"].apply(extract_crop)
    df["price_date"] = df["Source.Name"].apply(extract_date)

    # ---------------------------------------------------------
    # Rename columns
    # ---------------------------------------------------------

    df = df.rename(
        columns={
            "Market": "market_name",
            "Mandi Arrival Quantity": "arrival_quantity_quintals",
            "Mandi WholeSale Price": "modal_price_per_quintal",
        }
    )

    # ---------------------------------------------------------
    # Clean market names
    # ---------------------------------------------------------

    df["market_name"] = df["market_name"].apply(
        clean_market_name
    )

    # ---------------------------------------------------------
    # Keep only Tamil Nadu
    # ---------------------------------------------------------

    df = df[
        df["StateName"].str.strip().str.lower()
        == "tamil nadu"
    ].copy()

    # ---------------------------------------------------------
    # Remove aggregate "All Markets" rows
    #
    # These are totals/aggregates and should not be treated
    # as an individual mandi.
    # ---------------------------------------------------------

    before = len(df)

    df = df[
        df["market_name"].str.strip().str.lower()
        != "all markets"
    ].copy()

    removed = before - len(df)

    print(f"Removed aggregate rows: {removed}")

    # ---------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------

    df["arrival_quantity_quintals"] = pd.to_numeric(
        df["arrival_quantity_quintals"],
        errors="coerce",
    )

    df["modal_price_per_quintal"] = pd.to_numeric(
        df["modal_price_per_quintal"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Remove invalid rows
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "crop",
            "price_date",
            "market_name",
            "modal_price_per_quintal",
        ]
    ).copy()

    print(
        f"Removed invalid rows: "
        f"{before - len(df)}"
    )

    # ---------------------------------------------------------
    # Add district
    #
    # The current dataset is Tamil Nadu-wide, so district
    # cannot safely be inferred from the market name.
    #
    # We leave it as NULL for now rather than inventing data.
    # ---------------------------------------------------------

    df["district"] = None

    # ---------------------------------------------------------
    # Select final columns
    # ---------------------------------------------------------

    df = df[
        [
            "crop",
            "market_name",
            "district",
            "price_date",
            "modal_price_per_quintal",
            "arrival_quantity_quintals",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    df = df.sort_values(
        [
            "crop",
            "price_date",
            "market_name",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("CLEAN DATASET")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("Crops:")
    print(df["crop"].value_counts().to_string())

    print()
    print("Dates:")
    print(
        df["price_date"]
        .drop_duplicates()
        .sort_values()
        .to_string(index=False)
    )

    print()
    print("Markets:")
    print(
        df["market_name"]
        .nunique()
    )

    print()
    print("Price statistics:")
    print(
        df["modal_price_per_quintal"]
        .describe()
        .to_string()
    )

    print()
    print("Arrival statistics:")
    print(
        df["arrival_quantity_quintals"]
        .describe()
        .to_string()
    )

    print()
    print("First 10 rows:")
    print(
        df.head(10).to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()