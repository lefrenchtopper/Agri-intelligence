from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

RAW_ROOT = Path("data/raw/agmarknet/2026")
OUTPUT_DIR = Path("data/processed/agmarknet")
OUTPUT_FILE = OUTPUT_DIR / "onion_tamilnadu_weekly_2026.csv"

EXPECTED_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
]

EXPECTED_WEEKS = 4

# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_RENAME = {
    "District": "district",
    "Change (Over Previous Week)": "change_previous_week",
    "Change(Over Previous Month)": "change_previous_month",
    "Change (Over Previous Month)": "change_previous_month",
    "Change (Over Previous Year)": "change_previous_year",
}

# ============================================================
# HELPERS
# ============================================================

def clean_percentage(value):
    """
    Convert values such as:
        '2.7%'   -> 2.7
        '-34.6%' -> -34.6
        0.027    -> 0.027
        NaN      -> NaN
    """
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip().replace("%", "")

        if value in {"", "-", "—"}:
            return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def clean_price(value):
    """
    Convert price values to numeric.
    """
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.replace(",", "").strip()

        if value in {"", "-", "—"}:
            return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ============================================================
# PROCESS ONE FILE
# ============================================================
def process_file(file_path):
    # --------------------------------------------------------
    # Determine month and week from folder structure
    # --------------------------------------------------------

    week_folder = file_path.parent.name
    month_name = file_path.parent.parent.name

    if not week_folder.startswith("Week_"):
        raise ValueError(f"Unexpected folder structure: {file_path}")

    week = int(week_folder.replace("Week_", ""))

    if month_name not in EXPECTED_MONTHS:
        raise ValueError(f"Unexpected month: {month_name}")

    month = EXPECTED_MONTHS.index(month_name) + 1

    # --------------------------------------------------------
    # Read Excel
    # --------------------------------------------------------

    # Row 0 = report title
    # Row 1 = actual column headers
    # Row 2+ = data / explanatory rows

    df = pd.read_excel(
        file_path,
        header=1
    )

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    rename_map = {}
    price_columns_found = []

    for column in df.columns:

        column_str = str(column).strip()

        # The four price columns have changing dates,
        # so identify them using "Prices".
        if column_str.startswith("Prices"):
            price_columns_found.append(column)

        elif column_str in COLUMN_RENAME:
            rename_map[column] = COLUMN_RENAME[column_str]

    # --------------------------------------------------------
    # Validate price columns
    # --------------------------------------------------------

    price_names = [
        "current_price",
        "previous_week_price",
        "previous_month_price",
        "previous_year_price",
    ]

    if len(price_columns_found) != 4:
        raise ValueError(
            f"Expected 4 price columns, found "
            f"{len(price_columns_found)} in {file_path.name}: "
            f"{price_columns_found}"
        )

    for original_column, clean_name in zip(
        price_columns_found,
        price_names
    ):
        rename_map[original_column] = clean_name

    df = df.rename(columns=rename_map)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "district",
        "current_price",
        "previous_week_price",
        "previous_month_price",
        "previous_year_price",
        "change_previous_week",
        "change_previous_month",
        "change_previous_year",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}"
        )

    # --------------------------------------------------------
    # Clean district names
    # --------------------------------------------------------

    df["district"] = (
        df["district"]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Convert prices FIRST
    # --------------------------------------------------------

    price_columns = [
        "current_price",
        "previous_week_price",
        "previous_month_price",
        "previous_year_price",
    ]

    for column in price_columns:
        df[column] = df[column].apply(clean_price)

    # --------------------------------------------------------
    # Convert percentage changes
    # --------------------------------------------------------

    percentage_columns = [
        "change_previous_week",
        "change_previous_month",
        "change_previous_year",
    ]

    for column in percentage_columns:
        df[column] = df[column].apply(clean_percentage)

    # --------------------------------------------------------
    # Keep ONLY actual district rows
    # --------------------------------------------------------

    # A real district row must:
    #   1. Have a district name
    #   2. Have a current price
    #
    # This automatically removes:
    #   - Note:
    #   - Weighted Average Price = ...
    #   - Change (...) explanations
    #   - blank rows
    #   - footer rows

    df = df[
        df["district"].notna()
        & df["current_price"].notna()
    ].copy()

    # --------------------------------------------------------
    # Remove any remaining non-district rows
    # --------------------------------------------------------

    invalid_text = (
        df["district"]
        .str.lower()
        .str.startswith(
            (
                "note:",
                "weighted average",
                "change",
                "total",
                "average"
            ),
            na=False
        )
    )

    df = df.loc[~invalid_text].copy()

  
    # Add metadata


    df.insert(0, "year", 2026)
    df.insert(1, "month", month)
    df.insert(2, "month_name", month_name)
    df.insert(3, "week", week)
    # Select final columns
   

    final_columns = [
        "year",
        "month",
        "month_name",
        "week",
        "district",
        "current_price",
        "previous_week_price",
        "previous_month_price",
        "previous_year_price",
        "change_previous_week",
        "change_previous_month",
        "change_previous_year",
    ]

    return df[final_columns]

# MAIN

def main():

    print("=" * 70)
    print("AGMARKNET DATA CLEANING")
    print("=" * 70)
    print(f"Input : {RAW_ROOT}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    files = sorted(RAW_ROOT.rglob("*.xlsx"))

    print(f"Excel files found: {len(files)}")

    if not files:
        raise FileNotFoundError(
            f"No Excel files found in {RAW_ROOT}"
        )

    all_data = []
    failed_files = []

    # --------------------------------------------------------
    # Process every report
    # --------------------------------------------------------

    for index, file_path in enumerate(files, start=1):

        print(
            f"[{index:02d}/{len(files):02d}] "
            f"Processing {file_path.parent.parent.name} "
            f"{file_path.parent.name}"
        )

        try:
            df = process_file(file_path)
            all_data.append(df)

            print(f"       Rows kept: {len(df)}")

        except Exception as error:
            print(f"       ERROR: {error}")
            failed_files.append((file_path, error))

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if not all_data:
        raise RuntimeError("No files were successfully processed.")

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    combined = combined.sort_values(
        by=["year", "month", "week", "district"],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    before_duplicates = len(combined)

    combined = combined.drop_duplicates(
        subset=["year", "month", "week", "district"],
        keep="first"
    )

    duplicates_removed = before_duplicates - len(combined)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(f"Files found       : {len(files)}")
    print(f"Files processed   : {len(all_data)}")
    print(f"Files failed      : {len(failed_files)}")
    print(f"Rows in dataset   : {len(combined)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Districts         : {combined['district'].nunique()}")
    print(
        f"Month range       : "
        f"{combined['month'].min()} - {combined['month'].max()}"
    )
    print(
        f"Week range        : "
        f"{combined['week'].min()} - {combined['week'].max()}"
    )

    print()
    print("Rows by month:")

    month_counts = (
        combined
        .groupby(["month", "month_name"])
        .size()
    )

    print(month_counts.to_string())

    print()
    print("Missing values:")

    print(
        combined.isna()
        .sum()
        .to_string()
    )

    print()
    print(f"Saved to:")
    print(OUTPUT_FILE.resolve())

    # --------------------------------------------------------
    # Failed files
    # --------------------------------------------------------

    if failed_files:

        print()
        print("=" * 70)
        print("FAILED FILES")
        print("=" * 70)

        for file_path, error in failed_files:
            print(file_path)
            print(f"  {error}")


if __name__ == "__main__":
    main()