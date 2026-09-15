import pandas as pd

FILE = "data/raw/Tamilnadu.xlsx"

try:
    excel = pd.ExcelFile(FILE)
    print("SHEETS:", excel.sheet_names)

    for sheet in excel.sheet_names:
        print(f"\n{'=' * 60}")
        print(f"SHEET: {sheet}")
        print("=" * 60)

        df = pd.read_excel(FILE, sheet_name=sheet)
        print(f"Shape: {df.shape}")

        print("\nColumns:")
        for column in df.columns:
            print(f"  - {column}")

        print("\nFirst 5 rows:")
        print(df.head(5).to_string(index=False))

        # Automatically export sheet to CSV for inspect_market_csv.py
        csv_file = "data/raw/Tamilnadu.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Converted '{sheet}' to '{csv_file}'")

except Exception as e:
    print(f"Error reading Excel file: {e}")