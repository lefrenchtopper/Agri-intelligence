import pandas as pd

FILE = "data/raw/Tamilnadu.xlsx"

excel = pd.ExcelFile(FILE)

print("SHEETS:")
print(excel.sheet_names)

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