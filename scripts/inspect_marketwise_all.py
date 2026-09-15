from pathlib import Path
import pandas as pd

files = [
    Path("data/raw/agmarknet/marketwise_probe/marketwise_2024_07_w1.xlsx"),
    Path("data/raw/agmarknet/marketwise_probe/marketwise_2024_11_w1.xlsx"),
    Path("data/raw/agmarknet/marketwise_probe/marketwise_2023_11_w1.xlsx"),
]

for path in files:
    print("\n" + "=" * 80)
    print(path)

    sheets = pd.read_excel(
        path,
        sheet_name=None,
        header=None,
    )

    for sheet_name, df in sheets.items():
        print(f"\nSHEET: {sheet_name}")
        print(f"Shape: {df.shape}")
        print("\nFirst 20 rows:")
        print(df.head(20).to_string(index=False, header=False))

        text = df.astype(str)

        matches = text.apply(
            lambda col: col.str.contains(
                "coimbatore|market|district|tamil|onion",
                case=False,
                na=False,
                regex=True,
            )
        )

        rows = df.loc[matches.any(axis=1)]

        if not rows.empty:
            print("\nRelevant rows:")
            print(rows.head(30).to_string(index=False, header=False))