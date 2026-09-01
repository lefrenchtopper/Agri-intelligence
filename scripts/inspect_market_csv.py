import pandas as pd

FILE = "data/raw/Tamilnadu.csv"

df = pd.read_csv(FILE)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing values:")
print(df.isna().sum())

print("\nStates:")
print(df["StateName"].value_counts().head(20))

print("\nTamil Nadu rows:")
tn = df[df["StateName"].str.strip().str.lower() == "tamil nadu"].copy()

print("Rows:", len(tn))

print("\nFirst 20 Tamil Nadu markets:")
print(
    tn[
        [
            "Market",
            "Mandi Arrival Quantity",
            "Mandi WholeSale Price",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nPrice statistics:")
print(
    tn["Mandi WholeSale Price"].describe()
)

print("\nArrival statistics:")
print(
    tn["Mandi Arrival Quantity"].describe()
)

print("\nAll Markets rows:")
print(
    tn[
        tn["Market"].str.strip().str.lower() == "all markets"
    ].to_string(index=False)
)