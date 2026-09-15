import pandas as pd

FILE = "data/raw/Tamilnadu.csv"

try:
    df = pd.read_csv(FILE)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    # Case-insensitive column search helper
    cols = {col.lower(): col for col in df.columns}
    
    state_col = cols.get("statename", cols.get("state", None))
    market_col = cols.get("market", cols.get("marketname", None))
    price_col = cols.get("mandi wholesale price", cols.get("price", None))
    arrival_col = cols.get("mandi arrival quantity", cols.get("arrival", None))

    if state_col:
        print("\nStates:")
        print(df[state_col].value_counts().head(20))

        tn = df[df[state_col].astype(str).str.strip().str.lower() == "tamil nadu"].copy()
        print("\nTamil Nadu rows:", len(tn))

        if not tn.empty and market_col and price_col and arrival_col:
            print("\nFirst 20 Tamil Nadu markets:")
            print(tn[[market_col, arrival_col, price_col]].head(20).to_string(index=False))

            print("\nPrice statistics:")
            print(tn[price_col].describe())

            print("\nArrival statistics:")
            print(tn[arrival_col].describe())

            print("\nAll Markets rows:")
            all_m = tn[tn[market_col].astype(str).str.strip().str.lower() == "all markets"]
            print(all_m.to_string(index=False))

except Exception as e:
    print(f"Error inspecting CSV: {e}")