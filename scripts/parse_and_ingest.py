import sys
from pathlib import Path
from datetime import date
import pandas as pd

# Fix sys.path before importing backend modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_market_prices import parse_and_upsert_prices

def ingest_agmarknet_file(file_path: str, fallback_commodity: str = "Onion", record_date: date = None):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_excel(file_path) if path.suffix == ".xlsx" else pd.read_csv(file_path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    records = []
    default_date = record_date or date.today()

    for _, row in df.iterrows():
        market = str(row.get("market", row.get("marketname", ""))).strip()
        if not market or market.lower() in ["all markets", "nan", "", "unnamed: 0"]:
            continue

        modal_val = row.get("mandi wholesale price", row.get("modal price", None))
        if pd.isna(modal_val) or str(modal_val).strip() in ["-", ""]:
            continue

        try:
            modal_price = float(modal_val)
        except ValueError:
            continue

        arrival_val = row.get("mandi arrival quantity", row.get("arrival", None))

        records.append({
            "commodity": str(row.get("commodity", fallback_commodity)).strip(),
            "market_name": market,
            "district": str(row.get("district", row.get("districtname", "Coimbatore"))).strip(),
            "price_date": default_date,
            "min_price": None,
            "max_price": None,
            "modal_price": modal_price,
            "arrival_quantity": float(arrival_val) if pd.notna(arrival_val) and str(arrival_val).replace('.', '', 1).isdigit() else None,
        })

    print(f"📊 Parsed {len(records)} valid market rows from {file_path}")
    parse_and_upsert_prices(records)

if __name__ == "__main__":
    ingest_agmarknet_file("data/raw/sample_tn.csv", fallback_commodity="Onion", record_date=date(2023, 1, 1))