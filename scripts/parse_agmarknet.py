import sys
from pathlib import Path

# Must be before importing backend
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from backend.app.config.mappings import get_crop_id


def clean_agmarknet_dataframe(file_path: str):
    # Read CSV or Excel
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # Standardize column names (strip spaces, lowercase)
    df.columns = [col.strip().lower() for col in df.columns]

    cleaned_records = []
    
    # Iterate and map fields
    for _, row in df.iterrows():
        commodity_raw = str(row.get("commodity", "Onion"))
        crop_uuid = get_crop_id(commodity_raw)

        if not crop_uuid:
            continue  # Skip unmapped commodities

        cleaned_records.append({
            "crop_id": crop_uuid,
            "state": str(row.get("statename", "Tamil Nadu")).strip(),
            "district": str(row.get("districtname", "")).strip(),
            "market_name": str(row.get("market", "")).strip(),
            "arrival_quantity": float(row.get("mandi arrival quantity", 0) or 0),
            "wholesale_price": float(row.get("mandi wholesale price", 0) or 0),
        })

    return pd.DataFrame(cleaned_records)