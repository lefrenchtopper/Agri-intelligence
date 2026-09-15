import os
import glob
import re
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw", "agmarknet")
PROCESSED_DIR = os.path.join("data", "processed")
CANONICAL_FILE = os.path.join(PROCESSED_DIR, "canonical_market_prices_weekly.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

all_records = []

print("=== Starting Weekly Dataset Extraction & Canonical Build ===")

# Discover all files in raw directory
raw_files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True) + \
            glob.glob(os.path.join(RAW_DIR, "**", "*.xlsx"), recursive=True)

print(f"Processing {len(raw_files)} raw files...")

for file_path in raw_files:
    norm_path = os.path.normpath(file_path)
    
    # Extract year from path regex
    year_match = re.search(r'[\\/](202[0-9])[\\/]', norm_path)
    if not year_match:
        continue
    year = int(year_match.group(1))

    parts = norm_path.split(os.sep)
    month_str = parts[-2] if len(parts) >= 2 else "unknown"
    filename = parts[-1]
    
    week_match = re.search(r'\d+', filename)
    week_num = int(week_match.group()) if week_match else None

    df_raw = None

    # Attempt 1: Read as Excel file forcing openpyxl engine
    try:
        df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
    except Exception:
        # Attempt 2: Fallback to CSV parser if file is true text CSV
        try:
            df_raw = pd.read_csv(file_path, header=None)
        except Exception:
            continue

    if df_raw is None or df_raw.empty:
        continue

    # Extract non-empty records from parsed DataFrame
    for _, row in df_raw.iterrows():
        # Find non-null values across columns
        valid_cols = [str(val).strip() for val in row.dropna() if str(val).strip()]
        
        if len(valid_cols) < 2:
            continue

        market_val = valid_cols[0]

        # Ignore header/metadata rows
        if any(k in market_val.lower() for k in ["market", "average", "note:", "weekly analysis", "district", "s.no", "commodity"]):
            continue

        price_raw = valid_cols[1].replace(',', '').strip()

        # Parse numeric price or preserve missing value
        if price_raw in ['-', '', 'N/A', 'null', 'None', 'nan']:
            modal_price = np.nan
        else:
            try:
                modal_price = float(price_raw)
            except ValueError:
                modal_price = np.nan

        all_records.append({
            "commodity": "Onion",
            "state": "Tamil Nadu",
            "district": "Coimbatore",
            "market": market_val,
            "year": year,
            "month": month_str,
            "week": week_num,
            "modal_price_per_quintal": modal_price,
            "unit": "Rs./Quintal",
            "source_file": filename
        })

# Construct Canonical DataFrame
df = pd.DataFrame(all_records)

if not df.empty:
    df = df.drop_duplicates(subset=["district", "market", "year", "month", "week"])
    df.to_csv(CANONICAL_FILE, index=False)

    print("\n=== Processing Results ===")
    print(f"Total Canonical Records: {len(df)}")
    print(f"Valid Price Records: {df['modal_price_per_quintal'].notna().sum()}")
    print(f"Missing Observations Preserved: {df['modal_price_per_quintal'].isna().sum()}")
    print(f"Saved to: {CANONICAL_FILE}")
else:
    print("No records extracted. Let's inspect single file structure.")