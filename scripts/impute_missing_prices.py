import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

DB_PATH = os.path.join("data", "processed", "agri_intelligence.db")
OUTPUT_CSV = os.path.join("data", "processed", "canonical_market_prices_imputed_weekly.csv")

# Standard month string mapping for chronological sorting
MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def run_imputation_pipeline():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    print("=== Step 7: Starting Time-Series Price Imputation Pipeline ===")
    df = pd.read_sql("SELECT * FROM canonical_market_prices", con=engine)

    initial_missing = df['modal_price_per_quintal'].isna().sum()
    print(f"Total input records: {len(df)}")
    print(f"Missing price observations before imputation: {initial_missing}")

    # 1. Map months and construct chronological sort index
    df['month_num'] = df['month'].str.strip().str.lower().str[:3].map(MONTH_MAP)
    df['week_num'] = df['week'].fillna(1).astype(int)
    
    # Sort chronologically by market and time
    df = df.sort_values(by=['market', 'year', 'month_num', 'week_num']).reset_index(drop=True)

    # 2. Market-level Linear Interpolation + Forward/Backward Fill
    print("\n[1/2] Applying linear interpolation per market group...")
    df['price_imputed'] = (
        df.groupby('market')['modal_price_per_quintal']
        .transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
    )

    remaining_missing = df['price_imputed'].isna().sum()

    # 3. Fallback: Regional (District/State) Weekly Median for markets with no history in a period
    if remaining_missing > 0:
        print(f"[2/2] Applying district/year weekly median fallback for {remaining_missing} remaining values...")
        regional_medians = df.groupby(['district', 'year', 'month_num', 'week_num'])['modal_price_per_quintal'].transform('median')
        df['price_imputed'] = df['price_imputed'].fillna(regional_medians)

    # Global median fallback for any isolated outliers
    if df['price_imputed'].isna().sum() > 0:
        overall_median = df['modal_price_per_quintal'].median()
        df['price_imputed'] = df['price_imputed'].fillna(overall_median)

    # Flag imputed values
    df['is_imputed'] = df['modal_price_per_quintal'].isna()
    df['modal_price_per_quintal'] = df['price_imputed']
    
    # Clean up auxiliary sorting columns
    df = df.drop(columns=['month_num', 'week_num', 'price_imputed'])

    # 4. Save Imputed Dataset
    df.to_csv(OUTPUT_CSV, index=False)
    with engine.begin() as conn:
        df.to_sql("canonical_market_prices_imputed", con=conn, if_exists="replace", index=False)

    final_missing = df['modal_price_per_quintal'].isna().sum()

    print("\n=== Imputation Completed Successfully ===")
    print(f"Original Missing: {initial_missing}")
    print(f"Remaining Missing: {final_missing}")
    print(f"Total Imputed Records Flagged: {df['is_imputed'].sum()}")
    print(f"Imputed SQLite Table: 'canonical_market_prices_imputed'")
    print(f"Imputed CSV Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_imputation_pipeline()