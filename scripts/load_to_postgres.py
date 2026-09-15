import os
import pandas as pd
from sqlalchemy import create_engine, text

# Environment config
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agri_intelligence")

CSV_FILE = os.path.join("data", "processed", "canonical_market_prices_weekly.csv")
SQLITE_DB = os.path.join("data", "processed", "agri_intelligence.db")

def init_db_and_load():
    if not os.path.exists(CSV_FILE):
        print(f"Error: Processed CSV not found at {CSV_FILE}")
        return

    # Choose Database Engine
    if USE_POSTGRES:
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print("Connecting to PostgreSQL...")
    else:
        db_url = f"sqlite:///{SQLITE_DB}"
        print(f"Connecting to SQLite database at {SQLITE_DB}...")

    engine = create_engine(db_url)

    # Load Dataset
    print(f"Reading dataset from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    print(f"Inserting {len(df)} records into table 'canonical_market_prices'...")
    with engine.begin() as conn:
        df.to_sql(
            "canonical_market_prices",
            con=conn,
            if_exists="replace",  # Ensures clean table creation
            index=False
        )

    print("=== Bulk Migration Completed Successfully ===")

if __name__ == "__main__":
    init_db_and_load()