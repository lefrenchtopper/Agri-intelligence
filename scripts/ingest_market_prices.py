import sys
import uuid
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.app.database import engine
from backend.app.models.market_price import MarketPrice

# Map external commodity names to your database crop UUID string
CROP_UUID_MAP = {
    "onion": "c49a8637df4d4d399d27c7756c0ffa4f",
    "bellary onion": "c49a8637df4d4d399d27c7756c0ffa4f",
    "small onion": "c49a8637df4d4d399d27c7756c0ffa4f",
}


def parse_and_upsert_prices(records: list[dict]):
    """
    Upserts price records matching the MarketPrice SQLAlchemy model.
    Handles duplicate records using PostgreSQL ON CONFLICT DO UPDATE.
    """
    if not records:
        print("⚠️ No records provided for ingestion.")
        return

    db_rows = []

    for item in records:
        commodity_key = str(item.get("commodity", "")).strip().lower()
        crop_id_str = CROP_UUID_MAP.get(commodity_key)

        if not crop_id_str:
            print(f"⚠️ Skipping unmapped commodity: {item.get('commodity')}")
            continue

        db_rows.append({
            "crop_id": uuid.UUID(crop_id_str),
            "market_name": str(item["market_name"]).strip(),
            "district": str(item.get("district", "Coimbatore")).strip(),
            "price_date": item["price_date"],  # datetime.date object
            "min_price_per_quintal": item.get("min_price"),
            "max_price_per_quintal": item.get("max_price"),
            "modal_price_per_quintal": float(item["modal_price"]),
            "arrival_quantity_quintals": item.get("arrival_quantity"),
        })

    if not db_rows:
        print("⚠️ No valid rows ready for insertion.")
        return

    with Session(engine) as session:
        # Build PostgreSQL ON CONFLICT upsert statement
        stmt = insert(MarketPrice).values(db_rows)
        stmt = stmt.on_conflict_do_update(
            constraint="unique_market_crop_date",
            set_={
                "min_price_per_quintal": stmt.excluded.min_price_per_quintal,
                "max_price_per_quintal": stmt.excluded.max_price_per_quintal,
                "modal_price_per_quintal": stmt.excluded.modal_price_per_quintal,
                "arrival_quantity_quintals": stmt.excluded.arrival_quantity_quintals,
            },
        )

        session.execute(stmt)
        session.commit()
        print(f"✅ Successfully upserted {len(db_rows)} market price records.")


if __name__ == "__main__":
    # Test sample record
    sample_records = [
        {
            "commodity": "Onion",
            "market_name": "Madurai Market",
            "district": "Madurai",
            "price_date": date(2023, 1, 1),
            "min_price": 2000.00,
            "max_price": 2500.00,
            "modal_price": 2250.00,
            "arrival_quantity": 450.00,
        }
    ]

    parse_and_upsert_prices(sample_records)