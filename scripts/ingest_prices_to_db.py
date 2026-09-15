import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.orm import Session
from backend.app.database import engine
from backend.app.config.mappings import get_crop_id
from backend.app.models.market_price import MarketPrice  # Import your SQLAlchemy model

# Sample payload or parsed rows
sample_data = [
    {
        "commodity": "Onion",
        "state": "Tamil Nadu",
        "district": "Madurai",
        "market_name": "Madurai Market",
        "arrival_quantity": 450.0,
        "wholesale_price": 5500.0,
        "recorded_date": date.today()
    }
]

def load_data_to_db(records):
    with Session(engine) as session:
        inserted_count = 0
        for item in records:
            crop_id = get_crop_id(item["commodity"])
            if not crop_id:
                print(f"⚠️ Unmapped commodity: {item['commodity']}")
                continue

            price_entry = MarketPrice(
                crop_id=crop_id,
                state=item["state"],
                district=item.get("district"),
                market_name=item["market_name"],
                arrival_quantity=item.get("arrival_quantity", 0.0),
                wholesale_price=item["wholesale_price"],
                recorded_date=item["recorded_date"]
            )
            session.add(price_entry)
            inserted_count += 1
        
        session.commit()
        print(f"✅ Successfully inserted {inserted_count} price records into the database.")

if __name__ == "__main__":
    load_data_to_db(sample_data)