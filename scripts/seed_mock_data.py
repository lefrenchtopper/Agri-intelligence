"""
DEVELOPMENT / TESTING ONLY:
Generates synthetic 2021-2026 weekly market prices for local UI testing.
"""
import sys
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

# Setup Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.orm import Session
from backend.app.database import engine
from backend.app.models import Crop, MarketPrice

# Baseline configuration per crop
CROP_CONFIGS = {
    "Onion": {"base_2021": 2200, "annual_trend": 0.06, "volatility": 0.18, "peak_months": [10, 11, 12]},
    "Tomato": {"base_2021": 1800, "annual_trend": 0.05, "volatility": 0.22, "peak_months": [6, 7, 11]},
    "Coconut": {"base_2021": 2600, "annual_trend": 0.04, "volatility": 0.06, "peak_months": [4, 5]},
    "Rice": {"base_2021": 3100, "annual_trend": 0.035, "volatility": 0.04, "peak_months": [1, 2]},
    "Maize": {"base_2021": 1750, "annual_trend": 0.045, "volatility": 0.08, "peak_months": [3, 9]},
    "Groundnut": {"base_2021": 5600, "annual_trend": 0.04, "volatility": 0.07, "peak_months": [1, 7]},
}

DISTRICTS = ["Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Chennai", "Tirunelveli"]

def generate_weekly_dates(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        w_start = current - timedelta(days=current.weekday())
        w_end = w_start + timedelta(days=6)
        week_num = int(current.isocalendar().week)
        yield current, current.year, current.month, week_num, w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")
        current += timedelta(days=7)

def seed_mock_prices():
    random.seed(42)
    start_dt = datetime(2021, 1, 1)
    end_dt = datetime(2026, 8, 31)

    model_cols = {c.name for c in MarketPrice.__table__.columns}
    market_field = "market_name" if "market_name" in model_cols else "market"
    price_field = "modal_price_per_quintal" if "modal_price_per_quintal" in model_cols else "modal_price"

    with Session(engine) as session:
        print("⚠️ [DEV ONLY] Cleaning existing market prices...")
        session.query(MarketPrice).delete()
        session.commit()

        # Ensure Crop master records exist with native UUIDs
        existing_crops = {c.name.strip().title(): c for c in session.query(Crop).all()}
        for crop_name in CROP_CONFIGS:
            if crop_name not in existing_crops:
                new_c = Crop(id=uuid.uuid4(), name=crop_name)
                session.add(new_c)
                session.flush()
                existing_crops[crop_name] = new_c

        records = []
        print("🌾 Generating continuous 2021–2026 mock prices for 6 crops...")

        for crop_name, config in CROP_CONFIGS.items():
            crop_obj = existing_crops[crop_name]
            
            for district in DISTRICTS:
                market_val = f"{district} Central Market"
                
                for dt, year, month, week, w_start, w_end in generate_weekly_dates(start_dt, end_dt):
                    years_passed = (year - 2021) + (month - 1) / 12.0
                    trend_factor = 1.0 + (config["annual_trend"] * years_passed)
                    season_mult = 1.25 if month in config["peak_months"] else 0.95
                    noise = random.uniform(1.0 - config["volatility"], 1.0 + config["volatility"])
                    
                    modal_price = round(config["base_2021"] * trend_factor * season_mult * noise, 2)
                    min_price = round(modal_price * random.uniform(0.91, 0.96), 2)
                    max_price = round(modal_price * random.uniform(1.04, 1.09), 2)

                    kwargs = {
                        "id": uuid.uuid4(),  # Pass native UUID object, not string
                        "crop_id": crop_obj.id,
                        "district": district,
                        market_field: market_val,
                        price_field: modal_price,
                    }

                    if "min_price" in model_cols:
                        kwargs["min_price"] = min_price
                    if "max_price" in model_cols:
                        kwargs["max_price"] = max_price
                    if "price_date" in model_cols:
                        kwargs["price_date"] = dt.date()
                    if "year" in model_cols:
                        kwargs["year"] = year
                    if "month" in model_cols:
                        kwargs["month"] = month
                    if "week" in model_cols:
                        kwargs["week"] = week
                    if "week_start" in model_cols:
                        kwargs["week_start"] = w_start
                    if "week_end" in model_cols:
                        kwargs["week_end"] = w_end

                    records.append(MarketPrice(**kwargs))

        session.bulk_save_objects(records)
        session.commit()
        print(f"🎉 Successfully seeded {len(records)} mock price records (2021–2026) for local testing!")

if __name__ == "__main__":
    seed_mock_prices()