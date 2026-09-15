"""
Import verified real market observations into the database.
Strictly enforces Option 2 (Pure Real Data): fails if required crops or year coverage is missing.
"""
import csv
import re
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session

# Setup Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import engine
from backend.app.models import Crop, MarketPrice

TARGET_CROPS = {
    "onion": "Onion",
    "tomato": "Tomato",
    "coconut": "Coconut",
    "rice": "Rice",
    "maize": "Maize",
    "groundnut": "Groundnut",
}

TARGET_STATE = "Tamil Nadu"
MIN_HISTORY_YEARS = 5
REQUIRED_WINDOW = {2025, 2026}
MIN_RECORDS_PER_CROP = 50

SOURCE_FILES = (
    PROJECT_ROOT / "data" / "raw" / "Tamilnadu.csv",
    PROJECT_ROOT / "data" / "raw" / "upag_onion_history.csv",
    PROJECT_ROOT / "data" / "processed" / "market_prices_clean.csv",
)

def parse_date(value: str) -> date | None:
    value = str(value).strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value[:10], fmt).date()
        except ValueError:
            continue
    return None

def parse_number(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    cleaned = re.sub(r"[^0-9.-]", "", str(value))
    try:
        val = float(cleaned)
        return val if val > 0 else None
    except ValueError:
        return None

def normalize_row(row: dict[str, str]) -> dict[str, object] | None:
    lowered = {k.strip().casefold(): v for k, v in row.items()}
    
    # State check
    state = lowered.get("state", "Tamil Nadu")
    if str(state).strip().casefold() != TARGET_STATE.casefold():
        return None

    # Crop check
    crop_raw = lowered.get("commodity") or lowered.get("crop") or ""
    crop_key = str(crop_raw).strip().casefold()
    if crop_key not in TARGET_CROPS:
        return None

    # Price & Date parsing
    price_raw = (
        lowered.get("modal_price")
        or lowered.get("mandiwholesaleprice")
        or lowered.get("modal_price_per_quintal")
        or lowered.get("price")
    )
    date_raw = lowered.get("record_date") or lowered.get("price_date") or lowered.get("date") or lowered.get("arrival_date")

    parsed_dt = parse_date(str(date_raw or ""))
    modal_p = parse_number(str(price_raw or ""))

    if not parsed_dt or modal_p is None:
        return None

    min_p = parse_number(str(lowered.get("min_price", ""))) or modal_p
    max_p = parse_number(str(lowered.get("max_price", ""))) or modal_p
    district = str(lowered.get("district") or "Unknown").strip().title()
    market = str(lowered.get("market") or lowered.get("market_name") or "Central Market").strip().title()

    # Calculate weekly bounds
    year, month = parsed_dt.year, parsed_dt.month
    week = int(parsed_dt.isocalendar().week)

    return {
        "crop": TARGET_CROPS[crop_key],
        "district": district,
        "market": market,
        "modal_price": modal_p,
        "min_price": min_p,
        "max_price": max_p,
        "year": year,
        "month": month,
        "week": week,
        "week_start": parsed_dt.strftime("%Y-%m-%d"),
        "week_end": parsed_dt.strftime("%Y-%m-%d"),
        "price_date": parsed_dt,
    }

def get_source_paths() -> list[Path]:
    paths = [p for p in SOURCE_FILES if p.exists()]
    agmarknet_dir = PROJECT_ROOT / "data" / "raw" / "agmarknet"
    if agmarknet_dir.exists():
        for p in agmarknet_dir.rglob("*.csv"):
            if p not in paths:
                paths.append(p)
    return paths

def load_observations() -> list[dict[str, object]]:
    observations = []
    for path in get_source_paths():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as src:
                reader = csv.DictReader(src)
                for row in reader:
                    obs = normalize_row(row)
                    if obs:
                        observations.append(obs)
        except Exception as e:
            print(f"⚠️ Warning skipping {path.name}: {e}")
    return observations

def validate_coverage(observations: list[dict[str, object]]) -> None:
    by_crop = {crop: [] for crop in TARGET_CROPS.values()}
    for obs in observations:
        by_crop[obs["crop"]].append(obs)

    problems = []
    for crop_name, rows in by_crop.items():
        if not rows:
            problems.append(f"{crop_name}: no real observations found in data source")
            continue

        if len(rows) < MIN_RECORDS_PER_CROP:
            problems.append(f"{crop_name}: low observation count ({len(rows)} < {MIN_RECORDS_PER_CROP})")

        years = {r["year"] for r in rows}
        missing_window = REQUIRED_WINDOW - years
        if len(years) < MIN_HISTORY_YEARS:
            problems.append(f"{crop_name}: only {len(years)} distinct years found ({sorted(years)})")
        if missing_window:
            problems.append(f"{crop_name}: missing required window years {sorted(missing_window)}")

    if problems:
        print("\n❌ Pure Real Data Validation Failed:")
        for p in problems:
            print(f"  - {p}")
        raise RuntimeError("Refusing database insert. Incomplete source coverage.")

    print("✅ Real data coverage checks passed!")

def seed_real_data(observations: list[dict[str, object]]) -> int:
    with Session(engine) as session:
        # Load or create Crops
        crops = {c.name.casefold(): c for c in session.scalars(select(Crop))}
        for crop_name in TARGET_CROPS.values():
            if crop_name.casefold() not in crops:
                new_c = Crop(id=str(uuid.uuid4()), name=crop_name)
                session.add(new_c)
                session.flush()
                crops[crop_name.casefold()] = new_c

        # Batch check existing keys to avoid N+1 queries
        existing_keys = set(
            session.execute(
                select(MarketPrice.crop_id, MarketPrice.district, MarketPrice.market, MarketPrice.week_start)
            ).all()
        )

        new_records = []
        for obs in observations:
            crop_obj = crops[obs["crop"].casefold()]
            key = (crop_obj.id, obs["district"], obs["market"], obs["week_start"])
            if key in existing_keys:
                continue
            
            existing_keys.add(key)
            new_records.append(
                MarketPrice(
                    id=str(uuid.uuid4()),
                    crop_id=crop_obj.id,
                    district=obs["district"],
                    market=obs["market"],
                    modal_price=obs["modal_price"],
                    min_price=obs["min_price"],
                    max_price=obs["max_price"],
                    year=obs["year"],
                    month=obs["month"],
                    week=obs["week"],
                    week_start=obs["week_start"],
                    week_end=obs["week_end"]
                )
            )

        session.add_all(new_records)
        session.commit()
        return len(new_records)

def main() -> None:
    observations = load_observations()
    print(f"Loaded {len(observations)} valid real observations from source files.")
    validate_coverage(observations)
    inserted = seed_real_data(observations)
    print(f"🎉 Successfully imported {inserted} new real observations into Supabase!")

if __name__ == "__main__":
    main()