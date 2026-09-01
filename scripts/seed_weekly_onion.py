import sys
from pathlib import Path
from datetime import date
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import engine
from backend.app.models import Crop, MarketPriceWeekly


WEEK_START = date(2026, 1, 1)
WEEK_END = date(2026, 1, 8)


WEEKLY_PRICES = [
    {
        "market_name": "Kurichi(Uzhavar Sandhai)",
        "average_price": 3052.17,
        "previous_week": 3316.22,
        "previous_month": 2525.00,
        "previous_year": 4033.33,
        "change_week": -8.0,
        "change_month": 20.9,
        "change_year": -24.3,
    },
    {
        "market_name": "Mettupalayam(Uzhavar Sandhai)",
        "average_price": 3225.15,
        "previous_week": 3366.96,
        "previous_month": 2781.43,
        "previous_year": 4575.34,
        "change_week": -4.2,
        "change_month": 15.9,
        "change_year": -29.5,
    },
    {
        "market_name": "Pollachi(Uzhavar Sandhai)",
        "average_price": 3598.20,
        "previous_week": 3750.00,
        "previous_month": 3081.08,
        "previous_year": 5500.00,
        "change_week": -4.0,
        "change_month": 16.8,
        "change_year": -34.6,
    },
    {
        "market_name": "RSPuram(Uzhavar Sandhai)",
        "average_price": 2861.68,
        "previous_week": 3132.31,
        "previous_month": 2520.12,
        "previous_year": 4126.92,
        "change_week": -8.6,
        "change_month": 13.6,
        "change_year": -30.7,
    },
    {
        "market_name": "Singanallur(Uzhavar Sandhai)",
        "average_price": 3099.14,
        "previous_week": 3282.70,
        "previous_month": 2535.60,
        "previous_year": 4346.91,
        "change_week": -5.6,
        "change_month": 22.2,
        "change_year": -28.7,
    },
    {
        "market_name": "Sulur(Uzhavar Sandhai)",
        "average_price": 2846.97,
        "previous_week": 3257.67,
        "previous_month": 2646.96,
        "previous_year": 4350.85,
        "change_week": -12.6,
        "change_month": 7.6,
        "change_year": -34.6,
    },
    {
        "market_name": "Sundarapuram(Uzhavar Sandhai)",
        "average_price": 3027.82,
        "previous_week": 3340.73,
        "previous_month": 2528.05,
        "previous_year": 4114.73,
        "change_week": -9.4,
        "change_month": 19.8,
        "change_year": -26.4,
    },
    {
        "market_name": "Udumalpet APMC",
        "average_price": 3700.95,
        "previous_week": 3643.22,
        "previous_month": 3062.13,
        "previous_year": 5000.00,
        "change_week": 1.6,
        "change_month": 20.9,
        "change_year": -26.0,
    },
    {
        "market_name": "Vadavalli(Uzhavar Sandhai)",
        "average_price": 2816.14,
        "previous_week": 3132.74,
        "previous_month": 2450.20,
        "previous_year": 4127.04,
        "change_week": -10.1,
        "change_month": 14.9,
        "change_year": -31.8,
    },
]


with Session(engine) as session:
    onion = session.scalar(
        select(Crop).where(Crop.name == "Onion")
    )

    if onion is None:
        raise RuntimeError("Onion crop not found in crops table.")

    print(f"Onion ID: {onion.id}")

    added = 0
    skipped = 0

    for data in WEEKLY_PRICES:
        existing = session.scalar(
            select(MarketPriceWeekly).where(
                MarketPriceWeekly.crop_id == onion.id,
                MarketPriceWeekly.market_name == data["market_name"],
                MarketPriceWeekly.week_start == WEEK_START,
            )
        )

        if existing:
            print(f"Already exists: {data['market_name']}")
            skipped += 1
            continue

        record = MarketPriceWeekly(
            crop_id=onion.id,
            market_name=data["market_name"],
            district="Coimbatore",
            week_start=WEEK_START,
            week_end=WEEK_END,
            average_price_per_quintal=Decimal(str(data["average_price"])),
            previous_week_price=Decimal(str(data["previous_week"])),
            previous_month_price=Decimal(str(data["previous_month"])),
            previous_year_price=Decimal(str(data["previous_year"])),
            change_over_previous_week_pct=Decimal(str(data["change_week"])),
            change_over_previous_month_pct=Decimal(str(data["change_month"])),
            change_over_previous_year_pct=Decimal(str(data["change_year"])),
        )

        session.add(record)
        added += 1
        print(f"Added: {data['market_name']}")

    session.commit()

    print()
    print(f"Done. Added: {added}")
    print(f"Skipped: {skipped}")