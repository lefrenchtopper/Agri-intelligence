from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .database import engine
from .models import Crop, MarketPriceWeekly
from .routers import forecasts


app = FastAPI(
    title="Agri-Intelligence",
    description="Agricultural decision-support system for Coimbatore",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecasts.router)


# ---------------------------------------------------------
# CROPS
# ---------------------------------------------------------

@app.get("/api/crops")
def get_crops():
    with Session(engine) as session:
        crops = session.scalars(
            select(Crop)
        ).all()

        return [
            {
                "id": str(crop.id),
                "name": crop.name,
                "tamil_name": crop.tamil_name,
                "category": crop.category,
                "growing_days_min": crop.growing_days_min,
                "growing_days_max": crop.growing_days_max,
            }
            for crop in crops
        ]


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "project": "Agri-Intelligence",
        "version": "0.1.0",
    }


# ---------------------------------------------------------
# DATABASE TEST
# ---------------------------------------------------------

@app.get("/api/database-test")
def database_test():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()

        return {
            "database": "connected",
            "test_result": value,
        }

    except Exception as error:
        return {
            "database": "connection_failed",
            "error": str(error),
        }


# ---------------------------------------------------------
# WEEKLY MARKET PRICES
# ---------------------------------------------------------

@app.get("/api/market-prices/weekly")
def get_weekly_market_prices(
    crop_id: str | None = None,
    district: str | None = None,
    market_name: str | None = None,
    year: int | None = None,
    month: int | None = None,
    week: int | None = None,
):
    with Session(engine) as session:

        query = select(MarketPriceWeekly)

        # Crop
        if crop_id:
            query = query.where(
                MarketPriceWeekly.crop_id == crop_id
            )

        # District
        if district:
            query = query.where(
                MarketPriceWeekly.district == district
            )

        # Market
        if market_name:
            query = query.where(
                MarketPriceWeekly.market_name == market_name
            )

        # -------------------------------------------------
        # YEAR FILTER
        # -------------------------------------------------

        if year:
            query = query.where(
                MarketPriceWeekly.week_start >= date(year, 1, 1),
                MarketPriceWeekly.week_start < date(year + 1, 1, 1),
            )

        # -------------------------------------------------
        # MONTH FILTER
        # -------------------------------------------------

        if year and month:
            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)

            query = query.where(
                MarketPriceWeekly.week_start >= date(year, month, 1),
                MarketPriceWeekly.week_start < next_month,
            )

        # -------------------------------------------------
        # WEEK FILTER
        # -------------------------------------------------
        #
        # Our current data uses periods such as:
        #
        # Week 1 → 1–8 January
        # Week 2 → 8–15 January
        # Week 3 → 15–22 January
        # ...
        #
        # The database itself remains date-based.
        # We derive the week number from the month.
        # -------------------------------------------------

        if year and month and week:

            week_start_day = 1 + ((week - 1) * 7)

            if week_start_day <= 31:

                week_start = date(
                    year,
                    month,
                    week_start_day,
                )

                # Calculate the next boundary
                if month == 12:
                    next_month = date(year + 1, 1, 1)
                else:
                    next_month = date(year, month + 1, 1)

                candidate_end_day = week_start_day + 7

                if candidate_end_day <= 28:
                    week_end = date(
                        year,
                        month,
                        candidate_end_day,
                    )
                else:
                    week_end = next_month

                query = query.where(
                    MarketPriceWeekly.week_start >= week_start,
                    MarketPriceWeekly.week_start < week_end,
                )

        prices = session.scalars(
            query.order_by(
                MarketPriceWeekly.week_start,
                MarketPriceWeekly.market_name,
            )
        ).all()

        return [
            {
                "id": str(price.id),
                "crop_id": str(price.crop_id),

                "market_name": price.market_name,
                "district": price.district,

                "week_start": price.week_start,
                "week_end": price.week_end,

                # IMPORTANT:
                # Current database values are ₹ per quintal.
                "price_unit": "₹ per quintal",
                "quantity_unit": "quintal",

                "average_price_per_quintal": float(
                    price.average_price_per_quintal
                ),

                "previous_week_price": (
                    float(price.previous_week_price)
                    if price.previous_week_price is not None
                    else None
                ),

                "previous_month_price": (
                    float(price.previous_month_price)
                    if price.previous_month_price is not None
                    else None
                ),

                "previous_year_price": (
                    float(price.previous_year_price)
                    if price.previous_year_price is not None
                    else None
                ),

                "change_over_previous_week_pct": (
                    float(price.change_over_previous_week_pct)
                    if price.change_over_previous_week_pct is not None
                    else None
                ),

                "change_over_previous_month_pct": (
                    float(price.change_over_previous_month_pct)
                    if price.change_over_previous_month_pct is not None
                    else None
                ),

                "change_over_previous_year_pct": (
                    float(price.change_over_previous_year_pct)
                    if price.change_over_previous_year_pct is not None
                    else None
                ),
            }
            for price in prices
        ]


# ---------------------------------------------------------
# AVAILABLE TIME OPTIONS
# ---------------------------------------------------------

@app.get("/api/market-prices/weeks")
def get_available_weeks(
    crop_id: str | None = None,
    district: str | None = None,
    year: int | None = None,
    month: int | None = None,
):
    with Session(engine) as session:

        query = select(MarketPriceWeekly)

        # Crop
        if crop_id:
            query = query.where(
                MarketPriceWeekly.crop_id == crop_id
            )

        # District
        if district:
            query = query.where(
                MarketPriceWeekly.district == district
            )

        # Year
        if year:
            query = query.where(
                MarketPriceWeekly.week_start >= date(year, 1, 1),
                MarketPriceWeekly.week_start < date(year + 1, 1, 1),
            )

        # Month
        if year and month:

            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)

            query = query.where(
                MarketPriceWeekly.week_start >= date(
                    year,
                    month,
                    1,
                ),
                MarketPriceWeekly.week_start < next_month,
            )

        prices = session.scalars(
            query.order_by(
                MarketPriceWeekly.week_start
            )
        ).all()

        seen = set()
        weeks = []

        for price in prices:

            key = (
                price.week_start,
                price.week_end,
            )

            if key in seen:
                continue

            seen.add(key)

            # Determine week number inside the month.
            week_number = ((price.week_start.day - 1) // 7) + 1

            weeks.append(
                {
                    "year": price.week_start.year,
                    "month": price.week_start.month,
                    "week": week_number,
                    "week_start": price.week_start,
                    "week_end": price.week_end,
                }
            )

        return weeks