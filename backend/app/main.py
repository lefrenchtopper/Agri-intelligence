import uuid
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.api.v1.market_prices import router as market_prices_v1_router
from .database import engine
from .models import Crop, MarketPriceWeekly
from .routers import forecasts

# Initialize FastAPI ONCE with full metadata
app = FastAPI(
    title="Agri-Intelligence API",
    description="Agricultural decision-support system for Coimbatore",
    version="0.1.0",
)

MODEL_PATH = os.path.join("app", "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Run scripts/export_model.py first.")

model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    year: int = Field(..., example=2024)
    month_num: int = Field(..., ge=1, le=12, example=11)
    week_num: int = Field(..., ge=1, le=5, example=2)
    lag_1: float = Field(..., example=2400.0)
    lag_2: float = Field(..., example=2350.0)
    lag_4: float = Field(..., example=2200.0)
    rolling_mean_4: float = Field(..., example=2300.0)
    rolling_std_4: float = Field(..., example=85.5)

class PredictionResponse(BaseModel):
    predicted_modal_price: float
    unit: str = "Rs./Quintal"

@app.get("/")
def health_check():
    return {"status": "active", "service": "Agri-Intelligence API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([request.model_dump()])
        prediction = model.predict(input_data)[0]
        return PredictionResponse(predicted_modal_price=round(float(prediction), 2))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Configure CORS Middleware on the app instance
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(market_prices_v1_router, prefix="/api/v1")
app.include_router(forecasts.router)


# ---------------------------------------------------------
# ROOT & HEALTH
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Agri-Intelligence API is running"}


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
    try:
        parsed_crop_id = uuid.UUID(crop_id) if crop_id else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid crop_id") from exc

    with Session(engine) as session:

        query = select(MarketPriceWeekly)

        if parsed_crop_id:
            query = query.where(MarketPriceWeekly.crop_id == parsed_crop_id)

        if district:
            query = query.where(MarketPriceWeekly.district == district)

        if market_name:
            query = query.where(MarketPriceWeekly.market_name == market_name)

        if year:
            query = query.where(
                MarketPriceWeekly.week_start >= date(year, 1, 1),
                MarketPriceWeekly.week_start < date(year + 1, 1, 1),
            )

        if year and month:
            next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            query = query.where(
                MarketPriceWeekly.week_start >= date(year, month, 1),
                MarketPriceWeekly.week_start < next_month,
            )

        if year and month and week:
            week_start_day = 1 + ((week - 1) * 7)

            if week_start_day <= 31:
                week_start = date(year, month, week_start_day)
                next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
                candidate_end_day = week_start_day + 7

                week_end = (
                    date(year, month, candidate_end_day)
                    if candidate_end_day <= 28
                    else next_month
                )

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
                "price_unit": "₹ per quintal",
                "quantity_unit": "quintal",
                "average_price_per_quintal": float(price.average_price_per_quintal),
                "previous_week_price": float(price.previous_week_price) if price.previous_week_price is not None else None,
                "previous_month_price": float(price.previous_month_price) if price.previous_month_price is not None else None,
                "previous_year_price": float(price.previous_year_price) if price.previous_year_price is not None else None,
                "change_over_previous_week_pct": float(price.change_over_previous_week_pct) if price.change_over_previous_week_pct is not None else None,
                "change_over_previous_month_pct": float(price.change_over_previous_month_pct) if price.change_over_previous_month_pct is not None else None,
                "change_over_previous_year_pct": float(price.change_over_previous_year_pct) if price.change_over_previous_year_pct is not None else None,
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
    try:
        parsed_crop_id = uuid.UUID(crop_id) if crop_id else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid crop_id") from exc

    with Session(engine) as session:

        query = select(MarketPriceWeekly)

        if parsed_crop_id:
            query = query.where(MarketPriceWeekly.crop_id == parsed_crop_id)

        if district:
            query = query.where(MarketPriceWeekly.district == district)

        if year:
            query = query.where(
                MarketPriceWeekly.week_start >= date(year, 1, 1),
                MarketPriceWeekly.week_start < date(year + 1, 1, 1),
            )

        if year and month:
            next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            query = query.where(
                MarketPriceWeekly.week_start >= date(year, month, 1),
                MarketPriceWeekly.week_start < next_month,
            )

        prices = session.scalars(
            query.order_by(MarketPriceWeekly.week_start)
        ).all()

        seen = set()
        weeks = []

        for price in prices:
            key = (price.week_start, price.week_end)
            if key in seen:
                continue
            seen.add(key)

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