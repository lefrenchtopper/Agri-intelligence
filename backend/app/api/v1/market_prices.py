import uuid
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, and_
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import engine
from backend.app.models.market_price import MarketPrice

router = APIRouter(prefix="/market-prices", tags=["Market Prices"])


def get_db():
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class PricePointResponse(BaseModel):
    id: uuid.UUID
    market_name: str
    district: str
    price_date: date
    min_price_per_quintal: Optional[float] = None
    max_price_per_quintal: Optional[float] = None
    modal_price_per_quintal: float
    arrival_quantity_quintals: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class PriceTrendResponse(BaseModel):
    crop_id: uuid.UUID
    district: Optional[str] = None
    total_records: int
    data: List[PricePointResponse]


@router.get("/trends", response_model=PriceTrendResponse, status_code=status.HTTP_200_OK)
def get_historical_price_trends(
    crop_id: uuid.UUID = Query(..., description="UUID of the crop"),
    district: Optional[str] = Query(None, description="Filter by district name"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    filters = [MarketPrice.crop_id == crop_id]

    if district:
        filters.append(MarketPrice.district.ilike(f"%{district.strip()}%"))
    if start_date:
        filters.append(MarketPrice.price_date >= start_date)
    if end_date:
        filters.append(MarketPrice.price_date <= end_date)

    query = (
        select(MarketPrice)
        .where(and_(*filters))
        .order_by(MarketPrice.price_date.asc())
        .limit(limit)
    )

    records = db.scalars(query).all()

    return PriceTrendResponse(
        crop_id=crop_id,
        district=district,
        total_records=len(records),
        data=records,
    )