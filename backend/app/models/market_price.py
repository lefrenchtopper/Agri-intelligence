import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Numeric,
    String,
    UniqueConstraint,
    ForeignKey,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class MarketPrice(Base):
    __tablename__ = "market_prices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    crop_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("crops.id", ondelete="CASCADE"),
        nullable=False,
    )

    market_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    district: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="Coimbatore",
    )

    price_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    min_price_per_quintal: Mapped[float | None] = mapped_column(
        Numeric(8, 2),
        nullable=True,
    )

    max_price_per_quintal: Mapped[float | None] = mapped_column(
        Numeric(8, 2),
        nullable=True,
    )

    modal_price_per_quintal: Mapped[float] = mapped_column(
        Numeric(8, 2),
        nullable=False,
    )

    arrival_quantity_quintals: Mapped[float | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "crop_id",
            "market_name",
            "price_date",
            name="unique_market_crop_date",
        ),
    )
