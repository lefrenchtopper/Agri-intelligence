import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Numeric, String, UniqueConstraint, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class MarketPriceWeekly(Base):
    __tablename__ = "market_price_weekly"

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

    week_start: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    week_end: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    average_price_per_quintal: Mapped[float] = mapped_column(
        Numeric(8, 2),
        nullable=False,
    )

    previous_week_price: Mapped[float | None] = mapped_column(
        Numeric(8, 2),
        nullable=True,
    )

    previous_month_price: Mapped[float | None] = mapped_column(
        Numeric(8, 2),
        nullable=True,
    )

    previous_year_price: Mapped[float | None] = mapped_column(
        Numeric(8, 2),
        nullable=True,
    )

    change_over_previous_week_pct: Mapped[float | None] = mapped_column(
        Numeric(6, 2),
        nullable=True,
    )

    change_over_previous_month_pct: Mapped[float | None] = mapped_column(
        Numeric(6, 2),
        nullable=True,
    )

    change_over_previous_year_pct: Mapped[float | None] = mapped_column(
        Numeric(6, 2),
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
            "week_start",
            name="unique_weekly_market_crop",
        ),
    )