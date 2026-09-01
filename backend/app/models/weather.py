import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class Weather(Base):
    __tablename__ = "weather"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    district: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    taluk: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    record_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    rainfall_mm: Mapped[float] = mapped_column(
        Numeric(6, 2),
        nullable=False,
        default=0.0,
    )

    temp_max_c: Mapped[float | None] = mapped_column(
        Numeric(4, 1),
        nullable=True,
    )

    temp_min_c: Mapped[float | None] = mapped_column(
        Numeric(4, 1),
        nullable=True,
    )

    humidity_pct: Mapped[float | None] = mapped_column(
        Numeric(4, 1),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "district",
            "taluk",
            "record_date",
            name="unique_weather_day_location",
        ),
    )