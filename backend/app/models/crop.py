import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class Crop(Base):
    __tablename__ = "crops"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
    )

    tamil_name: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    category: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    growing_days_min: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    growing_days_max: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    min_temp_c: Mapped[float | None] = mapped_column(
        Numeric(4, 1),
        nullable=True,
    )

    max_temp_c: Mapped[float | None] = mapped_column(
        Numeric(4, 1),
        nullable=True,
    )

    min_rainfall_mm: Mapped[float | None] = mapped_column(
        Numeric(6, 1),
        nullable=True,
    )

    max_rainfall_mm: Mapped[float | None] = mapped_column(
        Numeric(6, 1),
        nullable=True,
    )

    water_requirement_mm: Mapped[float | None] = mapped_column(
        Numeric(6, 1),
        nullable=True,
    )

    soil_compatibility: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )