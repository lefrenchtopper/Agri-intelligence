from sqlalchemy import create_engine

from .config import DATABASE_URL


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the .env file")


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)