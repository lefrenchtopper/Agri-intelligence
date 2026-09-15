import os
from dotenv import load_dotenv

load_dotenv()

# Uses DATABASE_URL from your .env file if available; otherwise falls back to local PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/agri_db"
)