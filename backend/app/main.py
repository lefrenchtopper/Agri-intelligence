from fastapi import FastAPI
from sqlalchemy import text

from .database import engine


app = FastAPI(
    title="Agri-Intelligence",
    description="Agricultural decision-support system for Coimbatore",
    version="0.1.0",
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "project": "Agri-Intelligence",
        "version": "0.1.0",
    }


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