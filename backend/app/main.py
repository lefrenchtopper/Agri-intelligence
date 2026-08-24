from fastapi import FastAPI

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