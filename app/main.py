import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Agri-Intelligence API",
    description="Agricultural decision-support system for Coimbatore",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join("app", "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Run scripts/export_model.py first.")

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    year: int = Field(..., json_schema_extra={"example": 2024})
    month_num: int = Field(..., ge=1, le=12, json_schema_extra={"example": 11})
    week_num: int = Field(..., ge=1, le=5, json_schema_extra={"example": 2})
    lag_1: float = Field(..., json_schema_extra={"example": 2400.0})
    lag_2: float = Field(..., json_schema_extra={"example": 2350.0})
    lag_4: float = Field(..., json_schema_extra={"example": 2200.0})
    rolling_mean_4: float = Field(..., json_schema_extra={"example": 2300.0})
    rolling_std_4: float = Field(..., json_schema_extra={"example": 85.5})


class PredictionResponse(BaseModel):
    predicted_modal_price: float
    unit: str = "Rs./Quintal"


@app.get("/")
def root():
    return {"message": "Agri-Intelligence API is running", "status": "active"}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "project": "Agri-Intelligence", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([request.model_dump()])
        prediction = model.predict(input_data)[0]
        return PredictionResponse(predicted_modal_price=round(float(prediction), 2))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
