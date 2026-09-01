from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/api/v1/forecasts",
    tags=["Forecasts"],
)

PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "agmarknet"
    / "model_results"
    / "adaptive_predictions.csv"
)


@router.get("/{district}")
def get_latest_district_forecast(district: str):
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Forecast predictions not generated yet.",
        )

    df = pd.read_csv(PREDICTIONS_PATH)
    district_match = df[
        df["district"].astype(str).str.casefold() == district.casefold()
    ]

    if district_match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for district: {district}",
        )

    latest_record = district_match.iloc[-1]

    return {
        "status": "success",
        "district": latest_record["district"],
        "current_price": float(latest_record["current_price"]),
        "forecast": {
            "predicted_price": float(latest_record["predicted_price"]),
            "lower_bound": float(latest_record["lower_price"]),
            "upper_bound": float(latest_record["upper_price"]),
            "predicted_pct_change": round(
                float(latest_record["predicted_pct_change"]),
                4,
            ),
        },
        "regime": latest_record["volatility_regime"],
        "spike_alert": bool(latest_record["spike_alert_flag"]),
    }
