import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DEFAULT_MODEL = BASE_DIR / "models" / "agmarknet" / "onion_price_adaptive_v1.joblib"
MODEL_RESULTS_DIR = BASE_DIR / "data" / "processed" / "agmarknet" / "model_results"
DEFAULT_OUTPUT = MODEL_RESULTS_DIR / "adaptive_predictions.csv"
TELEMETRY_PATH = MODEL_RESULTS_DIR / "forecast_history_log.csv"


def main():
    parser = argparse.ArgumentParser(description="Predict next-week onion prices.")
    parser.add_argument("input", type=Path, help="CSV containing the latest feature rows")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = bundle["features"]
    frame = pd.read_csv(args.input)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = frame[features]
    predictions = bundle["forecaster"].predict(X)
    current_price = X["current_price"].to_numpy()
    volatility_scale = (
        X["relative_volatility_7w"].to_numpy()
        / bundle["volatility_threshold"]
    ).clip(min=1.0)
    expansion = (
        bundle["conformity_quantile"]
        * bundle["calibration_multiplier"]
        * volatility_scale
    )
    lower = (
        current_price
        + bundle["lower_pipeline"].predict(X)
        - expansion
    )
    upper = (
        current_price
        + bundle["upper_pipeline"].predict(X)
        + expansion
    )

    bundle_lower = bundle.get("normal_lower_multiplier", 0.70)
    bundle_upper = bundle.get("normal_upper_multiplier", 1.45)
    shock_lower = bundle.get("shock_lower_multiplier", 0.80)
    shock_upper = bundle.get("shock_upper_multiplier", 2.40)

    symmetric_lower = (
        current_price
        + bundle["lower_pipeline"].predict(X)
        - (
            bundle["conformity_quantile"]
            * bundle["calibration_multiplier"]
            * (X["relative_volatility_7w"].to_numpy() / bundle["volatility_threshold"]).clip(min=1.0)
        )
    )
    symmetric_upper = (
        current_price
        + bundle["upper_pipeline"].predict(X)
        + (
            bundle["conformity_quantile"]
            * bundle["calibration_multiplier"]
            * (X["relative_volatility_7w"].to_numpy() / bundle["volatility_threshold"]).clip(min=1.0)
        )
    )
    center = (symmetric_lower + symmetric_upper) / 2.0
    half_width = (symmetric_upper - symmetric_lower) / 2.0
    shock_mask = X["relative_volatility_7w"].to_numpy() > bundle["volatility_threshold"]
    lower_multiplier = np.where(shock_mask, shock_lower, bundle_lower)
    upper_multiplier = np.where(shock_mask, shock_upper, bundle_upper)
    lower = center - (half_width * lower_multiplier)
    upper = center + (half_width * upper_multiplier)

    result = frame[["district", "current_price"]].copy()
    result["predicted_price"] = predictions
    result["lower_price"] = lower
    result["upper_price"] = upper
    result["volatility_regime"] = "shock"
    result.loc[
        X["relative_volatility_7w"] <= bundle["volatility_threshold"],
        "volatility_regime",
    ] = "normal"
    result["predicted_pct_change"] = (
        result["predicted_price"] / result["current_price"] - 1
    )
    result["spike_alert_flag"] = (
        result["volatility_regime"].eq("shock")
            | (result["predicted_pct_change"] >= 0.10)
    )
    result["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    telemetry_columns = [
        "timestamp",
        "district",
        "current_price",
        "predicted_price",
        "lower_price",
        "upper_price",
        "volatility_regime",
        "predicted_pct_change",
        "spike_alert_flag",
    ]
    result[telemetry_columns].to_csv(
        TELEMETRY_PATH,
        mode="a",
        header=not TELEMETRY_PATH.exists(),
        index=False,
    )

    print(f"Saved: {args.output}")
    print(f"Telemetry: {TELEMETRY_PATH}")


if __name__ == "__main__":
    main()
