import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.adaptive_forecaster import (
    AdaptiveRegimeForecaster,
    fit_conformal_interval,
    make_pipeline,
    QuantileResidualRegressor,
)


FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)
MODEL_PATH = BASE_DIR / "models" / "agmarknet" / "onion_price_adaptive_v1.joblib"
TARGET = "target_next_week_price"
TARGET_COLUMNS = [TARGET, "target_next_week_change"]


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_columns = [column for column in df.columns if column not in TARGET_COLUMNS]
    X = df[feature_columns]
    y = df[TARGET]

    normal_model = make_pipeline(QuantileResidualRegressor(alpha=10.0), X)
    shock_model = make_pipeline(QuantileResidualRegressor(alpha=0.1), X)
    forecaster = AdaptiveRegimeForecaster(
        normal_model=normal_model,
        shock_model=shock_model,
        volatility_threshold=0.15,
    )
    forecaster.fit(X, y)

    _, _, conformity_quantile, lower_pipeline, upper_pipeline = (
        fit_conformal_interval(
            X,
            y,
            X,
            X,
            volatility_threshold=0.15,
            calibration_multiplier=1.5,
            return_pipelines=True,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "forecaster": forecaster,
            "lower_pipeline": lower_pipeline,
            "upper_pipeline": upper_pipeline,
            "features": feature_columns,
            "target": TARGET,
            "volatility_threshold": 0.15,
            "conformity_quantile": conformity_quantile,
            "calibration_fraction": 0.20,
            "calibration_multiplier": 1.5,
            "normal_lower_multiplier": 0.70,
            "normal_upper_multiplier": 1.45,
            "shock_lower_multiplier": 0.80,
            "shock_upper_multiplier": 2.40,
        },
        MODEL_PATH,
    )

    point = forecaster.predict(X)
    lower = X["current_price"].to_numpy() + lower_pipeline.predict(X)
    upper = X["current_price"].to_numpy() + upper_pipeline.predict(X)
    print(f"Saved: {MODEL_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Mean interval width: {np.mean(upper - lower):.2f}")
    print(f"Point range: {point.min():.2f} - {point.max():.2f}")


if __name__ == "__main__":
    main()
