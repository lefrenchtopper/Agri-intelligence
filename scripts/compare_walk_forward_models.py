import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.adaptive_forecaster import (
    AdaptiveRegimeForecaster,
    fit_conformal_interval,
    make_pipeline,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "agmarknet" / "features" / "onion_tamilnadu_forecasting_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "agmarknet" / "model_results"
RESULTS_PATH = OUTPUT_DIR / "walk_forward_model_comparison.csv"
AUDIT_PATH = OUTPUT_DIR / "walk_forward_audit_table.md"
TARGET = "target_next_week_price"
VOLATILITY_THRESHOLD = 0.15
CONFORMAL_MULTIPLIER = 1.5


def make_baseline_pipelines(X):
    return {
        "Linear Regression": make_pipeline(Ridge(alpha=1.0), X),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            X,
        ),
        "HistGradientBoosting": make_pipeline(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                loss="absolute_error",
                random_state=42,
            ),
            X,
        ),
    }


def metrics(frame):
    if frame.empty:
        return np.nan, np.nan
    return (
        mean_absolute_error(frame["actual"], frame["prediction"]),
        np.sqrt(mean_squared_error(frame["actual"], frame["prediction"])),
    )


def main():
    df = pd.read_csv(FEATURES_PATH).sort_values(
        ["time_index", "district"]
    ).reset_index(drop=True)
    required = {
        TARGET,
        "current_price",
        "time_index",
        "district",
        "relative_volatility_7w",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    spatial_columns = {
        "nashik_price_lag1",
        "nashik_price_lag2",
        "nashik_momentum_2w",
    }
    if spatial_columns.intersection(df.columns):
        missing_spatial = sorted(spatial_columns - set(df.columns))
        if missing_spatial:
            raise ValueError(
                "Spatial benchmark requires all upstream features; "
                f"missing: {missing_spatial}"
            )

    feature_columns = [
        column for column in df.columns
        if column not in {TARGET, "target_next_week_change"}
    ]
    periods = sorted(df["time_index"].dropna().unique())
    all_rows = []

    for test_period in periods[10:]:
        train_df = df[df["time_index"] < test_period].copy()
        test_df = df[df["time_index"] == test_period].copy()
        X_train = train_df[feature_columns]
        X_test = test_df[feature_columns]
        y_train = train_df[TARGET]
        y_test = test_df[TARGET].to_numpy()

        predictions = {
            "Naive (Lag-1)": test_df["current_price"].to_numpy()
        }
        for name, pipeline in make_baseline_pipelines(X_train).items():
            pipeline.fit(X_train, y_train)
            predictions[name] = pipeline.predict(X_test)

        adaptive = AdaptiveRegimeForecaster(
            volatility_threshold=VOLATILITY_THRESHOLD
        )
        adaptive.fit(X_train, y_train)
        predictions["Adaptive Forecaster"] = adaptive.predict(X_test)

        calibrated_interval = fit_conformal_interval(
            X_train,
            y_train,
            X_test,
            X_train,
            volatility_threshold=VOLATILITY_THRESHOLD,
            calibration_multiplier=CONFORMAL_MULTIPLIER,
        )
        predictions_lower = calibrated_interval[0]
        predictions_upper = calibrated_interval[1]
        conformity_quantile = calibrated_interval[2]

        for index, (_, row) in enumerate(test_df.iterrows()):
            volatility = row["relative_volatility_7w"]
            for name, values in predictions.items():
                all_rows.append(
                    {
                        "time_index": test_period,
                        "district": row["district"],
                        "model": name,
                        "actual": y_test[index],
                        "prediction": values[index],
                        "lower_bound": predictions_lower[index] if name == "Adaptive Forecaster" else np.nan,
                        "upper_bound": predictions_upper[index] if name == "Adaptive Forecaster" else np.nan,
                        "conformity_quantile": conformity_quantile if name == "Adaptive Forecaster" else np.nan,
                        "regime": "shock" if volatility > VOLATILITY_THRESHOLD else "normal",
                    }
                )

    results = pd.DataFrame(all_rows)
    rows = []
    for name in results["model"].unique():
        model_df = results[results["model"] == name]
        normal = model_df[model_df["regime"] == "normal"]
        shock = model_df[model_df["regime"] == "shock"]
        spike = model_df[model_df["time_index"].isin([29, 30])]
        overall_mae, overall_rmse = metrics(model_df)
        normal_mae, normal_rmse = metrics(normal)
        shock_mae, shock_rmse = metrics(shock)
        spike_mae, spike_rmse = metrics(spike)
        if name == "Adaptive Forecaster":
            coverage = (
                (model_df["actual"] >= model_df["lower_bound"])
                & (model_df["actual"] <= model_df["upper_bound"])
            ).mean() * 100
        else:
            coverage = np.nan
        rows.append(
            {
                "model": name,
                "overall_mae": overall_mae,
                "overall_rmse": overall_rmse,
                "normal_regime_mae": normal_mae,
                "normal_regime_rmse": normal_rmse,
                "shock_regime_mae": shock_mae,
                "shock_regime_rmse": shock_rmse,
                "spike_mae": spike_mae,
                "spike_rmse": spike_rmse,
                "coverage_10_90_percent": coverage,
            }
        )

    summary = pd.DataFrame(rows).set_index("model").reindex(
        [
            "Naive (Lag-1)",
            "Linear Regression",
            "Random Forest",
            "HistGradientBoosting",
            "Adaptive Forecaster",
        ]
    ).reset_index()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "walk_forward_model_period_comparison.csv", index=False)
    summary.to_csv(RESULTS_PATH, index=False)

    markdown = [
        "| Model | Overall MAE | Normal Regime MAE | Shock Regime MAE | Spike MAE (index 29-30) | 10-90% Coverage Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        coverage = "N/A" if pd.isna(row["coverage_10_90_percent"]) else f"{row['coverage_10_90_percent']:.2f}%"
        markdown.append(
            f"| {row['model']} | {row['overall_mae']:.2f} | {row['normal_regime_mae']:.2f} | {row['shock_regime_mae']:.2f} | {row['spike_mae']:.2f} | {coverage} |"
        )
    AUDIT_PATH.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Audit table: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
