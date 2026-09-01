from pathlib import Path
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "agmarknet"
    / "onion_price_model.pkl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "model_results"
)

PREDICTIONS_PATH = OUTPUT_DIR / "walk_forward_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "walk_forward_summary.csv"


def load_model(path):
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    try:
        model = joblib.load(path)
        print("Model loaded successfully with joblib:", type(model).__name__)
        return model
    except Exception as joblib_error:
        print("joblib loading failed.")
        print("Trying pickle fallback...")

        try:
            with open(path, "rb") as f:
                model = pickle.load(f)

            print("Model loaded successfully with pickle:", type(model).__name__)
            return model

        except Exception as pickle_error:
            print()
            print("Could not load the saved model.")
            print()
            print("joblib error:")
            print(joblib_error)
            print()
            print("pickle error:")
            print(pickle_error)
            raise


def identify_columns(df):
    target_candidates = [
        "target_next_week_price",
        "next_week_price",
        "target",
    ]

    target_column = None

    for column in target_candidates:
        if column in df.columns:
            target_column = column
            break

    if target_column is None:
        raise ValueError(
            "Could not identify the target column. "
            f"Available columns: {list(df.columns)}"
        )

    if "time_index" not in df.columns:
        raise ValueError("time_index column is required.")

    if "district" not in df.columns:
        raise ValueError("district column is required.")

    return target_column


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    non_zero = actual != 0

    if non_zero.any():
        mape = (
            np.mean(
                np.abs(
                    (actual[non_zero] - predicted[non_zero])
                    / actual[non_zero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return mae, rmse, mape


def main():

    print("=" * 70)
    print("AGMARKNET WALK-FORWARD BACKTEST")
    print("=" * 70)

    print(f"Features : {FEATURES_PATH}")
    print(f"Model    : {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURES_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    df = pd.read_csv(FEATURES_PATH)

    print(f"Rows     : {len(df)}")
    print(f"Cols     : {len(df.columns)}")
    print()

    template_model = load_model(MODEL_PATH)

    target_column = identify_columns(df)

    print()
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    print(f"Target column : {target_column}")

    df = df.sort_values(
        ["time_index", "district"]
    ).reset_index(drop=True)

    periods = sorted(
        df["time_index"].dropna().unique()
    )

    print(f"Time periods  : {len(periods)}")
    print(f"First period  : {periods[0]}")
    print(f"Last period   : {periods[-1]}")

    print()

    required_columns = [
        "time_index",
        "district",
        "current_price",
        target_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in {
            target_column,
            "target_next_week_change",
        }
    ]

    feature_columns = [
        column
        for column in feature_columns
        if column != "split"
    ]

    print(f"Feature columns : {len(feature_columns)}")

    print()
    print("=" * 70)
    print("WALK-FORWARD SETUP")
    print("=" * 70)

    min_train_periods = 10

    test_periods = periods[min_train_periods:]

    print(
        f"Minimum training periods : {min_train_periods}"
    )

    print(
        f"Walk-forward test periods : "
        f"{test_periods[0]} - {test_periods[-1]}"
    )

    print(
        f"Number of test periods    : "
        f"{len(test_periods)}"
    )

    print()

    all_predictions = []

    for test_period in test_periods:

        train_periods = [
            period
            for period in periods
            if period < test_period
        ]

        train_df = df[
            df["time_index"].isin(train_periods)
        ].copy()

        test_df = df[
            df["time_index"] == test_period
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        try:
            model = clone(template_model)
        except Exception:
            model = joblib.load(MODEL_PATH)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        result = test_df[
            [
                "time_index",
                "year",
                "month",
                "week",
                "district",
                "current_price",
                target_column,
            ]
        ].copy()

        result["prediction"] = predictions

        result["baseline_prediction"] = (
            result["current_price"]
        )

        result["prediction_error"] = (
            result[target_column]
            - result["prediction"]
        )

        result["absolute_error"] = (
            np.abs(result["prediction_error"])
        )

        result["baseline_error"] = (
            result[target_column]
            - result["baseline_prediction"]
        )

        result["baseline_absolute_error"] = (
            np.abs(result["baseline_error"])
        )

        result["model_direction"] = np.where(
            result["prediction"]
            > result["current_price"],
            1,
            0,
        )

        result["actual_direction"] = np.where(
            result[target_column]
            > result["current_price"],
            1,
            0,
        )

        result["direction_correct"] = (
            result["model_direction"]
            == result["actual_direction"]
        )

        result["train_start_period"] = (
            min(train_periods)
        )

        result["train_end_period"] = (
            max(train_periods)
        )

        all_predictions.append(result)

        mae, rmse, mape = calculate_metrics(
            y_test,
            predictions,
        )

        baseline_mae = mean_absolute_error(
            y_test,
            test_df["current_price"],
        )

        direction_accuracy = (
            result["direction_correct"].mean()
            * 100
        )

        print(
            f"Period {test_period:>2} | "
            f"Train {min(train_periods):>2}-{max(train_periods):>2} | "
            f"Rows {len(test_df):>3} | "
            f"MAE ₹{mae:>7.2f} | "
            f"Baseline ₹{baseline_mae:>7.2f} | "
            f"Direction {direction_accuracy:>6.2f}%"
        )

    if not all_predictions:
        raise RuntimeError(
            "No walk-forward predictions were generated."
        )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    print()
    print("=" * 70)
    print("OVERALL WALK-FORWARD RESULTS")
    print("=" * 70)

    actual = predictions_df[target_column]
    predicted = predictions_df["prediction"]
    baseline = predictions_df["baseline_prediction"]

    model_mae, model_rmse, model_mape = calculate_metrics(
        actual,
        predicted,
    )

    baseline_mae, baseline_rmse, baseline_mape = calculate_metrics(
        actual,
        baseline,
    )

    direction_accuracy = (
        predictions_df["direction_correct"].mean()
        * 100
    )

    mae_improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
        * 100
    )

    mape_improvement = (
        (baseline_mape - model_mape)
        / baseline_mape
        * 100
        if baseline_mape != 0
        else np.nan
    )

    print()
    print("MODEL")
    print(f"MAE  : ₹{model_mae:.2f}")
    print(f"RMSE : ₹{model_rmse:.2f}")
    print(f"MAPE : {model_mape:.2f}%")

    print()
    print("NAIVE BASELINE")
    print("Strategy: next week price = current week price")
    print(f"MAE  : ₹{baseline_mae:.2f}")
    print(f"RMSE : ₹{baseline_rmse:.2f}")
    print(f"MAPE : {baseline_mape:.2f}%")

    print()
    print("COMPARISON")
    print(
        f"MAE improvement  : {mae_improvement:.2f}%"
    )
    print(
        f"MAPE improvement : {mape_improvement:.2f}%"
    )

    if mae_improvement > 0:
        print(
            "Result: Model beats the naive baseline."
        )
    else:
        print(
            "Result: Model does NOT beat the naive baseline."
        )

    print()
    print("DIRECTIONAL ACCURACY")
    print(
        f"Accuracy : {direction_accuracy:.2f}%"
    )

    print()
    print("=" * 70)
    print("PERIOD-BY-PERIOD PERFORMANCE")
    print("=" * 70)

    period_summary = (
        predictions_df
        .groupby("time_index")
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "mae": np.mean(
                        np.abs(
                            group[target_column]
                            - group["prediction"]
                        )
                    ),
                    "baseline_mae": np.mean(
                        np.abs(
                            group[target_column]
                            - group["baseline_prediction"]
                        )
                    ),
                    "directional_accuracy": (
                        group["direction_correct"].mean()
                        * 100
                    ),
                    "actual_mean_price": (
                        group[target_column].mean()
                    ),
                    "predicted_mean_price": (
                        group["prediction"].mean()
                    ),
                    "current_mean_price": (
                        group["current_price"].mean()
                    ),
                }
            )
        )
        .reset_index()
    )

    print(
        period_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 70)
    print("BIAS ANALYSIS")
    print("=" * 70)

    errors = (
        predictions_df[target_column]
        - predictions_df["prediction"]
    )

    mean_error = errors.mean()
    median_error = errors.median()

    mean_prediction = (
        predictions_df["prediction"].mean()
    )

    mean_actual = (
        predictions_df[target_column].mean()
    )

    prediction_bias_percent = (
        (mean_prediction - mean_actual)
        / mean_actual
        * 100
    )

    print(
        f"Mean prediction error : ₹{mean_error:.2f}"
    )

    print(
        f"Median prediction error : ₹{median_error:.2f}"
    )

    print(
        f"Mean actual price : ₹{mean_actual:.2f}"
    )

    print(
        f"Mean predicted price : ₹{mean_prediction:.2f}"
    )

    print(
        f"Prediction bias : {prediction_bias_percent:.2f}%"
    )

    if prediction_bias_percent < 0:
        print(
            "Interpretation: Model systematically predicts lower prices."
        )
    elif prediction_bias_percent > 0:
        print(
            "Interpretation: Model systematically predicts higher prices."
        )
    else:
        print(
            "Interpretation: No material average prediction bias."
        )

    print()
    print("=" * 70)
    print("DISTRICT WALK-FORWARD PERFORMANCE")
    print("=" * 70)

    district_summary = (
        predictions_df
        .groupby("district")
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "mae": np.mean(
                        np.abs(
                            group[target_column]
                            - group["prediction"]
                        )
                    ),
                    "rmse": np.sqrt(
                        np.mean(
                            (
                                group[target_column]
                                - group["prediction"]
                            )
                            ** 2
                        )
                    ),
                    "baseline_mae": np.mean(
                        np.abs(
                            group[target_column]
                            - group["baseline_prediction"]
                        )
                    ),
                    "directional_accuracy": (
                        group["direction_correct"].mean()
                        * 100
                    ),
                }
            )
        )
        .reset_index()
        .sort_values("mae")
    )

    print(
        district_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    period_summary.to_csv(
        OUTPUT_DIR / "walk_forward_period_summary.csv",
        index=False,
    )

    district_summary.to_csv(
        OUTPUT_DIR / "walk_forward_district_summary.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "model_mae": model_mae,
                "model_rmse": model_rmse,
                "model_mape": model_mape,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_mape": baseline_mape,
                "mae_improvement_percent": mae_improvement,
                "mape_improvement_percent": mape_improvement,
                "directional_accuracy": direction_accuracy,
                "mean_prediction_error": mean_error,
                "median_prediction_error": median_error,
                "mean_actual_price": mean_actual,
                "mean_predicted_price": mean_prediction,
                "prediction_bias_percent": prediction_bias_percent,
                "walk_forward_test_periods": len(test_periods),
                "walk_forward_test_rows": len(predictions_df),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("WALK-FORWARD BACKTEST COMPLETE")
    print("=" * 70)

    print()
    print("Generated:")
    print(
        f"  - {PREDICTIONS_PATH.relative_to(BASE_DIR)}"
    )
    print(
        "  - "
        f"{(OUTPUT_DIR / 'walk_forward_period_summary.csv').relative_to(BASE_DIR)}"
    )
    print(
        "  - "
        f"{(OUTPUT_DIR / 'walk_forward_district_summary.csv').relative_to(BASE_DIR)}"
    )
    print(
        f"  - {SUMMARY_PATH.relative_to(BASE_DIR)}"
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Walk-forward MAE        : ₹{model_mae:.2f}"
    )

    print(
        f"Walk-forward baseline   : ₹{baseline_mae:.2f}"
    )

    print(
        f"MAE improvement         : {mae_improvement:.2f}%"
    )

    print(
        f"Walk-forward MAPE       : {model_mape:.2f}%"
    )

    print(
        f"Directional accuracy    : {direction_accuracy:.2f}%"
    )

    if mae_improvement > 0:
        print(
            "STATUS: MODEL BEATS BASELINE"
        )
    else:
        print(
            "STATUS: MODEL STILL NEEDS IMPROVEMENT"
        )


if __name__ == "__main__":
    main()