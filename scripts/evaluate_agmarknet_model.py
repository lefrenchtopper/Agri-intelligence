from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/processed/agmarknet/features/"
    "onion_tamilnadu_forecasting_features.csv"
)

MODEL_FILE = Path(
    "models/agmarknet/onion_price_model.pkl"
)

OUTPUT_DIR = Path(
    "data/processed/agmarknet/model_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "target_next_week_price"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = (
        mean_absolute_percentage_error(
            actual,
            predicted
        )
        * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def get_time_period(df):

    """
    Create a sequential time period.

    The dataset contains 4 weeks per month.
    This converts:

        January Week 1 -> 5
        January Week 2 -> 6
        ...
        February Week 1 -> 9

    The absolute value is not important.
    Only chronological ordering matters.
    """

    return (
        (df["month"].astype(int) - 1) * 4
        + df["week"].astype(int)
    )


def get_month_name(df):

    """
    Recover month names if the feature file does not contain
    month_name.

    This makes the evaluation script independent of whether
    feature engineering kept the original text column.
    """

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }

    return df["month"].astype(int).map(
        month_names
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("AGMARKNET MODEL EVALUATION")
    print("=" * 70)

    print(f"Features : {FEATURE_FILE}")
    print(f"Model    : {MODEL_FILE}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Rows     : {len(df)}"
    )

    print()

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model_package = joblib.load(
        MODEL_FILE
    )

    # Normally the training script saves the Pipeline directly.
    #
    # This also supports a dictionary-based model package
    # in case the training script is changed later.

    if isinstance(
        model_package,
        dict
    ):

        if "model" not in model_package:

            raise ValueError(
                "Model dictionary does not contain "
                "'model'."
            )

        model = model_package["model"]

        feature_columns = (
            model_package.get(
                "feature_columns"
            )
        )

    else:

        model = model_package

        feature_columns = None

    print(
        "Model loaded successfully: "
        f"{type(model).__name__}"
    )

    print()

    return (
        model,
        feature_columns,
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Recover month_name if feature engineering removed it.
    # --------------------------------------------------------

    if "month_name" not in df.columns:

        df["month_name"] = (
            get_month_name(df)
        )

    # --------------------------------------------------------
    # Create chronological time period.
    # --------------------------------------------------------

    df["time_period"] = (
        get_time_period(df)
    )

    # --------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "time_period",
            "district",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# CHRONOLOGICAL TEST SET
# ============================================================

def create_test_set(df):

    unique_periods = sorted(
        df["time_period"].unique()
    )

    n_periods = len(
        unique_periods
    )

    train_end = int(
        n_periods * 0.70
    )

    validation_end = int(
        n_periods * 0.85
    )

    train_periods = (
        unique_periods[:train_end]
    )

    validation_periods = (
        unique_periods[
            train_end:validation_end
        ]
    )

    test_periods = (
        unique_periods[
            validation_end:
        ]
    )

    test_df = df[
        df["time_period"].isin(
            test_periods
        )
    ].copy()

    print("=" * 70)
    print("CHRONOLOGICAL TEST SET")
    print("=" * 70)

    print(
        f"Total time periods : {n_periods}"
    )

    print(
        f"Train periods      : "
        f"{min(train_periods)} - "
        f"{max(train_periods)}"
    )

    print(
        f"Validation periods : "
        f"{min(validation_periods)} - "
        f"{max(validation_periods)}"
    )

    print(
        f"Test periods       : "
        f"{min(test_periods)} - "
        f"{max(test_periods)}"
    )

    print()

    print(
        f"Test rows          : "
        f"{len(test_df)}"
    )

    print()

    return test_df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def determine_features(
    test_df,
    feature_columns
):

    # --------------------------------------------------------
    # If the model package explicitly contains feature names,
    # use those.
    # --------------------------------------------------------

    if feature_columns is not None:

        missing_features = [
            column
            for column in feature_columns
            if column not in test_df.columns
        ]

        if missing_features:

            raise ValueError(
                "The following model features are missing "
                f"from the feature dataset:\n"
                f"{missing_features}"
            )

        return feature_columns

    # --------------------------------------------------------
    # Otherwise use the same columns used by the training
    # script: everything except target/output-only columns.
    # --------------------------------------------------------

    excluded_columns = {
        "target_next_week_price",
        "target_next_week_change",
        "month_name",
        "time_period",
    }

    feature_columns = [
        column
        for column in test_df.columns
        if column not in excluded_columns
    ]

    return feature_columns


def validate_features(
    test_df,
    feature_columns
):

    print("=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    print(
        f"Features supplied to model: "
        f"{len(feature_columns)}"
    )

    print()

    for column in feature_columns:

        print(
            f"  - {column}"
        )

    print()

    # --------------------------------------------------------
    # Critical leakage check.
    # --------------------------------------------------------

    forbidden = {
        "target_next_week_price",
        "target_next_week_change",
    }

    leakage = (
        set(feature_columns)
        & forbidden
    )

    if leakage:

        raise ValueError(
            "TARGET LEAKAGE DETECTED! "
            f"Forbidden columns found: {leakage}"
        )

    print(
        "Target leakage check: PASSED"
    )

    print()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test_df,
    feature_columns
):

    print("=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

    predictions = model.predict(
        X_test
    )

    test_df[
        "predicted_price"
    ] = predictions

    # --------------------------------------------------------
    # Prediction error:
    #
    # positive = model overpredicted
    # negative = model underpredicted
    # --------------------------------------------------------

    test_df[
        "prediction_error"
    ] = (
        test_df[
            "predicted_price"
        ]
        - test_df[
            TARGET
        ]
    )

    test_df[
        "absolute_error"
    ] = (
        test_df[
            "prediction_error"
        ].abs()
    )

    print(
        f"Predictions generated: "
        f"{len(predictions)}"
    )

    print()

    return (
        test_df,
        y_test,
        predictions,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def evaluate_model(
    y_test,
    predictions
):

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"MAE  : ₹{metrics['mae']:.2f}"
    )

    print(
        f"RMSE : ₹{metrics['rmse']:.2f}"
    )

    print(
        f"MAPE : {metrics['mape']:.2f}%"
    )

    print()

    return metrics


# ============================================================
# NAIVE BASELINE
# ============================================================

def evaluate_baseline(
    test_df,
    y_test
):

    print("=" * 70)
    print("NAIVE BASELINE")
    print("=" * 70)

    print(
        "Strategy: "
        "next week price = current week price"
    )

    baseline_predictions = (
        test_df[
            "current_price"
        ].values
    )

    metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    print(
        f"MAE  : ₹{metrics['mae']:.2f}"
    )

    print(
        f"RMSE : ₹{metrics['rmse']:.2f}"
    )

    print(
        f"MAPE : {metrics['mape']:.2f}%"
    )

    print()

    return (
        baseline_predictions,
        metrics,
    )


# ============================================================
# BASELINE COMPARISON
# ============================================================

def compare_baseline(
    model_metrics,
    baseline_metrics
):

    mae_improvement = (
        (
            baseline_metrics["mae"]
            - model_metrics["mae"]
        )
        / baseline_metrics["mae"]
        * 100
    )

    mape_improvement = (
        (
            baseline_metrics["mape"]
            - model_metrics["mape"]
        )
        / baseline_metrics["mape"]
        * 100
    )

    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"MAE improvement  : "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"MAPE improvement : "
        f"{mape_improvement:.2f}%"
    )

    if mae_improvement > 0:

        print(
            "Result: Model beats the naive baseline."
        )

    else:

        print(
            "Result: Model DOES NOT beat "
            "the naive baseline."
        )

    print()

    return (
        mae_improvement,
        mape_improvement,
    )


# ============================================================
# DIRECTIONAL ACCURACY
# ============================================================

def evaluate_direction(
    test_df,
    y_test,
    predictions
):

    current_prices = (
        test_df[
            "current_price"
        ].values
    )

    actual_prices = (
        y_test.values
    )

    predicted_prices = (
        predictions
    )

    # --------------------------------------------------------
    # Actual direction:
    #
    # Did the price actually increase next week?
    # --------------------------------------------------------

    actual_up = (
        actual_prices
        > current_prices
    )

    # --------------------------------------------------------
    # Predicted direction:
    #
    # Does the model predict a higher price than today?
    # --------------------------------------------------------

    predicted_up = (
        predicted_prices
        > current_prices
    )

    correct = (
        actual_up
        == predicted_up
    )

    directional_accuracy = (
        correct.mean()
        * 100
    )

    actual_up_count = (
        actual_up.sum()
    )

    actual_down_count = (
        (~actual_up).sum()
    )

    predicted_up_count = (
        predicted_up.sum()
    )

    predicted_down_count = (
        (~predicted_up).sum()
    )

    true_up = (
        actual_up
        & predicted_up
    ).sum()

    true_down = (
        (~actual_up)
        & (~predicted_up)
    ).sum()

    false_up = (
        (~actual_up)
        & predicted_up
    ).sum()

    false_down = (
        actual_up
        & (~predicted_up)
    ).sum()

    print("=" * 70)
    print("DIRECTIONAL ACCURACY")
    print("=" * 70)

    print(
        f"Correct direction : "
        f"{correct.sum()} / {len(correct)}"
    )

    print(
        f"Directional accuracy : "
        f"{directional_accuracy:.2f}%"
    )

    print()

    print("=" * 70)
    print("DIRECTION RESULTS")
    print("=" * 70)

    print(
        f"Actual UP / Predicted UP       : "
        f"{true_up}"
    )

    print(
        f"Actual DOWN / Predicted DOWN   : "
        f"{true_down}"
    )

    print(
        f"Actual DOWN / Predicted UP     : "
        f"{false_up}"
    )

    print(
        f"Actual UP / Predicted DOWN     : "
        f"{false_down}"
    )

    print()

    print(
        f"Actual UP weeks                : "
        f"{actual_up_count}"
    )

    print(
        f"Actual DOWN weeks              : "
        f"{actual_down_count}"
    )

    print(
        f"Predicted UP weeks             : "
        f"{predicted_up_count}"
    )

    print(
        f"Predicted DOWN weeks           : "
        f"{predicted_down_count}"
    )

    print()

    return {
        "directional_accuracy":
            directional_accuracy,

        "actual_up":
            actual_up_count,

        "actual_down":
            actual_down_count,

        "predicted_up":
            predicted_up_count,

        "predicted_down":
            predicted_down_count,

        "correct_up":
            true_up,

        "correct_down":
            true_down,

        "wrong_up":
            false_up,

        "wrong_down":
            false_down,
    }


# ============================================================
# DISTRICT PERFORMANCE
# ============================================================

def evaluate_districts(
    test_df
):

    results = []

    for district, group in (
        test_df.groupby(
            "district"
        )
    ):

        actual = group[
            TARGET
        ].values

        predicted = group[
            "predicted_price"
        ].values

        baseline = group[
            "current_price"
        ].values

        metrics = calculate_metrics(
            actual,
            predicted
        )

        baseline_metrics = (
            calculate_metrics(
                actual,
                baseline
            )
        )

        actual_direction = (
            actual
            > baseline
        )

        predicted_direction = (
            predicted
            > baseline
        )

        direction_accuracy = (
            (
                actual_direction
                == predicted_direction
            ).mean()
            * 100
        )

        improvement = (
            (
                baseline_metrics["mae"]
                - metrics["mae"]
            )
            / baseline_metrics["mae"]
            * 100
        )

        results.append({

            "district":
                district,

            "mae":
                metrics["mae"],

            "rmse":
                metrics["rmse"],

            "mape":
                metrics["mape"],

            "baseline_mae":
                baseline_metrics["mae"],

            "mae_improvement_percent":
                improvement,

            "directional_accuracy":
                direction_accuracy,
        })

    district_results = (
        pd.DataFrame(results)
        .sort_values(
            "mae"
        )
        .reset_index(
            drop=True
        )
    )

    print("=" * 70)
    print("DISTRICT PERFORMANCE")
    print("=" * 70)

    print(
        district_results.to_string(
            index=False
        )
    )

    print()

    # --------------------------------------------------------
    # Highlights
    # --------------------------------------------------------

    best_mae = (
        district_results.iloc[0]
    )

    worst_mae = (
        district_results.iloc[-1]
    )

    best_direction = (
        district_results.loc[
            district_results[
                "directional_accuracy"
            ].idxmax()
        ]
    )

    worst_direction = (
        district_results.loc[
            district_results[
                "directional_accuracy"
            ].idxmin()
        ]
    )

    print("=" * 70)
    print("DISTRICT HIGHLIGHTS")
    print("=" * 70)

    print(
        f"Lowest MAE district : "
        f"{best_mae['district']} "
        f"(₹{best_mae['mae']:.2f})"
    )

    print(
        f"Highest MAE district: "
        f"{worst_mae['district']} "
        f"(₹{worst_mae['mae']:.2f})"
    )

    print(
        f"Best directional accuracy : "
        f"{best_direction['district']} "
        f"({best_direction['directional_accuracy']:.2f}%)"
    )

    print(
        f"Worst directional accuracy : "
        f"{worst_direction['district']} "
        f"({worst_direction['directional_accuracy']:.2f}%)"
    )

    print()

    return district_results


# ============================================================
# ERROR ANALYSIS
# ============================================================

def error_analysis(
    test_df
):

    mean_error = (
        test_df[
            "prediction_error"
        ].mean()
    )

    median_error = (
        test_df[
            "prediction_error"
        ].median()
    )

    mean_absolute_error_value = (
        test_df[
            "absolute_error"
        ].mean()
    )

    median_absolute_error = (
        test_df[
            "absolute_error"
        ].median()
    )

    maximum_absolute_error = (
        test_df[
            "absolute_error"
        ].max()
    )

    prediction_bias_percent = (
        mean_error
        / test_df[
            TARGET
        ].mean()
        * 100
    )

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Mean prediction error : "
        f"₹{mean_error:.2f}"
    )

    print(
        f"Median prediction error : "
        f"₹{median_error:.2f}"
    )

    print(
        f"Mean absolute error : "
        f"₹{mean_absolute_error_value:.2f}"
    )

    print(
        f"Median absolute error : "
        f"₹{median_absolute_error:.2f}"
    )

    print(
        f"Maximum absolute error: "
        f"₹{maximum_absolute_error:.2f}"
    )

    print(
        f"Prediction bias : "
        f"{prediction_bias_percent:.2f}%"
    )

    if mean_error < 0:

        print(
            "Interpretation: "
            "Model has a downward prediction bias."
        )

    elif mean_error > 0:

        print(
            "Interpretation: "
            "Model has an upward prediction bias."
        )

    else:

        print(
            "Interpretation: "
            "No average prediction bias."
        )

    print()

    return {
        "mean_error":
            mean_error,

        "median_error":
            median_error,

        "mean_absolute_error":
            mean_absolute_error_value,

        "median_absolute_error":
            median_absolute_error,

        "maximum_absolute_error":
            maximum_absolute_error,

        "prediction_bias_percent":
            prediction_bias_percent,
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test_df
):

    prediction_file = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    output_columns = [
        "year",
        "month",
        "month_name",
        "week",
        "district",
        "current_price",
        "target_next_week_price",
        "predicted_price",
        "prediction_error",
        "absolute_error",
    ]

    # Only select columns that actually exist.
    # This makes the script robust to future feature changes.

    available_columns = [
        column
        for column in output_columns
        if column in test_df.columns
    ]

    test_df[
        available_columns
    ].to_csv(
        prediction_file,
        index=False
    )

    return prediction_file


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    model_metrics,
    baseline_metrics,
    mae_improvement,
    mape_improvement,
    direction_results,
    error_results,
):

    summary = pd.DataFrame([
        {

            "model_mae":
                model_metrics["mae"],

            "model_rmse":
                model_metrics["rmse"],

            "model_mape":
                model_metrics["mape"],

            "baseline_mae":
                baseline_metrics["mae"],

            "baseline_rmse":
                baseline_metrics["rmse"],

            "baseline_mape":
                baseline_metrics["mape"],

            "mae_improvement_percent":
                mae_improvement,

            "mape_improvement_percent":
                mape_improvement,

            "directional_accuracy":
                direction_results[
                    "directional_accuracy"
                ],

            "actual_up":
                direction_results[
                    "actual_up"
                ],

            "actual_down":
                direction_results[
                    "actual_down"
                ],

            "predicted_up":
                direction_results[
                    "predicted_up"
                ],

            "predicted_down":
                direction_results[
                    "predicted_down"
                ],

            "mean_prediction_error":
                error_results[
                    "mean_error"
                ],

            "median_prediction_error":
                error_results[
                    "median_error"
                ],

            "median_absolute_error":
                error_results[
                    "median_absolute_error"
                ],

            "maximum_absolute_error":
                error_results[
                    "maximum_absolute_error"
                ],

            "prediction_bias_percent":
                error_results[
                    "prediction_bias_percent"
                ],
        }
    ])

    summary_file = (
        OUTPUT_DIR
        / "evaluation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    return summary_file


# ============================================================
# SAVE DISTRICT RESULTS
# ============================================================

def save_district_results(
    district_results
):

    district_file = (
        OUTPUT_DIR
        / "district_performance.csv"
    )

    district_results.to_csv(
        district_file,
        index=False
    )

    return district_file


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load feature dataset
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------------

    (
        model,
        feature_columns,
    ) = load_model()

    # --------------------------------------------------------
    # 3. Prepare chronological data
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # 4. Create exact test set
    # --------------------------------------------------------

    test_df = create_test_set(
        df
    )

    # --------------------------------------------------------
    # 5. Determine model features
    # --------------------------------------------------------

    feature_columns = (
        determine_features(
            test_df,
            feature_columns
        )
    )

    # --------------------------------------------------------
    # 6. Validate features
    # --------------------------------------------------------

    validate_features(
        test_df,
        feature_columns
    )

    # --------------------------------------------------------
    # 7. Generate predictions
    # --------------------------------------------------------

    (
        test_df,
        y_test,
        predictions,
    ) = generate_predictions(
        model,
        test_df,
        feature_columns
    )

    # --------------------------------------------------------
    # 8. Evaluate ML model
    # --------------------------------------------------------

    model_metrics = (
        evaluate_model(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # 9. Evaluate naive baseline
    # --------------------------------------------------------

    (
        baseline_predictions,
        baseline_metrics,
    ) = evaluate_baseline(
        test_df,
        y_test
    )

    # --------------------------------------------------------
    # 10. Compare against baseline
    # --------------------------------------------------------

    (
        mae_improvement,
        mape_improvement,
    ) = compare_baseline(
        model_metrics,
        baseline_metrics
    )

    # --------------------------------------------------------
    # 11. Direction prediction
    # --------------------------------------------------------

    direction_results = (
        evaluate_direction(
            test_df,
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # 12. District-level evaluation
    # --------------------------------------------------------

    district_results = (
        evaluate_districts(
            test_df
        )
    )

    # --------------------------------------------------------
    # 13. Error / bias analysis
    # --------------------------------------------------------

    error_results = (
        error_analysis(
            test_df
        )
    )

    # --------------------------------------------------------
    # 14. Save prediction-level results
    # --------------------------------------------------------

    prediction_file = (
        save_predictions(
            test_df
        )
    )

    # --------------------------------------------------------
    # 15. Save district results
    # --------------------------------------------------------

    district_file = (
        save_district_results(
            district_results
        )
    )

    # --------------------------------------------------------
    # 16. Save evaluation summary
    # --------------------------------------------------------

    summary_file = (
        save_summary(
            model_metrics,
            baseline_metrics,
            mae_improvement,
            mape_improvement,
            direction_results,
            error_results,
        )
    )

    # --------------------------------------------------------
    # 17. Final status
    # --------------------------------------------------------

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Generated:")

    print(
        f"  - {prediction_file}"
    )

    print(
        f"  - {district_file}"
    )

    print(
        f"  - {summary_file}"
    )

    print()

    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Model MAPE        : "
        f"{model_metrics['mape']:.2f}%"
    )

    print(
        f"Baseline MAPE     : "
        f"{baseline_metrics['mape']:.2f}%"
    )

    print(
        f"Directional acc.  : "
        f"{direction_results['directional_accuracy']:.2f}%"
    )

    print(
        f"MAE improvement    : "
        f"{mae_improvement:.2f}%"
    )

    if mae_improvement > 0:

        print()
        print(
            "STATUS: MODEL BEATS BASELINE"
        )

    else:

        print()
        print(
            "STATUS: MODEL NEEDS IMPROVEMENT"
        )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
