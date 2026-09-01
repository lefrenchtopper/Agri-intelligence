from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/agmarknet/features/"
    "onion_tamilnadu_forecasting_features.csv"
)

MODEL_DIR = Path("models/agmarknet")

RESULTS_DIR = Path(
    "data/processed/agmarknet/model_results"
)

TARGET = "target_next_week_price"

# Columns that must NEVER be given to the model.
TARGET_COLUMNS = [
    "target_next_week_price",
    "target_next_week_change",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("AGMARKNET MODEL TRAINING")
    print("=" * 70)

    print(f"Input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows : {len(df)}")
    print(f"Cols : {len(df.columns)}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # Ensure chronological ordering
    # --------------------------------------------------------

    df = df.sort_values(
        by=["time_index", "district"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df[TARGET].copy()

    # --------------------------------------------------------
    # Remove target columns
    # --------------------------------------------------------

    X = df.drop(
        columns=TARGET_COLUMNS
    ).copy()

    return X, y, df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(X, y, df):

    # --------------------------------------------------------
    # We split by time_index rather than individual rows.
    #
    # This is important because all districts from a given
    # week belong to the same point in time.
    # --------------------------------------------------------

    unique_times = sorted(
        df["time_index"].unique()
    )

    n_times = len(unique_times)

    # 70% train
    # 15% validation
    # 15% test

    train_end = int(n_times * 0.70)
    validation_end = int(n_times * 0.85)

    train_times = unique_times[:train_end]
    validation_times = unique_times[
        train_end:validation_end
    ]
    test_times = unique_times[
        validation_end:
    ]

    train_mask = df["time_index"].isin(
        train_times
    )

    validation_mask = df["time_index"].isin(
        validation_times
    )

    test_mask = df["time_index"].isin(
        test_times
    )

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()

    X_validation = X.loc[
        validation_mask
    ].copy()

    y_validation = y.loc[
        validation_mask
    ].copy()

    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Total time periods : {n_times}"
    )

    print(
        f"Train periods      : "
        f"{min(train_times)} - {max(train_times)}"
    )

    print(
        f"Validation periods : "
        f"{min(validation_times)} - "
        f"{max(validation_times)}"
    )

    print(
        f"Test periods       : "
        f"{min(test_times)} - {max(test_times)}"
    )

    print()

    print(
        f"Training rows      : {len(X_train)}"
    )

    print(
        f"Validation rows    : {len(X_validation)}"
    )

    print(
        f"Test rows          : {len(X_test)}"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# FEATURE TYPES
# ============================================================

def identify_features(X):

    categorical_features = [
        "district"
    ]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    return (
        numeric_features,
        categorical_features,
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    numeric_features,
    categorical_features,
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    # Avoid division by zero.
    non_zero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (
                    actual[non_zero]
                    - predicted[non_zero]
                )
                / actual[non_zero]
            )
        )
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


# ============================================================
# BASELINE
# ============================================================

def evaluate_baseline(
    X_test,
    y_test,
):

    # --------------------------------------------------------
    # Naive forecast:
    #
    # Predict next week's price as the current week's price.
    # --------------------------------------------------------

    predictions = X_test[
        "current_price"
    ].values

    metrics = calculate_metrics(
        y_test.values,
        predictions,
    )

    print()
    print("=" * 70)
    print("BASELINE MODEL")
    print("=" * 70)

    print(
        "Strategy: next week price = current week price"
    )

    print(
        f"MAE  : ₹{metrics['MAE']:.2f}"
    )

    print(
        f"RMSE : ₹{metrics['RMSE']:.2f}"
    )

    print(
        f"MAPE : {metrics['MAPE']:.2f}%"
    )

    return metrics


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    models = {

        "linear_regression":
            LinearRegression(),

        "random_forest":
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),

        "hist_gradient_boosting":
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            ),
    }

    return models


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def train_models(
    models,
    preprocessor,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print()
        print("=" * 70)
        print(f"TRAINING: {model_name}")
        print("=" * 70)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_predictions = (
            pipeline.predict(
                X_validation
            )
        )

        validation_metrics = calculate_metrics(
            y_validation.values,
            validation_predictions,
        )

        print()
        print("Validation:")

        print(
            f"MAE  : ₹{validation_metrics['MAE']:.2f}"
        )

        print(
            f"RMSE : ₹{validation_metrics['RMSE']:.2f}"
        )

        print(
            f"MAPE : {validation_metrics['MAPE']:.2f}%"
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_predictions = (
            pipeline.predict(
                X_test
            )
        )

        test_metrics = calculate_metrics(
            y_test.values,
            test_predictions,
        )

        print()
        print("Test:")

        print(
            f"MAE  : ₹{test_metrics['MAE']:.2f}"
        )

        print(
            f"RMSE : ₹{test_metrics['RMSE']:.2f}"
        )

        print(
            f"MAPE : {test_metrics['MAPE']:.2f}%"
        )

        results.append(
            {
                "model": model_name,

                "validation_mae":
                    validation_metrics["MAE"],

                "validation_rmse":
                    validation_metrics["RMSE"],

                "validation_mape":
                    validation_metrics["MAPE"],

                "test_mae":
                    test_metrics["MAE"],

                "test_rmse":
                    test_metrics["RMSE"],

                "test_mape":
                    test_metrics["MAPE"],
            }
        )

        trained_models[
            model_name
        ] = pipeline

    return (
        pd.DataFrame(results),
        trained_models,
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    results,
    trained_models,
):

    # --------------------------------------------------------
    # Select based on validation MAE.
    #
    # We do NOT select based on test performance because the
    # test set should remain an unbiased final evaluation.
    # --------------------------------------------------------

    best_row = results.loc[
        results["validation_mae"].idxmin()
    ]

    best_model_name = best_row["model"]

    best_model = trained_models[
        best_model_name
    ]

    print()
    print("=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    print(
        f"Best model: {best_model_name}"
    )

    print(
        f"Validation MAE: "
        f"₹{best_row['validation_mae']:.2f}"
    )

    return (
        best_model_name,
        best_model,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    best_model_name,
    best_model,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save comparison table
    # --------------------------------------------------------

    results_file = (
        RESULTS_DIR
        / "model_comparison.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR
        / "onion_price_model.pkl"
    )

    joblib.dump(
        best_model,
        model_file,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Results: "
        f"{results_file.resolve()}"
    )

    print(
        f"Model  : "
        f"{model_file.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    X, y, df = prepare_data(
        df
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = chronological_split(
        X,
        y,
        df,
    )

    (
        numeric_features,
        categorical_features,
    ) = identify_features(X)

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_metrics = evaluate_baseline(
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    models = create_models()

    (
        results,
        trained_models,
    ) = train_models(
        models,
        preprocessor,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Add baseline to results
    # --------------------------------------------------------

    baseline_row = pd.DataFrame(
        [
            {
                "model": "naive_baseline",

                "validation_mae": np.nan,
                "validation_rmse": np.nan,
                "validation_mape": np.nan,

                "test_mae":
                    baseline_metrics["MAE"],

                "test_rmse":
                    baseline_metrics["RMSE"],

                "test_mape":
                    baseline_metrics["MAPE"],
            }
        ]
    )

    results = pd.concat(
        [
            baseline_row,
            results,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results
        .round(2)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Select best ML model
    # --------------------------------------------------------

    ml_results = results[
        results["model"]
        != "naive_baseline"
    ].copy()

    best_model_name, best_model = (
        select_best_model(
            ml_results,
            trained_models,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        best_model_name,
        best_model,
    )


if __name__ == "__main__":
    main()