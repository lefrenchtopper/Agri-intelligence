from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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

MODEL_DIR = Path(
    "models/agmarknet"
)

RESULTS_DIR = Path(
    "data/processed/agmarknet/model_results"
)

MODEL_FILE = (
    MODEL_DIR
    / "onion_direction_model.pkl"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "direction_model_comparison.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "direction_test_predictions.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "target_next_week_change"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("AGMARKNET PRICE DIRECTION MODEL TRAINING")
    print("=" * 70)

    print(f"Input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows : {len(df)}")
    print(f"Cols : {len(df.columns)}")

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_direction_target(df):

    print()
    print("=" * 70)
    print("CREATING DIRECTION TARGET")
    print("=" * 70)

    # --------------------------------------------------------
    # target_next_week_change represents the percentage change
    # from the current week to the next week.
    #
    # Positive  -> price increases
    # Zero/negative -> price does not increase
    # --------------------------------------------------------

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    df = df.copy()

    df["target_direction"] = (
        df[TARGET] > 0
    ).astype(int)

    print()
    print("Target definition:")
    print("  1 = next week's price increases")
    print("  0 = next week's price does not increase")

    print()

    counts = (
        df["target_direction"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    print("Class distribution:")

    for class_value, count in counts.items():

        percentage = (
            count / total * 100
        )

        if class_value == 1:
            label = "UP"
        else:
            label = "DOWN / STABLE"

        print(
            f"  {class_value} ({label})"
            f" : {count} "
            f"({percentage:.2f}%)"
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------

    df = df.sort_values(
        by=["time_index", "district"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df[
        "target_direction"
    ].copy()

    # --------------------------------------------------------
    # Columns that must NOT be used as features.
    #
    # target_next_week_price:
    #     Future price. Absolutely forbidden.
    #
    # target_next_week_change:
    #     Future percentage change. Absolutely forbidden.
    #
    # target_direction:
    #     The classification target itself.
    # --------------------------------------------------------

    forbidden_columns = {
        "target_next_week_price",
        "target_next_week_change",
        "target_direction",
        "month_name",
    }

    X = df.drop(
        columns=[
            column
            for column in forbidden_columns
            if column in df.columns
        ]
    ).copy()

    return X, y, df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    X,
    y,
    df,
):

    # --------------------------------------------------------
    # Never randomly split time-series data.
    #
    # Every district belonging to the same time period stays
    # together.
    # --------------------------------------------------------

    unique_times = sorted(
        df["time_index"].unique()
    )

    n_times = len(unique_times)

    train_end = int(
        n_times * 0.70
    )

    validation_end = int(
        n_times * 0.85
    )

    train_times = unique_times[
        :train_end
    ]

    validation_times = unique_times[
        train_end:validation_end
    ]

    test_times = unique_times[
        validation_end:
    ]

    train_mask = df[
        "time_index"
    ].isin(train_times)

    validation_mask = df[
        "time_index"
    ].isin(validation_times)

    test_mask = df[
        "time_index"
    ].isin(test_times)

    X_train = X.loc[
        train_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    X_validation = X.loc[
        validation_mask
    ].copy()

    y_validation = y.loc[
        validation_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Total time periods : {n_times}"
    )

    print(
        f"Train periods      : "
        f"{min(train_times)} - "
        f"{max(train_times)}"
    )

    print(
        f"Validation periods : "
        f"{min(validation_times)} - "
        f"{max(validation_times)}"
    )

    print(
        f"Test periods       : "
        f"{min(test_times)} - "
        f"{max(test_times)}"
    )

    print()

    print(
        f"Training rows      : "
        f"{len(X_train)}"
    )

    print(
        f"Validation rows    : "
        f"{len(X_validation)}"
    )

    print(
        f"Test rows          : "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        train_times,
        validation_times,
        test_times,
    )


# ============================================================
# FEATURE TYPES
# ============================================================

def identify_features(X):

    categorical_features = [
        "district"
    ]

    categorical_features = [
        column
        for column in categorical_features
        if column in X.columns
    ]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    print()
    print("=" * 70)
    print("FEATURES")
    print("=" * 70)

    print(
        f"Numeric features     : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_features)}"
    )

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

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        )
    ]

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return preprocessor


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "hist_gradient_boosting":
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            ),
    }

    return models


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    return {
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),

        "precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),

        "recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),

        "f1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),
    }


# ============================================================
# TRAIN MODELS
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

    test_predictions = {}

    for model_name, model in models.items():

        print()
        print("=" * 70)
        print(
            f"TRAINING: {model_name}"
        )
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
        # TRAIN
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_predictions = (
            pipeline.predict(
                X_validation
            )
        )

        validation_metrics = (
            calculate_metrics(
                y_validation,
                validation_predictions,
            )
        )

        print()
        print("Validation:")

        print(
            f"Accuracy  : "
            f"{validation_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{validation_metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{validation_metrics['recall'] * 100:.2f}%"
        )

        print(
            f"F1        : "
            f"{validation_metrics['f1'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_prediction = (
            pipeline.predict(
                X_test
            )
        )

        test_metrics = (
            calculate_metrics(
                y_test,
                test_prediction,
            )
        )

        print()
        print("Test:")

        print(
            f"Accuracy  : "
            f"{test_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{test_metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{test_metrics['recall'] * 100:.2f}%"
        )

        print(
            f"F1        : "
            f"{test_metrics['f1'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        matrix = confusion_matrix(
            y_test,
            test_prediction,
        )

        print()
        print("Test confusion matrix:")

        print(
            "                 Predicted"
        )

        print(
            "                 DOWN   UP"
        )

        print(
            f"Actual DOWN     "
            f"{matrix[0, 0]:6d} "
            f"{matrix[0, 1]:5d}"
        )

        print(
            f"Actual UP       "
            f"{matrix[1, 0]:6d} "
            f"{matrix[1, 1]:5d}"
        )

        results.append(
            {
                "model": model_name,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_precision":
                    validation_metrics[
                        "precision"
                    ],

                "validation_recall":
                    validation_metrics[
                        "recall"
                    ],

                "validation_f1":
                    validation_metrics[
                        "f1"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "test_precision":
                    test_metrics[
                        "precision"
                    ],

                "test_recall":
                    test_metrics[
                        "recall"
                    ],

                "test_f1":
                    test_metrics[
                        "f1"
                    ],
            }
        )

        trained_models[
            model_name
        ] = pipeline

        test_predictions[
            model_name
        ] = test_prediction

    return (
        pd.DataFrame(results),
        trained_models,
        test_predictions,
    )


# ============================================================
# BASELINE
# ============================================================

def evaluate_majority_baseline(
    y_train,
    y_test,
):

    # --------------------------------------------------------
    # Predict the most common class from training data.
    #
    # This gives us a classification baseline.
    # --------------------------------------------------------

    majority_class = (
        y_train
        .value_counts()
        .idxmax()
    )

    predictions = np.full(
        len(y_test),
        majority_class,
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print("MAJORITY BASELINE")
    print("=" * 70)

    if majority_class == 1:
        label = "UP"
    else:
        label = "DOWN / STABLE"

    print(
        f"Strategy: always predict {label}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1        : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    return metrics


# ============================================================
# MODEL SELECTION
# ============================================================

def select_best_model(
    results,
    trained_models,
):

    # --------------------------------------------------------
    # Select based on validation F1.
    #
    # F1 is more useful than raw accuracy when UP and DOWN
    # classes are not perfectly balanced.
    # --------------------------------------------------------

    best_row = results.loc[
        results[
            "validation_f1"
        ].idxmax()
    ]

    best_model_name = (
        best_row["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print()
    print("=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    print(
        f"Best model: "
        f"{best_model_name}"
    )

    print(
        f"Validation F1: "
        f"{best_row['validation_f1'] * 100:.2f}%"
    )

    return (
        best_model_name,
        best_model,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    best_model,
    best_model_name,
    results,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model package.
    # --------------------------------------------------------

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "target": "target_direction",
        "target_definition": (
            "1 = next week's price increases; "
            "0 = next week's price does not increase"
        ),
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model   : "
        f"{MODEL_FILE.resolve()}"
    )

    print(
        f"Results : "
        f"{RESULTS_FILE.resolve()}"
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_test_predictions(
    test_df,
    best_model_name,
    test_predictions,
):

    predictions = test_predictions[
        best_model_name
    ]

    output = test_df[
        [
            "year",
            "month",
            "week",
            "district",
            "current_price",
            "target_next_week_change",
        ]
    ].copy()

    output[
        "actual_direction"
    ] = (
        output[
            "target_next_week_change"
        ] > 0
    ).astype(int)

    output[
        "predicted_direction"
    ] = predictions

    output[
        "actual_direction_label"
    ] = np.where(
        output[
            "actual_direction"
        ] == 1,
        "UP",
        "DOWN / STABLE",
    )

    output[
        "predicted_direction_label"
    ] = np.where(
        output[
            "predicted_direction"
        ] == 1,
        "UP",
        "DOWN / STABLE",
    )

    output[
        "direction_correct"
    ] = (
        output[
            "actual_direction"
        ]
        ==
        output[
            "predicted_direction"
        ]
    )

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_FILE.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Create target
    # --------------------------------------------------------

    df = create_direction_target(
        df
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X,
        y,
        df,
    ) = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        train_times,
        validation_times,
        test_times,
    ) = chronological_split(
        X,
        y,
        df,
    )

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    (
        numeric_features,
        categorical_features,
    ) = identify_features(
        X
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_metrics = (
        evaluate_majority_baseline(
            y_train,
            y_test,
        )
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = create_models()

    (
        results,
        trained_models,
        test_predictions,
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
    # Add baseline
    # --------------------------------------------------------

    baseline_row = pd.DataFrame(
        [
            {
                "model":
                    "majority_baseline",

                "validation_accuracy":
                    np.nan,

                "validation_precision":
                    np.nan,

                "validation_recall":
                    np.nan,

                "validation_f1":
                    np.nan,

                "test_accuracy":
                    baseline_metrics[
                        "accuracy"
                    ],

                "test_precision":
                    baseline_metrics[
                        "precision"
                    ],

                "test_recall":
                    baseline_metrics[
                        "recall"
                    ],

                "test_f1":
                    baseline_metrics[
                        "f1"
                    ],
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
    # Comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    display_results = (
        results.copy()
    )

    metric_columns = [
        column
        for column in display_results.columns
        if column != "model"
    ]

    display_results[
        metric_columns
    ] = display_results[
        metric_columns
    ].round(4)

    print(
        display_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select ML model
    # --------------------------------------------------------

    ml_results = results[
        results["model"]
        != "majority_baseline"
    ].copy()

    (
        best_model_name,
        best_model,
    ) = select_best_model(
        ml_results,
        trained_models,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        best_model,
        best_model_name,
        results,
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_df = df[
        df["time_index"].isin(
            test_times
        )
    ].copy()

    save_test_predictions(
        test_df,
        best_model_name,
        test_predictions,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    best_row = ml_results.loc[
        ml_results[
            "model"
        ] == best_model_name
    ].iloc[0]

    print()
    print("=" * 70)
    print("DIRECTION MODEL COMPLETE")
    print("=" * 70)

    print(
        f"Selected model : "
        f"{best_model_name}"
    )

    print(
        f"Validation F1  : "
        f"{best_row['validation_f1'] * 100:.2f}%"
    )

    print(
        f"Test accuracy  : "
        f"{best_row['test_accuracy'] * 100:.2f}%"
    )

    print(
        f"Test precision : "
        f"{best_row['test_precision'] * 100:.2f}%"
    )

    print(
        f"Test recall    : "
        f"{best_row['test_recall'] * 100:.2f}%"
    )

    print(
        f"Test F1        : "
        f"{best_row['test_f1'] * 100:.2f}%"
    )

    print()
    print("Generated:")
    print(
        f"  - {MODEL_FILE}"
    )
    print(
        f"  - {RESULTS_FILE}"
    )
    print(
        f"  - {PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()