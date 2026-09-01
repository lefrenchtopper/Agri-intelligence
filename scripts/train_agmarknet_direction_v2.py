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
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# We define a meaningful price increase as:
#
# next week's price is at least THRESHOLD higher than
# the current week's price.
#
# Example:
#
# current = ₹3000
# threshold = 2%
#
# next week must be >= ₹3060
# to be classified as UP.
#
# ------------------------------------------------------------

INCREASE_THRESHOLD = 0.02

TARGET = "direction_target"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("AGMARKNET DIRECTION MODEL V2")
    print("=" * 70)

    print(f"Input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows : {len(df)}")
    print(f"Cols : {len(df.columns)}")

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(df):

    print()
    print("=" * 70)
    print("CREATING MEANINGFUL DIRECTION TARGET")
    print("=" * 70)

    # --------------------------------------------------------
    # Percentage change from current week to next week
    # --------------------------------------------------------

    next_change = (
        df["target_next_week_price"]
        / df["current_price"]
        - 1
    )

    df["next_week_change"] = next_change

    # --------------------------------------------------------
    # Target:
    #
    # 1 = meaningful increase
    # 0 = stable or decrease
    # --------------------------------------------------------

    df[TARGET] = (
        next_change >= INCREASE_THRESHOLD
    ).astype(int)

    print()
    print(
        f"Increase threshold : "
        f"{INCREASE_THRESHOLD * 100:.1f}%"
    )

    print()
    print("Target definition:")
    print(
        "1 = next week's price increases by at least "
        f"{INCREASE_THRESHOLD * 100:.1f}%"
    )
    print(
        "0 = next week's price is stable or decreases"
    )

    counts = df[TARGET].value_counts().sort_index()

    total = len(df)

    print()
    print("Class distribution:")

    for label in [0, 1]:

        count = counts.get(label, 0)

        percentage = (
            count / total * 100
        )

        if label == 0:
            name = "NOT UP"
        else:
            name = "UP"

        print(
            f"  {label} ({name:<7}) : "
            f"{count} ({percentage:.2f}%)"
        )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df):

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Use the existing time_index.
    #
    # All districts from the same week stay together.
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

    train_df = df[
        df["time_index"].isin(train_times)
    ].copy()

    validation_df = df[
        df["time_index"].isin(
            validation_times
        )
    ].copy()

    test_df = df[
        df["time_index"].isin(test_times)
    ].copy()

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
        f"{len(train_df)}"
    )

    print(
        f"Validation rows    : "
        f"{len(validation_df)}"
    )

    print(
        f"Test rows          : "
        f"{len(test_df)}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    # --------------------------------------------------------
    # Columns that cannot be used as predictors.
    #
    # target_next_week_price is future information.
    # target_next_week_change is future information.
    # direction_target is the target itself.
    # next_week_change is also future information.
    # --------------------------------------------------------

    excluded = {
        "target_next_week_price",
        "target_next_week_change",
        "direction_target",
        "next_week_change",
        "month_name",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].copy()

    return (
        X,
        y,
        feature_columns,
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

    # --------------------------------------------------------
    # Scaling is important for Logistic Regression.
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
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

def evaluate_predictions(
    y_true,
    predictions,
    probabilities,
):

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # AUC requires both classes to exist.
    # --------------------------------------------------------

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                C=0.5,
                random_state=42,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "hist_gradient_boosting":
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                random_state=42,
            ),
    }

    return models


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

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_probabilities = (
            pipeline.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_predictions = (
            validation_probabilities >= 0.50
        ).astype(int)

        validation_metrics = evaluate_predictions(
            y_validation,
            validation_predictions,
            validation_probabilities,
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

        print(
            f"ROC-AUC   : "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{validation_metrics['pr_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_probabilities = (
            pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        test_predictions = (
            test_probabilities >= 0.50
        ).astype(int)

        test_metrics = evaluate_predictions(
            y_test,
            test_predictions,
            test_probabilities,
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

        print(
            f"ROC-AUC   : "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{test_metrics['pr_auc']:.4f}"
        )

        print()
        print("Test confusion matrix:")

        print(
            "                 Predicted"
        )

        print(
            "                 NOT UP   UP"
        )

        print(
            f"Actual NOT UP    "
            f"{test_metrics['tn']:7d}"
            f"   {test_metrics['fp']:4d}"
        )

        print(
            f"Actual UP        "
            f"{test_metrics['fn']:7d}"
            f"   {test_metrics['tp']:4d}"
        )

        results.append(
            {
                "model": model_name,

                "validation_accuracy":
                    validation_metrics["accuracy"],

                "validation_precision":
                    validation_metrics["precision"],

                "validation_recall":
                    validation_metrics["recall"],

                "validation_f1":
                    validation_metrics["f1"],

                "validation_roc_auc":
                    validation_metrics["roc_auc"],

                "validation_pr_auc":
                    validation_metrics["pr_auc"],

                "test_accuracy":
                    test_metrics["accuracy"],

                "test_precision":
                    test_metrics["precision"],

                "test_recall":
                    test_metrics["recall"],

                "test_f1":
                    test_metrics["f1"],

                "test_roc_auc":
                    test_metrics["roc_auc"],

                "test_pr_auc":
                    test_metrics["pr_auc"],
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
# SELECT MODEL
# ============================================================

def select_best_model(
    results,
    trained_models,
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Select using validation F1.
    #
    # Test data must NOT influence model selection.
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
# SAVE
# ============================================================

def save_results(
    results,
    best_model_name,
    best_model,
    feature_columns,
):

    results_file = (
        RESULTS_DIR
        / "direction_v2_model_comparison.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    model_file = (
        MODEL_DIR
        / "onion_direction_model_v2.pkl"
    )

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "threshold": INCREASE_THRESHOLD,
        },
        model_file,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model   : "
        f"{model_file.resolve()}"
    )

    print(
        f"Results : "
        f"{results_file.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = create_target(
        df
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(
        df
    )

    (
        X_train,
        y_train,
        feature_columns,
    ) = prepare_features(
        train_df
    )

    (
        X_validation,
        y_validation,
        _,
    ) = prepare_features(
        validation_df
    )

    (
        X_test,
        y_test,
        _,
    ) = prepare_features(
        test_df
    )

    (
        numeric_features,
        categorical_features,
    ) = identify_features(
        X_train
    )

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

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

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

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results
        .round(4)
        .to_string(index=False)
    )

    (
        best_model_name,
        best_model,
    ) = select_best_model(
        results,
        trained_models,
    )

    save_results(
        results,
        best_model_name,
        best_model,
        feature_columns,
    )

    print()
    print("=" * 70)
    print("DIRECTION MODEL V2 COMPLETE")
    print("=" * 70)

    best_test = results[
        results["model"]
        == best_model_name
    ].iloc[0]

    print(
        f"Selected model : "
        f"{best_model_name}"
    )

    print(
        f"Test accuracy  : "
        f"{best_test['test_accuracy'] * 100:.2f}%"
    )

    print(
        f"Test precision : "
        f"{best_test['test_precision'] * 100:.2f}%"
    )

    print(
        f"Test recall    : "
        f"{best_test['test_recall'] * 100:.2f}%"
    )

    print(
        f"Test F1        : "
        f"{best_test['test_f1'] * 100:.2f}%"
    )

    print()
    print("Generated:")
    print(
        "  - models/agmarknet/"
        "onion_direction_model_v2.pkl"
    )
    print(
        "  - data/processed/agmarknet/"
        "model_results/"
        "direction_v2_model_comparison.csv"
    )


if __name__ == "__main__":
    main()