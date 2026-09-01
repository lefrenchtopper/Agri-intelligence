from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

WALK_FORWARD_FILE = ROOT / "scripts" / "walk_forward_agmarknet.py"
TRAIN_FILE = ROOT / "scripts" / "train_agmarknet_model.py"

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)

PREDICTIONS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "agmarknet"
    / "model_results"
    / "walk_forward_predictions.csv"
)


def read_file(path):
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8", errors="ignore")


def find_matches(text, patterns):
    results = []

    if text is None:
        return results

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        for match in matches:
            if isinstance(match, tuple):
                results.extend([x for x in match if x])
            else:
                results.append(match)

    return results


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def audit_target(walk_text, train_text):
    print_section("1. TARGET FORMULATION")

    combined = "\n".join(
        text for text in [walk_text, train_text] if text is not None
    )

    raw_price = "target_next_week_price" in combined
    change_target = "target_next_week_change" in combined

    print(f"Raw price target found       : {raw_price}")
    print(f"Percentage-change target     : {change_target}")

    if re.search(r'TARGET\s*=\s*["\']target_next_week_price["\']', combined):
        print("Primary target appears to be: RAW NEXT-WEEK PRICE")
    elif re.search(r'TARGET\s*=\s*["\']target_next_week_change["\']', combined):
        print("Primary target appears to be: NEXT-WEEK PERCENTAGE CHANGE")
    else:
        print("Could not determine primary TARGET assignment.")

    print()
    print("Relevant target references:")

    matches = find_matches(
        combined,
        [
            r'TARGET\s*=\s*["\'][^"\']+["\']',
            r'y\s*=\s*df\[[^\n]+\]',
        ],
    )

    for match in matches[:20]:
        print(f"  {match}")


def audit_features(walk_text, train_text):
    print_section("2. FEATURE MATRIX")

    combined = "\n".join(
        text for text in [walk_text, train_text] if text is not None
    )

    important_features = [
        "current_price",
        "price_change_1w",
        "price_change_2w",
        "price_change_4w",
        "week_vs_previous_week",
        "month_vs_previous_month",
        "year_vs_previous_year",
        "change_previous_week",
        "change_previous_month",
        "change_previous_year",
        "price_rolling_mean_4",
        "price_rolling_std_4",
        "price_rolling_mean_8",
        "district",
    ]

    print("Checking whether important features appear in training code:")

    for feature in important_features:
        found = feature in combined
        print(f"  {feature:<30} {found}")

    print()
    print("Feature matrix construction:")

    matches = find_matches(
        combined,
        [
            r'feature_columns\s*=\s*\[',
            r'X\s*=\s*df\[[^\n]+',
            r'X_train\s*=\s*train_df\[[^\n]+',
            r'X_test\s*=\s*test_df\[[^\n]+',
        ],
    )

    for match in matches[:20]:
        print(f"  {match}")


def audit_district(walk_text, train_text):
    print_section("3. DISTRICT REPRESENTATION")

    combined = "\n".join(
        text for text in [walk_text, train_text] if text is not None
    )

    checks = {
        "district column used": "district" in combined,
        "OneHotEncoder used": "OneHotEncoder" in combined,
        "ColumnTransformer used": "ColumnTransformer" in combined,
        "get_dummies used": "get_dummies" in combined,
        "separate model per district": bool(
            re.search(
                r'for\s+district\s+in\s+.*groupby',
                combined,
                flags=re.IGNORECASE,
            )
        ),
    }

    for name, value in checks.items():
        print(f"  {name:<32} {value}")

    print()
    print("Relevant categorical-processing references:")

    matches = find_matches(
        combined,
        [
            r'categorical_features\s*=\s*\[[^\]]*\]',
            r'OneHotEncoder[^\n]*',
            r'ColumnTransformer[^\n]*',
            r'get_dummies[^\n]*',
        ],
    )

    for match in matches[:20]:
        print(f"  {match}")


def audit_walk_forward(walk_text):
    print_section("4. WALK-FORWARD WINDOW STRATEGY")

    if walk_text is None:
        print("walk_forward_agmarknet.py was not found.")
        return

    checks = {
        "Time index sorting": bool(
            re.search(r'sort_values.*time_index', walk_text, re.IGNORECASE)
        ),
        "Expanding window terminology": bool(
            re.search(r'expanding', walk_text, re.IGNORECASE)
        ),
        "Rolling/sliding terminology": bool(
            re.search(r'rolling|sliding', walk_text, re.IGNORECASE)
        ),
        "X_train uses train_df": bool(
            re.search(r'X_train\s*=\s*train_df', walk_text)
        ),
        "X_test uses test_df": bool(
            re.search(r'X_test\s*=\s*test_df', walk_text)
        ),
        "model.fit(X_train, y_train)": bool(
            re.search(r'model\.fit\s*\(\s*X_train\s*,\s*y_train', walk_text)
        ),
    }

    for name, value in checks.items():
        print(f"  {name:<36} {value}")

    print()
    print("Relevant window/split lines:")

    lines = walk_text.splitlines()

    for i, line in enumerate(lines):
        lower = line.lower()

        if any(
            keyword in lower
            for keyword in [
                "train_df",
                "test_df",
                "time_index",
                "walk-forward",
                "walk_forward",
                "model.fit",
            ]
        ):
            print(f"  {i + 1}: {line.strip()}")


def audit_model_and_loss(walk_text, train_text):
    print_section("5. MODEL AND OBJECTIVE FUNCTION")

    combined = "\n".join(
        text for text in [walk_text, train_text] if text is not None
    )

    model_patterns = [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "RandomForest",
        "XGBRegressor",
        "LGBMRegressor",
        "CatBoostRegressor",
        "ExtraTreesRegressor",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
    ]

    print("Models referenced:")

    found_models = []

    for model in model_patterns:
        if model.lower() in combined.lower():
            found_models.append(model)

    if found_models:
        for model in found_models:
            print(f"  {model}")
    else:
        print("  No known regression model detected.")

    print()
    print("Loss/objective references:")

    loss_patterns = [
        r"loss\s*=\s*['\"][^'\"]+['\"]",
        r"criterion\s*=\s*['\"][^'\"]+['\"]",
        r"objective\s*=\s*['\"][^'\"]+['\"]",
        r"eval_metric\s*=\s*['\"][^'\"]+['\"]",
    ]

    matches = find_matches(combined, loss_patterns)

    if matches:
        for match in matches:
            print(f"  {match}")
    else:
        print("  No explicit loss/objective parameter found.")

    print()
    print("Model construction lines:")

    lines = combined.splitlines()

    for i, line in enumerate(lines):
        lower = line.lower()

        if any(
            keyword.lower() in lower
            for keyword in model_patterns
        ):
            print(f"  {i + 1}: {line.strip()}")


def audit_extrapolation():
    print_section("6. RAW-PRICE EXTRAPOLATION CHECK")

    if not FEATURE_FILE.exists():
        print(f"Feature file not found:\n{FEATURE_FILE}")
        return

    if not PREDICTIONS_FILE.exists():
        print(f"Prediction file not found:\n{PREDICTIONS_FILE}")
        return

    features = pd.read_csv(FEATURE_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required_features = {
        "time_index",
        "district",
        "current_price",
        "target_next_week_price",
    }

    required_predictions = {
        "time_index",
        "district",
        "current_price",
        "target_next_week_price",
        "prediction",
    }

    missing_features = required_features - set(features.columns)
    missing_predictions = required_predictions - set(predictions.columns)

    if missing_features:
        print(f"Missing feature columns: {sorted(missing_features)}")
        return

    if missing_predictions:
        print(f"Missing prediction columns: {sorted(missing_predictions)}")
        return

    features = features.sort_values(["time_index", "district"]).copy()
    predictions = predictions.sort_values(["time_index", "district"]).copy()

    results = []

    for time_index in sorted(predictions["time_index"].unique()):
        train = features[features["time_index"] < time_index]
        test = predictions[predictions["time_index"] == time_index]

        if train.empty:
            continue

        train_target_max = train["target_next_week_price"].max()
        train_target_min = train["target_next_week_price"].min()

        test_target_max = test["target_next_week_price"].max()
        test_prediction_max = test["prediction"].max()

        above_training_max = (
            test["target_next_week_price"] > train_target_max
        ).sum()

        predictions_above_training_max = (
            test["prediction"] > train_target_max
        ).sum()

        results.append(
            {
                "time_index": time_index,
                "training_target_min": train_target_min,
                "training_target_max": train_target_max,
                "test_target_max": test_target_max,
                "test_prediction_max": test_prediction_max,
                "actual_targets_above_training_max": above_training_max,
                "predictions_above_training_max": predictions_above_training_max,
                "test_rows": len(test),
            }
        )

    result = pd.DataFrame(results)

    print(
        result.to_string(index=False)
    )

    print()
    print("Important interpretation:")

    spike_periods = result[
        result["actual_targets_above_training_max"] > 0
    ]

    if spike_periods.empty:
        print(
            "  No test period contains targets above the historical "
            "training target maximum."
        )
    else:
        print(
            f"  {len(spike_periods)} test period(s) contain actual "
            "prices above the historical training maximum."
        )

        for _, row in spike_periods.iterrows():
            print(
                f"  time_index={int(row['time_index'])}: "
                f"training max={row['training_target_max']:.2f}, "
                f"actual max={row['test_target_max']:.2f}, "
                f"prediction max={row['test_prediction_max']:.2f}"
            )

    output_file = (
        ROOT
        / "data"
        / "processed"
        / "agmarknet"
        / "model_results"
        / "walk_forward_extrapolation_audit.csv"
    )

    result.to_csv(output_file, index=False)

    print()
    print(f"Saved: {output_file}")


def main():
    print("WALK-FORWARD IMPLEMENTATION AUDIT")
    print(f"Project root: {ROOT}")

    walk_text = read_file(WALK_FORWARD_FILE)
    train_text = read_file(TRAIN_FILE)

    print()
    print(f"Walk-forward script found: {walk_text is not None}")
    print(f"Training script found     : {train_text is not None}")

    audit_target(walk_text, train_text)
    audit_features(walk_text, train_text)
    audit_district(walk_text, train_text)
    audit_walk_forward(walk_text)
    audit_model_and_loss(walk_text, train_text)
    audit_extrapolation()

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print()
    print("Do not modify the forecasting model yet.")
    print("Use this output to decide whether the problem is:")
    print("  1. Target formulation")
    print("  2. Missing momentum features")
    print("  3. District representation")
    print("  4. Walk-forward window design")
    print("  5. Model/objective limitation")
    print("  6. Genuine regime shift beyond historical training range")


if __name__ == "__main__":
    main()