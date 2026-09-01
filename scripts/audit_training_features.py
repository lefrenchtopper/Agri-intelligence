from pathlib import Path
import ast

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)

TRAINING_SCRIPT = BASE_DIR / "scripts" / "walk_forward_agmarknet.py"

EXPECTED_FEATURES = [
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
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_4",
    "previous_week_price",
    "previous_month_price",
    "previous_year_price",
    "district",
]


def extract_feature_columns(source):
    tree = ast.parse(source)

    matches = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "feature_columns":
                    matches.append(node.value)

    if not matches:
        return None

    value = matches[-1]

    if isinstance(value, (ast.List, ast.Tuple)):
        features = []

        for item in value.elts:
            if isinstance(item, ast.Constant):
                features.append(item.value)

        return features

    return None


def main():
    print("TRAINING FEATURE AUDIT")
    print(f"Training script: {TRAINING_SCRIPT}")
    print(f"Feature file:    {FEATURE_FILE}")
    print()

    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    source = TRAINING_SCRIPT.read_text(encoding="utf-8")

    feature_columns = extract_feature_columns(source)

    if feature_columns is None:
        print("Could not automatically extract feature_columns.")
        print("Inspect the training script manually.")
        return

    print("Features actually listed in feature_columns:")
    for feature in feature_columns:
        print(f"  {feature}")

    print()
    print("Feature coverage:")
    print()

    for feature in EXPECTED_FEATURES:
        status = "USED" if feature in feature_columns else "MISSING"
        print(f"{feature:<30} {status}")

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in feature_columns
    ]

    print()
    print(f"Total features used:     {len(feature_columns)}")
    print(f"Expected relevant:       {len(EXPECTED_FEATURES)}")
    print(f"Missing relevant:         {len(missing)}")

    if missing:
        print()
        print("Missing features:")
        for feature in missing:
            print(f"  {feature}")
    else:
        print()
        print("All expected forecasting features are included.")


if __name__ == "__main__":
    main()