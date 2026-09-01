from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

print(f"Scanning: {SCRIPTS_DIR}")
print()

patterns = {
    "models": [
        r"RandomForest",
        r"GradientBoosting",
        r"RandomForestRegressor",
        r"GradientBoostingRegressor",
        r"RandomForestClassifier",
        r"XGBRegressor",
        r"XGBClassifier",
        r"LGBMRegressor",
        r"CatBoostRegressor",
        r"HistGradientBoosting",
        r"ExtraTrees",
        r"RandomForest",
        r"LinearRegression",
        r"Ridge",
        r"Lasso",
        r"ElasticNet",
    ],
    "training": [
        r"\.fit\(",
        r"train_test_split",
        r"TimeSeriesSplit",
        r"walk.?forward",
        r"expanding",
        r"rolling",
    ],
    "targets": [
        r"target_next_week_price",
        r"target_next_week_change",
        r"current_price",
        r"y\s*=",
        r"target\s*=",
    ],
    "features": [
        r"feature_columns",
        r"features\s*=",
        r"X\s*=",
        r"drop\(",
        r"district",
        r"OneHotEncoder",
        r"get_dummies",
        r"ColumnTransformer",
    ],
    "prediction": [
        r"\.predict\(",
        r"prediction",
        r"predictions",
        r"walk_forward_predictions",
    ],
}

py_files = sorted(SCRIPTS_DIR.glob("*.py"))

if not py_files:
    print("No Python scripts found in scripts/")
    raise SystemExit(1)

print(f"Python scripts found: {len(py_files)}")
print()

for path in py_files:
    text = path.read_text(encoding="utf-8", errors="ignore")

    matches = []

    for category, regexes in patterns.items():
        for pattern in regexes:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.append(category)
                break

    if not matches:
        continue

    print("=" * 70)
    print(path.name)
    print("Categories:", ", ".join(sorted(set(matches))))
    print("=" * 70)

    lines = text.splitlines()

    for line_number, line in enumerate(lines, start=1):
        matched = False

        for category, regexes in patterns.items():
            for pattern in regexes:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    matched = True
                    break

            if matched:
                break

        if matched:
            clean = line.strip()

            if clean:
                print(f"{line_number:4}: {clean}")

    print()

print("=" * 70)
print("Pipeline inspection complete.")
print("=" * 70)