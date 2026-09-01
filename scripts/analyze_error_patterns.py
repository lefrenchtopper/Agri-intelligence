from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "agmarknet" / "model_results" / "walk_forward_predictions.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "agmarknet" / "model_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading walk-forward predictions...")
print(f"File: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

required = [
    "time_index",
    "district",
    "current_price",
    "target_next_week_price",
    "prediction",
    "baseline_prediction",
    "absolute_error",
    "baseline_absolute_error",
]

missing = [col for col in required if col not in df.columns]

if missing:
    print("ERROR: Missing required columns:")
    for col in missing:
        print(f"  - {col}")
    print()
    print("Available columns:")
    print(list(df.columns))
    raise SystemExit(1)

# Actual percentage movement from current price to next week's price
df["actual_change"] = (
    df["target_next_week_price"] - df["current_price"]
)

df["actual_change_percent"] = (
    df["actual_change"] / df["current_price"].replace(0, np.nan)
) * 100

# Model percentage movement
df["predicted_change"] = (
    df["prediction"] - df["current_price"]
)

df["predicted_change_percent"] = (
    df["predicted_change"] / df["current_price"].replace(0, np.nan)
) * 100

# How much the model under/over-predicts the actual movement
df["movement_error"] = (
    df["predicted_change"] - df["actual_change"]
)

df["movement_error_percent"] = (
    df["predicted_change_percent"] - df["actual_change_percent"]
)

# Absolute movement size
df["absolute_actual_change"] = df["actual_change"].abs()
df["absolute_actual_change_percent"] = df["actual_change_percent"].abs()

# Categorize market movement
def classify_movement(x):
    if pd.isna(x):
        return "unknown"

    if x <= -10:
        return "large_decrease"

    if x <= -5:
        return "moderate_decrease"

    if x < 5:
        return "stable"

    if x < 10:
        return "moderate_increase"

    return "large_increase"


df["movement_category"] = df["actual_change_percent"].apply(
    classify_movement
)

# Identify unusually large errors
df["large_error"] = df["absolute_error"] >= 500
df["very_large_error"] = df["absolute_error"] >= 750

# Identify cases where model loses to naive baseline
df["model_worse_than_baseline"] = (
    df["absolute_error"] > df["baseline_absolute_error"]
)

print("Error pattern analysis complete.")
print()

# 1. Overall movement analysis
movement_summary = (
    df.groupby("movement_category", dropna=False)
    .agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        mean_actual_change_percent=("actual_change_percent", "mean"),
        mean_predicted_change_percent=("predicted_change_percent", "mean"),
        mean_movement_error_percent=("movement_error_percent", "mean"),
        directional_accuracy=("direction_correct", "mean"),
        large_errors=("large_error", "sum"),
        very_large_errors=("very_large_error", "sum"),
    )
    .reset_index()
)

movement_summary["directional_accuracy"] *= 100

movement_summary["mae_improvement_percent"] = (
    (movement_summary["baseline_mae"] - movement_summary["mae"])
    / movement_summary["baseline_mae"]
) * 100

movement_summary = movement_summary.sort_values(
    "movement_category"
)

movement_file = OUTPUT_DIR / "walk_forward_change_bucket_analysis.csv"
movement_summary.to_csv(movement_file, index=False)

# 2. Period-level spike analysis
period_summary = (
    df.groupby("time_index")
    .agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        mean_actual_price=("target_next_week_price", "mean"),
        mean_current_price=("current_price", "mean"),
        mean_actual_change_percent=("actual_change_percent", "mean"),
        mean_predicted_change_percent=("predicted_change_percent", "mean"),
        mean_absolute_change_percent=("absolute_actual_change_percent", "mean"),
        large_errors=("large_error", "sum"),
        very_large_errors=("very_large_error", "sum"),
    )
    .reset_index()
)

period_summary["mae_improvement_percent"] = (
    (period_summary["baseline_mae"] - period_summary["mae"])
    / period_summary["baseline_mae"]
) * 100

period_file = OUTPUT_DIR / "walk_forward_spike_analysis.csv"
period_summary.to_csv(period_file, index=False)

# 3. District error patterns
district_summary = (
    df.groupby("district")
    .agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        mean_actual_change_percent=("actual_change_percent", "mean"),
        mean_predicted_change_percent=("predicted_change_percent", "mean"),
        mean_absolute_change_percent=("absolute_actual_change_percent", "mean"),
        mean_movement_error_percent=("movement_error_percent", "mean"),
        directional_accuracy=("direction_correct", "mean"),
        large_errors=("large_error", "sum"),
        very_large_errors=("very_large_error", "sum"),
    )
    .reset_index()
)

district_summary["directional_accuracy"] *= 100

district_summary["mae_improvement_percent"] = (
    (district_summary["baseline_mae"] - district_summary["mae"])
    / district_summary["baseline_mae"]
) * 100

district_summary = district_summary.sort_values("mae", ascending=False)

district_file = OUTPUT_DIR / "walk_forward_district_error_patterns.csv"
district_summary.to_csv(district_file, index=False)

# 4. Worst predictions
worst = df[
    [
        "time_index",
        "year",
        "month",
        "week",
        "district",
        "current_price",
        "target_next_week_price",
        "prediction",
        "baseline_prediction",
        "actual_change_percent",
        "predicted_change_percent",
        "movement_error_percent",
        "absolute_error",
        "baseline_absolute_error",
        "direction_correct",
    ]
].copy()

worst = worst.sort_values(
    "absolute_error",
    ascending=False
).head(50)

worst_file = OUTPUT_DIR / "walk_forward_worst_predictions_detailed.csv"
worst.to_csv(worst_file, index=False)

# 5. Print useful results
print("Movement categories:")
print(
    movement_summary[
        [
            "movement_category",
            "rows",
            "mae",
            "baseline_mae",
            "mae_improvement_percent",
            "mean_actual_change_percent",
            "mean_predicted_change_percent",
            "directional_accuracy",
            "large_errors",
            "very_large_errors",
        ]
    ].to_string(index=False)
)

print()
print("Worst periods by MAE:")
print(
    period_summary.sort_values("mae", ascending=False)
    .head(10)
    [
        [
            "time_index",
            "mae",
            "baseline_mae",
            "mae_improvement_percent",
            "mean_actual_change_percent",
            "mean_predicted_change_percent",
            "large_errors",
            "very_large_errors",
        ]
    ]
    .to_string(index=False)
)

print()
print("Worst districts:")
print(
    district_summary.head(10)
    [
        [
            "district",
            "mae",
            "baseline_mae",
            "mae_improvement_percent",
            "directional_accuracy",
            "large_errors",
            "very_large_errors",
        ]
    ]
    .to_string(index=False)
)

print()
print("Largest prediction errors:")
print(
    worst.head(15)
    [
        [
            "time_index",
            "district",
            "current_price",
            "target_next_week_price",
            "prediction",
            "absolute_error",
            "actual_change_percent",
            "predicted_change_percent",
        ]
    ]
    .to_string(index=False)
)

print()
print("Generated:")
print(f"  {movement_file}")
print(f"  {period_file}")
print(f"  {district_file}")
print(f"  {worst_file}")