from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "model_results"
    / "walk_forward_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "model_results"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AGMARKNET WALK-FORWARD ERROR ANALYSIS")
print("=" * 70)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREDICTIONS")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nPrediction file not found:\n{INPUT_FILE}\n\n"
        "Run walk_forward_agmarknet.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "time_index",
    "district",
    "current_price",
    "target_next_week_price",
    "prediction",
    "baseline_prediction",
    "prediction_error",
    "absolute_error",
    "baseline_error",
    "baseline_absolute_error",
    "model_direction",
    "actual_direction",
    "direction_correct",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"\nMissing required columns:\n{missing}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )

print("Data validation: PASSED")


# ============================================================
# CLEAN TYPES
# ============================================================

numeric_columns = [
    "time_index",
    "current_price",
    "target_next_week_price",
    "prediction",
    "baseline_prediction",
    "prediction_error",
    "absolute_error",
    "baseline_error",
    "baseline_absolute_error",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["direction_correct"] = (
    df["direction_correct"]
    .astype(str)
    .str.upper()
    .map({"TRUE": True, "FALSE": False})
)


# ============================================================
# DERIVED ERROR METRICS
# ============================================================

df["percentage_error"] = (
    df["prediction_error"].abs()
    / df["target_next_week_price"].replace(0, np.nan)
) * 100

df["signed_percentage_error"] = (
    df["prediction_error"]
    / df["target_next_week_price"].replace(0, np.nan)
) * 100

df["baseline_percentage_error"] = (
    df["baseline_absolute_error"]
    / df["target_next_week_price"].replace(0, np.nan)
) * 100

df["model_beats_baseline"] = (
    df["absolute_error"] < df["baseline_absolute_error"]
)

df["model_worse_than_baseline"] = (
    df["absolute_error"] > df["baseline_absolute_error"]
)

df["large_error_10_percent"] = (
    df["percentage_error"] > 10
)

df["large_error_20_percent"] = (
    df["percentage_error"] > 20
)

df["large_error_30_percent"] = (
    df["percentage_error"] > 30
)


# ============================================================
# 1. WORST INDIVIDUAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("1. WORST INDIVIDUAL PREDICTIONS")
print("=" * 70)

worst = (
    df.sort_values("absolute_error", ascending=False)
    .copy()
)

worst_columns = [
    "time_index",
    "year",
    "month",
    "week",
    "district",
    "current_price",
    "target_next_week_price",
    "prediction",
    "baseline_prediction",
    "absolute_error",
    "baseline_absolute_error",
    "percentage_error",
    "model_direction",
    "actual_direction",
    "direction_correct",
]

worst_columns = [
    col for col in worst_columns
    if col in worst.columns
]

worst_output = worst[worst_columns]

worst_output.to_csv(
    OUTPUT_DIR / "worst_predictions.csv",
    index=False
)

print("\nTop 20 largest model errors:\n")
print(
    worst_output.head(20).to_string(index=False)
)


# ============================================================
# 2. DISTRICT ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("2. DISTRICT ERROR ANALYSIS")
print("=" * 70)

district_analysis = (
    df.groupby("district")
    .agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
        rmse=("prediction_error", lambda x: np.sqrt(np.mean(x ** 2))),
        mean_percentage_error=("percentage_error", "mean"),
        mean_signed_error=("prediction_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        directional_accuracy=("direction_correct", "mean"),
        model_wins=("model_beats_baseline", "sum"),
        model_losses=("model_worse_than_baseline", "sum"),
        large_errors_10pct=("large_error_10_percent", "sum"),
        large_errors_20pct=("large_error_20_percent", "sum"),
        large_errors_30pct=("large_error_30_percent", "sum"),
    )
    .reset_index()
)

district_analysis["directional_accuracy"] *= 100

district_analysis["model_win_rate"] = (
    district_analysis["model_wins"]
    / district_analysis["rows"]
) * 100

district_analysis["baseline_improvement_percent"] = (
    (
        district_analysis["baseline_mae"]
        - district_analysis["mae"]
    )
    / district_analysis["baseline_mae"]
) * 100

district_analysis = district_analysis.sort_values(
    "mae",
    ascending=True
)

district_analysis.to_csv(
    OUTPUT_DIR / "district_error_analysis.csv",
    index=False
)

print("\nBest districts:\n")
print(
    district_analysis.head(10).to_string(index=False)
)

print("\nWorst districts:\n")
print(
    district_analysis.tail(10)
    .sort_values("mae", ascending=False)
    .to_string(index=False)
)


# ============================================================
# 3. PERIOD ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. PERIOD ERROR ANALYSIS")
print("=" * 70)

period_analysis = (
    df.groupby("time_index")
    .agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
        rmse=("prediction_error", lambda x: np.sqrt(np.mean(x ** 2))),
        mean_percentage_error=("percentage_error", "mean"),
        mean_signed_error=("prediction_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        directional_accuracy=("direction_correct", "mean"),
        actual_mean_price=("target_next_week_price", "mean"),
        predicted_mean_price=("prediction", "mean"),
        current_mean_price=("current_price", "mean"),
        model_wins=("model_beats_baseline", "sum"),
        model_losses=("model_worse_than_baseline", "sum"),
    )
    .reset_index()

period_analysis["directional_accuracy"] *= 100

period_analysis["model_win_rate"] = (
    period_analysis["model_wins"]
    / period_analysis["rows"]
) * 100

period_analysis["improvement_percent"] = (
    (
        period_analysis["baseline_mae"]
        - period_analysis["mae"]
    )
    / period_analysis["baseline_mae"]
) * 100

period_analysis.to_csv(
    OUTPUT_DIR / "period_error_analysis.csv",
    index=False
)

print("\nWorst periods:\n")
print(
    period_analysis
    .sort_values("mae", ascending=False)
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 4. DIRECTIONAL ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. DIRECTIONAL ERROR ANALYSIS")
print("=" * 70)

direction_analysis = (
    df.groupby(
        ["model_direction", "actual_direction"]
    )
    .agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("prediction_error", "mean"),
        mean_percentage_error=("percentage_error", "mean"),
    )
    .reset_index()
)

direction_analysis["direction_label"] = (
    direction_analysis["model_direction"]
    .map({
        0: "DOWN / NO INCREASE",
        1: "UP"
    })
    + " -> "
    + direction_analysis["actual_direction"]
    .map({
        0: "DOWN / NO INCREASE",
        1: "UP"
    })
)

direction_analysis.to_csv(
    OUTPUT_DIR / "direction_error_analysis.csv",
    index=False
)

print(direction_analysis.to_string(index=False))


# ============================================================
# 5. BASELINE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("5. MODEL VS BASELINE")
print("=" * 70)

total_rows = len(df)

model_wins = int(df["model_beats_baseline"].sum())
model_losses = int(df["model_worse_than_baseline"].sum())
ties = total_rows - model_wins - model_losses

baseline_comparison = pd.DataFrame(
    [
        {
            "total_predictions": total_rows,
            "model_better_than_baseline": model_wins,
            "model_worse_than_baseline": model_losses,
            "ties": ties,
            "model_win_rate_percent": (
                model_wins / total_rows * 100
            ),
            "model_loss_rate_percent": (
                model_losses / total_rows * 100
            ),
            "model_mae": df["absolute_error"].mean(),
            "baseline_mae": df["baseline_absolute_error"].mean(),
        }
    ]
)

baseline_comparison.to_csv(
    OUTPUT_DIR / "baseline_comparison.csv",
    index=False
)

print(
    baseline_comparison.to_string(index=False)
)


# ============================================================
# 6. BIAS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. PREDICTION BIAS ANALYSIS")
print("=" * 70)

mean_error = df["prediction_error"].mean()
median_error = df["prediction_error"].median()
mean_actual = df["target_next_week_price"].mean()
mean_prediction = df["prediction"].mean()

bias_percent = (
    mean_error / mean_actual
) * 100

under_predictions = int(
    (df["prediction"] < df["target_next_week_price"]).sum()
)

over_predictions = int(
    (df["prediction"] > df["target_next_week_price"]).sum()
)

exact_predictions = int(
    (df["prediction"] == df["target_next_week_price"]).sum()
)

bias_analysis = pd.DataFrame(
    [
        {
            "mean_prediction_error": mean_error,
            "median_prediction_error": median_error,
            "mean_actual_price": mean_actual,
            "mean_predicted_price": mean_prediction,
            "bias_percent": bias_percent,
            "under_predictions": under_predictions,
            "over_predictions": over_predictions,
            "exact_predictions": exact_predictions,
            "under_prediction_rate_percent": (
                under_predictions / total_rows * 100
            ),
            "over_prediction_rate_percent": (
                over_predictions / total_rows * 100
            ),
        }
    ]
)

bias_analysis.to_csv(
    OUTPUT_DIR / "bias_analysis.csv",
    index=False
)

print(bias_analysis.to_string(index=False))

if bias_percent < 0:
    print("\nInterpretation: MODEL TENDS TO UNDERSHOOT.")
elif bias_percent > 0:
    print("\nInterpretation: MODEL TENDS TO OVERSHOOT.")
else:
    print("\nInterpretation: NO MEAN SYSTEMATIC BIAS.")


# ============================================================
# 7. ERROR SIZE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("7. ERROR SIZE DISTRIBUTION")
print("=" * 70)

error_distribution = pd.DataFrame(
    [
        {
            "category": "Error <= 5%",
            "count": int(
                (df["percentage_error"] <= 5).sum()
            ),
        },
        {
            "category": "Error 5-10%",
            "count": int(
                (
                    (df["percentage_error"] > 5)
                    & (df["percentage_error"] <= 10)
                ).sum()
            ),
        },
        {
            "category": "Error 10-20%",
            "count": int(
                (
                    (df["percentage_error"] > 10)
                    & (df["percentage_error"] <= 20)
                ).sum()
            ),
        },
        {
            "category": "Error 20-30%",
            "count": int(
                (
                    (df["percentage_error"] > 20)
                    & (df["percentage_error"] <= 30)
                ).sum()
            ),
        },
        {
            "category": "Error > 30%",
            "count": int(
                (df["percentage_error"] > 30).sum()
            ),
        },
    ]
)

error_distribution["percentage"] = (
    error_distribution["count"]
    / total_rows
) * 100

print(error_distribution.to_string(index=False))


# ============================================================
# 8. OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("8. OVERALL ERROR ANALYSIS SUMMARY")
print("=" * 70)

summary = {
    "total_predictions": total_rows,

    "model_mae": df["absolute_error"].mean(),

    "model_rmse": np.sqrt(
        np.mean(df["prediction_error"] ** 2)
    ),

    "model_mape": df["percentage_error"].mean(),

    "baseline_mae": df["baseline_absolute_error"].mean(),

    "mae_improvement_percent": (
        (
            df["baseline_absolute_error"].mean()
            - df["absolute_error"].mean()
        )
        / df["baseline_absolute_error"].mean()
    ) * 100,

    "directional_accuracy": (
        df["direction_correct"].mean() * 100
    ),

    "mean_prediction_error": mean_error,

    "median_prediction_error": median_error,

    "bias_percent": bias_percent,

    "model_win_rate": (
        model_wins / total_rows * 100
    ),

    "model_loss_rate": (
        model_losses / total_rows * 100
    ),

    "errors_over_10_percent": int(
        df["large_error_10_percent"].sum()
    ),

    "errors_over_20_percent": int(
        df["large_error_20_percent"].sum()
    ),

    "errors_over_30_percent": int(
        df["large_error_30_percent"].sum()
    ),

    "worst_absolute_error": df["absolute_error"].max(),

    "worst_percentage_error": df["percentage_error"].max(),
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    OUTPUT_DIR / "error_analysis_summary.csv",
    index=False
)

print(summary_df.to_string(index=False))


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL ERROR ANALYSIS")
print("=" * 70)

worst_row = df.loc[
    df["absolute_error"].idxmax()
]

best_district = district_analysis.iloc[0]
worst_district = district_analysis.iloc[-1]

worst_period = period_analysis.loc[
    period_analysis["mae"].idxmax()
]

print(
    f"\nWorst individual prediction:"
    f"\n  District : {worst_row['district']}"
    f"\n  Period   : {worst_row['time_index']}"
    f"\n  Actual   : ₹{worst_row['target_next_week_price']:.2f}"
    f"\n  Predicted: ₹{worst_row['prediction']:.2f}"
    f"\n  Error    : ₹{worst_row['absolute_error']:.2f}"
)

print(
    f"\nBest district:"
    f"\n  {best_district['district']}"
    f"\n  MAE: ₹{best_district['mae']:.2f}"
    f"\n  Direction: {best_district['directional_accuracy']:.2f}%"
)

print(
    f"\nWorst district:"
    f"\n  {worst_district['district']}"
    f"\n  MAE: ₹{worst_district['mae']:.2f}"
    f"\n  Direction: {worst_district['directional_accuracy']:.2f}%"
)

print(
    f"\nWorst time period:"
    f"\n  Period: {worst_period['time_index']}"
    f"\n  MAE: ₹{worst_period['mae']:.2f}"
    f"\n  Baseline MAE: ₹{worst_period['baseline_mae']:.2f}"
    f"\n  Direction: {worst_period['directional_accuracy']:.2f}%"
)

print("\n" + "=" * 70)
print("GENERATED FILES")
print("=" * 70)

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  - {file}")

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
