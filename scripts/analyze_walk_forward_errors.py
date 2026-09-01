from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "agmarknet" / "model_results" / "walk_forward_predictions.csv"
OUTPUT = ROOT / "data" / "processed" / "agmarknet" / "model_results"

df = pd.read_csv(INPUT)

print("Loaded:", INPUT)
print("Rows:", len(df))
print("Columns:", len(df.columns))

required = [
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

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["model_better"] = (
    df["absolute_error"] < df["baseline_absolute_error"]
)

df["model_worse"] = (
    df["absolute_error"] > df["baseline_absolute_error"]
)

df["error_difference"] = (
    df["absolute_error"] - df["baseline_absolute_error"]
)

df["percentage_error"] = (
    df["absolute_error"] / df["target_next_week_price"].abs()
) * 100

district = (
    df.groupby("district")
    .agg(
        rows=("district", "size"),
        mae=("absolute_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        mean_error=("prediction_error", "mean"),
        median_error=("prediction_error", "median"),
        mean_absolute_percentage_error=("percentage_error", "mean"),
        directional_accuracy=("direction_correct", "mean"),
        model_wins=("model_better", "sum"),
        model_losses=("model_worse", "sum"),
    )
    .reset_index()
)

district["mae_improvement"] = (
    (district["baseline_mae"] - district["mae"])
    / district["baseline_mae"]
) * 100

district["directional_accuracy"] *= 100

district = district.sort_values("mae", ascending=True)

period = (
    df.groupby("time_index")
    .agg(
        rows=("time_index", "size"),
        mae=("absolute_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        mean_error=("prediction_error", "mean"),
        median_error=("prediction_error", "median"),
        directional_accuracy=("direction_correct", "mean"),
        model_wins=("model_better", "sum"),
        model_losses=("model_worse", "sum"),
    )
    .reset_index()
)

period["mae_improvement"] = (
    (period["baseline_mae"] - period["mae"])
    / period["baseline_mae"]
) * 100

period["directional_accuracy"] *= 100

worst = (
    df[
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
            "prediction_error",
            "absolute_error",
            "baseline_absolute_error",
            "error_difference",
            "model_direction",
            "actual_direction",
            "direction_correct",
        ]
    ]
    .sort_values("absolute_error", ascending=False)
)

summary = pd.DataFrame(
    {
        "metric": [
            "rows",
            "model_mae",
            "baseline_mae",
            "mae_improvement_percent",
            "model_rmse",
            "baseline_rmse",
            "model_mape_percent",
            "baseline_mape_percent",
            "directional_accuracy_percent",
            "mean_prediction_error",
            "median_prediction_error",
            "model_wins",
            "model_losses",
            "model_win_rate_percent",
        ],
        "value": [
            len(df),
            df["absolute_error"].mean(),
            df["baseline_absolute_error"].mean(),
            (
                (df["baseline_absolute_error"].mean() - df["absolute_error"].mean())
                / df["baseline_absolute_error"].mean()
            ) * 100,
            (df["prediction_error"] ** 2).mean() ** 0.5,
            (df["baseline_error"] ** 2).mean() ** 0.5,
            (
                df["absolute_error"]
                / df["target_next_week_price"].abs()
            ).mean() * 100,
            (
                df["baseline_absolute_error"]
                / df["target_next_week_price"].abs()
            ).mean() * 100,
            df["direction_correct"].mean() * 100,
            df["prediction_error"].mean(),
            df["prediction_error"].median(),
            df["model_better"].sum(),
            df["model_worse"].sum(),
            df["model_better"].mean() * 100,
        ],
    }
)

OUTPUT.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUTPUT / "walk_forward_error_analysis_summary.csv",
    index=False,
)

district.to_csv(
    OUTPUT / "walk_forward_error_by_district.csv",
    index=False,
)

period.to_csv(
    OUTPUT / "walk_forward_error_by_period.csv",
    index=False,
)

worst.to_csv(
    OUTPUT / "walk_forward_worst_predictions.csv",
    index=False,
)

print()
print("Error analysis complete.")
print()
print("Generated:")
print("  walk_forward_error_analysis_summary.csv")
print("  walk_forward_error_by_district.csv")
print("  walk_forward_error_by_period.csv")
print("  walk_forward_worst_predictions.csv")
print()
print("Top 10 worst predictions:")
print(
    worst[
        [
            "time_index",
            "district",
            "target_next_week_price",
            "prediction",
            "absolute_error",
            "baseline_absolute_error",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("Districts where the model performs worst:")
print(
    district[
        [
            "district",
            "mae",
            "baseline_mae",
            "mae_improvement",
            "directional_accuracy",
        ]
    ]
    .sort_values("mae", ascending=False)
    .head(10)
    .to_string(index=False)
)