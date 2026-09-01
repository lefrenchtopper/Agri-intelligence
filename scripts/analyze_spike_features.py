from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "model_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading forecasting features...")
print(f"File: {FEATURE_FILE}")

df = pd.read_csv(FEATURE_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

required = [
    "time_index",
    "district",
    "current_price",
    "target_next_week_price",
]

missing = [column for column in required if column not in df.columns]

if missing:
    print("ERROR: Missing required columns:")
    for column in missing:
        print(f"  - {column}")

    print()
    print("Available columns:")
    for column in df.columns:
        print(f"  - {column}")

    raise SystemExit(1)

print("Inspecting feature columns...")
print()

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numeric feature columns: {len(numeric_columns)}")

for column in numeric_columns:
    if column not in required:
        print(f"  {column}")

print()

# Calculate the actual next-week movement from the data available at each period.
df["actual_change"] = (
    df["target_next_week_price"] - df["current_price"]
)

df["actual_change_percent"] = (
    df["actual_change"]
    / df["current_price"].replace(0, np.nan)
) * 100

df["absolute_change_percent"] = df["actual_change_percent"].abs()

# Identify the periods that contain the major movements found in the
# walk-forward error analysis.
df["movement_category"] = "stable"

df.loc[df["actual_change_percent"] <= -10, "movement_category"] = "large_decrease"

df.loc[
    (df["actual_change_percent"] > -10)
    & (df["actual_change_percent"] <= -5),
    "movement_category",
] = "moderate_decrease"

df.loc[
    (df["actual_change_percent"] >= 5)
    & (df["actual_change_percent"] < 10),
    "movement_category",
] = "moderate_increase"

df.loc[
    df["actual_change_percent"] >= 10,
    "movement_category",
] = "large_increase"

# The main spike period identified by the previous analysis.
spike_periods = [29, 30]

spike_df = df[df["time_index"].isin(spike_periods)].copy()

normal_df = df[~df["time_index"].isin(spike_periods)].copy()

print("Spike-period rows:")
print(
    spike_df[
        [
            "time_index",
            "district",
            "current_price",
            "target_next_week_price",
            "actual_change_percent",
        ]
    ]
    .sort_values(["time_index", "district"])
    .to_string(index=False)
)

print()

# Compare numerical feature distributions between spike and non-spike periods.
comparison_rows = []

for column in numeric_columns:
    if column in [
        "time_index",
        "year",
        "month",
        "week",
        "target_next_week_price",
    ]:
        continue

    spike_values = pd.to_numeric(
        spike_df[column],
        errors="coerce",
    ).dropna()

    normal_values = pd.to_numeric(
        normal_df[column],
        errors="coerce",
    ).dropna()

    if len(spike_values) == 0 or len(normal_values) == 0:
        continue

    spike_mean = spike_values.mean()
    normal_mean = normal_values.mean()

    spike_median = spike_values.median()
    normal_median = normal_values.median()

    spike_std = spike_values.std()
    normal_std = normal_values.std()

    normal_abs_mean = normal_values.abs().mean()

    if normal_abs_mean == 0:
        relative_difference = np.nan
    else:
        relative_difference = (
            abs(spike_mean - normal_mean)
            / normal_abs_mean
        ) * 100

    comparison_rows.append(
        {
            "feature": column,
            "spike_mean": spike_mean,
            "normal_mean": normal_mean,
            "spike_median": spike_median,
            "normal_median": normal_median,
            "spike_std": spike_std,
            "normal_std": normal_std,
            "relative_difference_percent": relative_difference,
        }
    )

comparison = pd.DataFrame(comparison_rows)

if not comparison.empty:
    comparison = comparison.sort_values(
        "relative_difference_percent",
        ascending=False,
    )

comparison_file = (
    OUTPUT_DIR
    / "walk_forward_spike_feature_comparison.csv"
)

comparison.to_csv(comparison_file, index=False)

# Compare feature values immediately before the major spike.
# This is more useful than simply comparing period 29 itself because
# period 29's target is the future price that the model was trying to predict.
available_periods = sorted(df["time_index"].dropna().unique())

pre_spike_periods = []

for spike_period in spike_periods:
    previous_periods = [
        period
        for period in available_periods
        if period < spike_period
    ]

    pre_spike_periods.extend(previous_periods[-3:])

pre_spike_periods = sorted(set(pre_spike_periods))

pre_spike_df = df[
    df["time_index"].isin(pre_spike_periods)
].copy()

pre_spike_summary_rows = []

for column in numeric_columns:
    if column in [
        "time_index",
        "year",
        "month",
        "week",
        "target_next_week_price",
    ]:
        continue

    spike_values = pd.to_numeric(
        spike_df[column],
        errors="coerce",
    ).dropna()

    pre_values = pd.to_numeric(
        pre_spike_df[column],
        errors="coerce",
    ).dropna()

    if len(spike_values) == 0 or len(pre_values) == 0:
        continue

    pre_mean = pre_values.mean()
    spike_mean = spike_values.mean()

    if pre_mean == 0:
        percentage_change = np.nan
    else:
        percentage_change = (
            (spike_mean - pre_mean)
            / abs(pre_mean)
        ) * 100

    pre_spike_summary_rows.append(
        {
            "feature": column,
            "pre_spike_mean": pre_mean,
            "spike_period_mean": spike_mean,
            "change_percent": percentage_change,
        }
    )

pre_spike_summary = pd.DataFrame(
    pre_spike_summary_rows
)

if not pre_spike_summary.empty:
    pre_spike_summary["absolute_change_percent"] = (
        pre_spike_summary["change_percent"].abs()
    )

    pre_spike_summary = pre_spike_summary.sort_values(
        "absolute_change_percent",
        ascending=False,
    )

pre_spike_file = (
    OUTPUT_DIR
    / "walk_forward_pre_spike_feature_changes.csv"
)

pre_spike_summary.to_csv(
    pre_spike_file,
    index=False,
)

# Analyze price acceleration by period.
period_price_summary = (
    df.groupby("time_index")
    .agg(
        rows=("current_price", "size"),
        mean_current_price=("current_price", "mean"),
        mean_target_price=("target_next_week_price", "mean"),
        mean_actual_change_percent=("actual_change_percent", "mean"),
        median_actual_change_percent=("actual_change_percent", "median"),
        mean_absolute_change_percent=("absolute_change_percent", "mean"),
    )
    .reset_index()
)

period_price_summary["previous_period_change_percent"] = (
    period_price_summary["mean_current_price"]
    .pct_change()
    * 100
)

period_price_summary["price_acceleration"] = (
    period_price_summary["previous_period_change_percent"]
    .diff()
)

period_file = (
    OUTPUT_DIR
    / "walk_forward_price_acceleration_analysis.csv"
)

period_price_summary.to_csv(
    period_file,
    index=False,
)

print("Feature comparison:")
print(
    comparison.head(20).to_string(index=False)
)

print()

print("Pre-spike feature changes:")
print(
    pre_spike_summary.head(20).to_string(index=False)
)

print()

print("Price acceleration around the final periods:")
print(
    period_price_summary.tail(10).to_string(index=False)
)

print()

print("Generated:")
print(f"  {comparison_file}")
print(f"  {pre_spike_file}")
print(f"  {period_file}")

print()
print("Spike feature analysis complete.")