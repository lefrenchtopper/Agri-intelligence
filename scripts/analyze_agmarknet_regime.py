from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/processed/agmarknet/features/"
    "onion_tamilnadu_forecasting_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/agmarknet/model_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_PRICE = "target_next_week_price"

CURRENT_PRICE = "current_price"

TARGET_CHANGE = "target_next_week_change"

DIRECTION_THRESHOLD = 0.02


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AGMARKNET TEMPORAL / REGIME ANALYSIS")
print("=" * 70)

print(f"Input: {FEATURE_FILE}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print(f"Rows : {len(df)}")
print(f"Cols : {len(df.columns)}")
print()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "year",
    "month",
    "week",
    "district",
    "time_index",
    CURRENT_PRICE,
    TARGET_PRICE,
    TARGET_CHANGE,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# CHRONOLOGICAL ORDER
# ============================================================

df = df.sort_values(
    [
        "time_index",
        "district",
    ]
).reset_index(drop=True)


# ============================================================
# TIME PERIOD SUMMARY
# ============================================================

print("=" * 70)
print("TIME PERIOD ANALYSIS")
print("=" * 70)

time_summary = (
    df.groupby("time_index")
    .agg(
        year=("year", "first"),
        month=("month", "first"),
        week=("week", "first"),

        districts=("district", "nunique"),

        current_price_mean=(
            CURRENT_PRICE,
            "mean"
        ),

        current_price_median=(
            CURRENT_PRICE,
            "median"
        ),

        current_price_min=(
            CURRENT_PRICE,
            "min"
        ),

        current_price_max=(
            CURRENT_PRICE,
            "max"
        ),

        next_price_mean=(
            TARGET_PRICE,
            "mean"
        ),

        next_price_median=(
            TARGET_PRICE,
            "median"
        ),

        next_change_mean=(
            TARGET_CHANGE,
            "mean"
        ),

        next_change_median=(
            TARGET_CHANGE,
            "median"
        ),
    )
    .reset_index()
)


# ============================================================
# DIRECTION TARGET
# ============================================================

df["direction_target"] = (
    df[TARGET_CHANGE]
    >= DIRECTION_THRESHOLD
).astype(int)


# ============================================================
# DIRECTION BY TIME
# ============================================================

direction_summary = (
    df.groupby("time_index")
    .agg(
        up_count=(
            "direction_target",
            "sum"
        ),

        total_count=(
            "direction_target",
            "count"
        ),

        average_change=(
            TARGET_CHANGE,
            "mean"
        ),
    )
    .reset_index()
)

direction_summary["up_percentage"] = (
    direction_summary["up_count"]
    / direction_summary["total_count"]
    * 100
)


time_summary = time_summary.merge(
    direction_summary,
    on="time_index",
    how="left"
)


# ============================================================
# PRINT TIME SUMMARY
# ============================================================

print()

print(
    time_summary[
        [
            "time_index",
            "year",
            "month",
            "week",
            "current_price_mean",
            "next_price_mean",
            "next_change_mean",
            "up_percentage",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

print()


# ============================================================
# SPLIT PERIODS
# ============================================================

unique_times = sorted(
    df["time_index"].unique()
)

n_times = len(unique_times)

train_end = int(n_times * 0.70)

validation_end = int(n_times * 0.85)

train_times = unique_times[
    :train_end
]

validation_times = unique_times[
    train_end:validation_end
]

test_times = unique_times[
    validation_end:
]


print("=" * 70)
print("TEMPORAL SPLIT ANALYSIS")
print("=" * 70)

print(
    f"Train      : "
    f"{min(train_times)} - {max(train_times)}"
)

print(
    f"Validation : "
    f"{min(validation_times)} - "
    f"{max(validation_times)}"
)

print(
    f"Test       : "
    f"{min(test_times)} - {max(test_times)}"
)

print()


# ============================================================
# CREATE SPLIT LABEL
# ============================================================

def assign_split(time_index):

    if time_index in train_times:
        return "train"

    if time_index in validation_times:
        return "validation"

    if time_index in test_times:
        return "test"

    return "unknown"


df["split"] = df[
    "time_index"
].apply(assign_split)


# ============================================================
# SPLIT STATISTICS
# ============================================================

print("=" * 70)
print("SPLIT DISTRIBUTION")
print("=" * 70)

split_summary = (
    df.groupby("split")
    .agg(
        rows=("district", "count"),

        districts=("district", "nunique"),

        current_price_mean=(
            CURRENT_PRICE,
            "mean"
        ),

        current_price_median=(
            CURRENT_PRICE,
            "median"
        ),

        current_price_std=(
            CURRENT_PRICE,
            "std"
        ),

        next_price_mean=(
            TARGET_PRICE,
            "mean"
        ),

        next_price_median=(
            TARGET_PRICE,
            "median"
        ),

        next_price_std=(
            TARGET_PRICE,
            "std"
        ),

        next_change_mean=(
            TARGET_CHANGE,
            "mean"
        ),

        next_change_std=(
            TARGET_CHANGE,
            "std"
        ),

        up_percentage=(
            "direction_target",
            "mean"
        ),
    )
    .reset_index()
)

split_summary[
    "up_percentage"
] *= 100


print(
    split_summary
    .round(3)
    .to_string(index=False)
)

print()


# ============================================================
# TRAIN VS TEST PRICE SHIFT
# ============================================================

train_df = df[
    df["split"] == "train"
]

validation_df = df[
    df["split"] == "validation"
]

test_df = df[
    df["split"] == "test"
]


def percentage_difference(
    old,
    new,
):

    if old == 0:
        return np.nan

    return (
        (new - old)
        / abs(old)
        * 100
    )


train_price = train_df[
    CURRENT_PRICE
].mean()

validation_price = validation_df[
    CURRENT_PRICE
].mean()

test_price = test_df[
    CURRENT_PRICE
].mean()


print("=" * 70)
print("PRICE REGIME SHIFT")
print("=" * 70)

print(
    f"Train mean current price      : "
    f"₹{train_price:.2f}"
)

print(
    f"Validation mean current price : "
    f"₹{validation_price:.2f}"
)

print(
    f"Test mean current price       : "
    f"₹{test_price:.2f}"
)

print()

print(
    "Validation vs train:"
)

print(
    f"  {percentage_difference(train_price, validation_price):.2f}%"
)

print(
    "Test vs train:"
)

print(
    f"  {percentage_difference(train_price, test_price):.2f}%"
)

print()


# ============================================================
# TARGET DISTRIBUTION SHIFT
# ============================================================

train_change = train_df[
    TARGET_CHANGE
].mean()

validation_change = validation_df[
    TARGET_CHANGE
].mean()

test_change = test_df[
    TARGET_CHANGE
].mean()


print("=" * 70)
print("TARGET CHANGE SHIFT")
print("=" * 70)

print(
    f"Train mean next-week change      : "
    f"{train_change * 100:.2f}%"
)

print(
    f"Validation mean next-week change : "
    f"{validation_change * 100:.2f}%"
)

print(
    f"Test mean next-week change       : "
    f"{test_change * 100:.2f}%"
)

print()


# ============================================================
# DIRECTION DISTRIBUTION
# ============================================================

print("=" * 70)
print("DIRECTION DISTRIBUTION")
print("=" * 70)

for split_name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    total = len(split_df)

    up = (
        split_df["direction_target"]
        .sum()
    )

    not_up = total - up

    print()
    print(split_name)

    print(
        f"  UP     : {up} "
        f"({up / total * 100:.2f}%)"
    )

    print(
        f"  NOT UP : {not_up} "
        f"({not_up / total * 100:.2f}%)"
    )


print()


# ============================================================
# DISTRICT REGIME ANALYSIS
# ============================================================

print("=" * 70)
print("DISTRICT REGIME ANALYSIS")
print("=" * 70)


district_shift = []


for district, group in df.groupby(
    "district"
):

    train_group = group[
        group["split"] == "train"
    ]

    test_group = group[
        group["split"] == "test"
    ]

    if len(train_group) == 0:
        continue

    if len(test_group) == 0:
        continue

    train_mean = train_group[
        CURRENT_PRICE
    ].mean()

    test_mean = test_group[
        CURRENT_PRICE
    ].mean()

    train_change = train_group[
        TARGET_CHANGE
    ].mean()

    test_change = test_group[
        TARGET_CHANGE
    ].mean()

    train_up = (
        train_group[
            "direction_target"
        ].mean()
        * 100
    )

    test_up = (
        test_group[
            "direction_target"
        ].mean()
        * 100
    )

    district_shift.append(
        {
            "district": district,

            "train_price":
                train_mean,

            "test_price":
                test_mean,

            "price_shift_percent":
                percentage_difference(
                    train_mean,
                    test_mean,
                ),

            "train_change_percent":
                train_change * 100,

            "test_change_percent":
                test_change * 100,

            "train_up_percent":
                train_up,

            "test_up_percent":
                test_up,

            "direction_shift_percentage_points":
                test_up - train_up,
        }
    )


district_shift = pd.DataFrame(
    district_shift
)

district_shift = district_shift.sort_values(
    "price_shift_percent"
)


print(
    district_shift
    .round(2)
    .to_string(index=False)
)

print()


# ============================================================
# FEATURE CORRELATION ANALYSIS
# ============================================================

print("=" * 70)
print("FEATURE / TARGET CORRELATION")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()


excluded_for_correlation = [
    TARGET_PRICE,
    "direction_target",
    "split",
]


correlations = []


for column in numeric_columns:

    if column in excluded_for_correlation:
        continue

    try:

        correlation = df[
            column
        ].corr(
            df[TARGET_CHANGE]
        )

        correlations.append(
            {
                "feature": column,
                "correlation_with_next_change":
                    correlation,
                "absolute_correlation":
                    abs(correlation),
            }
        )

    except Exception:

        pass


correlations = pd.DataFrame(
    correlations
).sort_values(
    "absolute_correlation",
    ascending=False
)


print(
    correlations
    .head(20)
    .round(4)
    .to_string(index=False)
)

print()


# ============================================================
# TRAIN / TEST CORRELATION COMPARISON
# ============================================================

print("=" * 70)
print("TRAIN VS TEST FEATURE RELATIONSHIP")
print("=" * 70)


comparison = []


candidate_features = [
    column
    for column in numeric_columns
    if column not in [
        TARGET_PRICE,
        "direction_target",
    ]
]


for feature in candidate_features:

    try:

        train_corr = train_df[
            feature
        ].corr(
            train_df[TARGET_CHANGE]
        )

        test_corr = test_df[
            feature
        ].corr(
            test_df[TARGET_CHANGE]
        )

        comparison.append(
            {
                "feature": feature,
                "train_correlation":
                    train_corr,
                "test_correlation":
                    test_corr,
                "correlation_change":
                    test_corr - train_corr,
            }
        )

    except Exception:

        pass


comparison = pd.DataFrame(
    comparison
)

comparison["absolute_change"] = (
    comparison[
        "correlation_change"
    ].abs()
)

comparison = comparison.sort_values(
    "absolute_change",
    ascending=False
)


print(
    comparison
    .head(20)
    .round(4)
    .to_string(index=False)
)

print()


# ============================================================
# EXTREME PRICE MOVEMENTS
# ============================================================

print("=" * 70)
print("EXTREME NEXT-WEEK MOVEMENTS")
print("=" * 70)

extreme = df[
    [
        "year",
        "month",
        "week",
        "district",
        CURRENT_PRICE,
        TARGET_PRICE,
        TARGET_CHANGE,
        "split",
    ]
].copy()


extreme["change_percent"] = (
    extreme[TARGET_CHANGE]
    * 100
)


largest_increases = (
    extreme
    .sort_values(
        "change_percent",
        ascending=False
    )
    .head(10)
)


largest_decreases = (
    extreme
    .sort_values(
        "change_percent",
        ascending=True
    )
    .head(10)
)


print()
print("Largest increases:")

print(
    largest_increases
    .round(2)
    .to_string(index=False)
)


print()
print("Largest decreases:")

print(
    largest_decreases
    .round(2)
    .to_string(index=False)
)

print()


# ============================================================
# SAVE RESULTS
# ============================================================

time_file = (
    OUTPUT_DIR
    / "temporal_regime_summary.csv"
)

split_file = (
    OUTPUT_DIR
    / "temporal_split_summary.csv"
)

district_file = (
    OUTPUT_DIR
    / "district_regime_shift.csv"
)

correlation_file = (
    OUTPUT_DIR
    / "feature_target_correlations.csv"
)

relationship_file = (
    OUTPUT_DIR
    / "train_test_feature_relationships.csv"
)

extreme_file = (
    OUTPUT_DIR
    / "extreme_price_movements.csv"
)


time_summary.to_csv(
    time_file,
    index=False
)

split_summary.to_csv(
    split_file,
    index=False
)

district_shift.to_csv(
    district_file,
    index=False
)

correlations.to_csv(
    correlation_file,
    index=False
)

comparison.to_csv(
    relationship_file,
    index=False
)

extreme.to_csv(
    extreme_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("REGIME ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Generated:")

print(
    f"  - {time_file}"
)

print(
    f"  - {split_file}"
)

print(
    f"  - {district_file}"
)

print(
    f"  - {correlation_file}"
)

print(
    f"  - {relationship_file}"
)

print(
    f"  - {extreme_file}"
)

print()

print("=" * 70)
print("KEY QUESTIONS FOR NEXT STEP")
print("=" * 70)

print(
    "1. Did the test period have a different price regime?"
)

print(
    "2. Did UP/DOWN frequency change substantially?"
)

print(
    "3. Which features changed their relationship with future price?"
)

print(
    "4. Which districts behave consistently?"
)

print(
    "5. Are the available 27 time periods sufficient?"
)

print()

print(
    "Do NOT retrain another model yet."
)

print(
    "Use this analysis to decide the next modeling strategy."
)

print()