from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    ROOT
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)

df = pd.read_csv(FEATURES)

print("Loaded:", FEATURES)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Time index distribution:")
print(df["time_index"].value_counts().sort_index())

periods = [27, 28, 29, 30]

subset = df[df["time_index"].isin(periods)].copy()

print()
print("Rows for periods 27-30:")
print(subset.to_string(index=False))

numeric = subset.select_dtypes(include="number")

summary = (
    numeric
    .groupby(subset["time_index"])
    .agg(["mean", "min", "max", "std"])
)

print()
print("Numeric feature summary for periods 27-30:")
print(summary.to_string())

period_means = numeric.groupby(subset["time_index"]).mean()

print()
print("Percentage changes between periods:")

for previous, current in zip(periods[:-1], periods[1:]):
    if previous not in period_means.index or current not in period_means.index:
        continue

    changes = (
        (period_means.loc[current] - period_means.loc[previous])
        / period_means.loc[previous].replace(0, pd.NA)
    ) * 100

    changes = changes.dropna()
    changes = changes.reindex(
        changes.abs().sort_values(ascending=False).index
    )

    print()
    print(f"{previous} -> {current}")
    print(changes.head(15).to_string())

print()
print("Actual price statistics:")

price_columns = [
    column
    for column in [
        "current_price",
        "target_next_week_price",
    ]
    if column in df.columns
]

if price_columns:
    print(
        subset[
            ["time_index", "district"] + price_columns
        ].sort_values(
            ["time_index", "district"]
        ).to_string(index=False)
    )