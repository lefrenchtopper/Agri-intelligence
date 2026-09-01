from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/agmarknet/onion_tamilnadu_weekly_2026.csv"
)

OUTPUT_DIR = Path("data/processed/agmarknet/eda")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AGMARKNET EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"Input: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print()


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("DATASET INFO")
print("=" * 70)

print(df.info())

print()


# ============================================================
# MISSING VALUES
# ============================================================

print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

print(missing.to_string())

print()


# ============================================================
# DUPLICATES
# ============================================================

print("=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicates = df.duplicated(
    subset=["year", "month", "week", "district"]
).sum()

print(f"Duplicate district-week observations: {duplicates}")

print()


# ============================================================
# DISTRICTS
# ============================================================

print("=" * 70)
print("DISTRICTS")
print("=" * 70)

districts = sorted(df["district"].unique())

print(f"Number of districts: {len(districts)}")

for district in districts:
    print(district)

print()


# ============================================================
# WEEKLY SUMMARY
# ============================================================

print("=" * 70)
print("WEEKLY PRICE SUMMARY")
print("=" * 70)

weekly = (
    df
    .groupby(["month", "month_name", "week"])
    ["current_price"]
    .agg(
        mean="mean",
        median="median",
        minimum="min",
        maximum="max",
        std="std"
    )
    .reset_index()
)

print(
    weekly.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

weekly.to_csv(
    OUTPUT_DIR / "weekly_summary.csv",
    index=False
)

print()


# ============================================================
# DISTRICT SUMMARY
# ============================================================

print("=" * 70)
print("DISTRICT PRICE SUMMARY")
print("=" * 70)

district_summary = (
    df
    .groupby("district")
    .agg(
        mean=("current_price", "mean"),
        median=("current_price", "median"),
        minimum=("current_price", "min"),
        maximum=("current_price", "max"),
        std=("current_price", "std")
    )
    .sort_values("mean", ascending=False)
)

print(
    district_summary.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

district_summary.to_csv(
    OUTPUT_DIR / "district_summary.csv"
)

print()


# ============================================================
# CORRELATION
# ============================================================

print("=" * 70)
print("NUMERIC CORRELATION")
print("=" * 70)

numeric_columns = [
    "current_price",
    "previous_week_price",
    "previous_month_price",
    "previous_year_price",
    "change_previous_week",
    "change_previous_month",
    "change_previous_year",
]

correlation = df[numeric_columns].corr()

print(
    correlation.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

correlation.to_csv(
    OUTPUT_DIR / "correlation.csv"
)

print()


# ============================================================
# TOP / BOTTOM OBSERVATIONS
# ============================================================

print("=" * 70)
print("HIGHEST CURRENT PRICES")
print("=" * 70)

highest = (
    df[
        [
            "month_name",
            "week",
            "district",
            "current_price"
        ]
    ]
    .sort_values("current_price", ascending=False)
    .head(10)
)

print(
    highest.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()

print("=" * 70)
print("LOWEST CURRENT PRICES")
print("=" * 70)

lowest = (
    df[
        [
            "month_name",
            "week",
            "district",
            "current_price"
        ]
    ]
    .sort_values("current_price")
    .head(10)
)

print(
    lowest.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ============================================================
# PLOT 1 — STATEWIDE WEEKLY PRICE
# ============================================================

df["time_index"] = (
    (df["month"] - 1) * 4 + df["week"]
)

statewide = (
    df
    .groupby("time_index")["current_price"]
    .mean()
)

plt.figure(figsize=(12, 6))

plt.plot(
    statewide.index,
    statewide.values,
    marker="o"
)

plt.title(
    "Average Onion Wholesale Price — Tamil Nadu"
)

plt.xlabel("Week")
plt.ylabel("Price (Rs./Quintal)")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "statewide_weekly_price.png",
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2 — DISTRICT PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(14, 7))

district_order = (
    df
    .groupby("district")["current_price"]
    .mean()
    .sort_values()
    .index
)

df.boxplot(
    column="current_price",
    by="district",
    figsize=(14, 7),
    rot=90
)

plt.suptitle("")

plt.title(
    "Onion Wholesale Price Distribution by District"
)

plt.xlabel("District")

plt.ylabel("Price (Rs./Quintal)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "district_price_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3 — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index.tolist()
)

plt.title("Price Feature Correlation")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=150
)

plt.close()


# ============================================================
# SAVE CLEAN EDA DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "eda_dataset.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"Output directory: {OUTPUT_DIR.resolve()}")

print()
print("Generated:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {file.name}")