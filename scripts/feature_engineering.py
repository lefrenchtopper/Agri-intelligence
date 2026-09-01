from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/agmarknet/onion_tamilnadu_weekly_2026.csv"
)

OUTPUT_DIR = Path("data/processed/agmarknet/features")

OUTPUT_FILE = OUTPUT_DIR / "onion_tamilnadu_forecasting_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("AGMARKNET FEATURE ENGINEERING")
    print("=" * 70)

    print(f"Input : {INPUT_FILE}")
    print(f"Rows  : {len(df)}")
    print()

    return df


# ============================================================
# SORT DATA
# ============================================================

def sort_data(df):

    df = df.sort_values(
        by=["district", "year", "month", "week"]
    ).reset_index(drop=True)

    return df


# ============================================================
# CREATE TIME FEATURES
# ============================================================

def create_time_features(df):

    # --------------------------------------------------------
    # Sequential week number
    # --------------------------------------------------------

    df["time_index"] = (
        (df["month"] - 1) * 4
        + (df["week"] - 1)
    )

    # --------------------------------------------------------
    # Cyclical representation of month
    # --------------------------------------------------------

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # --------------------------------------------------------
    # Cyclical representation of week
    # --------------------------------------------------------

    df["week_sin"] = np.sin(
        2 * np.pi * df["week"] / 4
    )

    df["week_cos"] = np.cos(
        2 * np.pi * df["week"] / 4
    )

    return df


# ============================================================
# CREATE LAG FEATURES
# ============================================================

def create_lag_features(df):

    # IMPORTANT:
    #
    # shift(1) means the value from the previous observation.
    #
    # Therefore these features only use information that was
    # already available before the prediction point.

    grouped = df.groupby("district")["current_price"]

    df["price_lag_1"] = grouped.shift(1)
    df["price_lag_2"] = grouped.shift(2)
    df["price_lag_3"] = grouped.shift(3)
    df["price_lag_4"] = grouped.shift(4)

    return df


# ============================================================
# ROLLING PRICE FEATURES
# ============================================================

def create_rolling_features(df):

    grouped = df.groupby("district")["current_price"]

    # --------------------------------------------------------
    # Shift first, then rolling.
    #
    # This is VERY important.
    #
    # We do NOT calculate:
    #
    # rolling(4)
    #
    # directly on current_price because that could include
    # the price we are trying to predict.
    # --------------------------------------------------------

    previous_prices = grouped.shift(1)

    df["price_rolling_mean_4"] = (
        previous_prices
        .groupby(df["district"])
        .transform(
            lambda x: x.rolling(4, min_periods=2).mean()
        )
    )

    df["price_rolling_std_4"] = (
        previous_prices
        .groupby(df["district"])
        .transform(
            lambda x: x.rolling(4, min_periods=2).std()
        )
    )

    # Longer historical window

    df["price_rolling_mean_8"] = (
        previous_prices
        .groupby(df["district"])
        .transform(
            lambda x: x.rolling(8, min_periods=2).mean()
        )
    )

    df["price_mean_7d"] = (
        previous_prices
        .groupby(df["district"])
        .transform(
            lambda x: x.rolling(7, min_periods=2).mean()
        )
    )

    df["price_rolling_std_7w"] = (
        previous_prices
        .groupby(df["district"])
        .transform(
            lambda x: x.rolling(7, min_periods=2).std()
        )
    )

    df["price_lag_7d"] = grouped.shift(7)

    df["rolling_max_90d"] = (
        previous_prices
        .groupby(df["district"])
        .transform(
            lambda x: x.rolling(90, min_periods=2).max()
        )
    )

    df["relative_volatility_7w"] = (
        df["price_rolling_std_7w"] / df["price_mean_7d"]
    )

    df["price_velocity_7d"] = (
        (df["current_price"] - df["price_lag_7d"])
        / df["price_lag_7d"]
    )

    df["historical_max_ratio"] = (
        df["current_price"] / df["rolling_max_90d"]
    )

    return df


# ============================================================
# PRICE MOMENTUM
# ============================================================

def create_momentum_features(df):

    # --------------------------------------------------------
    # Percentage movement from previous week
    # --------------------------------------------------------

    df["price_change_1w"] = (
        df["current_price"]
        / df["price_lag_1"]
        - 1
    )

    # --------------------------------------------------------
    # Percentage movement from two weeks ago
    # --------------------------------------------------------

    df["price_change_2w"] = (
        df["current_price"]
        / df["price_lag_2"]
        - 1
    )

    # --------------------------------------------------------
    # Percentage movement from four weeks ago
    # --------------------------------------------------------

    df["price_change_4w"] = (
        df["current_price"]
        / df["price_lag_4"]
        - 1
    )

    return df


# ============================================================
# PREVIOUS PRICE FEATURES
# ============================================================

def create_reference_features(df):

    # These are historical reference prices already present
    # in the Agmarknet report.

    df["week_vs_previous_week"] = (
        df["current_price"]
        / df["previous_week_price"]
        - 1
    )

    df["month_vs_previous_month"] = (
        df["current_price"]
        / df["previous_month_price"]
        - 1
    )

    df["year_vs_previous_year"] = (
        df["current_price"]
        / df["previous_year_price"]
        - 1
    )

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(df):

    # --------------------------------------------------------
    # TARGET:
    #
    # next week's actual price
    #
    # current week information → next week price
    # --------------------------------------------------------

    df["target_next_week_price"] = (
        df.groupby("district")["current_price"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # Target percentage change
    #
    # This will eventually be useful for the recommendation
    # system.
    # --------------------------------------------------------

    df["target_next_week_change"] = (
        df["target_next_week_price"]
        / df["current_price"]
        - 1
    )

    return df


# ============================================================
# CLEAN TRAINING DATA
# ============================================================

def prepare_training_data(df):

    # The first few rows of each district naturally have
    # missing lag features.
    #
    # The final row has no next-week target.
    #
    # We do NOT fill these with future information.

    required_features = [
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_lag_4",
        "price_rolling_mean_4",
        "price_rolling_std_4",
        "price_rolling_mean_8",
        "price_mean_7d",
        "price_rolling_std_7w",
        "price_lag_7d",
        "rolling_max_90d",
        "relative_volatility_7w",
        "price_velocity_7d",
        "historical_max_ratio",
        "target_next_week_price",
    ]

    before = len(df)

    df = df.dropna(
        subset=required_features
    ).copy()

    after = len(df)

    print()
    print("=" * 70)
    print("TRAINING DATA PREPARATION")
    print("=" * 70)

    print(f"Rows before removing incomplete history : {before}")
    print(f"Rows after removing incomplete history  : {after}")
    print(f"Rows removed                            : {before - after}")

    return df


# ============================================================
# SELECT FINAL FEATURES
# ============================================================

def select_columns(df):

    feature_columns = [

        # Identification
        "year",
        "month",
        "week",
        "district",

        # Current known price
        "current_price",

        # Historical reference prices
        "previous_week_price",
        "previous_month_price",
        "previous_year_price",

        # Historical changes
        "change_previous_week",
        "change_previous_month",
        "change_previous_year",

        # Lag prices
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "price_lag_4",

        # Rolling statistics
        "price_rolling_mean_4",
        "price_rolling_std_4",
        "price_rolling_mean_8",
        "price_mean_7d",
        "price_rolling_std_7w",
        "price_lag_7d",
        "rolling_max_90d",
        "relative_volatility_7w",
        "price_velocity_7d",
        "historical_max_ratio",

        # Momentum
        "price_change_1w",
        "price_change_2w",
        "price_change_4w",

        # Reference movements
        "week_vs_previous_week",
        "month_vs_previous_month",
        "year_vs_previous_year",

        # Time
        "time_index",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",

        # TARGET
        "target_next_week_price",
        "target_next_week_change",
    ]

    return df[feature_columns]


# ============================================================
# VALIDATION
# ============================================================

def validate(df):

    print()
    print("=" * 70)
    print("FEATURE DATASET VALIDATION")
    print("=" * 70)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print()
    print("Missing values:")

    print(
        df.isna()
        .sum()
        .to_string()
    )

    print()
    print("Target summary:")

    print(
        df[
            [
                "target_next_week_price",
                "target_next_week_change"
            ]
        ]
        .describe()
        .round(3)
        .to_string()
    )


# ============================================================
# SAVE
# ============================================================

def save_data(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(f"Saved to:")
    print(OUTPUT_FILE.resolve())


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = sort_data(df)

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_momentum_features(df)

    df = create_reference_features(df)

    df = create_target(df)

    df = prepare_training_data(df)

    df = select_columns(df)

    validate(df)

    save_data(df)


if __name__ == "__main__":
    main()