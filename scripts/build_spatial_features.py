import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "district",
    "price_date",
    "modal_price_per_quintal",
}
UPSTREAM_STATIONS = {"nashik", "lasalgaon", "ahmednagar"}


def build_features(tamil_nadu_path, upstream_path, output_path):
    tamil_nadu = pd.read_csv(tamil_nadu_path)
    upstream = pd.read_csv(upstream_path)
    missing = sorted(REQUIRED_COLUMNS - set(upstream.columns))
    if missing:
        raise ValueError(f"Upstream source missing columns: {missing}")

    upstream["station"] = upstream["district"].fillna(
        upstream.get("market_name", "")
    ).astype(str).str.casefold()
    upstream = upstream[
        upstream["station"].str.contains(
            "|".join(UPSTREAM_STATIONS), regex=True
        )
    ].copy()
    if upstream.empty:
        raise ValueError("No Nashik/Lasalgaon/Ahmednagar rows found.")

    upstream["price_date"] = pd.to_datetime(upstream["price_date"])
    if {"source_year", "source_month", "source_week"}.issubset(upstream.columns):
        upstream["year"] = upstream["source_year"]
        upstream["month"] = upstream["source_month"]
        upstream["week"] = upstream["source_week"]
    else:
        upstream["year"] = upstream["price_date"].dt.year
        upstream["month"] = upstream["price_date"].dt.month
        upstream["week"] = ((upstream["price_date"].dt.day - 1) // 7) + 1
    weekly = (
        upstream.groupby(["station", "year", "month", "week"], as_index=False)
        ["modal_price_per_quintal"]
        .mean()
        .rename(columns={"modal_price_per_quintal": "nashik_price"})
    )
    weekly["nashik_price_lag1"] = weekly.groupby("station")[
        "nashik_price"
    ].shift(1)
    weekly["nashik_price_lag2"] = weekly.groupby("station")[
        "nashik_price"
    ].shift(2)
    weekly["nashik_momentum_2w"] = (
        weekly["nashik_price_lag1"]
        - weekly.groupby("station")["nashik_price"].shift(3)
    ) / weekly.groupby("station")["nashik_price"].shift(3)
    result = tamil_nadu.merge(
        weekly[[
            "year",
            "month",
            "week",
            "nashik_price_lag1",
            "nashik_price_lag2",
            "nashik_momentum_2w",
        ]].drop_duplicates(["year", "month", "week"]),
        on=["year", "month", "week"],
        how="left",
        validate="many_to_one",
    )
    if result[
        [
            "nashik_price_lag1",
            "nashik_price_lag2",
            "nashik_momentum_2w",
        ]
    ].isna().any().any():
        raise ValueError("Aligned spatial features contain missing values.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tamil_nadu", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build_features(args.tamil_nadu, args.upstream, args.output)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
