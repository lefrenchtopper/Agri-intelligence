import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.adaptive_forecaster import (
    AdaptiveRegimeForecaster,
    fit_conformal_interval,
)

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "features"
    / "onion_tamilnadu_forecasting_features.csv"
)
TARGET = "target_next_week_price"
TARGET_COLUMNS = {TARGET, "target_next_week_change"}


def run_conformal_audit():
    print("=" * 80)
    print("      CONFORMAL PREDICTION INTERVAL CALIBRATION AUDIT (HARD GATE)      ")
    print("=" * 80)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature dataset missing: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH).sort_values(["time_index", "district"]).reset_index(drop=True)
    feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS]
    periods = sorted(df["time_index"].dropna().unique())
    eval_periods = periods[10:]

    records = []

    # Run out-of-fold evaluation matching production baseline
    for period in eval_periods:
        train_df = df[df["time_index"] < period].copy()
        test_df = df[df["time_index"] == period].copy()
        if test_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET]

        model = AdaptiveRegimeForecaster(volatility_threshold=0.15)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        lower, upper, _ = fit_conformal_interval(
            X_train,
            y_train,
            X_test,
            X_train,
            volatility_threshold=0.15,
            calibration_multiplier=1.5,
        )

        for i in range(len(test_df)):
            vol = test_df["relative_volatility_7w"].iloc[i]
            records.append({
                "time_index": period,
                "district": test_df["district"].iloc[i],
                "actual": y_test.iloc[i],
                "pred": pred[i],
                "q_lower": lower[i],
                "q_upper": upper[i],
                "volatility": vol,
                "regime": "shock" if vol > 0.15 else "normal",
            })

    audit_df = pd.DataFrame(records)

    # Calculate metrics
    audit_df["is_covered"] = (audit_df["actual"] >= audit_df["q_lower"]) & (audit_df["actual"] <= audit_df["q_upper"])
    audit_df["lower_miss"] = audit_df["actual"] < audit_df["q_lower"]
    audit_df["upper_miss"] = audit_df["actual"] > audit_df["q_upper"]
    audit_df["interval_width"] = audit_df["q_upper"] - audit_df["q_lower"]

    total_n = len(audit_df)
    overall_cov = audit_df["is_covered"].mean() * 100
    overall_low_miss = audit_df["lower_miss"].mean() * 100
    overall_up_miss = audit_df["upper_miss"].mean() * 100
    avg_width = audit_df["interval_width"].mean()

    print("\n--- 1. OVERALL COVERAGE & MISS TYPES ---")
    print(f"Total Samples        : {total_n}")
    print(f"Overall Coverage     : {overall_cov:.2f}% (Target ~80.0%)")
    print(f"Lower-bound Misses   : {audit_df['lower_miss'].sum()} ({overall_low_miss:.2f}%) [Actual < Lower]")
    print(f"Upper-bound Misses   : {audit_df['upper_miss'].sum()} ({overall_up_miss:.2f}%) [Actual > Upper]")
    print(f"Mean Interval Width  : ₹{avg_width:.2f}")

    print("\n--- 2. BREAKDOWN BY REGIME ---")
    for r in ["normal", "shock"]:
        sub = audit_df[audit_df["regime"] == r]
        if sub.empty:
            continue
        print(f"[{r.upper()} REGIME] (n={len(sub)})")
        print(f"  • Coverage        : {sub['is_covered'].mean() * 100:.2f}%")
        print(f"  • Lower Misses    : {sub['lower_miss'].mean() * 100:.2f}%")
        print(f"  • Upper Misses    : {sub['upper_miss'].mean() * 100:.2f}%")
        print(f"  • Mean Width      : ₹{sub['interval_width'].mean():.2f}")

    print("\n--- 3. CRITICAL SPIKE PERIODS 29–30 ---")
    spike_df = audit_df[audit_df["time_index"].isin([29, 30])]
    if not spike_df.empty:
        print(f"Spike Samples (29-30): {len(spike_df)}")
        print(f"Spike Coverage       : {spike_df['is_covered'].mean() * 100:.2f}%")
        print(f"Spike Lower Misses   : {spike_df['lower_miss'].sum()} ({spike_df['lower_miss'].mean() * 100:.2f}%)")
        print(f"Spike Upper Misses   : {spike_df['upper_miss'].sum()} ({spike_df['upper_miss'].mean() * 100:.2f}%)")
        print(f"Spike Mean Width     : ₹{spike_df['interval_width'].mean():.2f}")

        for p in [29, 30]:
            p_sub = spike_df[spike_df["time_index"] == p]
            print(f"\n  [Period {p} Details] (n={len(p_sub)}):")
            print(f"    - Coverage      : {p_sub['is_covered'].mean() * 100:.2f}%")
            print(f"    - Avg Actual    : ₹{p_sub['actual'].mean():.2f}")
            print(f"    - Avg Prediction: ₹{p_sub['pred'].mean():.2f}")
            print(f"    - Avg Interval  : [₹{p_sub['q_lower'].mean():.2f}, ₹{p_sub['q_upper'].mean():.2f}]")
    else:
        print("No samples found for spike periods 29–30.")

    print("\n--- 4. TIME PERIOD QUARTILE BREAKDOWN ---")
    audit_df["period_quartile"] = pd.qcut(audit_df["time_index"], q=4, labels=["Q1 (Early)", "Q2 (Mid-Early)", "Q3 (Mid-Late)", "Q4 (Late)"])
    for q_label in ["Q1 (Early)", "Q2 (Mid-Early)", "Q3 (Mid-Late)", "Q4 (Late)"]:
        q_sub = audit_df[audit_df["period_quartile"] == q_label]
        print(f"{q_label:<15} | Coverage: {q_sub['is_covered'].mean() * 100:6.2f}% | Upper Miss: {q_sub['upper_miss'].mean() * 100:5.2f}% | Lower Miss: {q_sub['lower_miss'].mean() * 100:5.2f}% | Avg Width: ₹{q_sub['interval_width'].mean():.2f}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_conformal_audit()