import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
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


def fit_asymmetric_conformal_interval(
    X_train,
    y_train,
    X_test,
    volatility_threshold=0.15,
    normal_m_lower=0.70,
    normal_m_upper=1.45,
    shock_m_lower=0.80,
    shock_m_upper=2.40,
):
    """
    Derives asymmetric bounds directly from baseline symmetric conformal bounds.
    Reclaims unused bandwidth from the lower bound and allocates it to the upper bound.
    """
    sym_lower, sym_upper, _ = fit_conformal_interval(
        X_train,
        y_train,
        X_test,
        X_train,
        volatility_threshold=volatility_threshold,
        calibration_multiplier=1.5,
    )

    # Derive symmetric center and half-width directly from symmetric bounds
    center = (sym_upper + sym_lower) / 2.0
    half_width = (sym_upper - sym_lower) / 2.0

    test_vols = (
        X_test["relative_volatility_7w"].values
        if "relative_volatility_7w" in X_test.columns
        else np.zeros(len(X_test))
    )
    shock_mask = test_vols > volatility_threshold

    # Apply regime-specific asymmetric multipliers
    m_lower = np.where(shock_mask, shock_m_lower, normal_m_lower)
    m_upper = np.where(shock_mask, shock_m_upper, normal_m_upper)

    q_lower = center - (half_width * m_lower)
    q_upper = center + (half_width * m_upper)

    return q_lower, q_upper, center


def run_asymmetric_experiment():
    print("=" * 80)
    print("      ASYMMETRIC CONFORMAL CALIBRATION EXPERIMENT (HARD GATE)      ")
    print("=" * 80)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature dataset missing: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH).sort_values(["time_index", "district"]).reset_index(drop=True)
    feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS]
    periods = sorted(df["time_index"].dropna().unique())
    eval_periods = periods[10:]

    sym_records = []
    asym_records = []

    for period in eval_periods:
        train_df = df[df["time_index"] < period].copy()
        test_df = df[df["time_index"] == period].copy()
        if test_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET]

        # Point Prediction (Hard Gate - Unchanged)
        model = AdaptiveRegimeForecaster(volatility_threshold=0.15)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # 1. Baseline Symmetric Calibration
        sym_lower, sym_upper, _ = fit_conformal_interval(
            X_train,
            y_train,
            X_test,
            X_train,
            volatility_threshold=0.15,
            calibration_multiplier=1.5,
        )

        # 2. Experimental Asymmetric Calibration
        asym_lower, asym_upper, _ = fit_asymmetric_conformal_interval(
            X_train,
            y_train,
            X_test,
            volatility_threshold=0.15,
            normal_m_lower=0.70,  # Reclaim lower waste
            normal_m_upper=1.45,  # Shift protection upward
            shock_m_lower=0.80,
            shock_m_upper=2.40,  # Expansion during upward surge initiation
        )

        for i in range(len(test_df)):
            vol = test_df["relative_volatility_7w"].iloc[i]
            regime = "shock" if vol > 0.15 else "normal"
            act = y_test.iloc[i]

            base_info = {
                "time_index": period,
                "district": test_df["district"].iloc[i],
                "actual": act,
                "pred": pred[i],
                "volatility": vol,
                "regime": regime,
            }

            sym_records.append({
                **base_info,
                "q_lower": sym_lower[i],
                "q_upper": sym_upper[i],
            })

            asym_records.append({
                **base_info,
                "q_lower": asym_lower[i],
                "q_upper": asym_upper[i],
            })

    def evaluate_calibration(records):
        res_df = pd.DataFrame(records)
        res_df["abs_err"] = np.abs(res_df["actual"] - res_df["pred"])
        res_df["is_covered"] = (res_df["actual"] >= res_df["q_lower"]) & (res_df["actual"] <= res_df["q_upper"])
        res_df["lower_miss"] = res_df["actual"] < res_df["q_lower"]
        res_df["upper_miss"] = res_df["actual"] > res_df["q_upper"]
        res_df["interval_width"] = res_df["q_upper"] - res_df["q_lower"]

        overall_mae = res_df["abs_err"].mean()
        
        spike_df = res_df[res_df["time_index"].isin([29, 30])]
        spike_mae = spike_df["abs_err"].mean() if not spike_df.empty else np.nan

        norm_df = res_df[res_df["regime"] == "normal"]
        shock_df = res_df[res_df["regime"] == "shock"]
        p29_df = res_df[res_df["time_index"] == 29]
        p30_df = res_df[res_df["time_index"] == 30]

        return {
            "Overall Coverage": res_df["is_covered"].mean() * 100,
            "Normal Coverage": norm_df["is_covered"].mean() * 100 if not norm_df.empty else 0.0,
            "Shock Coverage": shock_df["is_covered"].mean() * 100 if not shock_df.empty else 0.0,
            "Period 29 Coverage": p29_df["is_covered"].mean() * 100 if not p29_df.empty else 0.0,
            "Period 30 Coverage": p30_df["is_covered"].mean() * 100 if not p30_df.empty else 0.0,
            "Upper-Miss Rate": res_df["upper_miss"].mean() * 100,
            "Lower-Miss Rate": res_df["lower_miss"].mean() * 100,
            "Mean Interval Width": res_df["interval_width"].mean(),
            "Overall MAE": overall_mae,
            "Spike MAE": spike_mae,
        }

    sym_metrics = evaluate_calibration(sym_records)
    asym_metrics = evaluate_calibration(asym_records)

    metrics_display = [
        ("Overall Coverage (%)", "Overall Coverage", "{:.2f}%"),
        ("Normal-Regime Coverage (%)", "Normal Coverage", "{:.2f}%"),
        ("Shock-Regime Coverage (%)", "Shock Coverage", "{:.2f}%"),
        ("Period 29 Coverage (%)", "Period 29 Coverage", "{:.2f}%"),
        ("Period 30 Coverage (%)", "Period 30 Coverage", "{:.2f}%"),
        ("Upper-Miss Rate (%)", "Upper-Miss Rate", "{:.2f}%"),
        ("Lower-Miss Rate (%)", "Lower-Miss Rate", "{:.2f}%"),
        ("Mean Interval Width (₹)", "Mean Interval Width", "₹{:.2f}"),
        ("Point MAE (Overall)", "Overall MAE", "₹{:.2f}"),
        ("Spike MAE (Periods 29-30)", "Spike MAE", "₹{:.2f}"),
    ]

    print(f"\n{'Metric':<30} | {'Symmetric (Baseline)':<22} | {'Asymmetric (Experiment)':<22} | {'Delta':<12}")
    print("-" * 92)

    for label, key, fmt in metrics_display:
        v_sym = sym_metrics[key]
        v_asym = asym_metrics[key]

        if "MAE" in key:
            diff_str = "0.00 (EXACT)" if abs(v_asym - v_sym) < 1e-4 else f"{v_asym - v_sym:+.2f}"
        else:
            diff_str = f"{v_asym - v_sym:+.2f}"

        print(f"{label:<30} | {fmt.format(v_sym):<22} | {fmt.format(v_asym):<22} | {diff_str:<12}")

    print("=" * 92)
    print("EXPERIMENT COMPLETE — Model artifact onion_price_adaptive_v1.joblib remains UNCHANGED.")
    print("=" * 92)


if __name__ == "__main__":
    run_asymmetric_experiment()