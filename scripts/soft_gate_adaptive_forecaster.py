import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.adaptive_forecaster import (
    AdaptiveRegimeForecaster,
    fit_conformal_interval,
    make_pipeline,
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


def logistic_gate(volatility, threshold=0.15, slope=20.0):
    volatility = np.asarray(volatility, dtype=float)
    return 1.0 / (1.0 + np.exp(-slope * (volatility - threshold)))


class SoftGateAdaptiveForecaster:
    """Adaptive forecaster using a continuous logistic soft gate."""

    def __init__(self, volatility_threshold=0.15, slope=20.0):
        self.tau = volatility_threshold
        self.k = slope
        self.normal_base = None
        self.normal_res = None
        self.normal_q10 = None
        self.normal_q90 = None
        self.shock_base = None
        self.shock_res = None
        self.shock_q10 = None
        self.shock_q90 = None

    def _fit_regime(self, X, y, regime_name):
        if regime_name == "normal":
            base_model = Ridge(alpha=10.0)
            residual_model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
            q10_model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.10,
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
            q90_model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.90,
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
        else:
            base_model = Ridge(alpha=0.1)
            residual_model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )
            q10_model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.10,
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )
            q90_model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.90,
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )

        base_pipeline = make_pipeline(base_model, X)
        residual_pipeline = make_pipeline(residual_model, X)
        q10_pipeline = make_pipeline(q10_model, X)
        q90_pipeline = make_pipeline(q90_model, X)

        base_pipeline.fit(X, y)
        residuals = y.to_numpy() - base_pipeline.predict(X)
        residual_pipeline.fit(X, residuals)
        q10_pipeline.fit(X, residuals)
        q90_pipeline.fit(X, residuals)

        return base_pipeline, residual_pipeline, q10_pipeline, q90_pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series):
        vol = X["relative_volatility_7w"].to_numpy()
        normal_mask = vol <= self.tau
        shock_mask = vol > self.tau

        if np.any(normal_mask):
            X_norm = X.loc[normal_mask].copy()
            y_norm = y.loc[normal_mask].copy()
            (
                self.normal_base,
                self.normal_res,
                self.normal_q10,
                self.normal_q90,
            ) = self._fit_regime(X_norm, y_norm, "normal")
        else:
            self.normal_base = make_pipeline(Ridge(alpha=10.0), X)
            self.normal_base.fit(X, y)
            self.normal_res = make_pipeline(
                HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                ),
                X,
            )
            self.normal_q10 = make_pipeline(
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.10,
                    max_iter=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                ),
                X,
            )
            self.normal_q90 = make_pipeline(
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.90,
                    max_iter=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                ),
                X,
            )
            residuals = y.to_numpy() - self.normal_base.predict(X)
            self.normal_res.fit(X, residuals)
            self.normal_q10.fit(X, residuals)
            self.normal_q90.fit(X, residuals)

        if np.any(shock_mask):
            X_shock = X.loc[shock_mask].copy()
            y_shock = y.loc[shock_mask].copy()
            (
                self.shock_base,
                self.shock_res,
                self.shock_q10,
                self.shock_q90,
            ) = self._fit_regime(X_shock, y_shock, "shock")
        else:
            self.shock_base = make_pipeline(Ridge(alpha=0.1), X)
            self.shock_base.fit(X, y)
            self.shock_res = make_pipeline(
                HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                ),
                X,
            )
            self.shock_q10 = make_pipeline(
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.10,
                    max_iter=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                ),
                X,
            )
            self.shock_q90 = make_pipeline(
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.90,
                    max_iter=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                ),
                X,
            )
            residuals = y.to_numpy() - self.shock_base.predict(X)
            self.shock_res.fit(X, residuals)
            self.shock_q10.fit(X, residuals)
            self.shock_q90.fit(X, residuals)

    def predict(self, X: pd.DataFrame):
        vols = X["relative_volatility_7w"].to_numpy()
        w = logistic_gate(vols, threshold=self.tau, slope=self.k)

        y_norm = self.normal_base.predict(X) + self.normal_res.predict(X)
        y_shock = self.shock_base.predict(X) + self.shock_res.predict(X)
        y_pred = (1.0 - w) * y_norm + w * y_shock

        q10_norm = self.normal_base.predict(X) + self.normal_q10.predict(X)
        q90_norm = self.normal_base.predict(X) + self.normal_q90.predict(X)
        q10_shock = self.shock_base.predict(X) + self.shock_q10.predict(X)
        q90_shock = self.shock_base.predict(X) + self.shock_q90.predict(X)

        q10 = (1.0 - w) * q10_norm + w * q10_shock
        q90 = (1.0 - w) * q90_norm + w * q90_shock
        return np.asarray(y_pred, dtype=float), np.asarray(q10, dtype=float), np.asarray(q90, dtype=float)


def compute_directional_accuracy(y_true, y_pred, y_lag1):
    actual_dir = np.sign(np.asarray(y_true, dtype=float) - np.asarray(y_lag1, dtype=float))
    pred_dir = np.sign(np.asarray(y_pred, dtype=float) - np.asarray(y_lag1, dtype=float))
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def summarize_metrics(y_true, y_pred, q10, q90, volatilities, times, lag1_values):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)
    times = np.asarray(times)
    lag1_values = np.asarray(lag1_values, dtype=float)

    overall_mae = float(np.mean(np.abs(y_true - y_pred)))
    norm_mask = volatilities <= 0.15
    shock_mask = volatilities > 0.15
    spike_mask = np.isin(times, [29, 30])

    norm_mae = float(np.mean(np.abs(y_true[norm_mask] - y_pred[norm_mask]))) if np.any(norm_mask) else np.nan
    shock_mae = float(np.mean(np.abs(y_true[shock_mask] - y_pred[shock_mask]))) if np.any(shock_mask) else np.nan
    spike_mae = float(np.mean(np.abs(y_true[spike_mask] - y_pred[spike_mask]))) if np.any(spike_mask) else np.nan
    dir_acc = compute_directional_accuracy(y_true, y_pred, lag1_values)
    coverage = float(np.mean((y_true >= q10) & (y_true <= q90)) * 100.0)

    return {
        "Overall MAE": overall_mae,
        "Normal MAE": norm_mae,
        "Shock MAE": shock_mae,
        "Spike MAE (29-30)": spike_mae,
        "Directional Acc (%)": dir_acc,
        "Interval Coverage (%)": coverage,
    }


def run_experiment():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature file missing: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH).sort_values(["time_index", "district"]).reset_index(drop=True)
    required = {TARGET, "current_price", "time_index", "district", "relative_volatility_7w"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS]
    periods = sorted(df["time_index"].dropna().unique())
    eval_periods = periods[10:]

    configurations = {
        "Hard Gate (Baseline)": {"type": "hard", "k": None},
        "Soft Gate (k=5, Smooth)": {"type": "soft", "k": 5.0},
        "Soft Gate (k=10, Medium)": {"type": "soft", "k": 10.0},
        "Soft Gate (k=20, Steep)": {"type": "soft", "k": 20.0},
    }

    results = []
    for config_name, cfg in configurations.items():
        y_trues, y_preds, q10s, q90s, lag1s, volatilities, times = [], [], [], [], [], [], []

        for period in eval_periods:
            train_df = df[df["time_index"] < period].copy()
            test_df = df[df["time_index"] == period].copy()
            if test_df.empty:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df[TARGET]
            X_test = test_df[feature_cols]
            y_test = test_df[TARGET]

            if cfg["type"] == "hard":
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
                q10 = lower
                q90 = upper
            else:
                model = SoftGateAdaptiveForecaster(volatility_threshold=0.15, slope=cfg["k"])
                model.fit(X_train, y_train)
                pred, q10, q90 = model.predict(X_test)

            y_trues.extend(y_test.to_numpy())
            y_preds.extend(np.asarray(pred, dtype=float))
            q10s.extend(np.asarray(q10, dtype=float))
            q90s.extend(np.asarray(q90, dtype=float))
            lag1s.extend(test_df["price_lag_1"].to_numpy())
            volatilities.extend(test_df["relative_volatility_7w"].to_numpy())
            times.extend(test_df["time_index"].to_numpy())

        metrics = summarize_metrics(y_trues, y_preds, q10s, q90s, volatilities, times, lag1s)
        metrics["Configuration"] = config_name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\n==========================================================================================")
    print("                     P2 SOFT-GATING BENCHMARK EVALUATION RESULTS                          ")
    print("==========================================================================================\n")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\n==========================================================================================\n")

    best_soft = results_df[results_df["Configuration"].str.contains("Soft Gate")].sort_values(["Overall MAE", "Spike MAE (29-30)"]).iloc[0]

    result_path = BASE_DIR / "soft_gate_experiment_results.md"
    markdown_lines = [
        "# P2 Soft-Gating Network Experiment Report",
        "",
        "| Configuration | Overall MAE | Normal MAE | Shock MAE | Spike MAE (29-30) | Directional Acc (%) | Interval Coverage (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results_df.iterrows():
        markdown_lines.append(
            f"| {row['Configuration']} | {row['Overall MAE']:.2f} | {row['Normal MAE']:.2f} | {row['Shock MAE']:.2f} | {row['Spike MAE (29-30)']:.2f} | {row['Directional Acc (%)']:.2f} | {row['Interval Coverage (%)']:.2f} |"
        )
    markdown_lines.extend([
        "",
        "*Decision Rule:* Keep the hard gate unless a soft gate improves both Overall MAE below ₹168.57 and Spike MAE below ₹473.41.",
    ])
    if best_soft["Overall MAE"] < 168.57 and best_soft["Spike MAE (29-30)"] < 473.41:
        markdown_lines.append(
            f"Recommendation: adopt the best soft gate ({best_soft['Configuration']}) because it beats the decision thresholds."
        )
        print("Recommendation: adopt the best soft gate configuration.")
    else:
        markdown_lines.append(
            f"Recommendation: keep the hard gate because the soft gate did not beat the required thresholds. Best soft gate: {best_soft['Configuration']}."
        )
        print("Recommendation: keep the hard gate because the soft gate did not beat the required thresholds.")

    result_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"Report written to: {result_path}")


if __name__ == "__main__":
    run_experiment()
