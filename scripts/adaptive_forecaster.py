import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class QuantileResidualRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, alpha=10.0, quantile=0.5):
        self.alpha = alpha
        self.quantile = quantile

    def fit(self, X, y):
        self.linear_model = Ridge(alpha=self.alpha)
        self.residual_tree = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.quantile,
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.5,
            random_state=42,
        )
        self.linear_model.fit(X, y)
        residuals = y - self.linear_model.predict(X)
        self.residual_tree.fit(X, residuals)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return (
            self.linear_model.predict(X)
            + self.residual_tree.predict(X)
        )


class AdaptiveRegimeForecaster(RegressorMixin, BaseEstimator):

    def __init__(self, normal_model=None, shock_model=None, volatility_threshold=0.15):
        self.normal_model = normal_model
        self.shock_model = shock_model
        self.volatility_threshold = volatility_threshold

    def fit(self, X, y):
        if self.normal_model is None:
            self.normal_model = make_pipeline(
                QuantileResidualRegressor(alpha=10.0), X
            )
        if self.shock_model is None:
            self.shock_model = make_pipeline(
                QuantileResidualRegressor(alpha=0.1), X
            )
        self.normal_pipeline_ = clone(self.normal_model)
        self.shock_pipeline_ = clone(self.shock_model)
        target_difference = y - X["current_price"].to_numpy()
        self.normal_pipeline_.fit(X, target_difference)
        self.shock_pipeline_.fit(X, target_difference)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        volatility = X["relative_volatility_7w"].to_numpy()
        shock_mask = volatility > self.volatility_threshold
        predictions = np.empty(len(X), dtype=float)

        if (~shock_mask).any():
            predictions[~shock_mask] = (
                X.loc[~shock_mask, "current_price"].to_numpy()
                + self.normal_pipeline_.predict(X.loc[~shock_mask])
            )

        if shock_mask.any():
            predictions[shock_mask] = (
                X.loc[shock_mask, "current_price"].to_numpy()
                + self.shock_pipeline_.predict(X.loc[shock_mask])
            )

        return predictions


def fit_conformal_interval(
    X_train,
    y_train,
    X_eval,
    feature_frame,
    alpha=0.20,
    calibration_fraction=0.20,
    volatility_threshold=0.15,
    calibration_multiplier=1.5,
    normal_lower_multiplier=0.70,
    normal_upper_multiplier=1.45,
    shock_lower_multiplier=0.80,
    shock_upper_multiplier=2.40,
    return_pipelines=False,
):
    split = max(
        1,
        int(len(X_train) * (1 - calibration_fraction)),
    )
    if split >= len(X_train):
        split = len(X_train) - 1
    if split < 1:
        raise ValueError("At least two training rows are required for calibration.")

    X_fit = X_train.iloc[:split]
    y_fit = y_train.iloc[:split]
    X_calibration = X_train.iloc[split:]
    y_calibration = y_train.iloc[split:].to_numpy()
    fit_difference = y_fit.to_numpy() - X_fit["current_price"].to_numpy()

    lower_pipeline = make_pipeline(
        QuantileResidualRegressor(alpha=10.0, quantile=0.10),
        feature_frame,
    )
    upper_pipeline = make_pipeline(
        QuantileResidualRegressor(alpha=0.1, quantile=0.90),
        feature_frame,
    )
    lower_pipeline.fit(X_fit, fit_difference)
    upper_pipeline.fit(X_fit, fit_difference)

    calibration_current = X_calibration["current_price"].to_numpy()
    calibration_lower = (
        calibration_current + lower_pipeline.predict(X_calibration)
    )
    calibration_upper = (
        calibration_current + upper_pipeline.predict(X_calibration)
    )
    nonconformity = np.maximum(
        calibration_lower - y_calibration,
        y_calibration - calibration_upper,
    )
    quantile_index = min(
        len(nonconformity) - 1,
        int(np.ceil((len(nonconformity) + 1) * (1 - alpha))) - 1,
    )
    conformity_quantile = np.sort(nonconformity)[quantile_index]

    eval_current = X_eval["current_price"].to_numpy()
    eval_volatility = X_eval["relative_volatility_7w"].to_numpy()
    volatility_scale = np.maximum(
        1.0,
        eval_volatility / volatility_threshold,
    )
    expansion = (
        conformity_quantile
        * calibration_multiplier
        * volatility_scale
    )

    symmetric_lower = eval_current + lower_pipeline.predict(X_eval) - expansion
    symmetric_upper = eval_current + upper_pipeline.predict(X_eval) + expansion
    center = (symmetric_lower + symmetric_upper) / 2.0
    half_width = (symmetric_upper - symmetric_lower) / 2.0

    shock_mask = eval_volatility > volatility_threshold
    lower_multiplier = np.where(
        shock_mask,
        shock_lower_multiplier,
        normal_lower_multiplier,
    )
    upper_multiplier = np.where(
        shock_mask,
        shock_upper_multiplier,
        normal_upper_multiplier,
    )

    lower = center - (half_width * lower_multiplier)
    upper = center + (half_width * upper_multiplier)
    if return_pipelines:
        return (
            lower,
            upper,
            conformity_quantile,
            lower_pipeline,
            upper_pipeline,
        )
    return lower, upper, conformity_quantile


def make_pipeline(model, feature_frame):
    categorical_features = ["district"]
    numeric_features = [
        column
        for column in feature_frame.columns
        if column not in categorical_features
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
