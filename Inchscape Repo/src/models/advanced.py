"""Tree-based forecasting models extracted from the original notebooks.

These are optional: install xgboost and lightgbm to run the relevant functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def engineer_time_features(data: pd.DataFrame, external: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the lag, rolling, ratio, and zero-streak features used in the experiments."""
    data = data.copy().sort_values(["sku_id", "month"])
    data["month"] = pd.to_datetime(data["month"])
    grouped = data.groupby("sku_id")["demand"]
    for lag in range(1, 25):
        data[f"lag_{lag}"] = grouped.shift(lag)
    for window in (3, 6, 12, 24):
        data[f"rolling_mean_{window}"] = grouped.transform(lambda series: series.rolling(window, min_periods=1).mean().shift(1))
    data["rolling_mean_3_div_12"] = data["rolling_mean_3"] / (data["rolling_mean_12"] + 1e-6)
    data["rolling_mean_6_div_24"] = data["rolling_mean_6"] / (data["rolling_mean_24"] + 1e-6)
    data["rolling_mean_3_minus_12"] = data["rolling_mean_3"] - data["rolling_mean_12"]
    data["rolling_mean_6_minus_24"] = data["rolling_mean_6"] - data["rolling_mean_24"]
    zero_run = grouped.transform(lambda series: series.eq(0).cumsum() - series.eq(0).cumsum().where(series.gt(0)).ffill().fillna(0))
    data["zero_streak"] = zero_run.groupby(data["sku_id"]).shift(1).fillna(0).astype(int)
    if external is not None:
        ext = external.copy()
        ext["month"] = pd.to_datetime(ext["date"])
        data = data.merge(ext.drop(columns=["date", "year"], errors="ignore"), on="month", how="left")
    return data


def _encoded_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = pd.get_dummies(data, columns=[col for col in ("Brand", "Channel") if col in data], dtype=float)
    excluded = {"sku_id", "month", "demand", "demand_type", "collision_flag", "Country", "REGION"}
    features = [col for col in data.columns if col not in excluded and pd.api.types.is_numeric_dtype(data[col])]
    return data, features


def _fix_test_features_leakage(engineered: pd.DataFrame, train_end_month: pd.Timestamp) -> pd.DataFrame:
    """
    Fix data leakage by recalculating test rows' rolling features using only training data.
    
    When features are engineered on combined train+test data, rolling features for test rows
    can inadvertently use values from other test rows. This function ensures that test rows'
    rolling features only use training-period values, preventing leakage in backtesting.
    """
    engineered = engineered.copy()
    train_data = engineered[engineered["month"] < train_end_month]
    test_mask = engineered["month"] >= train_end_month
    
    # For each window size, recalculate test rows' rolling features from training data only
    for window in [3, 6, 12, 24]:
        col = f"rolling_mean_{window}"
        if col in engineered.columns:
            for idx in engineered[test_mask].index:
                sku = engineered.loc[idx, "sku_id"]
                sku_train = train_data[train_data["sku_id"] == sku]
                if len(sku_train) >= window:
                    # Use the last `window` values from training data
                    engineered.loc[idx, col] = sku_train["demand"].tail(window).mean()
                else:
                    # If not enough training data, use what we have
                    engineered.loc[idx, col] = sku_train["demand"].mean() if len(sku_train) > 0 else np.nan
    
    return engineered


def tree_forecast(train: pd.DataFrame, predict: pd.DataFrame, model_type: str, external: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast with the original XGBoost, LightGBM, or Random Forest experiment."""
    combined, features = _encoded_features(engineer_time_features(pd.concat([train, predict], ignore_index=True), external))
    train_months = pd.to_datetime(train["month"])
    predict_months = pd.to_datetime(predict["month"])
    train_end_month = train_months.max()
    
    # Fix data leakage: recalculate test rows' rolling features using only training data
    combined = _fix_test_features_leakage(combined, train_end_month)
    
    train_rows = combined.loc[combined["month"].isin(train_months)]
    predict_rows = combined.loc[combined["month"].isin(predict_months)]
    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError("model_type must be xgboost, lightgbm, or random_forest")
    model.fit(train_rows[features].fillna(0), train_rows["demand"])
    return predict_rows[["sku_id", "month"]].assign(demand=np.maximum(0, model.predict(predict_rows[features].fillna(0))))


def lumpy_hurdle_forecast(train: pd.DataFrame, predict: pd.DataFrame, threshold: float = 0.5, weight_factor: float = 1.0, external: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two-stage LightGBM occurrence/size model from the original lumpy-demand section."""
    import lightgbm as lgb
    combined, features = _encoded_features(engineer_time_features(pd.concat([train, predict], ignore_index=True), external))
    train_end_month = pd.to_datetime(train["month"]).max()
    
    # Fix data leakage: recalculate test rows' rolling features using only training data
    combined = _fix_test_features_leakage(combined, train_end_month)
    
    train_rows = combined.loc[combined["month"].isin(pd.to_datetime(train["month"]))]
    predict_rows = combined.loc[combined["month"].isin(pd.to_datetime(predict["month"]))]
    x_train, x_test, target = train_rows[features].fillna(0), predict_rows[features].fillna(0), train_rows["demand"]
    occurred = target.gt(0).astype(int)
    weight = (len(occurred) - occurred.sum()) / max(occurred.sum(), 1) * weight_factor
    classifier = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, scale_pos_weight=weight).fit(x_train, occurred)
    positive = target.gt(0)
    regressor = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1).fit(x_train.loc[positive], target.loc[positive])
    occurs = classifier.predict_proba(x_test)[:, 1] >= threshold
    return predict_rows[["sku_id", "month"]].assign(demand=np.maximum(0, occurs * regressor.predict(x_test)))


@dataclass
class Chronos2RidgeConfig:
    """Configuration for Chronos 2 with optional ridge correction layer."""

    model_id: str = "amazon/chronos-2"
    device_map: str = "cpu"
    context_length: int = 12
    prediction_length: int = 1
    batch_size: int = 256
    num_samples: int = 100
    temperature: float = 0.6
    ridge_alpha: float = 25.0
    ridge_feature_columns: list[str] = field(default_factory=list)


class Chronos2RidgeForecaster:
    """Skeleton forecaster that combines Chronos 2 base predictions with optional ridge correction.

    Notes:
    - The Chronos base model is loaded lazily when predictions are requested.
    - The ridge layer is optional and can be attached later from training artifacts.
    - This class is inference-oriented for serving SKU forecasts.
    """

    def __init__(self, config: Chronos2RidgeConfig, ridge_pipeline=None) -> None:
        self.config = config
        self.ridge_pipeline = ridge_pipeline
        self._pipeline = None
        self._pipeline_type = None

    @classmethod
    def from_artifact(cls, artifact: dict) -> "Chronos2RidgeForecaster":
        cfg_dict = artifact.get("chronos_config", {})
        config = Chronos2RidgeConfig(**cfg_dict)
        ridge_pipeline = artifact.get("ridge_pipeline")
        return cls(config=config, ridge_pipeline=ridge_pipeline)

    def to_artifact(self) -> dict:
        return {
            "artifact_type": "chronos_2_ridge",
            "chronos_config": {
                "model_id": self.config.model_id,
                "device_map": self.config.device_map,
                "context_length": self.config.context_length,
                "prediction_length": self.config.prediction_length,
                "batch_size": self.config.batch_size,
                "num_samples": self.config.num_samples,
                "temperature": self.config.temperature,
                "ridge_alpha": self.config.ridge_alpha,
                "ridge_feature_columns": list(self.config.ridge_feature_columns),
            },
            "ridge_pipeline": self.ridge_pipeline,
        }

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from chronos import Chronos2Pipeline, ChronosPipeline
        except ImportError as exc:
            raise RuntimeError("chronos-forecasting is not installed") from exc

        try:
            self._pipeline = Chronos2Pipeline.from_pretrained(self.config.model_id, device_map=self.config.device_map)
            self._pipeline_type = "chronos2"
        except Exception:
            self._pipeline = ChronosPipeline.from_pretrained(self.config.model_id, device_map=self.config.device_map)
            self._pipeline_type = "chronos1"

    @staticmethod
    def _extract_point_forecast(prediction_output) -> float:
        values = prediction_output.detach().cpu().numpy() if hasattr(prediction_output, "detach") else np.asarray(prediction_output)
        return float(np.median(values))

    def _predict_base_batch(self, histories: list[np.ndarray]) -> list[float]:
        self._ensure_pipeline()
        if self._pipeline_type == "chronos2":
            outputs = self._pipeline.predict(
                histories,
                prediction_length=self.config.prediction_length,
                batch_size=self.config.batch_size,
            )
            return [self._extract_point_forecast(output) for output in outputs]

        import torch

        contexts = [torch.tensor(history, dtype=torch.float32) for history in histories]
        outputs = self._pipeline.predict(
            contexts,
            prediction_length=self.config.prediction_length,
            num_samples=self.config.num_samples,
            temperature=self.config.temperature,
        )
        if hasattr(outputs, "shape"):
            output_rows = [outputs[i] for i in range(outputs.shape[0])]
        else:
            output_rows = list(outputs)
        return [self._extract_point_forecast(output) for output in output_rows]

    def _apply_ridge_correction(self, base_predictions: pd.Series, correction_features: pd.DataFrame | None) -> pd.Series:
        if self.ridge_pipeline is None:
            return base_predictions.clip(lower=0.0)
        if correction_features is None:
            return base_predictions.clip(lower=0.0)
        needed = [col for col in self.config.ridge_feature_columns if col in correction_features.columns]
        if not needed:
            return base_predictions.clip(lower=0.0)
        corrected = self.ridge_pipeline.predict(correction_features[needed])
        corrected_series = pd.Series(np.asarray(corrected, dtype=float), index=base_predictions.index)
        return corrected_series.clip(lower=0.0)

    def predict_horizon_for_sku(
        self,
        sku_id: str,
        sku_history: pd.DataFrame,
        horizon_months: int,
        correction_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Predict future horizon for one SKU.

        Parameters
        ----------
        sku_id:
            SKU identifier.
        sku_history:
            DataFrame with columns [sku_id, month, demand].
        horizon_months:
            Number of future months to forecast.
        correction_features:
            Optional month-level features for ridge correction. It can be provided
            now or added later once feature pipelines are finalized.
        """
        history = sku_history.copy().sort_values("month")
        history["month"] = pd.to_datetime(history["month"])
        if history.empty:
            return pd.DataFrame(columns=["sku_id", "month", "demand"])

        predictions: list[dict] = []
        running_values = history["demand"].astype(float).tolist()
        last_month = pd.Timestamp(history["month"].max())

        for step in range(1, horizon_months + 1):
            context = np.asarray(running_values[-self.config.context_length :], dtype=np.float32)
            base_pred = self._predict_base_batch([context])[0]
            next_month = last_month + pd.DateOffset(months=step)
            base_series = pd.Series([base_pred], index=[0], dtype=float)
            corrected = self._apply_ridge_correction(base_series, correction_features)
            final_pred = float(corrected.iloc[0])
            running_values.append(final_pred)
            predictions.append({"sku_id": sku_id, "month": next_month, "demand": max(final_pred, 0.0)})

        return pd.DataFrame(predictions)
