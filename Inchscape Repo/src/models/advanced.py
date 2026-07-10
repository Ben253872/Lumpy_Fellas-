"""Tree-based forecasting models extracted from the original notebooks.

These are optional: install xgboost and lightgbm to run the relevant functions.
"""

from __future__ import annotations

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


def tree_forecast(train: pd.DataFrame, predict: pd.DataFrame, model_type: str, external: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast with the original XGBoost, LightGBM, or Random Forest experiment."""
    combined, features = _encoded_features(engineer_time_features(pd.concat([train, predict], ignore_index=True), external))
    train_months = pd.to_datetime(train["month"])
    predict_months = pd.to_datetime(predict["month"])
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
