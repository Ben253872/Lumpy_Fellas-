"""Baseline and intermittent-demand models from the original notebook."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


ForecastModel = Callable[[pd.DataFrame, list[pd.Timestamp]], pd.DataFrame]


def wmape(months: pd.Series, actual: pd.Series, forecast: pd.Series, sku_id: pd.Series | None = None) -> float:
    """Pooled WMAPE: sum(|actual - forecast|) / sum(actual) across all SKU-months.
    
    Zero-actual rows contribute 0 to the denominator but any forecast error still
    adds to the numerator, correctly penalising over-forecasting on zero-demand periods.
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    denom = np.abs(actual_arr).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(actual_arr - forecast_arr).sum() / denom * 100)


def wmape_per_sku(months: pd.Series, actual: pd.Series, forecast: pd.Series, sku_id: pd.Series) -> tuple[dict, float]:
    """Pooled WMAPE per SKU: sum(|error|) / sum(actual) for each SKU individually.
    
    Individual SKU WMAPEs are NaN when a SKU had zero actual demand in the evaluation period.
    The overall metric is the pooled WMAPE across all SKU-months (not the mean of SKU WMAPEs).
    
    Returns:
        Tuple of (dict mapping sku_id to sku_wmape_percent, pooled_wmape_across_all_skus)
    """
    scores = pd.DataFrame({"actual": np.asarray(actual, dtype=float),
                           "forecast": np.asarray(forecast, dtype=float),
                           "sku_id": sku_id})
    scores["absolute_error"] = (scores["actual"] - scores["forecast"]).abs()
    
    # Per-SKU pooled WMAPE: sum(|error|) / sum(actual) per SKU
    sku_agg = scores.groupby("sku_id").agg(actual_sum=("actual", "sum"), error_sum=("absolute_error", "sum"))
    sku_wmapes = {}
    for sku, row in sku_agg.iterrows():
        if row["actual_sum"] > 0:
            sku_wmapes[sku] = float(row["error_sum"] / row["actual_sum"] * 100)
        else:
            sku_wmapes[sku] = np.nan  # SKU had no actual demand in this period
    
    # Overall metric: pooled across all SKU-months
    total_actual = scores["actual"].sum()
    total_error = scores["absolute_error"].sum()
    pooled_wmape = float(total_error / total_actual * 100) if total_actual > 0 else np.nan
    
    return sku_wmapes, pooled_wmape


def rolling_origin_validation(data: pd.DataFrame, model: ForecastModel, horizon: int = 1, initial_train_months: int = 36) -> pd.DataFrame:
    """Evaluate a model with the rolling-origin strategy used in the original work."""
    data = data[["sku_id", "month", "demand"]].copy()
    data["month"] = pd.to_datetime(data["month"])
    months = sorted(data["month"].unique())
    evaluations = []
    for start in range(initial_train_months, len(months), horizon):
        test_months = months[start : start + horizon]
        if not test_months:
            break
        train = data.loc[data["month"] <= months[start - 1]]
        actual = data.loc[data["month"].isin(test_months)]
        predicted = model(train, test_months).rename(columns={"demand": "forecast"})
        evaluated = actual.merge(predicted, on=["sku_id", "month"], how="left")
        evaluations.append(evaluated.assign(forecast=lambda df: df["forecast"].fillna(0.0)))
    return pd.concat(evaluations, ignore_index=True) if evaluations else pd.DataFrame(columns=["sku_id", "month", "demand", "forecast"])


def _constant_forecast(train: pd.DataFrame, months: list[pd.Timestamp], values: pd.Series) -> pd.DataFrame:
    rows = [{"sku_id": sku, "month": month, "demand": max(0.0, float(value))} for sku, value in values.items() for month in months]
    return pd.DataFrame(rows)


def naive(train: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    return _constant_forecast(train, months, train.sort_values("month").groupby("sku_id")["demand"].last())


def seasonal_naive(train: pd.DataFrame, months: list[pd.Timestamp], seasonal_period: int = 12) -> pd.DataFrame:
    rows = []
    for sku, history in train.groupby("sku_id"):
        history = history.sort_values("month").set_index("month")["demand"]
        fallback = float(history.iloc[-1])
        for month in months:
            rows.append({"sku_id": sku, "month": month, "demand": float(history.get(month - pd.DateOffset(months=seasonal_period), fallback))})
    return pd.DataFrame(rows)


def simple_moving_average(train: pd.DataFrame, months: list[pd.Timestamp], window_size: int = 6) -> pd.DataFrame:
    values = train.sort_values("month").groupby("sku_id")["demand"].apply(lambda s: s.tail(window_size).mean())
    return _constant_forecast(train, months, values)


def croston(train: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    values = {}
    for sku, history in train.groupby("sku_id"):
        demand = history.sort_values("month")["demand"].to_numpy()
        positions = np.flatnonzero(demand > 0)
        if len(positions) < 2:
            values[sku] = demand[-1] if len(demand) else 0.0
        else:
            values[sku] = demand[positions].mean() / np.diff(positions).mean()
    return _constant_forecast(train, months, pd.Series(values))


def sba(train: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    values = {}
    for sku, history in train.groupby("sku_id"):
        demand = history.sort_values("month")["demand"].to_numpy()
        positions = np.flatnonzero(demand > 0)
        if len(positions) < 2:
            values[sku] = demand[-1] if len(demand) else 0.0
        else:
            interval = np.diff(positions).mean()
            values[sku] = demand[positions].mean() / interval * (1 - 1 / interval)
    return _constant_forecast(train, months, pd.Series(values))


def tsb(train: pd.DataFrame, months: list[pd.Timestamp], alpha: float = 0.1, beta: float = 0.1) -> pd.DataFrame:
    values = {}
    for sku, history in train.groupby("sku_id"):
        demand = history.sort_values("month")["demand"].to_numpy()
        nonzero = np.flatnonzero(demand > 0)
        if len(nonzero) == 0:
            values[sku] = 0.0
            continue
        size, probability, previous = demand[nonzero[0]], 1.0, nonzero[0]
        for position in range(nonzero[0] + 1, len(demand)):
            if demand[position] > 0:
                size = alpha * demand[position] + (1 - alpha) * size
                probability = beta * (1 / (position - previous)) + (1 - beta) * probability
                previous = position
            else:
                probability *= 1 - beta
        values[sku] = size * max(probability, 1e-6)
    return _constant_forecast(train, months, pd.Series(values))
