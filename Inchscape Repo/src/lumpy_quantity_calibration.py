from __future__ import annotations

import numpy as np
import pandas as pd


def quantity_statistics(train: pd.DataFrame, sku_ids: set, history_months: int = 24) -> pd.DataFrame:
    rows = []
    for sku_id, group in train.loc[train.sku_id.isin(sku_ids)].groupby("sku_id", sort=False):
        values = group.sort_values("month").demand.astype(float).clip(lower=0).to_numpy()[-history_months:]
        positive = values[values > 0]
        rows.append(
            {
                "sku_id": sku_id,
                "positive_median": float(np.median(positive)) if len(positive) else 0.0,
                "positive_mean": float(positive.mean()) if len(positive) else 0.0,
                "positive_p75": float(np.quantile(positive, 0.75)) if len(positive) else 0.0,
                "positive_max": float(positive.max()) if len(positive) else 0.0,
                "positive_count": int(len(positive)),
            }
        )
    return pd.DataFrame(rows)


def apply_quantity_transform(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    mode: str,
    parameter: float,
    pooled_ratio: float = 1.0,
) -> pd.DataFrame:
    result = frame.merge(statistics, on="sku_id", how="left")
    forecast = result.forecast.astype(float).clip(lower=0).to_numpy()
    positive = forecast > 0
    if mode == "scale":
        adjusted = forecast * float(parameter)
    elif mode == "pooled_ratio":
        multiplier = 1.0 + float(parameter) * (float(pooled_ratio) - 1.0)
        adjusted = forecast * multiplier
    elif mode.startswith("cap_"):
        column = mode.replace("cap_", "positive_")
        cap = result[column].fillna(0.0).to_numpy(float) * float(parameter)
        adjusted = np.where(positive & (cap > 0), np.minimum(forecast, cap), forecast)
    elif mode.startswith("blend_"):
        column = mode.replace("blend_", "positive_")
        reference = result[column].fillna(0.0).to_numpy(float)
        adjusted = np.where(
            positive & (reference > 0),
            float(parameter) * forecast + (1.0 - float(parameter)) * reference,
            forecast,
        )
    else:
        raise ValueError(f"Unknown quantity transform: {mode}")
    result["forecast"] = np.maximum(0.0, np.nan_to_num(adjusted, nan=0.0, posinf=0.0))
    return result


def transform_grid() -> pd.DataFrame:
    rows = []
    for parameter in (0.5, 0.75, 1.0, 1.25):
        rows.append({"mode": "scale", "parameter": parameter})
    for parameter in (0.5, 0.75, 1.0):
        rows.append({"mode": "pooled_ratio", "parameter": parameter})
    for mode in ("cap_median", "cap_p75", "cap_max"):
        for parameter in (1.0, 1.5, 2.0):
            rows.append({"mode": mode, "parameter": parameter})
    for mode in ("blend_median", "blend_mean", "blend_p75"):
        for parameter in (0.25, 0.5, 0.75):
            rows.append({"mode": mode, "parameter": parameter})
    result = pd.DataFrame(rows)
    result["transform_id"] = result.apply(lambda row: f"{row['mode']}__p{row['parameter']:.2f}", axis=1)
    return result
