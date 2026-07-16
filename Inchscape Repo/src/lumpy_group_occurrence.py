from __future__ import annotations

import numpy as np
import pandas as pd


def group_block_priors(
    train: pd.DataFrame,
    sku_ids: set,
    block_starts: list[pd.Timestamp],
    group_column: str,
    smoothing: float = 6.0,
) -> pd.DataFrame:
    history = train.loc[train.sku_id.isin(sku_ids)].copy()
    history["month"] = pd.to_datetime(history.month)
    metadata = (
        history.sort_values(["sku_id", "month"])
        .groupby("sku_id", as_index=False)[group_column]
        .agg(lambda values: values.dropna().iloc[-1] if values.notna().any() else "unknown")
        .rename(columns={group_column: "group_value"})
    )
    history = history.merge(metadata, on="sku_id", how="left")
    history["month_number"] = history.month.dt.month
    history["occurred"] = history.demand.gt(0).astype(float)
    grouped = history.groupby(["group_value", "month_number"]).occurred.agg(["sum", "count"])
    global_month = history.groupby("month_number").occurred.mean()
    rows = []
    for row in metadata.itertuples(index=False):
        raw = []
        for block_number, block_start in enumerate(block_starts, 1):
            months = [(pd.Timestamp(block_start) + pd.DateOffset(months=offset)).month for offset in range(3)]
            block_score = 0.0
            for month in months:
                if (row.group_value, month) in grouped.index:
                    stats = grouped.loc[(row.group_value, month)]
                    prior = float(global_month.get(month, 0.0))
                    rate = (float(stats["sum"]) + smoothing * prior) / (float(stats["count"]) + smoothing)
                else:
                    rate = float(global_month.get(month, 0.0))
                block_score += rate
            raw.append(block_score)
        raw = np.asarray(raw, dtype=float)
        mean = float(raw.mean())
        factors = raw / mean if mean > 0 else np.ones(len(raw))
        for block_number, (block_start, factor) in enumerate(zip(block_starts, factors), 1):
            rows.append({"sku_id": row.sku_id, "block_number": block_number, "block_start": pd.Timestamp(block_start), "group_factor": float(factor)})
    return pd.DataFrame(rows)


def adjust_probabilities(
    frame: pd.DataFrame,
    probability: np.ndarray,
    factors: np.ndarray,
    power: float,
) -> np.ndarray:
    working = frame[["sku_id"]].copy()
    working["base"] = np.asarray(probability, dtype=float).clip(0.0, 1.0)
    working["adjusted"] = working.base * np.power(np.maximum(np.asarray(factors, dtype=float), 1e-6), float(power))
    base_sum = working.groupby("sku_id").base.transform("sum")
    adjusted_sum = working.groupby("sku_id").adjusted.transform("sum").replace(0.0, np.nan)
    result = (working.adjusted * base_sum / adjusted_sum).fillna(working.base)
    return result.to_numpy(float).clip(0.0, 1.0)
