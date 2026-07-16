from __future__ import annotations

import numpy as np
import pandas as pd


BLOCK_MONTHS = 3


def target_blocks(test: pd.DataFrame, sku_ids: set) -> pd.DataFrame:
    frame = test.loc[test.sku_id.isin(sku_ids)].sort_values(["sku_id", "month"]).copy()
    frame["block_number"] = frame.groupby("sku_id").cumcount() // BLOCK_MONTHS + 1
    blocks = frame.groupby(["sku_id", "block_number"], as_index=False).agg(
        block_start=("month", "min"), target=("demand", "sum")
    )
    return blocks.loc[blocks.block_number.le(6)].copy()


def total_targets(test: pd.DataFrame, sku_ids: set) -> pd.Series:
    return test.loc[test.sku_id.isin(sku_ids)].groupby("sku_id").demand.sum().reindex(sorted(sku_ids), fill_value=0.0)


def timing_weights(
    train: pd.DataFrame,
    sku_ids: set,
    block_starts: list[pd.Timestamp],
    group_column: str,
) -> pd.DataFrame:
    train = train.loc[train.sku_id.isin(sku_ids)].copy()
    train["month"] = pd.to_datetime(train.month)
    if group_column == "__cohort__" or group_column not in train.columns:
        metadata = pd.DataFrame({"sku_id": sorted(sku_ids), "timing_group": "all"})
    else:
        metadata = (
            train.sort_values(["sku_id", "month"])
            .groupby("sku_id", as_index=False)[group_column]
            .agg(lambda values: values.dropna().iloc[-1] if values.notna().any() else "unknown")
            .rename(columns={group_column: "timing_group"})
        )
        metadata = pd.DataFrame({"sku_id": sorted(sku_ids)}).merge(metadata, on="sku_id", how="left").fillna({"timing_group": "unknown"})
    history = train.merge(metadata, on="sku_id", how="left")
    history["month_number"] = history.month.dt.month
    seasonal = history.groupby(["timing_group", "month_number"]).demand.mean()
    cohort_seasonal = history.groupby("month_number").demand.mean()
    rows = []
    for group, members in metadata.groupby("timing_group"):
        scores = []
        for block_start in block_starts:
            months = [(pd.Timestamp(block_start) + pd.DateOffset(months=offset)).month for offset in range(BLOCK_MONTHS)]
            score = sum(float(seasonal.get((group, month), cohort_seasonal.get(month, 0.0))) for month in months)
            scores.append(score)
        scores = np.asarray(scores, dtype=float)
        scores = scores / scores.sum() if scores.sum() > 0 else np.repeat(1.0 / len(block_starts), len(block_starts))
        for sku in members.sku_id:
            for number, (block_start, weight) in enumerate(zip(block_starts, scores), 1):
                rows.append({"sku_id": sku, "block_number": number, "block_start": pd.Timestamp(block_start), "timing_weight": float(weight)})
    return pd.DataFrame(rows)


def distribute_totals(
    total_forecasts: pd.DataFrame,
    actual_blocks: pd.DataFrame,
    weights: pd.DataFrame,
    naive_scale: pd.Series | None = None,
) -> pd.DataFrame:
    result = actual_blocks.merge(total_forecasts[["sku_id", "forecast_total"]], on="sku_id", how="left")
    result = result.merge(weights, on=["sku_id", "block_number", "block_start"], how="left")
    result["forecast"] = result.forecast_total.fillna(0.0).clip(lower=0.0) * result.timing_weight.fillna(1.0 / 6.0)
    if naive_scale is None:
        result["block_naive_scale"] = 0.0
    else:
        result["block_naive_scale"] = result.sku_id.map(naive_scale).fillna(0.0)
    return result


def empirical_total_forecast(features: pd.DataFrame, method: str) -> np.ndarray:
    if method == "recent12_rate":
        return np.maximum(0.0, features.recent_12m_total.to_numpy(float) * 1.5)
    if method == "recent6_rate":
        return np.maximum(0.0, features.recent_6m_total.to_numpy(float) * 3.0)
    if method == "history_rate":
        months = np.maximum(features.history_months.to_numpy(float), 1.0)
        return np.maximum(0.0, features.history_total_demand.to_numpy(float) * 18.0 / months)
    if method == "bayes_rate":
        own_months = np.maximum(features.history_months.to_numpy(float), 1.0)
        own_rate = features.history_total_demand.to_numpy(float) / own_months
        global_rate = float(features.history_total_demand.sum() / max(features.history_months.sum(), 1.0))
        reliability = own_months / (own_months + 12.0)
        return np.maximum(0.0, 18.0 * (reliability * own_rate + (1.0 - reliability) * global_rate))
    raise ValueError(f"Unknown empirical method: {method}")


def naive_scales(train: pd.DataFrame, sku_ids: set) -> pd.Series:
    def scale(values):
        array = values.to_numpy(float)
        return float(np.mean(np.abs(np.diff(array)))) if len(array) > 1 else 0.0
    return train.loc[train.sku_id.isin(sku_ids)].sort_values(["sku_id", "month"]).groupby("sku_id").demand.apply(scale)
