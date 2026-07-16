from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"sku_id", "block_number", "block_start", "target", "forecast"}


def normalize_forecasts(frame: pd.DataFrame, segment: str, source: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    result = frame[list(REQUIRED_COLUMNS)].copy()
    result["segment"] = segment
    result["champion_source"] = source
    result["block_start"] = pd.to_datetime(result.block_start)
    return result


def consolidate(frames: list[pd.DataFrame], expected_skus: int = 690) -> pd.DataFrame:
    result = pd.concat(frames, ignore_index=True)
    duplicates = result.duplicated(["sku_id", "block_number"], keep=False)
    if duplicates.any():
        raise ValueError("Champion routes overlap on SKU and block")
    block_counts = result.groupby("sku_id").block_number.nunique()
    if result.sku_id.nunique() != expected_skus or not block_counts.eq(6).all():
        raise ValueError(f"Expected {expected_skus} SKUs with six blocks")
    return result.sort_values(["segment", "sku_id", "block_number"]).reset_index(drop=True)


def score(frame: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    groups = [("all", frame)] if group_column is None else frame.groupby(group_column, sort=False)
    rows = []
    for name, group in groups:
        sku = group.groupby("sku_id", as_index=False).agg(actual_total=("target", "sum"), forecast_total=("forecast", "sum"))
        errors = group.assign(error=(group.target - group.forecast).abs()).groupby("sku_id").error.sum()
        sku["absolute_error"] = sku.sku_id.map(errors)
        positive = sku.loc[sku.actual_total.gt(0)].copy()
        positive["wmape"] = 100 * positive.absolute_error / positive.actual_total
        actual_total = float(sku.actual_total.sum())
        forecast_total = float(sku.forecast_total.sum())
        rows.append({"segment": name, "all_skus": len(sku), "positive_skus": len(positive), "under_50": int(positive.wmape.lt(50).sum()), "under_70": int(positive.wmape.lt(70).sum()), "under_100": int(positive.wmape.lt(100).sum()), "median_wmape": float(positive.wmape.median()) if len(positive) else np.nan, "portfolio_wmape": float(100*sku.absolute_error.sum()/actual_total) if actual_total else np.nan, "actual_total": actual_total, "forecast_total": forecast_total, "bias_pct": float(100*(forecast_total-actual_total)/actual_total) if actual_total else np.nan})
    result = pd.DataFrame(rows)
    for threshold in (50, 70, 100):
        result[f"pct_below_{threshold}"] = 100 * result[f"under_{threshold}"] / result.positive_skus.replace(0, np.nan)
    return result
