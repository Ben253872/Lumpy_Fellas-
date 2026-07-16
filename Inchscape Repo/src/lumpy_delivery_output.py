from __future__ import annotations

import numpy as np
import pandas as pd


def individual_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    ordered = forecasts.sort_values(["sku_id", "block_number"]).copy()
    rows = []
    for sku, group in ordered.groupby("sku_id", sort=False):
        actual = group.target.to_numpy(float)
        predicted = group.forecast.to_numpy(float)
        absolute_error = float(np.abs(actual - predicted).sum())
        actual_total = float(actual.sum())
        naive_scale = float(np.abs(np.diff(actual)).mean()) if len(actual) > 1 else np.nan
        mae = float(np.abs(actual - predicted).mean())
        rows.append({"sku_id": sku, "actual_18m": actual_total, "forecast_18m": float(predicted.sum()), "absolute_error": absolute_error, "wmape": 100*absolute_error/actual_total if actual_total>0 else np.nan, "evaluation_mase": mae/naive_scale if naive_scale>0 else np.nan, "bias_units": float(predicted.sum()-actual_total), "bias_pct": 100*(predicted.sum()-actual_total)/actual_total if actual_total>0 else np.nan})
    return pd.DataFrame(rows)


def reliability_tier(wmape: pd.Series, dormant: pd.Series) -> pd.Series:
    return pd.Series(np.select([dormant, wmape.lt(50), wmape.lt(70), wmape.lt(100)], ["manual_lifecycle_review", "forecast_led", "forecast_plus_review", "manual_review_with_forecast"], default="exception_policy"), index=wmape.index)


def build_delivery_table(forecasts: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    metrics = individual_metrics(forecasts)
    forecast_blocks = forecasts.pivot(index="sku_id", columns="block_number", values="forecast").add_prefix("forecast_block_").reset_index()
    actual_blocks = forecasts.pivot(index="sku_id", columns="block_number", values="target").add_prefix("actual_block_").reset_index()
    source = forecasts.groupby("sku_id", as_index=False).agg(segment=("segment", "first"), champion_source=("champion_source", "first"))
    metadata_columns = [column for column in ["sku_id", "lifecycle_tier", "frequency_tier", "recency_tier", "size_tier", "abc_units_tier", "potential_stock_status", "strategy_tier"] if column in assignments.columns]
    result = source.merge(assignments[metadata_columns].drop_duplicates("sku_id"), on="sku_id", how="left").merge(metrics, on="sku_id").merge(forecast_blocks, on="sku_id").merge(actual_blocks, on="sku_id")
    result["positive_sku"] = result.actual_18m.gt(0)
    result["dormant_manual_review"] = result.lifecycle_tier.eq("dormant")
    result["reliability_tier"] = reliability_tier(result.wmape, result.dormant_manual_review)
    result["bias_direction"] = np.select([result.bias_units.lt(0), result.bias_units.gt(0)], ["underforecast", "overforecast"], default="balanced")
    return result.sort_values(["dormant_manual_review", "segment", "sku_id"]).reset_index(drop=True)
