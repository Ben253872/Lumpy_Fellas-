from __future__ import annotations

import numpy as np
import pandas as pd


def sku_error_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    sku = forecasts.groupby(["segment", "sku_id"], as_index=False).agg(actual_total=("target", "sum"), forecast_total=("forecast", "sum"))
    errors = forecasts.assign(absolute_error=(forecasts.target - forecasts.forecast).abs()).groupby(["segment", "sku_id"], as_index=False).absolute_error.sum()
    sku = sku.merge(errors, on=["segment", "sku_id"])
    sku["signed_error"] = sku.forecast_total - sku.actual_total
    sku["bias_pct"] = 100 * sku.signed_error / sku.actual_total.replace(0, np.nan)
    sku["wmape"] = 100 * sku.absolute_error / sku.actual_total.replace(0, np.nan)
    sku["actual_volume_quartile"] = sku.groupby("segment").actual_total.transform(lambda values: pd.qcut(values.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    return sku


def concentration_table(sku: pd.DataFrame, shares: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5)) -> pd.DataFrame:
    positive = sku.loc[sku.actual_total.gt(0)].sort_values("absolute_error", ascending=False).copy()
    total_error = positive.absolute_error.sum()
    rows = []
    for share in shares:
        count = max(1, int(np.ceil(len(positive) * share)))
        top = positive.head(count)
        rows.append({"top_sku_share": share, "sku_count": count, "absolute_error_share": float(top.absolute_error.sum() / total_error), "actual_volume_share": float(top.actual_total.sum() / positive.actual_total.sum()), "signed_error": float(top.signed_error.sum())})
    return pd.DataFrame(rows)


def block_error_attribution(forecasts: pd.DataFrame, event_threshold: float = 0.5) -> pd.DataFrame:
    frame = forecasts.copy()
    actual_event = frame.target.gt(0)
    forecast_event = frame.forecast.ge(event_threshold)
    frame["error_type"] = np.select(
        [actual_event & ~forecast_event, ~actual_event & forecast_event, actual_event & forecast_event & frame.forecast.lt(frame.target), actual_event & forecast_event & frame.forecast.ge(frame.target)],
        ["missed_event", "false_event", "matched_event_under", "matched_event_over"],
        default="quiet_block",
    )
    frame["absolute_error"] = (frame.target - frame.forecast).abs()
    frame["signed_error"] = frame.forecast - frame.target
    result = frame.groupby(["segment", "error_type"], as_index=False).agg(blocks=("sku_id", "size"), actual_units=("target", "sum"), forecast_units=("forecast", "sum"), absolute_error=("absolute_error", "sum"), signed_error=("signed_error", "sum"))
    result["segment_error_share"] = result.absolute_error / result.groupby("segment").absolute_error.transform("sum")
    return result
