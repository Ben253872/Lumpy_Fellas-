from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def recency_band(months_since_positive: pd.Series) -> pd.Series:
    return pd.cut(months_since_positive, bins=[11, 17, 23, 35, np.inf], labels=["12_17", "18_23", "24_35", "36_plus"]).astype("object")


def case_table(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    columns = ["sku_id", "months_since_positive", "positive_mean", "SUBFAMILY_DESCRIPTION"]
    available = [column for column in columns if column in features.columns]
    cases = targets.merge(features[available].drop_duplicates("sku_id"), on="sku_id", how="left", validate="many_to_one")
    cases["recency_band"] = recency_band(cases.months_since_positive)
    cases["event"] = cases.target.gt(0).astype(float)
    return cases


def reactivation_candidates(
    history: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    group_modes: Iterable[str] = ("recency", "subfamily_recency"),
    smoothing_strengths: Iterable[float] = (5.0, 15.0, 40.0),
    individual_size_weights: Iterable[float] = (0.0, 0.5, 0.8),
    thresholds: Iterable[float] = (0.0, 0.25, 0.5),
    scales: Iterable[float] = (0.75, 1.0, 1.25),
) -> pd.DataFrame:
    if history.empty:
        raise ValueError("At least one completed dormant cohort is required")
    current = case_table(features, targets)
    global_stats = history.groupby("block_number").agg(events=("event", "sum"), exposures=("event", "size"), positive_total=("target", lambda values: values[values > 0].sum()), positive_count=("target", lambda values: values.gt(0).sum())).reset_index()
    global_stats["global_probability"] = (global_stats.events + 1.0) / (global_stats.exposures + 2.0)
    global_stats["global_size"] = global_stats.positive_total / global_stats.positive_count.replace(0, np.nan)
    output = []
    for group_mode in group_modes:
        group_columns = ["block_number", "recency_band"]
        if group_mode == "subfamily_recency":
            group_columns.append("SUBFAMILY_DESCRIPTION")
        grouped = history.groupby(group_columns, dropna=False).agg(events=("event", "sum"), exposures=("event", "size"), positive_total=("target", lambda values: values[values > 0].sum()), positive_count=("target", lambda values: values.gt(0).sum())).reset_index()
        for strength in smoothing_strengths:
            stats = grouped.merge(global_stats[["block_number", "global_probability", "global_size"]], on="block_number", how="left")
            stats["event_probability"] = (stats.events + strength * stats.global_probability) / (stats.exposures + strength)
            stats["group_size"] = (stats.positive_total + strength * stats.global_size) / (stats.positive_count + strength)
            forecast = current.merge(stats[group_columns + ["event_probability", "group_size"]], on=group_columns, how="left").merge(global_stats[["block_number", "global_probability", "global_size"]], on="block_number", how="left")
            forecast["event_probability"] = forecast.event_probability.fillna(forecast.global_probability)
            forecast["group_size"] = forecast.group_size.fillna(forecast.global_size).fillna(0.0)
            for size_weight in individual_size_weights:
                size = size_weight * forecast.positive_mean.fillna(0.0) + (1.0 - size_weight) * forecast.group_size
                for threshold in thresholds:
                    base = forecast.event_probability * size if threshold == 0 else np.where(forecast.event_probability >= threshold, size, 0.0)
                    mode = "expected" if threshold == 0 else "gated"
                    for scale in scales:
                        candidate = forecast[["sku_id", "block_number", "block_start", "target", "event_probability"]].copy()
                        candidate["forecast"] = np.maximum(0.0, base * scale)
                        candidate["candidate_id"] = f"{group_mode}__smooth_{strength:.0f}__size_{size_weight:.1f}__{mode}_{threshold:.2f}__scale_{scale:.2f}"
                        output.append(candidate)
    return pd.concat(output, ignore_index=True)


def inventory_policy(forecasts: pd.DataFrame) -> pd.DataFrame:
    result = forecasts.groupby("sku_id", as_index=False).agg(expected_units=("forecast", "sum"), no_event_probability=("event_probability", lambda values: float(np.prod(1.0 - values.clip(0, 1)))))
    result["reactivation_probability"] = 1.0 - result.no_event_probability
    result["inventory_policy"] = np.select([result.reactivation_probability.ge(0.75), result.reactivation_probability.ge(0.35)], ["safety_stock_review", "monitor_reactivation"], default="no_stock_manual_review")
    return result.drop(columns="no_event_probability")
