from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def combine_candidates(
    time_series: pd.DataFrame,
    analogues: pd.DataFrame,
    weights: Iterable[float] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    keys = [column for column in ("fold_id", "sku_id", "block_number", "block_start") if column in time_series.columns]
    target_keys = keys + ["target"]
    output = []
    for model_key, frame in time_series.groupby("model_key", sort=False):
        raw = frame.copy(); raw["candidate_id"] = f"short__{model_key}"; output.append(raw)
    for analogue_key, frame in analogues.groupby("model_key", sort=False):
        raw = frame.copy(); raw["candidate_id"] = f"analogue__{analogue_key}"; output.append(raw)
    for short_key, short in time_series.groupby("model_key", sort=False):
        left = short[target_keys + ["forecast"]].rename(columns={"forecast": "short_forecast"})
        for analogue_key, analogue in analogues.groupby("model_key", sort=False):
            right = analogue[keys + ["forecast"]].rename(columns={"forecast": "analogue_forecast"})
            merged = left.merge(right, on=keys, how="inner")
            for weight in weights:
                candidate = merged[target_keys].copy()
                candidate["forecast"] = weight * merged.short_forecast + (1.0 - weight) * merged.analogue_forecast
                candidate["model_key"] = f"blend_{short_key}_{analogue_key}_{weight:.2f}"
                candidate["candidate_id"] = f"blend__{short_key}__{analogue_key}__short_weight_{weight:.2f}"
                output.append(candidate)
    return pd.concat(output, ignore_index=True)


def score_candidates(forecasts: pd.DataFrame) -> pd.DataFrame:
    unit_keys = [column for column in ("fold_id", "sku_id") if column in forecasts.columns]
    rows = []
    for candidate_id, group in forecasts.groupby("candidate_id", sort=False):
        sku = group.groupby(unit_keys, as_index=False).agg(actual_total=("target", "sum"), forecast_total=("forecast", "sum"))
        errors = group.assign(error=(group.target - group.forecast).abs()).groupby(unit_keys).error.sum().reset_index(name="absolute_error")
        sku = sku.merge(errors, on=unit_keys)
        positive = sku.loc[sku.actual_total.gt(0)].copy(); positive["wmape"] = 100 * positive.absolute_error / positive.actual_total
        rows.append({"candidate_id": candidate_id, "positive_cases": len(positive), "under_50": int(positive.wmape.lt(50).sum()), "under_70": int(positive.wmape.lt(70).sum()), "under_100": int(positive.wmape.lt(100).sum()), "median_wmape": float(positive.wmape.median()), "portfolio_wmape": float(100*sku.absolute_error.sum()/sku.actual_total.sum()), "actual_total": float(sku.actual_total.sum()), "forecast_total": float(sku.forecast_total.sum())})
    return pd.DataFrame(rows)


def rank_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    for threshold in (50, 70, 100):
        ranked[f"share_{threshold}"] = ranked[f"under_{threshold}"] / ranked.positive_cases
    return ranked.sort_values(["share_70", "share_50", "share_100", "median_wmape", "portfolio_wmape"], ascending=[False, False, False, True, True]).reset_index(drop=True)


def pair_blends(
    forecasts: pd.DataFrame,
    left_model: str,
    right_model: str,
    left_weights: Iterable[float] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    keys = [column for column in ("fold_id", "sku_id", "block_number", "block_start") if column in forecasts.columns]
    target_keys = keys + ["target"]
    left = forecasts.loc[forecasts.model_key.eq(left_model), target_keys + ["forecast"]].rename(columns={"forecast": "left_forecast"})
    right = forecasts.loc[forecasts.model_key.eq(right_model), keys + ["forecast"]].rename(columns={"forecast": "right_forecast"})
    merged = left.merge(right, on=keys, how="inner")
    output = []
    for weight in left_weights:
        candidate = merged[target_keys].copy()
        candidate["forecast"] = weight * merged.left_forecast + (1.0 - weight) * merged.right_forecast
        candidate["model_key"] = f"pair_{left_model}_{right_model}_{weight:.2f}"
        candidate["candidate_id"] = f"pair__{left_model}__{right_model}__left_weight_{weight:.2f}"
        output.append(candidate)
    return pd.concat(output, ignore_index=True)


def scale_candidate_grid(forecasts: pd.DataFrame, scales: Iterable[float]) -> pd.DataFrame:
    output = []
    for scale in scales:
        candidate = forecasts.copy()
        candidate["forecast"] = candidate.forecast.clip(lower=0.0) * float(scale)
        candidate["candidate_id"] = candidate.candidate_id.astype(str) + f"__scale_{float(scale):.2f}"
        candidate["calibration_scale"] = float(scale)
        output.append(candidate)
    return pd.concat(output, ignore_index=True)
