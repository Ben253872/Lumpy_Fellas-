from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicySpec:
    name: str
    selector: str
    individual_weight: float = 1.0
    blend_weight: float = 1.0
    scale: float = 1.0


POLICIES = (
    PolicySpec("sku_only", "sku"),
    PolicySpec("hierarchical_75", "hierarchical", individual_weight=0.75),
    PolicySpec("hierarchical_50", "hierarchical", individual_weight=0.50),
    PolicySpec("frequency_champion", "frequency"),
    PolicySpec("strategy_tier_champion", "strategy_tier"),
    PolicySpec("sku_global_blend_75", "sku_global_blend", blend_weight=0.75),
    PolicySpec("sku_global_blend_50", "sku_global_blend", blend_weight=0.50),
)


def add_strategy_tier(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    point_priority = (
        result["frequency_tier"].eq("recurring_4_6")
        & result["recency_tier"].eq("recent_0_2")
        & result["abc_units_tier"].isin(["A", "B"])
        & result["lifecycle_tier"].isin(["established", "developing"])
    )
    inventory_priority = (
        result["lifecycle_tier"].isin(["cold_start", "dormant"])
        | result["frequency_tier"].eq("rare_0_1")
        | result["recency_tier"].isin(["stale_6_11", "dormant_12_plus", "no_history"])
    )
    result["strategy_tier"] = np.select(
        [point_priority, inventory_priority],
        ["point_forecast_priority", "inventory_policy_priority"],
        default="cautious_point_forecast",
    )
    result["forecast_mode"] = result["strategy_tier"].map(
        {
            "point_forecast_priority": "point_forecast_plus_range",
            "cautious_point_forecast": "cautious_point_plus_range",
            "inventory_policy_priority": "inventory_policy_plus_range",
        }
    )
    return result


def add_block_last_month(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["block_start"] = pd.to_datetime(result["block_start"])
    result["block_last_month"] = result["block_start"] + pd.DateOffset(months=2)
    return result


def known_forecasts_at_cutoff(
    forecasts: pd.DataFrame,
    cutoff: pd.Timestamp,
    identity_columns: Iterable[str] = ("sku_id", "model_key", "block_start"),
) -> pd.DataFrame:
    result = add_block_last_month(forecasts)
    result = result.loc[result.block_last_month.le(pd.Timestamp(cutoff))].copy()
    sort_columns = [column for column in [*identity_columns, "train_end", "fold_id"] if column in result.columns]
    if "train_end" in result.columns:
        result["train_end"] = pd.to_datetime(result["train_end"])
    result = result.sort_values(sort_columns).drop_duplicates(list(identity_columns), keep="last")
    return result.reset_index(drop=True)


def _sku_model_scores(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["absolute_error"] = (frame["target"] - frame["forecast"]).abs()
    scores = frame.groupby(["sku_id", "model_key"], as_index=False).agg(
        actual_total=("target", "sum"),
        absolute_error=("absolute_error", "sum"),
    )
    scores["wmape"] = np.where(
        scores.actual_total.gt(0), 100.0 * scores.absolute_error / scores.actual_total, np.nan
    )
    return scores


def _rank_model_summary(scores: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in scores.groupby(group_columns + ["model_key"], dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        valid = group.loc[group.wmape.notna()]
        row = dict(zip(group_columns + ["model_key"], keys))
        row.update(
            under_50=int(valid.wmape.lt(50).sum()),
            under_70=int(valid.wmape.lt(70).sum()),
            under_100=int(valid.wmape.lt(100).sum()),
            median_wmape=float(valid.wmape.median()) if len(valid) else np.inf,
            pooled_wmape=float(100.0 * group.absolute_error.sum() / group.actual_total.sum())
            if group.actual_total.sum() > 0
            else np.inf,
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        group_columns + ["under_50", "under_70", "under_100", "median_wmape", "pooled_wmape", "model_key"],
        ascending=[True] * len(group_columns) + [False, False, False, True, True, True],
    )


def _global_champion(scores: pd.DataFrame) -> str:
    return str(_rank_model_summary(scores, []).iloc[0].model_key)


def _selection_table(history: pd.DataFrame, features: pd.DataFrame, spec: PolicySpec) -> tuple[pd.DataFrame, str]:
    scores = _sku_model_scores(history)
    feature_columns = ["sku_id", "frequency_tier", "strategy_tier"]
    scores = scores.merge(features[feature_columns].drop_duplicates("sku_id"), on="sku_id", how="left")
    global_key = _global_champion(scores)

    if spec.selector == "sku" or spec.selector == "sku_global_blend":
        selected = (
            scores.assign(rank_score=scores.wmape.fillna(np.inf))
            .sort_values(["sku_id", "rank_score", "absolute_error", "model_key"])
            .groupby("sku_id", as_index=False)
            .head(1)[["sku_id", "model_key"]]
        )
    elif spec.selector in {"frequency", "strategy_tier"}:
        segment = spec.selector + ("_tier" if spec.selector == "frequency" else "")
        summary = _rank_model_summary(scores, [segment])
        winners = summary.groupby(segment, as_index=False).head(1)[[segment, "model_key"]]
        selected = features[["sku_id", segment]].merge(winners, on=segment, how="left")
    elif spec.selector == "hierarchical":
        segment_summary = _rank_model_summary(scores, ["strategy_tier"])
        segment_metric = segment_summary[["strategy_tier", "model_key", "median_wmape"]].rename(
            columns={"median_wmape": "segment_wmape"}
        )
        ranked = scores.merge(segment_metric, on=["strategy_tier", "model_key"], how="left")
        ranked["rank_score"] = (
            spec.individual_weight * ranked.wmape.fillna(ranked.segment_wmape)
            + (1.0 - spec.individual_weight) * ranked.segment_wmape
        )
        selected = (
            ranked.sort_values(["sku_id", "rank_score", "absolute_error", "model_key"])
            .groupby("sku_id", as_index=False)
            .head(1)[["sku_id", "model_key"]]
        )
    else:
        raise ValueError(f"Unknown selector: {spec.selector}")

    selected["model_key"] = selected.model_key.fillna(global_key)
    return selected.rename(columns={"model_key": "selected_model_key"}), global_key


def apply_policy(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    spec: PolicySpec,
) -> pd.DataFrame:
    selections, global_key = _selection_table(history, features, spec)
    candidate = candidates.merge(selections, on="sku_id", how="left")
    candidate["selected_model_key"] = candidate.selected_model_key.fillna(global_key)
    chosen = candidate.loc[candidate.model_key.eq(candidate.selected_model_key)].copy()

    if spec.selector == "sku_global_blend":
        global_rows = candidates.loc[candidates.model_key.eq(global_key), ["sku_id", "block_start", "forecast"]].rename(
            columns={"forecast": "global_forecast"}
        )
        chosen = chosen.merge(global_rows, on=["sku_id", "block_start"], how="left")
        chosen["forecast"] = (
            spec.blend_weight * chosen.forecast
            + (1.0 - spec.blend_weight) * chosen.global_forecast.fillna(chosen.forecast)
        )
        chosen["model"] = f"{spec.name} blend"
        chosen["model_key"] = spec.name
    chosen["forecast"] = np.maximum(0.0, spec.scale * chosen.forecast.astype(float))
    chosen["segmented_policy"] = spec.name
    return chosen


def score_forecasts(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = forecasts.copy()
    frame["absolute_error"] = (frame.target - frame.forecast).abs()
    if "block_naive_scale" not in frame.columns:
        frame["block_naive_scale"] = np.nan
    sku = frame.groupby("sku_id", as_index=False).agg(
        block_count=("target", "size"),
        actual_total=("target", "sum"),
        forecast_total=("forecast", "sum"),
        absolute_error=("absolute_error", "sum"),
        mean_block_absolute_error=("absolute_error", "mean"),
        block_naive_scale=("block_naive_scale", "first"),
    )
    sku["block_wmape_percent"] = np.where(
        sku.actual_total.gt(0), 100.0 * sku.absolute_error / sku.actual_total, np.nan
    )
    sku["block_mase"] = np.where(
        sku.block_naive_scale.gt(0), sku.mean_block_absolute_error / sku.block_naive_scale, np.nan
    )
    valid = sku.loc[sku.block_wmape_percent.notna()]
    actual = float(sku.actual_total.sum())
    forecast = float(sku.forecast_total.sum())
    summary = {
        "all_sku_count": int(sku.sku_id.nunique()),
        "valid_positive_sku_count": int(len(valid)),
        "under_50_skus": int(valid.block_wmape_percent.lt(50).sum()),
        "under_70_skus": int(valid.block_wmape_percent.lt(70).sum()),
        "under_100_skus": int(valid.block_wmape_percent.lt(100).sum()),
        "under_50_share": float(valid.block_wmape_percent.lt(50).mean()) if len(valid) else np.nan,
        "under_70_share": float(valid.block_wmape_percent.lt(70).mean()) if len(valid) else np.nan,
        "median_sku_block_wmape": float(valid.block_wmape_percent.median()) if len(valid) else np.nan,
        "p75_sku_block_wmape": float(valid.block_wmape_percent.quantile(0.75)) if len(valid) else np.nan,
        "median_sku_block_mase": float(sku.block_mase.median()) if sku.block_mase.notna().any() else np.nan,
        "portfolio_block_wmape": float(100.0 * sku.absolute_error.sum() / actual) if actual > 0 else np.nan,
        "actual_total": actual,
        "forecast_total": forecast,
        "bias_pct": float(100.0 * (forecast - actual) / actual) if actual > 0 else np.nan,
    }
    return sku, summary


def rank_policy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["under_50_skus", "under_70_skus", "under_100_skus", "median_sku_block_wmape", "portfolio_block_wmape", "policy"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)


def fit_interval_radius(residuals: pd.DataFrame, features: pd.DataFrame, quantile: float = 0.80) -> pd.DataFrame:
    frame = residuals.drop(columns=["strategy_tier"], errors="ignore").merge(
        features[["sku_id", "strategy_tier"]], on="sku_id", how="left"
    )
    frame["absolute_error"] = (frame.target - frame.forecast).abs()
    global_radius = float(frame.absolute_error.quantile(quantile)) if len(frame) else 0.0
    rows = []
    for tier, group in frame.groupby("strategy_tier", dropna=False):
        radius = float(group.absolute_error.quantile(quantile)) if len(group) >= 20 else global_radius
        rows.append({"strategy_tier": tier, "interval_radius": radius, "residual_rows": len(group)})
    return pd.DataFrame(rows)


def attach_intervals(
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    radii: pd.DataFrame,
) -> pd.DataFrame:
    result = forecasts.drop(columns=["strategy_tier", "forecast_mode"], errors="ignore").merge(
        features[["sku_id", "strategy_tier", "forecast_mode"]].drop_duplicates("sku_id"),
        on="sku_id",
        how="left",
    ).merge(radii, on="strategy_tier", how="left")
    fallback = float(radii.interval_radius.median()) if len(radii) else 0.0
    result["interval_radius"] = result.interval_radius.fillna(fallback)
    result["forecast_lower_80"] = np.maximum(0.0, result.forecast - result.interval_radius)
    result["forecast_upper_80"] = result.forecast + result.interval_radius
    return result
