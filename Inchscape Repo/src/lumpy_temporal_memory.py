from __future__ import annotations

import numpy as np
import pandas as pd


def add_within_horizon_rank(errors: pd.DataFrame) -> pd.DataFrame:
    result = errors.copy()
    result["error_rank"] = result.groupby(["horizon_id", "sku_id"]).wmape.rank(
        method="average", pct=True
    )
    return result


def temporal_selections(
    error_history: pd.DataFrame,
    cohort_map: pd.DataFrame,
    horizon_order: list[str],
    score_mode: str = "wmape",
    decay: float = 1.0,
    individual_weight: float = 1.0,
) -> pd.DataFrame:
    if not horizon_order:
        raise ValueError("At least one historical horizon is required.")
    history = add_within_horizon_rank(error_history)
    history = history.loc[history.horizon_id.isin(horizon_order)].copy()
    history = history.drop(columns=["cohort"], errors="ignore")
    value_column = "wmape" if score_mode == "wmape" else "error_rank"
    history[value_column] = history[value_column].astype(float).clip(upper=500.0 if score_mode == "wmape" else 1.0)
    weights = {horizon: float(decay ** (len(horizon_order) - 1 - index)) for index, horizon in enumerate(horizon_order)}
    history["time_weight"] = history.horizon_id.map(weights)
    history["weighted_value"] = history[value_column] * history.time_weight
    individual = history.groupby(["sku_id", "candidate_id"], as_index=False).agg(
        weighted_value=("weighted_value", "sum"), time_weight=("time_weight", "sum")
    )
    individual["individual_score"] = individual.weighted_value / individual.time_weight
    mapped = history.merge(cohort_map[["sku_id", "cohort"]].drop_duplicates(), on="sku_id", how="left")
    cohort = mapped.groupby(["cohort", "candidate_id", "horizon_id"], as_index=False).agg(
        horizon_score=(value_column, "median")
    )
    cohort["time_weight"] = cohort.horizon_id.map(weights)
    cohort["weighted_value"] = cohort.horizon_score * cohort.time_weight
    cohort = cohort.groupby(["cohort", "candidate_id"], as_index=False).agg(
        weighted_value=("weighted_value", "sum"), time_weight=("time_weight", "sum")
    )
    cohort["cohort_score"] = cohort.weighted_value / cohort.time_weight
    scored = individual.merge(cohort_map[["sku_id", "cohort"]].drop_duplicates(), on="sku_id", how="left")
    scored = scored.merge(cohort[["cohort", "candidate_id", "cohort_score"]], on=["cohort", "candidate_id"], how="left")
    scored["routing_score"] = (
        float(individual_weight) * scored.individual_score
        + (1.0 - float(individual_weight)) * scored.cohort_score
    )
    return (
        scored.sort_values(["sku_id", "routing_score", "cohort_score", "candidate_id"])
        .groupby("sku_id", as_index=False)
        .head(1)[["sku_id", "candidate_id", "routing_score"]]
        .rename(columns={"candidate_id": "selected_candidate_id"})
    )


def persistence_table(errors: pd.DataFrame) -> pd.DataFrame:
    ranked = add_within_horizon_rank(errors)
    best = ranked.sort_values(["horizon_id", "sku_id", "wmape", "candidate_id"]).groupby(
        ["horizon_id", "sku_id"], as_index=False
    ).head(1)[["horizon_id", "sku_id", "candidate_id"]].drop_duplicates(["horizon_id", "sku_id"])
    horizons = list(dict.fromkeys(errors.horizon_id.tolist()))
    rows = []
    for left, right in zip(horizons[:-1], horizons[1:]):
        first = best.loc[best.horizon_id.eq(left), ["sku_id", "candidate_id"]].rename(columns={"candidate_id": "left_candidate"})
        second = best.loc[best.horizon_id.eq(right), ["sku_id", "candidate_id"]].rename(columns={"candidate_id": "right_candidate"})
        joined = first.merge(second, on="sku_id")
        rows.append({"from_horizon": left, "to_horizon": right, "sku_count": len(joined), "same_exact_winner": int(joined.left_candidate.eq(joined.right_candidate).sum()), "same_exact_share": float(joined.left_candidate.eq(joined.right_candidate).mean())})
    return pd.DataFrame(rows)
