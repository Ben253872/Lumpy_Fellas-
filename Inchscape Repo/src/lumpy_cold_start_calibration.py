from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def calibrated_candidates(
    forecasts: pd.DataFrame,
    scales: Iterable[float],
    top_blocks: Iterable[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """Expand analogue forecasts with level and sparse block-placement choices."""
    required = {"sku_id", "block_number", "forecast", "model_key"}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    group_columns = [column for column in ("fold_id", "sku_id", "model_key") if column in forecasts.columns]
    output: list[pd.DataFrame] = []
    for scale in scales:
        for keep in top_blocks:
            candidate = forecasts.copy()
            if keep:
                ranks = candidate.groupby(group_columns, sort=False)["forecast"].rank(
                    method="first", ascending=False
                )
                candidate["forecast"] = candidate["forecast"].where(ranks.le(keep), 0.0)
            candidate["forecast"] = candidate["forecast"].clip(lower=0.0) * float(scale)
            placement = "all_blocks" if keep == 0 else f"top_{keep}_blocks"
            candidate["source_model_key"] = candidate["model_key"]
            candidate["calibration_scale"] = float(scale)
            candidate["placement_rule"] = placement
            candidate["candidate_id"] = (
                candidate["model_key"].astype(str)
                + "__"
                + placement
                + f"__scale_{float(scale):.2f}"
            )
            output.append(candidate)
    return pd.concat(output, ignore_index=True)


def candidate_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in forecasts.groupby("candidate_id", sort=False):
        sku = (
            group.groupby("sku_id", as_index=False)
            .agg(
                absolute_error=("forecast", lambda values: 0.0),
                actual_total=("target", "sum"),
                forecast_total=("forecast", "sum"),
            )
        )
        errors = group.assign(error=(group.target - group.forecast).abs()).groupby("sku_id").error.sum()
        sku["absolute_error"] = sku.sku_id.map(errors)
        positive = sku.loc[sku.actual_total.gt(0)].copy()
        positive["wmape"] = 100.0 * positive.absolute_error / positive.actual_total
        rows.append(
            {
                "candidate_id": candidate_id,
                "positive_skus": len(positive),
                "under_50": int(positive.wmape.lt(50).sum()),
                "under_70": int(positive.wmape.lt(70).sum()),
                "under_100": int(positive.wmape.lt(100).sum()),
                "median_wmape": float(positive.wmape.median()) if len(positive) else np.nan,
                "portfolio_wmape": (
                    float(100.0 * sku.absolute_error.sum() / sku.actual_total.sum())
                    if sku.actual_total.sum() > 0
                    else np.nan
                ),
                "actual_total": float(sku.actual_total.sum()),
                "forecast_total": float(sku.forecast_total.sum()),
            }
        )
    return pd.DataFrame(rows)


def rank_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["under_70", "under_50", "under_100", "median_wmape", "portfolio_wmape"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
