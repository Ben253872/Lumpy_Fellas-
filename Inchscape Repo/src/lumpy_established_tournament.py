from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import lumpy_block_hybrid as bh


CALIBRATION_SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
BASE_MODEL_KEYS = tuple(bh.BASELINE_LABELS) + tuple(bh.MODEL_SPECS)


def variant_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_key in bh.BASELINE_LABELS:
        for scale in CALIBRATION_SCALES:
            rows.append(
                {
                    "base_model_key": model_key,
                    "variant_id": f"{model_key}__baseline__s{scale:.2f}",
                    "mode": "baseline",
                    "power": 1.0,
                    "threshold": 0.0,
                    "scale": scale,
                    "baseline": "none",
                    "blend": 0.0,
                }
            )
    for model_key, spec in bh.MODEL_SPECS.items():
        modes = [("direct", 1.0, "none", 0.0)] if spec.architecture == "direct" else [
            ("expected", 1.0, "none", 0.0),
            ("soft", 0.75, "none", 0.0),
            ("soft", 0.75, "recent6", 0.25),
        ]
        for mode, power, baseline, blend in modes:
            for scale in CALIBRATION_SCALES:
                rows.append(
                    {
                        "base_model_key": model_key,
                        "variant_id": (
                            f"{model_key}__{mode}__p{power:.2f}__{baseline}__b{blend:.2f}__s{scale:.2f}"
                        ),
                        "mode": mode,
                        "power": power,
                        "threshold": 0.0,
                        "scale": scale,
                        "baseline": baseline,
                        "blend": blend,
                    }
                )
    return pd.DataFrame(rows)


def baseline_variants(prepared: dict[str, Any]) -> pd.DataFrame:
    baseline = bh.baseline_forecasts(prepared)
    frames = []
    variants = variant_grid()
    for row in variants.loc[variants.base_model_key.isin(bh.BASELINE_LABELS)].itertuples(index=False):
        frame = baseline.loc[baseline.model_key.eq(row.base_model_key)].copy()
        frame["forecast"] = frame.forecast.astype(float) * float(row.scale)
        frame["base_model_key"] = row.base_model_key
        frame["variant_id"] = row.variant_id
        frame["calibration_scale"] = float(row.scale)
        frame["variant_mode"] = row.mode
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def model_variants(
    model_key: str,
    prepared: dict[str, Any],
    random_state: int = 42,
) -> pd.DataFrame:
    components = bh.fit_components(model_key, prepared, random_state)
    variants = variant_grid().loc[lambda x: x.base_model_key.eq(model_key)]
    frames = []
    for row in variants.itertuples(index=False):
        recipe = {
            "mode": row.mode,
            "power": float(row.power),
            "threshold": float(row.threshold),
            "scale": float(row.scale),
            "baseline": row.baseline,
            "blend": float(row.blend),
        }
        frame = bh.compose(model_key, components, recipe)
        frame["base_model_key"] = model_key
        frame["variant_id"] = row.variant_id
        frame["calibration_scale"] = float(row.scale)
        frame["variant_mode"] = row.mode
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_forecast(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = frame.copy()
    scored["absolute_error"] = (scored.target - scored.forecast).abs()
    sku = scored.groupby("sku_id", as_index=False).agg(
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
        "under_100_share": float(valid.block_wmape_percent.lt(100).mean()) if len(valid) else np.nan,
        "median_sku_block_wmape": float(valid.block_wmape_percent.median()) if len(valid) else np.nan,
        "p75_sku_block_wmape": float(valid.block_wmape_percent.quantile(0.75)) if len(valid) else np.nan,
        "median_sku_block_mase": float(sku.block_mase.median()) if sku.block_mase.notna().any() else np.nan,
        "portfolio_block_wmape": float(100.0 * sku.absolute_error.sum() / actual) if actual > 0 else np.nan,
        "actual_total": actual,
        "forecast_total": forecast,
        "bias_pct": float(100.0 * (forecast - actual) / actual) if actual > 0 else np.nan,
    }
    return sku, summary


def rank_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "under_70_skus",
            "under_50_skus",
            "under_100_skus",
            "median_sku_block_wmape",
            "portfolio_block_wmape",
            "abs_bias_pct",
            "variant_id",
        ],
        ascending=[False, False, False, True, True, True, True],
    ).reset_index(drop=True)


def candidate_summary(forecasts: pd.DataFrame, sku_ids: set) -> pd.DataFrame:
    cohort = forecasts.loc[forecasts.sku_id.isin(sku_ids)]
    rows = []
    for (model_key, variant_id), group in cohort.groupby(["base_model_key", "variant_id"], sort=False):
        _, summary = score_forecast(group)
        rows.append(
            {
                "base_model_key": model_key,
                "variant_id": variant_id,
                "variant_mode": group.variant_mode.iloc[0],
                "calibration_scale": float(group.calibration_scale.iloc[0]),
                **summary,
                "abs_bias_pct": abs(summary["bias_pct"]),
            }
        )
    return rank_summary(pd.DataFrame(rows))


def tuned_model_variants(tuning_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        tuning_summary.sort_values(
            [
                "base_model_key",
                "under_70_skus",
                "under_50_skus",
                "under_100_skus",
                "median_sku_block_wmape",
                "portfolio_block_wmape",
                "abs_bias_pct",
                "variant_id",
            ],
            ascending=[True, False, False, False, True, True, True, True],
        )
        .groupby("base_model_key", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def filter_tuned_variants(forecasts: pd.DataFrame, tuned: pd.DataFrame) -> pd.DataFrame:
    keys = tuned[["base_model_key", "variant_id"]].drop_duplicates()
    return forecasts.merge(keys, on=["base_model_key", "variant_id"], how="inner")
