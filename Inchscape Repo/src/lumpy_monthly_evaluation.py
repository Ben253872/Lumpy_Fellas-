"""Canonical evaluation for monthly lumpy-demand forecasts.

The operational contract is:

* a forecast is issued every month;
* each scored forecast is matched to the actual two months later; and
* accuracy is smoothed with complete, overlapping three-month WMAPE windows.

The functions here contain no model-specific logic so tuning, model selection,
scorecards and exports can all use exactly the same metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonthlyEvaluationConfig:
    forecast_lead_months: int = 2
    forecast_horizon_months: int = 18
    rolling_wmape_months: int = 3
    train_months: int = 48
    minimum_train_months: int = 18


def month_start(value: object) -> pd.Timestamp:
    """Normalise a date-like value to the first day of its month."""
    return pd.Timestamp(value).to_period("M").to_timestamp()


def build_monthly_origin_jobs(
    data: pd.DataFrame,
    target_months: Iterable[object],
    month_column: str,
    config: MonthlyEvaluationConfig | None = None,
) -> pd.DataFrame:
    """Create one chronology-safe training cutoff for every target month."""
    config = config or MonthlyEvaluationConfig()
    available_months = pd.Series(
        pd.to_datetime(data[month_column].dropna().unique())
    ).sort_values()
    if available_months.empty:
        return pd.DataFrame()

    first_month = month_start(available_months.iloc[0])
    rows = []
    for job_id, target_value in enumerate(sorted(set(target_months)), start=1):
        target_month = month_start(target_value)
        forecast_origin = target_month - pd.DateOffset(
            months=config.forecast_lead_months
        )
        train_end = forecast_origin
        desired_train_start = train_end - pd.DateOffset(
            months=config.train_months - 1
        )
        train_start = max(first_month, desired_train_start)
        observed_train_months = int(
            available_months.between(train_start, train_end).sum()
        )
        if observed_train_months < config.minimum_train_months:
            continue
        rows.append(
            {
                "job_id": job_id,
                "forecast_origin": forecast_origin,
                "target_month": target_month,
                "train_start": train_start,
                "train_end": train_end,
                "observed_train_months": observed_train_months,
                "forecast_lead_months": config.forecast_lead_months,
            }
        )
    return pd.DataFrame(rows)


def run_monthly_origin_backtest(
    data: pd.DataFrame,
    jobs: pd.DataFrame,
    model_names: Iterable[str],
    run_model: Callable[[str, pd.DataFrame, pd.DataFrame], pd.DataFrame],
    sku_column: str,
    month_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each candidate using only information available at its origin."""
    forecasts = []
    errors = []
    for job in jobs.itertuples(index=False):
        train = data.loc[
            pd.to_datetime(data[month_column]).between(
                pd.Timestamp(job.train_start), pd.Timestamp(job.train_end)
            )
        ].copy()
        test = data.loc[
            pd.to_datetime(data[month_column]).eq(pd.Timestamp(job.target_month))
        ].copy()
        for model_name in model_names:
            try:
                forecast = run_model(model_name, train, test).copy()
                forecast["model_key"] = model_name
                for column in jobs.columns:
                    forecast[column] = getattr(job, column)
                forecasts.append(forecast)
            except Exception as exc:
                errors.append(
                    {
                        "job_id": job.job_id,
                        "target_month": job.target_month,
                        "model_key": model_name,
                        "error": repr(exc),
                    }
                )
    forecast_frame = (
        pd.concat(forecasts, ignore_index=True, sort=False)
        if forecasts
        else pd.DataFrame()
    )
    if not forecast_frame.empty:
        duplicate_keys = [sku_column, month_column, "model_key"]
        if forecast_frame.duplicated(duplicate_keys).any():
            raise AssertionError(
                f"Duplicate forecast rows found for {duplicate_keys}."
            )
    return forecast_frame, pd.DataFrame(errors)


def add_complete_rolling_wmape(
    forecasts: pd.DataFrame,
    sku_column: str,
    month_column: str,
    actual_column: str,
    forecast_column: str = "forecast",
    model_column: str = "model_key",
    rolling_months: int = 3,
) -> pd.DataFrame:
    """Add complete overlapping three-month-total WMAPE windows.

    Each SKU is aggregated inside its own three-month window before the
    absolute error is calculated. This smooths one-month timing differences
    without allowing errors from different SKUs to cancel.
    """
    scored = forecasts.copy()
    scored[month_column] = pd.to_datetime(scored[month_column])
    scored[actual_column] = pd.to_numeric(
        scored[actual_column], errors="coerce"
    ).fillna(0.0)
    scored[forecast_column] = pd.to_numeric(
        scored[forecast_column], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    scored["absolute_error"] = (
        scored[actual_column] - scored[forecast_column]
    ).abs()
    groups = [sku_column, model_column]
    scored = scored.sort_values(groups + [month_column]).reset_index(drop=True)
    scored["rolling_actual_units"] = scored.groupby(
        groups, dropna=False
    )[actual_column].transform(
        lambda values: values.rolling(
            rolling_months, min_periods=rolling_months
        ).sum()
    )
    scored["rolling_forecast_units"] = scored.groupby(
        groups, dropna=False
    )[forecast_column].transform(
        lambda values: values.rolling(
            rolling_months, min_periods=rolling_months
        ).sum()
    )
    scored["rolling_absolute_error_units"] = (
        scored["rolling_actual_units"] - scored["rolling_forecast_units"]
    ).abs()
    scored["rolling_3m_wmape_percent"] = np.where(
        scored["rolling_actual_units"].gt(0),
        100.0
        * scored["rolling_absolute_error_units"]
        / scored["rolling_actual_units"],
        np.nan,
    )
    return scored


def score_skus(
    scored_forecasts: pd.DataFrame,
    sku_column: str,
    actual_column: str,
    model_column: str = "model_key",
) -> pd.DataFrame:
    """Return one comparable mean rolling-three-month WMAPE per SKU/model."""
    return (
        scored_forecasts.groupby([sku_column, model_column], as_index=False)
        .agg(
            actual_units=(actual_column, "sum"),
            forecast_units=("forecast", "sum"),
            absolute_error_units=("absolute_error", "sum"),
            rolling_windows_scored=("rolling_3m_wmape_percent", "count"),
            rolling_3m_wmape_percent=(
                "rolling_3m_wmape_percent",
                "mean",
            ),
        )
    )


def pooled_group_rolling_wmape(
    forecasts: pd.DataFrame,
    group_columns: list[str],
    month_column: str,
    actual_column: str,
    forecast_column: str = "forecast",
    rolling_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool SKU rolling-total absolute errors without cross-SKU cancellation."""
    source = forecasts.copy()
    source[actual_column] = pd.to_numeric(
        source[actual_column], errors="coerce"
    ).fillna(0.0)
    source[forecast_column] = pd.to_numeric(
        source[forecast_column], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    source = add_complete_rolling_wmape(
        source.assign(model_key="group_metric"),
        sku_column="sku_id",
        month_column=month_column,
        actual_column=actual_column,
        forecast_column=forecast_column,
        model_column="model_key",
        rolling_months=rolling_months,
    )
    monthly = (
        source.groupby(group_columns + [month_column], as_index=False)
        .agg(
            actual_units=(actual_column, "sum"),
            forecast_units=(forecast_column, "sum"),
            rolling_actual_units=("rolling_actual_units", "sum"),
            rolling_absolute_error_units=(
                "rolling_absolute_error_units",
                "sum",
            ),
        )
        .sort_values(group_columns + [month_column])
    )
    monthly["rolling_3m_wmape_percent"] = np.where(
        monthly["rolling_actual_units"].gt(0),
        100.0
        * monthly["rolling_absolute_error_units"]
        / monthly["rolling_actual_units"],
        np.nan,
    )
    summary = (
        monthly.groupby(group_columns, as_index=False)
        .agg(
            months=(month_column, "nunique"),
            rolling_windows_scored=("rolling_3m_wmape_percent", "count"),
            rolling_3m_wmape_percent=(
                "rolling_3m_wmape_percent",
                "mean",
            ),
        )
    )
    return monthly, summary


def individual_threshold_summary(
    sku_scores: pd.DataFrame,
    score_column: str = "rolling_3m_wmape_percent",
) -> dict[str, float | int]:
    """Summarise mutually exclusive individual-SKU accuracy thresholds."""

    valid = sku_scores.loc[sku_scores[score_column].notna()].copy()
    score = pd.to_numeric(valid[score_column], errors="coerce")
    return {
        "positive_skus": int(len(valid)),
        "under_50": int(score.lt(50).sum()),
        "under_70": int(score.lt(70).sum()),
        "under_100": int(score.lt(100).sum()),
        "median_wmape": float(score.median()) if len(valid) else np.nan,
        "mean_wmape": float(score.mean()) if len(valid) else np.nan,
    }


def rank_individual_candidates(
    summaries: pd.DataFrame,
    candidate_column: str = "candidate_id",
) -> pd.DataFrame:
    """Rank candidates by under-50, under-70, under-100, then median."""

    required = {
        candidate_column,
        "under_50",
        "under_70",
        "under_100",
        "median_wmape",
    }
    missing = required.difference(summaries.columns)
    if missing:
        raise KeyError(f"Missing ranking columns: {sorted(missing)}")
    return summaries.sort_values(
        [
            "under_50",
            "under_70",
            "under_100",
            "median_wmape",
            candidate_column,
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
