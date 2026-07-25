"""End-to-end monthly forecasting workflow for the final notebook.

The workflow uses a monthly forecast origin, a two-month target lead and
complete overlapping three-month rolling WMAPE. Model and calibration choices
are made on development data before the untouched holdout is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd

import lumpy_forecasting as lf
import lumpy_monthly_evaluation as evaluation
import lumpy_sku_router as router


@dataclass(frozen=True)
class PipelineConfig:
    """Store the fixed forecast and evaluation settings."""

    forecast_horizon_months: int = 18
    forecast_lead_months: int = 2
    rolling_wmape_months: int = 3
    train_months: int = 48
    minimum_train_months: int = 18
    random_state: int = 42


MODEL_NAMES = (
    "sba_croston",
    "sba_croston_tuned",
    "seasonal_sba_croston",
    "recent_mean_6",
    "tsb",
    "boosted_sba_hybrid",
    "aggregate_allocation",
    "hurdle_random_forest",
)

CALIBRATION_SCALES = (0.50, 0.65, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.25, 1.50)


def model_config(config: PipelineConfig) -> lf.LumpyConfig:
    """Translate the notebook settings into the shared model configuration."""

    return lf.LumpyConfig(
        variant="all_sku_history",
        train_months=config.train_months,
        gap_months=config.forecast_lead_months - 1,
        test_months=config.forecast_horizon_months,
        step_months=1,
        min_train_months=config.minimum_train_months,
        max_folds=None,
        external_mode="off",
        random_state=config.random_state,
    )


def load_inputs(
    project_root: Path,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source sales and build the complete monthly model frame."""

    shared_config = model_config(config)
    sales, external = lf.load_lumpy_inputs(project_root, shared_config)
    model_data, _ = lf.build_lumpy_model_frame(sales, external, shared_config)
    model_data[lf.MONTH_COLUMN] = pd.to_datetime(model_data[lf.MONTH_COLUMN])
    return sales, external, model_data


def evaluation_calendar(
    model_data: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DataFrame]:
    """Create adjacent development and untouched 18-month holdout windows."""

    last_month = pd.Timestamp(model_data[lf.MONTH_COLUMN].max())
    holdout_months = pd.date_range(
        last_month - pd.DateOffset(months=config.forecast_horizon_months - 1),
        last_month,
        freq="MS",
    )
    development_months = pd.date_range(
        holdout_months.min()
        - pd.DateOffset(months=config.forecast_horizon_months),
        holdout_months.min() - pd.DateOffset(months=1),
        freq="MS",
    )
    evaluation_config = evaluation.MonthlyEvaluationConfig(
        forecast_lead_months=config.forecast_lead_months,
        forecast_horizon_months=config.forecast_horizon_months,
        rolling_wmape_months=config.rolling_wmape_months,
        train_months=config.train_months,
        minimum_train_months=config.minimum_train_months,
    )
    jobs = evaluation.build_monthly_origin_jobs(
        model_data,
        development_months.append(holdout_months),
        lf.MONTH_COLUMN,
        evaluation_config,
    )
    return development_months, holdout_months, jobs


def classify_population(
    sales: pd.DataFrame,
    model_data: pd.DataFrame,
    holdout_months: pd.DatetimeIndex,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Freeze lifecycle segments before holdout and flag positive holdout demand."""

    classification_cutoff = holdout_months.min() - pd.DateOffset(
        months=config.forecast_lead_months
    )
    classification_start = classification_cutoff - pd.DateOffset(
        months=config.train_months - 1
    )
    classification_train = model_data.loc[
        model_data[lf.MONTH_COLUMN].between(
            classification_start, classification_cutoff
        )
    ].copy()
    universe = sorted(model_data[lf.SKU_COLUMN].unique())
    metadata = router.extract_static_metadata(sales)
    population = router.history_feature_table(
        classification_train,
        universe,
        metadata,
    )
    population["segment"] = np.where(
        population["lifecycle_tier"].eq("established"),
        population["frequency_tier"].map(
            {
                "recurring_4_6": "recurring",
                "occasional_2_3": "occasional",
                "rare_0_1": "rare",
            }
        ),
        population["lifecycle_tier"],
    )
    holdout_actual = (
        model_data.loc[model_data[lf.MONTH_COLUMN].isin(holdout_months)]
        .groupby(lf.SKU_COLUMN)[lf.TARGET_COLUMN]
        .sum()
    )
    population["classification_cutoff"] = classification_cutoff
    population["non_dormant_at_cutoff"] = population[
        "lifecycle_tier"
    ].ne("dormant")
    population["positive_holdout_demand"] = (
        population[lf.SKU_COLUMN].map(holdout_actual).fillna(0.0).gt(0)
    )
    population["reporting_eligible"] = (
        population["non_dormant_at_cutoff"]
        & population["positive_holdout_demand"]
    )
    return population


def run_candidate_backtest(
    model_data: pd.DataFrame,
    jobs: pd.DataFrame,
    config: PipelineConfig,
    model_names: Iterable[str] = MODEL_NAMES,
    checkpoint_directory: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every candidate at every monthly origin using only known data."""

    candidates = tuple(model_names)
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    forecasts: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    total = len(jobs) * len(candidates)
    completed = 0
    started = time.perf_counter()
    for job in jobs.itertuples(index=False):
        train = model_data.loc[
            model_data[lf.MONTH_COLUMN].between(job.train_start, job.train_end)
        ].copy()
        test = model_data.loc[
            model_data[lf.MONTH_COLUMN].eq(job.target_month)
        ].copy()
        for model_name in candidates:
            completed += 1
            checkpoint = (
                checkpoint_directory
                / f"{pd.Timestamp(job.target_month):%Y-%m}__{model_name}.csv"
                if checkpoint_directory is not None
                else None
            )
            try:
                if checkpoint is not None and checkpoint.exists():
                    forecast = pd.read_csv(
                        checkpoint,
                        parse_dates=[
                            lf.MONTH_COLUMN,
                            "forecast_origin",
                            "target_month",
                            "train_start",
                            "train_end",
                        ],
                    )
                    status = "loaded"
                else:
                    forecast = lf.run_model(
                        model_name,
                        train,
                        test,
                        replace(model_config(config), test_months=1),
                    )
                    forecast["model_key"] = model_name
                    for column in jobs.columns:
                        forecast[column] = getattr(job, column)
                    if checkpoint is not None:
                        forecast.to_csv(checkpoint, index=False)
                    status = "fit"
                forecasts.append(forecast)
                print(
                    f"[{completed}/{total}] {job.target_month:%Y-%m} "
                    f"{model_name}: {status}",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "target_month": pd.Timestamp(job.target_month),
                        "model_key": model_name,
                        "error": repr(exc),
                    }
                )
    result = (
        pd.concat(forecasts, ignore_index=True, sort=False)
        if forecasts
        else pd.DataFrame()
    )
    print(
        f"Candidate run finished in {(time.perf_counter() - started) / 60:,.1f} minutes.",
        flush=True,
    )
    return result, pd.DataFrame(errors)


def complete_candidate_grid(
    forecasts: pd.DataFrame,
    model_data: pd.DataFrame,
    sku_ids: Iterable[object],
    months: Iterable[pd.Timestamp],
    model_names: Iterable[str],
) -> pd.DataFrame:
    """Fill absent pre-launch SKU months with zero actual and zero forecast."""

    sku_ids = list(sku_ids)
    months = pd.DatetimeIndex(months)
    base = pd.MultiIndex.from_product(
        [sku_ids, months], names=[lf.SKU_COLUMN, lf.MONTH_COLUMN]
    ).to_frame(index=False)
    actual = (
        model_data.loc[
            model_data[lf.SKU_COLUMN].isin(sku_ids)
            & model_data[lf.MONTH_COLUMN].isin(months),
            [lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN],
        ]
        .groupby([lf.SKU_COLUMN, lf.MONTH_COLUMN], as_index=False)[
            lf.TARGET_COLUMN
        ]
        .sum()
    )
    base = base.merge(
        actual, on=[lf.SKU_COLUMN, lf.MONTH_COLUMN], how="left"
    )
    base[lf.TARGET_COLUMN] = base[lf.TARGET_COLUMN].fillna(0.0)

    frames = []
    for model_name in model_names:
        model_forecast = forecasts.loc[
            forecasts["model_key"].eq(model_name),
            [lf.SKU_COLUMN, lf.MONTH_COLUMN, "forecast"],
        ].drop_duplicates([lf.SKU_COLUMN, lf.MONTH_COLUMN], keep="last")
        frame = base.merge(
            model_forecast,
            on=[lf.SKU_COLUMN, lf.MONTH_COLUMN],
            how="left",
        )
        frame["forecast"] = frame["forecast"].fillna(0.0).clip(lower=0.0)
        frame["model_key"] = model_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def lock_candidates_on_development(
    development_grid: pd.DataFrame,
    scales: Iterable[float] = CALIBRATION_SCALES,
) -> pd.DataFrame:
    """Choose one calibration per model using development WMAPE only."""

    rows = []
    for model_name, group in development_grid.groupby("model_key", sort=True):
        for scale in scales:
            candidate = group.copy()
            candidate["forecast"] = candidate["forecast"] * float(scale)
            candidate["portfolio"] = "Overall"
            _, score = evaluation.pooled_group_rolling_wmape(
                candidate,
                group_columns=["portfolio"],
                month_column=lf.MONTH_COLUMN,
                actual_column=lf.TARGET_COLUMN,
                rolling_months=3,
            )
            rows.append(
                {
                    "model_key": model_name,
                    "scale": float(scale),
                    "development_rolling_3m_wmape_percent": float(
                        score.iloc[0]["rolling_3m_wmape_percent"]
                    ),
                }
            )
    all_scores = pd.DataFrame(rows)
    return (
        all_scores.sort_values(
            [
                "model_key",
                "development_rolling_3m_wmape_percent",
                "scale",
            ]
        )
        .groupby("model_key", as_index=False)
        .head(1)
        .sort_values(
            [
                "development_rolling_3m_wmape_percent",
                "model_key",
                "scale",
            ]
        )
        .reset_index(drop=True)
    )


def nested_development_selection(
    development_grid: pd.DataFrame,
    development_months: Iterable[pd.Timestamp],
    scales: Iterable[float] = CALIBRATION_SCALES,
) -> pd.DataFrame:
    """Calibrate on nine months, then select the model on the next nine."""

    months = pd.DatetimeIndex(development_months).sort_values()
    if len(months) != 18:
        raise ValueError("Nested selection requires an 18-month development window.")
    calibration_months = months[:9]
    validation_months = months[9:]
    calibration_locks = lock_candidates_on_development(
        development_grid.loc[
            development_grid[lf.MONTH_COLUMN].isin(calibration_months)
        ],
        scales=scales,
    )
    rows = []
    for lock in calibration_locks.itertuples(index=False):
        candidate = development_grid.loc[
            development_grid[lf.MONTH_COLUMN].isin(validation_months)
            & development_grid["model_key"].eq(lock.model_key)
        ].copy()
        candidate["forecast"] *= float(lock.scale)
        candidate["portfolio"] = "Overall"
        _, score = evaluation.pooled_group_rolling_wmape(
            candidate,
            group_columns=["portfolio"],
            month_column=lf.MONTH_COLUMN,
            actual_column=lf.TARGET_COLUMN,
            rolling_months=3,
        )
        rows.append(
            {
                "model_key": lock.model_key,
                "locked_scale": float(lock.scale),
                "calibration_rolling_3m_wmape_percent": float(
                    lock.development_rolling_3m_wmape_percent
                ),
                "validation_rolling_3m_wmape_percent": float(
                    score.iloc[0]["rolling_3m_wmape_percent"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "validation_rolling_3m_wmape_percent",
            "model_key",
            "locked_scale",
        ]
    ).reset_index(drop=True)


def score_locked_candidates(
    holdout_grid: pd.DataFrame,
    locks: pd.DataFrame,
) -> pd.DataFrame:
    """Apply development-locked calibrations to the untouched holdout."""

    rows = []
    for lock in locks.itertuples(index=False):
        scale = float(
            lock.locked_scale if hasattr(lock, "locked_scale") else lock.scale
        )
        calibration_score = float(
            lock.calibration_rolling_3m_wmape_percent
            if hasattr(lock, "calibration_rolling_3m_wmape_percent")
            else lock.development_rolling_3m_wmape_percent
        )
        validation_score = float(
            lock.validation_rolling_3m_wmape_percent
            if hasattr(lock, "validation_rolling_3m_wmape_percent")
            else lock.development_rolling_3m_wmape_percent
        )
        candidate = holdout_grid.loc[
            holdout_grid["model_key"].eq(lock.model_key)
        ].copy()
        candidate["forecast"] *= scale
        candidate["portfolio"] = "Overall"
        _, score = evaluation.pooled_group_rolling_wmape(
            candidate,
            group_columns=["portfolio"],
            month_column=lf.MONTH_COLUMN,
            actual_column=lf.TARGET_COLUMN,
            rolling_months=3,
        )
        rows.append(
            {
                "model_key": lock.model_key,
                "locked_scale": scale,
                "calibration_rolling_3m_wmape_percent": calibration_score,
                "validation_rolling_3m_wmape_percent": validation_score,
                "holdout_rolling_3m_wmape_percent": float(
                    score.iloc[0]["rolling_3m_wmape_percent"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "validation_rolling_3m_wmape_percent",
            "model_key",
        ]
    ).reset_index(drop=True)


def _apply_route(
    candidate_grid: pd.DataFrame,
    route: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a locked model and scale to each route group."""

    frames = []
    for lock in route.itertuples(index=False):
        if lock.route_group == "all":
            mask = pd.Series(True, index=candidate_grid.index)
        elif lock.route_group == "recurring":
            mask = candidate_grid["segment"].eq("recurring")
        elif lock.route_group == "nonrecurring":
            mask = candidate_grid["segment"].ne("recurring")
        else:
            raise ValueError(f"Unknown route group: {lock.route_group}")
        frame = candidate_grid.loc[
            mask & candidate_grid["model_key"].eq(lock.model_key)
        ].copy()
        frame["forecast"] *= float(lock.locked_scale)
        frame["model_key"] = (
            f"scaled_{lock.model_key}_{float(lock.locked_scale):.2f}"
        )
        frame["route_group"] = lock.route_group
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    duplicate_keys = [lf.SKU_COLUMN, lf.MONTH_COLUMN]
    if result.duplicated(duplicate_keys).any():
        raise AssertionError("A SKU-month was assigned to more than one route.")
    return result


def select_guardrailed_sba_route(
    development_grid: pd.DataFrame,
    development_months: Iterable[pd.Timestamp],
    category_tolerance_points: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a stable SBA route using calibration and validation only.

    The best overall validation recipe defines the benchmark. Recipes within
    one WMAPE point remain eligible, and the lowest mean category WMAPE wins.
    This prevents a small headline gain from hiding severe category imbalance.
    """

    months = pd.DatetimeIndex(development_months).sort_values()
    candidate_locks = nested_development_selection(
        development_grid.drop(columns=["segment"]),
        months,
    )
    lock_lookup = candidate_locks.set_index("model_key")
    required = {"sba_croston", "sba_croston_tuned"}
    if not required.issubset(lock_lookup.index):
        raise ValueError("Both standard and tuned SBA candidates are required.")

    global_winner = candidate_locks.iloc[0]
    standard_scale = float(lock_lookup.loc["sba_croston", "locked_scale"])
    tuned_scale = float(lock_lookup.loc["sba_croston_tuned", "locked_scale"])
    recipes = {
        "global_validation_winner": pd.DataFrame(
            [
                {
                    "route_group": "all",
                    "model_key": global_winner["model_key"],
                    "locked_scale": float(global_winner["locked_scale"]),
                }
            ]
        ),
        "global_tuned_sba": pd.DataFrame(
            [
                {
                    "route_group": "all",
                    "model_key": "sba_croston_tuned",
                    "locked_scale": tuned_scale,
                }
            ]
        ),
        "frequency_guardrailed_sba": pd.DataFrame(
            [
                {
                    "route_group": "recurring",
                    "model_key": "sba_croston",
                    "locked_scale": standard_scale,
                },
                {
                    "route_group": "nonrecurring",
                    "model_key": "sba_croston_tuned",
                    "locked_scale": tuned_scale,
                },
            ]
        ),
    }

    validation_grid = development_grid.loc[
        development_grid[lf.MONTH_COLUMN].isin(months[9:])
    ].copy()
    rows = []
    route_outputs = {}
    for recipe_id, route in recipes.items():
        routed = _apply_route(validation_grid, route)
        route_outputs[recipe_id] = routed
        overall_input = routed.copy()
        overall_input["portfolio"] = "Overall"
        _, overall = evaluation.pooled_group_rolling_wmape(
            overall_input,
            group_columns=["portfolio"],
            month_column=lf.MONTH_COLUMN,
            actual_column=lf.TARGET_COLUMN,
            rolling_months=3,
        )
        _, categories = evaluation.pooled_group_rolling_wmape(
            routed,
            group_columns=["segment"],
            month_column=lf.MONTH_COLUMN,
            actual_column=lf.TARGET_COLUMN,
            rolling_months=3,
        )
        rows.append(
            {
                "recipe_id": recipe_id,
                "validation_portfolio_rolling_3m_wmape_percent": float(
                    overall.iloc[0]["rolling_3m_wmape_percent"]
                ),
                "validation_mean_category_rolling_3m_wmape_percent": float(
                    categories["rolling_3m_wmape_percent"].mean()
                ),
                "validation_worst_category_rolling_3m_wmape_percent": float(
                    categories["rolling_3m_wmape_percent"].max()
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    best_portfolio = comparison[
        "validation_portfolio_rolling_3m_wmape_percent"
    ].min()
    comparison["within_portfolio_tolerance"] = comparison[
        "validation_portfolio_rolling_3m_wmape_percent"
    ].le(best_portfolio + category_tolerance_points)
    eligible = comparison.loc[comparison["within_portfolio_tolerance"]]
    selected_id = eligible.sort_values(
        [
            "validation_mean_category_rolling_3m_wmape_percent",
            "validation_portfolio_rolling_3m_wmape_percent",
            "recipe_id",
        ]
    ).iloc[0]["recipe_id"]
    comparison["selected"] = comparison["recipe_id"].eq(selected_id)
    comparison = comparison.sort_values(
        [
            "selected",
            "validation_portfolio_rolling_3m_wmape_percent",
            "recipe_id",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return comparison, recipes[selected_id], candidate_locks


def routed_holdout(
    holdout_grid: pd.DataFrame,
    route: pd.DataFrame,
    population: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create final result tables for a development-locked SKU route."""

    segments = population.loc[
        population["reporting_eligible"], [lf.SKU_COLUMN, "segment"]
    ]
    grid = holdout_grid.drop(columns=["segment"], errors="ignore").merge(
        segments, on=lf.SKU_COLUMN, how="inner"
    )
    champion = _apply_route(grid, route)
    scored = evaluation.add_complete_rolling_wmape(
        champion,
        sku_column=lf.SKU_COLUMN,
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=config.rolling_wmape_months,
    )
    sku_scores = evaluation.score_skus(
        scored,
        sku_column=lf.SKU_COLUMN,
        actual_column=lf.TARGET_COLUMN,
    ).merge(segments, on=lf.SKU_COLUMN, how="left")
    sku_scores["complete_rolling_windows"] = (
        config.forecast_horizon_months - config.rolling_wmape_months + 1
    )
    sku_scores["forecast_lead_months"] = config.forecast_lead_months
    sku_scores["rolling_wmape_months"] = config.rolling_wmape_months
    sku_scores = sku_scores.rename(
        columns={
            "rolling_3m_wmape_percent": "sku_rolling_3m_wmape_percent",
            "model_key": "model",
        }
    )

    monthly = scored.merge(
        sku_scores[[lf.SKU_COLUMN, "sku_rolling_3m_wmape_percent"]],
        on=lf.SKU_COLUMN,
        how="left",
    )
    monthly["forecast_origin"] = monthly[lf.MONTH_COLUMN] - pd.DateOffset(
        months=config.forecast_lead_months
    )
    monthly = monthly.rename(
        columns={
            lf.TARGET_COLUMN: "actual_units",
            "forecast": "forecast_units",
            "absolute_error": "absolute_error_units",
            "rolling_3m_wmape_percent": "window_rolling_3m_wmape_percent",
        }
    )

    _, category_scores = evaluation.pooled_group_rolling_wmape(
        champion,
        group_columns=["segment"],
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=config.rolling_wmape_months,
    )
    overall_input = champion.copy()
    overall_input["category"] = "Overall"
    _, overall_score = evaluation.pooled_group_rolling_wmape(
        overall_input,
        group_columns=["category"],
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=config.rolling_wmape_months,
    )
    return monthly, sku_scores, category_scores, overall_score


def champion_holdout(
    holdout_grid: pd.DataFrame,
    champion_model: str,
    champion_scale: float,
    population: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the final monthly and per-SKU holdout result tables."""

    segments = population.loc[
        population["reporting_eligible"], [lf.SKU_COLUMN, "segment"]
    ]
    champion = holdout_grid.loc[
        holdout_grid["model_key"].eq(champion_model)
        & holdout_grid[lf.SKU_COLUMN].isin(segments[lf.SKU_COLUMN])
    ].copy()
    champion["forecast"] *= float(champion_scale)
    champion["model_key"] = f"scaled_{champion_model}"
    champion = champion.merge(segments, on=lf.SKU_COLUMN, how="inner")
    scored = evaluation.add_complete_rolling_wmape(
        champion,
        sku_column=lf.SKU_COLUMN,
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=config.rolling_wmape_months,
    )
    sku_scores = evaluation.score_skus(
        scored,
        sku_column=lf.SKU_COLUMN,
        actual_column=lf.TARGET_COLUMN,
    ).merge(segments, on=lf.SKU_COLUMN, how="left")
    sku_scores["complete_rolling_windows"] = (
        config.forecast_horizon_months - config.rolling_wmape_months + 1
    )
    sku_scores["forecast_lead_months"] = config.forecast_lead_months
    sku_scores["rolling_wmape_months"] = config.rolling_wmape_months
    sku_scores = sku_scores.rename(
        columns={
            "rolling_3m_wmape_percent": "sku_rolling_3m_wmape_percent",
            "model_key": "model",
        }
    )

    monthly = scored.merge(
        sku_scores[[lf.SKU_COLUMN, "sku_rolling_3m_wmape_percent"]],
        on=lf.SKU_COLUMN,
        how="left",
    )
    monthly["forecast_origin"] = monthly[lf.MONTH_COLUMN] - pd.DateOffset(
        months=config.forecast_lead_months
    )
    monthly = monthly.rename(
        columns={
            lf.TARGET_COLUMN: "actual_units",
            "forecast": "forecast_units",
            "absolute_error": "absolute_error_units",
            "rolling_3m_wmape_percent": "window_rolling_3m_wmape_percent",
        }
    )

    _, category_scores = evaluation.pooled_group_rolling_wmape(
        champion,
        group_columns=["segment"],
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=config.rolling_wmape_months,
    )
    overall_input = champion.copy()
    overall_input["category"] = "Overall"
    _, overall_score = evaluation.pooled_group_rolling_wmape(
        overall_input,
        group_columns=["category"],
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=config.rolling_wmape_months,
    )
    return monthly, sku_scores, category_scores, overall_score


def individual_accuracy_bands(sku_scores: pd.DataFrame) -> pd.DataFrame:
    """Build mutually exclusive individual-SKU accuracy bands by category."""

    frame = sku_scores.copy()
    frame["accuracy_band"] = pd.cut(
        frame["sku_rolling_3m_wmape_percent"],
        bins=[-np.inf, 50, 70, 100, np.inf],
        labels=["Under 50%", "50% to <70%", "70% to <100%", "100% or above"],
        right=False,
    )
    counts = (
        frame.groupby(["segment", "accuracy_band"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"segment": "category"})
    )
    overall = pd.DataFrame(
        [
            {
                "category": "Overall",
                **frame["accuracy_band"].value_counts().to_dict(),
            }
        ]
    )
    result = pd.concat([counts, overall], ignore_index=True).fillna(0)
    band_columns = ["Under 50%", "50% to <70%", "70% to <100%", "100% or above"]
    for column in band_columns:
        if column not in result:
            result[column] = 0
        result[column] = result[column].astype(int)
    result["positive_skus"] = result[band_columns].sum(axis=1)
    for column in band_columns:
        result[f"{column} pct"] = (
            100.0 * result[column] / result["positive_skus"]
        )
    return result[
        ["category", "positive_skus"]
        + [value for column in band_columns for value in (column, f"{column} pct")]
    ]


def future_champion_forecast(
    model_data: pd.DataFrame,
    champion_model: str,
    champion_scale: float,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Fit the locked model to all history and forecast the next 18 months."""

    future_months = pd.date_range(
        pd.Timestamp(model_data[lf.MONTH_COLUMN].max()) + pd.DateOffset(months=1),
        periods=config.forecast_horizon_months,
        freq="MS",
    )
    future_grid = pd.MultiIndex.from_product(
        [sorted(model_data[lf.SKU_COLUMN].unique()), future_months],
        names=[lf.SKU_COLUMN, lf.MONTH_COLUMN],
    ).to_frame(index=False)
    future_grid[lf.TARGET_COLUMN] = 0.0
    forecast = lf.run_model(
        champion_model,
        model_data,
        future_grid,
        replace(model_config(config), test_months=config.forecast_horizon_months),
    )
    forecast["forecast"] *= float(champion_scale)
    forecast["model_key"] = f"scaled_{champion_model}"
    forecast["forecast_run_month"] = pd.Timestamp(
        model_data[lf.MONTH_COLUMN].max()
    )
    return forecast


def future_routed_forecast(
    model_data: pd.DataFrame,
    route: pd.DataFrame,
    population: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Fit each locked route model to all history and forecast 18 months."""

    model_frames = []
    for model_name in route["model_key"].unique():
        model_frames.append(
            future_champion_forecast(
                model_data,
                model_name,
                1.0,
                config,
            ).assign(model_key=model_name)
        )
    candidates = pd.concat(model_frames, ignore_index=True, sort=False)
    candidates = candidates.merge(
        population[[lf.SKU_COLUMN, "segment"]],
        on=lf.SKU_COLUMN,
        how="left",
    )
    routed = _apply_route(candidates, route)
    routed["forecast_run_month"] = pd.Timestamp(
        model_data[lf.MONTH_COLUMN].max()
    )
    return routed
