"""Transparent end-to-end candidate search and monthly model selection.

This module contains the reusable model helpers used by the submission
notebook.  The notebook remains the orchestration layer: it loads the data,
shows each intermediate table, calls these helpers, evaluates the final
18-month backtest, and writes the two delivery CSVs only at the end.

Every target month is forecast from an origin two months earlier.  Model
selection uses only candidate results whose target actual would already be
known at that origin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier, XGBRegressor

import lumpy_forecasting as lf
import lumpy_monthly_evaluation as evaluation
import lumpy_monthly_pipeline as monthly
import lumpy_sku_router as sku_router


CALIBRATION_SCALES = (
    0.25,
    0.40,
    0.50,
    0.65,
    0.75,
    0.85,
    1.00,
    1.15,
    1.25,
    1.50,
    2.00,
)
TEMPORAL_WINDOWS = (2, 3, 4, 6, 9, 12)
POSITIVE_WINDOWS = (3, 6, 12)
SELECTOR_MEMORY_OPTIONS = (3, 5, 9, 12)
SELECTOR_POOL_OPTIONS = (1, 4, 16, 64)


@dataclass(frozen=True)
class CandidateRunConfig:
    """Store fixed settings for the complete candidate run."""

    random_state: int = 42
    xgb_estimators: int = 180
    checkpoint_directory: Path | None = None


def candidate_catalogue() -> pd.DataFrame:
    """Return the base forecasting methods generated at every monthly origin."""

    rows: list[dict[str, str]] = []

    def add(key: str, family: str, explanation: str) -> None:
        rows.append(
            {
                "model_key": key,
                "method_family": family,
                "explanation": explanation,
            }
        )

    for key, explanation in (
        ("zero", "Forecast zero demand."),
        ("sba_croston", "SBA-adjusted Croston intermittent-demand forecast."),
        ("sba_croston_tuned", "SBA Croston with its smoothing value selected from earlier data."),
        ("seasonal_sba_croston", "SBA Croston adjusted for recurring calendar patterns."),
        ("recent_mean_6", "Mean demand from the latest six known months."),
        ("tsb", "TSB forecast that updates demand size and occurrence probability."),
        ("boosted_sba_hybrid", "Tree-based monthly total forecast blended with SBA."),
        ("aggregate_allocation", "Forecast a group total and allocate it back to SKUs."),
        ("hurdle_random_forest", "Random-forest occurrence and demand-size forecast."),
        ("hurdle_random_forest__external", "Random-forest forecast with safe external information."),
    ):
        add(key, "Core and regression models", explanation)

    add("temporal__lag2", "Recent-history and seasonal averages", "Use demand at the forecast origin.")
    for window in TEMPORAL_WINDOWS:
        add(
            f"temporal__recent_mean_{window}",
            "Recent-history and seasonal averages",
            f"Use mean demand from the latest {window} known months.",
        )
    for window in POSITIVE_WINDOWS:
        add(
            f"temporal__positive_mean_{window}",
            "Recent-history and seasonal averages",
            f"Use the positive-demand mean from the latest {window} known months.",
        )
    add(
        "temporal__seasonal_mean_12_14",
        "Recent-history and seasonal averages",
        "Use demand from the same calendar month in earlier years.",
    )
    add(
        "temporal__seasonal_recent_blend",
        "Recent-history and seasonal averages",
        "Blend the seasonal estimate with the latest six-month mean.",
    )

    for method in (
        "peer_knn5",
        "peer_knn15",
        "subfamily_mean",
        "empirical_bayes5",
        "empirical_bayes15",
    ):
        add(
            f"analogue__cold_{method}",
            "Similar-product forecasts",
            "Use demand rates from products with similar descriptions and subfamilies.",
        )
    add(
        "pseudo_cold_validated_analogue",
        "Similar-product forecasts",
        "The similar-product method used when a SKU has no usable history.",
    )

    for key, explanation in (
        ("xgb_tweedie_depth2_reg10", "Depth-two Tweedie XGBoost using internal history."),
        ("xgb_tweedie_depth3_reg5", "Depth-three Tweedie XGBoost using internal history."),
        ("xgb_tweedie_depth2_reg10__external", "Depth-two Tweedie XGBoost with safe external information."),
        ("xgb_tweedie_depth3_reg5__external", "Depth-three Tweedie XGBoost with safe external information."),
        ("hurdle_xgb_threshold_010__external", "XGBoost occurrence and positive-demand models with external information."),
    ):
        add(key, "Regression models", explanation)

    for key, explanation in (
        ("ensemble__sba_mean", "Average the intermittent-demand forecasts."),
        ("ensemble__recent_sba", "Average recent demand and SBA forecasts."),
        ("ensemble__seasonal_sba", "Average seasonal and SBA forecasts."),
        ("ensemble__internal_xgb_mean", "Average the internal XGBoost forecasts."),
        ("ensemble__external_xgb_mean", "Average the external XGBoost forecasts."),
        ("ensemble__learned_median", "Use the median across a diverse set of forecasts."),
    ):
        add(key, "Blended forecasts", explanation)

    return pd.DataFrame(rows)


def adjusted_candidate_catalogue() -> pd.DataFrame:
    """Expand the base catalogue into every fixed calibration candidate."""

    catalogue = candidate_catalogue()
    rows = []
    for row in catalogue.itertuples(index=False):
        if row.model_key == "pseudo_cold_validated_analogue":
            rows.append(
                {
                    **row._asdict(),
                    "scale": 1.0,
                    "candidate_id": row.model_key,
                }
            )
            continue
        for scale in CALIBRATION_SCALES:
            rows.append(
                {
                    **row._asdict(),
                    "scale": float(scale),
                    "candidate_id": f"{row.model_key}__scale_{scale:.2f}",
                }
            )
    return pd.DataFrame(rows)


def _forecast_frame(
    test: pd.DataFrame,
    values: np.ndarray | pd.Series,
    model_key: str,
) -> pd.DataFrame:
    frame = test[[lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN]].copy()
    frame["forecast"] = np.maximum(
        0.0, np.nan_to_num(np.asarray(values, dtype=float), posinf=0.0)
    )
    frame["model_key"] = model_key
    return frame


def _temporal_candidates(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Create recent-history and seasonal forecasts from known demand only."""

    groups = {
        sku: rows.sort_values(lf.MONTH_COLUMN)
        for sku, rows in train.groupby(lf.SKU_COLUMN, sort=False)
    }
    output = []
    for row in test[[lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN]].itertuples(
        index=False
    ):
        history = groups.get(row.sku_id)
        values = (
            history[lf.TARGET_COLUMN].astype(float).clip(lower=0).to_numpy()
            if history is not None
            else np.array([], dtype=float)
        )
        months = (
            pd.DatetimeIndex(history[lf.MONTH_COLUMN])
            if history is not None
            else pd.DatetimeIndex([])
        )
        forecasts: dict[str, float] = {
            "temporal__lag2": float(values[-1]) if len(values) else 0.0
        }
        for window in TEMPORAL_WINDOWS:
            forecasts[f"temporal__recent_mean_{window}"] = (
                float(values[-window:].mean()) if len(values) else 0.0
            )
        for window in POSITIVE_WINDOWS:
            recent = values[-window:]
            positive = recent[recent > 0]
            forecasts[f"temporal__positive_mean_{window}"] = (
                float(positive.mean()) if len(positive) else 0.0
            )
        same_month = values[months.month == pd.Timestamp(row.month).month]
        seasonal = float(same_month[-2:].mean()) if len(same_month) else 0.0
        recent_six = float(values[-6:].mean()) if len(values) else 0.0
        forecasts["temporal__seasonal_mean_12_14"] = seasonal
        forecasts["temporal__seasonal_recent_blend"] = (
            0.5 * seasonal + 0.5 * recent_six
        )
        for model_key, forecast in forecasts.items():
            output.append(
                {
                    lf.SKU_COLUMN: row.sku_id,
                    lf.MONTH_COLUMN: pd.Timestamp(row.month),
                    lf.TARGET_COLUMN: float(row.demand),
                    "forecast": max(0.0, float(forecast)),
                    "model_key": model_key,
                }
            )
    return pd.DataFrame(output)


def _description_text(metadata: pd.DataFrame) -> pd.Series:
    subfamily = metadata.get(
        "SUBFAMILY_DESCRIPTION", pd.Series("unknown", index=metadata.index)
    ).fillna("unknown")
    material = metadata.get(
        "MATERIAL_DESCRIPTION", pd.Series("unknown", index=metadata.index)
    ).fillna("unknown")
    return (
        "subfamily_"
        + subfamily.astype(str).str.replace(" ", "_", regex=False)
        + " "
        + material.astype(str)
    ).str.lower()


def _analogue_neighbours(sales: pd.DataFrame) -> dict[object, dict[str, list[object]]]:
    """Find fixed description neighbours; demand is not used in matching."""

    metadata = sku_router.extract_static_metadata(sales).drop_duplicates(
        lf.SKU_COLUMN
    )
    metadata = metadata.sort_values(lf.SKU_COLUMN).reset_index(drop=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1
    )
    matrix = vectorizer.fit_transform(_description_text(metadata))
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, -1.0)
    subfamily = metadata.get(
        "SUBFAMILY_DESCRIPTION", pd.Series("unknown", index=metadata.index)
    ).fillna("unknown").astype(str).to_numpy()
    skus = metadata[lf.SKU_COLUMN].to_numpy()
    result: dict[object, dict[str, list[object]]] = {}
    for index, sku in enumerate(skus):
        same = np.flatnonzero(subfamily == subfamily[index])
        same = same[same != index]
        eligible = same if len(same) else np.flatnonzero(np.arange(len(skus)) != index)
        ranked = eligible[np.argsort(similarity[index, eligible])[::-1]]
        result[sku] = {
            "ranked": skus[ranked[:15]].tolist(),
            "subfamily": skus[same].tolist(),
        }
    return result


def _analogue_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    neighbours: dict[object, dict[str, list[object]]],
) -> pd.DataFrame:
    """Forecast each SKU from description-matched peer demand rates."""

    ordered = train.sort_values([lf.SKU_COLUMN, lf.MONTH_COLUMN])
    overall = ordered.groupby(lf.SKU_COLUMN)[lf.TARGET_COLUMN].mean()
    target_month = pd.Timestamp(test[lf.MONTH_COLUMN].iloc[0]).month
    seasonal = (
        ordered.loc[ordered[lf.MONTH_COLUMN].dt.month.eq(target_month)]
        .groupby(lf.SKU_COLUMN)[lf.TARGET_COLUMN]
        .mean()
    )
    peer_rate = 0.5 * seasonal.reindex(overall.index).fillna(overall) + 0.5 * overall
    output = []
    for row in test[[lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN]].itertuples(
        index=False
    ):
        mapping = neighbours.get(row.sku_id, {"ranked": [], "subfamily": []})
        ranked = mapping["ranked"]
        subfamily = mapping["subfamily"]

        def mean_for(ids: list[object]) -> float:
            values = peer_rate.reindex(ids).dropna()
            return float(values.mean()) if len(values) else float(overall.mean())

        knn5 = mean_for(ranked[:5])
        knn15 = mean_for(ranked[:15])
        subfamily_mean = mean_for(subfamily)
        forecasts = {
            "analogue__cold_peer_knn5": knn5,
            "analogue__cold_peer_knn15": knn15,
            "analogue__cold_subfamily_mean": subfamily_mean,
            "analogue__cold_empirical_bayes5": 0.65 * knn5
            + 0.35 * subfamily_mean,
            "analogue__cold_empirical_bayes15": 0.65 * knn15
            + 0.35 * subfamily_mean,
            "pseudo_cold_validated_analogue": subfamily_mean,
        }
        for model_key, forecast in forecasts.items():
            output.append(
                {
                    lf.SKU_COLUMN: row.sku_id,
                    lf.MONTH_COLUMN: pd.Timestamp(row.month),
                    lf.TARGET_COLUMN: float(row.demand),
                    "forecast": max(0.0, float(forecast)),
                    "model_key": model_key,
                }
            )
    return pd.DataFrame(output)


def _history_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_external: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build two-month-ahead XGBoost samples without using unknown demand."""

    train = train.sort_values([lf.SKU_COLUMN, lf.MONTH_COLUMN]).copy()
    group = train.groupby(lf.SKU_COLUMN, sort=False)[lf.TARGET_COLUMN]
    for lag in (2, 3, 4, 6, 12):
        train[f"history_lag_{lag}"] = group.shift(lag)
    for window in (3, 6, 12):
        train[f"history_mean_{window}"] = group.transform(
            lambda values: values.shift(2).rolling(window, min_periods=1).mean()
        )
        train[f"history_positive_rate_{window}"] = group.transform(
            lambda values: values.shift(2).gt(0).rolling(window, min_periods=1).mean()
        )
    train["history_positive_mean_12"] = group.transform(
        lambda values: values.shift(2).where(values.shift(2).gt(0)).rolling(
            12, min_periods=1
        ).mean()
    )
    train["target_month_sin"] = np.sin(
        2 * np.pi * train[lf.MONTH_COLUMN].dt.month / 12
    )
    train["target_month_cos"] = np.cos(
        2 * np.pi * train[lf.MONTH_COLUMN].dt.month / 12
    )

    feature_columns = [
        column
        for column in train.columns
        if column.startswith("history_")
        or column in {"target_month_sin", "target_month_cos"}
    ]
    if include_external:
        feature_columns += [
            column
            for column in train.columns
            if column.startswith("external_known__")
        ]

    samples = train.loc[
        train["history_lag_12"].notna(), feature_columns + [lf.TARGET_COLUMN]
    ].copy()
    x_train = samples[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = samples[lf.TARGET_COLUMN].astype(float).clip(lower=0.0)

    test_rows = []
    groups = {
        sku: rows.sort_values(lf.MONTH_COLUMN)
        for sku, rows in train.groupby(lf.SKU_COLUMN, sort=False)
    }
    for row in test.itertuples(index=False):
        history = groups.get(getattr(row, lf.SKU_COLUMN))
        values = (
            history[lf.TARGET_COLUMN].astype(float).clip(lower=0).to_numpy()
            if history is not None
            else np.array([], dtype=float)
        )
        features: dict[str, float] = {}
        for lag in (2, 3, 4, 6, 12):
            position = lag - 2
            features[f"history_lag_{lag}"] = (
                float(values[-1 - position])
                if len(values) > position
                else 0.0
            )
        for window in (3, 6, 12):
            recent = values[-window:]
            features[f"history_mean_{window}"] = (
                float(recent.mean()) if len(recent) else 0.0
            )
            features[f"history_positive_rate_{window}"] = (
                float(np.mean(recent > 0)) if len(recent) else 0.0
            )
        recent = values[-12:]
        positive = recent[recent > 0]
        features["history_positive_mean_12"] = (
            float(positive.mean()) if len(positive) else 0.0
        )
        month_number = pd.Timestamp(getattr(row, lf.MONTH_COLUMN)).month
        features["target_month_sin"] = float(
            np.sin(2 * np.pi * month_number / 12)
        )
        features["target_month_cos"] = float(
            np.cos(2 * np.pi * month_number / 12)
        )
        if include_external:
            for column in feature_columns:
                if column.startswith("external_known__"):
                    features[column] = float(
                        pd.to_numeric(getattr(row, column), errors="coerce")
                        if pd.notna(getattr(row, column))
                        else 0.0
                    )
        test_rows.append(features)
    x_test = pd.DataFrame(test_rows, columns=feature_columns).fillna(0.0)
    return x_train, y_train, x_test


def _xgb_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_external: bool,
    config: CandidateRunConfig,
) -> pd.DataFrame:
    x_train, y_train, x_test = _history_features(
        train, test, include_external=include_external
    )
    suffix = "__external" if include_external else ""
    outputs = []
    for depth, reg_lambda in ((2, 10.0), (3, 5.0)):
        model = XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.35,
            n_estimators=config.xgb_estimators,
            learning_rate=0.035,
            max_depth=depth,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=reg_lambda,
            n_jobs=-1,
            random_state=config.random_state,
        )
        model.fit(x_train, y_train)
        prediction = np.maximum(0.0, model.predict(x_test))
        outputs.append(
            _forecast_frame(
                test,
                prediction,
                f"xgb_tweedie_depth{depth}_reg{int(reg_lambda)}{suffix}",
            )
        )

    if include_external:
        occurred = y_train.gt(0).astype(int)
        if occurred.nunique() > 1:
            classifier = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=config.xgb_estimators,
                learning_rate=0.035,
                max_depth=2,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=10.0,
                n_jobs=-1,
                random_state=config.random_state,
            )
            classifier.fit(x_train, occurred)
            probability = classifier.predict_proba(x_test)[:, 1]
        else:
            probability = np.repeat(float(occurred.iloc[0]), len(x_test))
        positive = y_train.gt(0)
        if positive.any():
            amount_model = XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=1.35,
                n_estimators=config.xgb_estimators,
                learning_rate=0.035,
                max_depth=2,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=10.0,
                n_jobs=-1,
                random_state=config.random_state + 1,
            )
            amount_model.fit(x_train.loc[positive], y_train.loc[positive])
            amount = np.maximum(0.0, amount_model.predict(x_test))
        else:
            amount = np.zeros(len(x_test))
        hurdle = np.where(probability >= 0.10, probability * amount, 0.0)
        outputs.append(
            _forecast_frame(
                test,
                hurdle,
                "hurdle_xgb_threshold_010__external",
            )
        )
    return pd.concat(outputs, ignore_index=True)


def _ensemble_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build transparent averages and a median from the base forecasts."""

    key_columns = [lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN]
    wide = candidates.pivot_table(
        index=key_columns,
        columns="model_key",
        values="forecast",
        aggfunc="first",
    )
    recipes = {
        "ensemble__sba_mean": [
            "sba_croston",
            "sba_croston_tuned",
            "tsb",
        ],
        "ensemble__recent_sba": ["recent_mean_6", "sba_croston_tuned"],
        "ensemble__seasonal_sba": [
            "seasonal_sba_croston",
            "temporal__seasonal_recent_blend",
        ],
        "ensemble__internal_xgb_mean": [
            "xgb_tweedie_depth2_reg10",
            "xgb_tweedie_depth3_reg5",
        ],
        "ensemble__external_xgb_mean": [
            "xgb_tweedie_depth2_reg10__external",
            "xgb_tweedie_depth3_reg5__external",
            "hurdle_xgb_threshold_010__external",
        ],
    }
    frames = []
    for model_key, members in recipes.items():
        available = [member for member in members if member in wide.columns]
        if not available:
            continue
        values = wide[available].mean(axis=1)
        frame = values.rename("forecast").reset_index()
        frame["model_key"] = model_key
        frames.append(frame)
    diverse = [
        key
        for key in (
            "sba_croston_tuned",
            "tsb",
            "temporal__recent_mean_6",
            "temporal__positive_mean_12",
            "analogue__cold_subfamily_mean",
            "xgb_tweedie_depth2_reg10",
            "xgb_tweedie_depth2_reg10__external",
        )
        if key in wide.columns
    ]
    median = wide[diverse].median(axis=1).rename("forecast").reset_index()
    median["model_key"] = "ensemble__learned_median"
    frames.append(median)
    return pd.concat(frames, ignore_index=True)


def _complete_origin_grid(
    frame: pd.DataFrame,
    sku_ids: list[object],
    target_month: pd.Timestamp,
) -> pd.DataFrame:
    """Add zero rows for SKUs that have not entered the source history yet."""

    model_keys = candidate_catalogue()["model_key"].tolist()
    base = pd.MultiIndex.from_product(
        [sku_ids, model_keys],
        names=[lf.SKU_COLUMN, "model_key"],
    ).to_frame(index=False)
    actual = (
        frame[[lf.SKU_COLUMN, lf.TARGET_COLUMN]]
        .drop_duplicates(lf.SKU_COLUMN)
        .set_index(lf.SKU_COLUMN)[lf.TARGET_COLUMN]
    )
    forecast = frame.set_index([lf.SKU_COLUMN, "model_key"])["forecast"]
    base[lf.MONTH_COLUMN] = pd.Timestamp(target_month)
    base[lf.TARGET_COLUMN] = (
        base[lf.SKU_COLUMN].map(actual).fillna(0.0).astype(float)
    )
    keys = pd.MultiIndex.from_frame(base[[lf.SKU_COLUMN, "model_key"]])
    base["forecast"] = forecast.reindex(keys).fillna(0.0).to_numpy(float)
    for column in frame.columns:
        if column in base.columns:
            continue
        values = frame[column].dropna()
        if len(values) and values.nunique(dropna=False) == 1:
            base[column] = values.iloc[0]
    return base


def run_complete_candidate_search(
    model_data_internal: pd.DataFrame,
    model_data_external: pd.DataFrame,
    sales: pd.DataFrame,
    jobs: pd.DataFrame,
    pipeline_config: monthly.PipelineConfig,
    run_config: CandidateRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every base method at every monthly origin."""

    run_config = run_config or CandidateRunConfig(
        random_state=pipeline_config.random_state
    )
    neighbours = _analogue_neighbours(sales)
    universe_skus = sorted(sales[lf.SKU_COLUMN].unique())
    checkpoints = run_config.checkpoint_directory
    if checkpoints is not None:
        checkpoints = Path(checkpoints)
        checkpoints.mkdir(parents=True, exist_ok=True)
    all_forecasts = []
    errors = []
    started = time.perf_counter()
    for position, job in enumerate(jobs.itertuples(index=False), start=1):
        checkpoint = (
            checkpoints / f"candidates_{pd.Timestamp(job.target_month):%Y-%m}.csv"
            if checkpoints is not None
            else None
        )
        if checkpoint is not None and checkpoint.exists():
            frame = pd.read_csv(
                checkpoint,
                parse_dates=[lf.MONTH_COLUMN, "forecast_origin", "target_month"],
            )
            frame = _complete_origin_grid(
                frame, universe_skus, pd.Timestamp(job.target_month)
            )
            all_forecasts.append(frame)
            print(
                f"[{position}/{len(jobs)}] {job.target_month:%Y-%m}: loaded checkpoint",
                flush=True,
            )
            continue
        try:
            train_internal = model_data_internal.loc[
                model_data_internal[lf.MONTH_COLUMN].between(
                    job.train_start, job.train_end
                )
            ].copy()
            test_internal = model_data_internal.loc[
                model_data_internal[lf.MONTH_COLUMN].eq(job.target_month)
            ].copy()
            train_external = model_data_external.loc[
                model_data_external[lf.MONTH_COLUMN].between(
                    job.train_start, job.train_end
                )
            ].copy()
            test_external = model_data_external.loc[
                model_data_external[lf.MONTH_COLUMN].eq(job.target_month)
            ].copy()

            frames = []
            for model_key in monthly.MODEL_NAMES:
                forecast = lf.run_model(
                    model_key,
                    train_internal,
                    test_internal,
                    replace(monthly.model_config(pipeline_config), test_months=1),
                )
                forecast["model_key"] = model_key
                frames.append(
                    forecast[
                        [
                            lf.SKU_COLUMN,
                            lf.MONTH_COLUMN,
                            lf.TARGET_COLUMN,
                            "forecast",
                            "model_key",
                        ]
                    ]
                )
            zero = _forecast_frame(
                test_internal, np.zeros(len(test_internal)), "zero"
            )
            frames.append(zero)
            external_rf = lf.run_model(
                "hurdle_random_forest",
                train_external,
                test_external,
                replace(
                    monthly.model_config(pipeline_config),
                    test_months=1,
                    external_mode="selected",
                ),
            )
            external_rf["model_key"] = "hurdle_random_forest__external"
            frames.append(
                external_rf[
                    [
                        lf.SKU_COLUMN,
                        lf.MONTH_COLUMN,
                        lf.TARGET_COLUMN,
                        "forecast",
                        "model_key",
                    ]
                ]
            )
            frames.append(_temporal_candidates(train_internal, test_internal))
            frames.append(
                _analogue_candidates(train_internal, test_internal, neighbours)
            )
            frames.append(
                _xgb_candidates(
                    train_internal,
                    test_internal,
                    include_external=False,
                    config=run_config,
                )
            )
            frames.append(
                _xgb_candidates(
                    train_external,
                    test_external,
                    include_external=True,
                    config=run_config,
                )
            )
            base = pd.concat(frames, ignore_index=True, sort=False)
            base = base.drop_duplicates(
                [lf.SKU_COLUMN, lf.MONTH_COLUMN, "model_key"], keep="last"
            )
            ensemble = _ensemble_candidates(base)
            frame = pd.concat([base, ensemble], ignore_index=True, sort=False)
            frame["forecast_origin"] = pd.Timestamp(job.forecast_origin)
            frame["target_month"] = pd.Timestamp(job.target_month)
            frame["train_start"] = pd.Timestamp(job.train_start)
            frame["train_end"] = pd.Timestamp(job.train_end)
            frame = _complete_origin_grid(
                frame, universe_skus, pd.Timestamp(job.target_month)
            )
            if checkpoint is not None:
                frame.to_csv(checkpoint, index=False)
            all_forecasts.append(frame)
            print(
                f"[{position}/{len(jobs)}] {job.target_month:%Y-%m}: "
                f"{frame.model_key.nunique()} base methods",
                flush=True,
            )
        except Exception as exc:
            errors.append(
                {
                    "target_month": pd.Timestamp(job.target_month),
                    "error": repr(exc),
                }
            )
            print(
                f"[{position}/{len(jobs)}] {job.target_month:%Y-%m}: ERROR {exc!r}",
                flush=True,
            )
    result = (
        pd.concat(all_forecasts, ignore_index=True, sort=False)
        if all_forecasts
        else pd.DataFrame()
    )
    print(
        f"Complete candidate search finished in "
        f"{(time.perf_counter() - started) / 60:,.1f} minutes.",
        flush=True,
    )
    return result, pd.DataFrame(errors)


def _candidate_id(model_key: str, scale: float) -> str:
    if model_key == "pseudo_cold_validated_analogue":
        return model_key
    return f"{model_key}__scale_{scale:.2f}"


def _candidate_score_matrix(
    history: pd.DataFrame,
    sku_ids: list[object],
    memory_months: int,
    rolling_months: int = 3,
) -> pd.DataFrame:
    """Return one chronology-safe historical score per SKU/candidate."""

    months = sorted(pd.to_datetime(history[lf.MONTH_COLUMN].unique()))
    months = months[-memory_months:]
    source = history.loc[
        history[lf.SKU_COLUMN].isin(sku_ids)
        & history[lf.MONTH_COLUMN].isin(months)
    ].copy()
    actual = (
        source[[lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN]]
        .drop_duplicates([lf.SKU_COLUMN, lf.MONTH_COLUMN])
        .pivot(index=lf.SKU_COLUMN, columns=lf.MONTH_COLUMN, values=lf.TARGET_COLUMN)
        .reindex(index=sku_ids, columns=months)
        .fillna(0.0)
        .to_numpy(float)
    )
    model_keys = sorted(source.model_key.unique())
    values = {}
    for model_key in model_keys:
        forecast = (
            source.loc[source.model_key.eq(model_key)]
            .pivot(index=lf.SKU_COLUMN, columns=lf.MONTH_COLUMN, values="forecast")
            .reindex(index=sku_ids, columns=months)
            .fillna(0.0)
            .to_numpy(float)
        )
        for scale in CALIBRATION_SCALES:
            if model_key == "pseudo_cold_validated_analogue" and scale != 1.0:
                continue
            adjusted = forecast * scale
            if len(months) >= rolling_months:
                actual_roll = np.lib.stride_tricks.sliding_window_view(
                    actual, rolling_months, axis=1
                ).sum(axis=2)
                forecast_roll = np.lib.stride_tricks.sliding_window_view(
                    adjusted, rolling_months, axis=1
                ).sum(axis=2)
                valid = actual_roll > 0
                ratios = np.divide(
                    np.abs(actual_roll - forecast_roll),
                    actual_roll,
                    out=np.full_like(actual_roll, np.nan, dtype=float),
                    where=valid,
                )
                valid_windows = np.sum(~np.isnan(ratios), axis=1)
                score = np.divide(
                    np.nansum(ratios, axis=1),
                    valid_windows,
                    out=np.full(len(sku_ids), np.nan),
                    where=valid_windows > 0,
                ) * 100.0
            else:
                denominator = np.abs(actual).sum(axis=1)
                score = np.divide(
                    np.abs(actual - adjusted).sum(axis=1),
                    denominator,
                    out=np.full(len(sku_ids), np.nan),
                    where=denominator > 0,
                ) * 100.0
            values[_candidate_id(model_key, scale)] = score
    return pd.DataFrame(values, index=pd.Index(sku_ids, name=lf.SKU_COLUMN))


def _rank_candidates(score_matrix: pd.DataFrame) -> list[str]:
    rows = []
    for candidate_id in score_matrix.columns:
        values = score_matrix[candidate_id].dropna()
        rows.append(
            {
                "candidate_id": candidate_id,
                "under_50": int(values.lt(50).sum()),
                "under_70": int(values.lt(70).sum()),
                "under_100": int(values.lt(100).sum()),
                "median_wmape": float(values.median())
                if len(values)
                else np.inf,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        [
            "under_50",
            "under_70",
            "under_100",
            "median_wmape",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True],
    )
    return ranked.candidate_id.tolist()


def _parse_candidate(candidate_id: str) -> tuple[str, float]:
    marker = "__scale_"
    if marker not in candidate_id:
        return candidate_id, 1.0
    model_key, scale = candidate_id.rsplit(marker, 1)
    return model_key, float(scale)


def _select_target(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    target_month: pd.Timestamp,
    sku_ids: list[object],
    memory_months: int,
    candidate_pool: int,
) -> pd.DataFrame:
    scores = _candidate_score_matrix(
        history, sku_ids=sku_ids, memory_months=memory_months
    )
    ranked = _rank_candidates(scores)
    return _materialise_target_selection(
        candidates=candidates,
        target_month=target_month,
        sku_ids=sku_ids,
        scores=scores,
        ranked_candidates=ranked,
        memory_months=memory_months,
        candidate_pool=candidate_pool,
    )


def _materialise_target_selection(
    candidates: pd.DataFrame,
    target_month: pd.Timestamp,
    sku_ids: list[object],
    scores: pd.DataFrame,
    ranked_candidates: list[str],
    memory_months: int,
    candidate_pool: int,
) -> pd.DataFrame:
    """Choose and materialise forecasts from an already-scored candidate set."""

    ranked = ranked_candidates
    pool = ranked[: min(candidate_pool, len(ranked))]
    if not pool:
        raise ValueError("No candidates were available to the monthly selector.")
    target = candidates.loc[
        candidates[lf.MONTH_COLUMN].eq(pd.Timestamp(target_month))
        & candidates[lf.SKU_COLUMN].isin(sku_ids)
    ]
    lookup = target.set_index([lf.SKU_COLUMN, "model_key"])["forecast"]
    rows = []
    for sku in sku_ids:
        own = scores.loc[sku, pool].dropna().sort_values()
        selected = own.index[0] if len(own) else pool[0]
        model_key, scale = _parse_candidate(selected)
        key = (sku, model_key)
        if key not in lookup.index:
            selected = pool[0]
            model_key, scale = _parse_candidate(selected)
            key = (sku, model_key)
        actual = target.loc[
            target[lf.SKU_COLUMN].eq(sku), lf.TARGET_COLUMN
        ].iloc[0]
        rows.append(
            {
                lf.SKU_COLUMN: sku,
                lf.MONTH_COLUMN: pd.Timestamp(target_month),
                lf.TARGET_COLUMN: float(actual),
                "forecast": max(0.0, float(lookup.loc[key]) * scale),
                "selected_model": selected,
                "model_key": "online_expert_router",
                "memory_months": int(memory_months),
                "candidate_pool": int(candidate_pool),
            }
        )
    return pd.DataFrame(rows)


def _score_selected(frame: pd.DataFrame) -> dict[str, float | int]:
    scored = evaluation.add_complete_rolling_wmape(
        frame,
        sku_column=lf.SKU_COLUMN,
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=3,
    )
    sku = evaluation.score_skus(
        scored, sku_column=lf.SKU_COLUMN, actual_column=lf.TARGET_COLUMN
    )
    return evaluation.individual_threshold_summary(sku)


def tune_selector_by_category(
    candidates: pd.DataFrame,
    population: pd.DataFrame,
    development_months: Iterable[pd.Timestamp],
    forecast_lead_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose memory and pool size using only the development window."""

    months = pd.DatetimeIndex(development_months).sort_values()
    validation_months = months[len(months) // 2 :]
    rows = []
    selected_frames = []
    for category, category_population in population.loc[
        population.reporting_eligible
    ].groupby("segment", sort=True):
        sku_ids = category_population[lf.SKU_COLUMN].tolist()
        for memory in SELECTOR_MEMORY_OPTIONS:
            cache: dict[pd.Timestamp, tuple[pd.DataFrame, list[str]]] = {}
            for target_month in validation_months:
                known_through = pd.Timestamp(target_month) - pd.DateOffset(
                    months=forecast_lead_months
                )
                history = candidates.loc[
                    candidates[lf.MONTH_COLUMN].le(known_through)
                ]
                scores = _candidate_score_matrix(
                    history, sku_ids=sku_ids, memory_months=memory
                )
                cache[pd.Timestamp(target_month)] = (
                    scores,
                    _rank_candidates(scores),
                )
            for pool_size in SELECTOR_POOL_OPTIONS:
                output = []
                for target_month in validation_months:
                    scores, ranked = cache[pd.Timestamp(target_month)]
                    output.append(
                        _materialise_target_selection(
                            candidates=candidates,
                            target_month=pd.Timestamp(target_month),
                            sku_ids=sku_ids,
                            scores=scores,
                            ranked_candidates=ranked,
                            memory_months=memory,
                            candidate_pool=pool_size,
                        )
                    )
                selected = pd.concat(output, ignore_index=True)
                summary = _score_selected(selected)
                setting_id = f"memory_{memory}__pool_{pool_size}"
                rows.append(
                    {
                        "segment": category,
                        "setting_id": setting_id,
                        "memory_months": memory,
                        "candidate_pool": pool_size,
                        **summary,
                    }
                )
                selected["setting_id"] = setting_id
                selected["segment"] = category
                selected_frames.append(selected)
    tuning = pd.DataFrame(rows).sort_values(
        [
            "segment",
            "under_50",
            "under_70",
            "under_100",
            "median_wmape",
            "setting_id",
        ],
        ascending=[True, False, False, False, True, True],
    )
    locks = tuning.groupby("segment", as_index=False).head(1).reset_index(drop=True)
    return locks, pd.concat(selected_frames, ignore_index=True)


def apply_monthly_selector(
    candidates: pd.DataFrame,
    population: pd.DataFrame,
    selector_locks: pd.DataFrame,
    holdout_months: Iterable[pd.Timestamp],
    forecast_lead_months: int = 2,
) -> pd.DataFrame:
    """Apply development-locked settings to each holdout forecast origin."""

    output = []
    locks = selector_locks.set_index("segment")
    reporting = population.loc[population.reporting_eligible]
    for target_month in pd.DatetimeIndex(holdout_months).sort_values():
        known_through = pd.Timestamp(target_month) - pd.DateOffset(
            months=forecast_lead_months
        )
        history = candidates.loc[candidates[lf.MONTH_COLUMN].le(known_through)]
        for category, category_population in reporting.groupby(
            "segment", sort=True
        ):
            lock = locks.loc[category]
            output.append(
                _select_target(
                    candidates,
                    history,
                    pd.Timestamp(target_month),
                    category_population[lf.SKU_COLUMN].tolist(),
                    int(lock.memory_months),
                    int(lock.candidate_pool),
                ).assign(segment=category)
            )
    return pd.concat(output, ignore_index=True).sort_values(
        [lf.SKU_COLUMN, lf.MONTH_COLUMN]
    )


def build_delivery_tables(
    selected_holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the final per-SKU and monthly result tables."""

    selected = selected_holdout.copy()
    selected["model_key"] = "online_expert_router"
    scored = evaluation.add_complete_rolling_wmape(
        selected,
        sku_column=lf.SKU_COLUMN,
        month_column=lf.MONTH_COLUMN,
        actual_column=lf.TARGET_COLUMN,
        rolling_months=3,
    )
    sku_scores = evaluation.score_skus(
        scored,
        sku_column=lf.SKU_COLUMN,
        actual_column=lf.TARGET_COLUMN,
    )
    dominant = (
        selected.groupby([lf.SKU_COLUMN, "selected_model"])
        .size()
        .rename("months_selected")
        .reset_index()
        .sort_values(
            [lf.SKU_COLUMN, "months_selected", "selected_model"],
            ascending=[True, False, True],
        )
        .groupby(lf.SKU_COLUMN, as_index=False)
        .head(1)
        .rename(columns={"selected_model": "dominant_expert"})
    )
    sku_results = (
        sku_scores.rename(
            columns={
                "rolling_3m_wmape_percent": "sku_rolling_3m_wmape_percent",
                "model_key": "model",
            }
        )
        .merge(
            selected[[lf.SKU_COLUMN, "segment"]].drop_duplicates(),
            on=lf.SKU_COLUMN,
            how="left",
        )
        .merge(dominant, on=lf.SKU_COLUMN, how="left")
    )
    sku_results = sku_results[
        [
            lf.SKU_COLUMN,
            "segment",
            "sku_rolling_3m_wmape_percent",
            "model",
            "dominant_expert",
            "months_selected",
        ]
    ].sort_values(["segment", "sku_rolling_3m_wmape_percent", lf.SKU_COLUMN])

    monthly_results = selected.merge(
        sku_results[[lf.SKU_COLUMN, "sku_rolling_3m_wmape_percent"]],
        on=lf.SKU_COLUMN,
        how="left",
    ).rename(
        columns={
            lf.TARGET_COLUMN: "actual_units",
            "forecast": "forecast_units",
        }
    )
    monthly_results = monthly_results[
        [
            lf.SKU_COLUMN,
            lf.MONTH_COLUMN,
            "segment",
            "actual_units",
            "forecast_units",
            "selected_model",
            "sku_rolling_3m_wmape_percent",
        ]
    ].sort_values([lf.SKU_COLUMN, lf.MONTH_COLUMN])
    return sku_results.reset_index(drop=True), monthly_results.reset_index(
        drop=True
    )
