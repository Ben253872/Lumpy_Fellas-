from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

import lumpy_internal_hybrid as history_tools


BLOCK_MONTHS = 3
ID_COLUMNS = {"sku_id", "block_start", "block_number", "target", "occurred", "block_naive_scale"}


@dataclass(frozen=True)
class BlockModelSpec:
    label: str
    architecture: str
    max_depth: int
    n_estimators: int
    learning_rate: float
    min_child_weight: float
    reg_lambda: float
    occurrence_weight: str = "sqrt"
    size_objective: str = "log"


MODEL_SPECS: dict[str, BlockModelSpec] = {
    "block_hurdle_d2_sqrt_log": BlockModelSpec(
        "Block hurdle XGB depth2 sqrt-weight log-size", "hurdle", 2, 350, 0.035, 8, 8.0, "sqrt", "log"
    ),
    "block_hurdle_d2_full_log": BlockModelSpec(
        "Block hurdle XGB depth2 full-weight log-size", "hurdle", 2, 350, 0.035, 8, 8.0, "full", "log"
    ),
    "block_hurdle_d3_sqrt_log": BlockModelSpec(
        "Block hurdle XGB depth3 sqrt-weight log-size", "hurdle", 3, 300, 0.030, 12, 12.0, "sqrt", "log"
    ),
    "block_direct_d2_tweedie": BlockModelSpec(
        "Block direct XGB depth2 Tweedie", "direct", 2, 400, 0.030, 8, 10.0, "none", "tweedie"
    ),
}

BASELINE_LABELS = {
    "block_sba": "Block SBA Croston",
    "block_tsb": "Block TSB",
    "block_recent6": "Block recent 6-month mean",
    "block_seasonal": "Block historical month-of-year mean",
}


def _month_distance(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def _seasonal_block(history_values: np.ndarray, history_months: pd.DatetimeIndex, target_start: pd.Timestamp) -> float:
    if len(history_values) == 0:
        return 0.0
    total = 0.0
    for offset in range(BLOCK_MONTHS):
        month = target_start + pd.DateOffset(months=offset)
        mask = history_months.month == month.month
        total += float(np.mean(history_values[mask])) if mask.any() else float(np.mean(history_values))
    return total


def _block_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    complete = len(values) // BLOCK_MONTHS
    if complete < 2:
        return np.nan
    blocks = values[-complete * BLOCK_MONTHS:].reshape(complete, BLOCK_MONTHS).sum(axis=1)
    differences = np.abs(np.diff(blocks))
    return float(differences.mean()) if len(differences) and differences.mean() > 0 else np.nan


def _feature_row(history_values: np.ndarray, history_months: pd.DatetimeIndex, target_start: pd.Timestamp, block_number: int, gap_months: int) -> dict[str, float]:
    months_ahead = gap_months + 1 + (block_number - 1) * BLOCK_MONTHS
    summary = history_tools.summarize_history(history_values, elapsed_months=months_ahead)
    angle = 2.0 * np.pi * (target_start.month - 1) / 12.0
    summary.update(
        {
            "block_number_feature": float(block_number),
            "months_ahead_start": float(months_ahead),
            "block_start_month_sin": float(np.sin(angle)),
            "block_start_month_cos": float(np.cos(angle)),
            "baseline_sba": float(summary["hybrid_sba_alpha_02"] * BLOCK_MONTHS),
            "baseline_tsb": float(summary["hybrid_tsb_rate"] * BLOCK_MONTHS),
            "baseline_recent6": float(summary["hybrid_recent_6m_mean"] * BLOCK_MONTHS),
            "baseline_seasonal": _seasonal_block(history_values, history_months, target_start),
        }
    )
    return summary


def build_training_samples(
    train: pd.DataFrame,
    sku_column: str,
    date_column: str,
    target_column: str,
    gap_months: int,
    horizon_months: int,
    min_history_months: int = 12,
    anchor_step_months: int = 3,
) -> pd.DataFrame:
    block_count = horizon_months // BLOCK_MONTHS
    samples = []
    for sku, rows in train.sort_values([sku_column, date_column]).groupby(sku_column, sort=False):
        values = rows[target_column].astype(float).clip(lower=0).to_numpy()
        months = pd.DatetimeIndex(pd.to_datetime(rows[date_column]).to_numpy())
        first_cutoff = min_history_months - 1
        for cutoff in range(first_cutoff, len(values), anchor_step_months):
            history_values = values[:cutoff + 1]
            history_months = months[:cutoff + 1]
            for block_index in range(block_count):
                start = cutoff + gap_months + 1 + block_index * BLOCK_MONTHS
                end = start + BLOCK_MONTHS
                if end > len(values):
                    break
                target_start = pd.Timestamp(months[start])
                target = float(values[start:end].sum())
                feature = _feature_row(history_values, history_months, target_start, block_index + 1, gap_months)
                feature.update(
                    {
                        "sku_id": sku,
                        "block_start": target_start,
                        "block_number": block_index + 1,
                        "target": target,
                        "occurred": int(target > 0),
                        "block_naive_scale": _block_scale(history_values),
                    }
                )
                samples.append(feature)
    return pd.DataFrame(samples)


def build_test_samples(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sku_column: str,
    date_column: str,
    target_column: str,
    gap_months: int,
    horizon_months: int,
) -> pd.DataFrame:
    block_count = horizon_months // BLOCK_MONTHS
    samples = []
    train_groups = {sku: rows.sort_values(date_column) for sku, rows in train.groupby(sku_column, sort=False)}
    for sku, test_rows in test.sort_values([sku_column, date_column]).groupby(sku_column, sort=False):
        history = train_groups.get(sku)
        if history is None or history.empty:
            continue
        history_values = history[target_column].astype(float).clip(lower=0).to_numpy()
        history_months = pd.DatetimeIndex(pd.to_datetime(history[date_column]).to_numpy())
        test_rows = test_rows.sort_values(date_column).reset_index(drop=True)
        for block_index in range(block_count):
            start = block_index * BLOCK_MONTHS
            block = test_rows.iloc[start:start + BLOCK_MONTHS]
            if len(block) < BLOCK_MONTHS:
                continue
            target_start = pd.Timestamp(block[date_column].iloc[0])
            target = float(block[target_column].astype(float).clip(lower=0).sum())
            feature = _feature_row(history_values, history_months, target_start, block_index + 1, gap_months)
            feature.update(
                {
                    "sku_id": sku,
                    "block_start": target_start,
                    "block_number": block_index + 1,
                    "target": target,
                    "occurred": int(target > 0),
                    "block_naive_scale": _block_scale(history_values),
                }
            )
            samples.append(feature)
    return pd.DataFrame(samples)


def prepare_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sku_column: str,
    date_column: str,
    target_column: str,
    gap_months: int,
    horizon_months: int,
) -> dict[str, Any]:
    train_samples = build_training_samples(
        train, sku_column, date_column, target_column, gap_months, horizon_months
    )
    test_samples = build_test_samples(
        train, test, sku_column, date_column, target_column, gap_months, horizon_months
    )
    if train_samples.empty:
        raise ValueError(
            "No block training samples can be constructed from this split. "
            "The available history is shorter than the minimum history, gap, and target block combined."
        )
    if test_samples.empty:
        raise ValueError("No complete three-month test blocks can be constructed from this split.")
    feature_columns = sorted(column for column in train_samples.columns if column not in ID_COLUMNS)
    feature_columns = [column for column in feature_columns if train_samples[column].notna().any()]
    if not feature_columns:
        raise ValueError("No usable block-model features were constructed from this split.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_samples[feature_columns])
    x_test = imputer.transform(test_samples.reindex(columns=feature_columns))
    return {
        "train_samples": train_samples,
        "test_samples": test_samples,
        "x_train": x_train,
        "x_test": x_test,
        "feature_columns": feature_columns,
    }


def _occurrence_weight(mode: str, occurred: np.ndarray) -> float:
    positives = max(1, int(occurred.sum()))
    negatives = max(1, int(len(occurred) - positives))
    ratio = negatives / positives
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return float(np.sqrt(ratio))
    if mode == "full":
        return float(ratio)
    raise ValueError(f"Unknown occurrence weight: {mode}")


def fit_components(model_key: str, prepared: dict[str, Any], random_state: int = 42) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    train_samples = prepared["train_samples"]
    target = train_samples["target"].astype(float).clip(lower=0).to_numpy()
    occurred = (target > 0).astype(int)
    positive = target > 0
    common = dict(
        n_estimators=spec.n_estimators,
        learning_rate=spec.learning_rate,
        max_depth=spec.max_depth,
        min_child_weight=spec.min_child_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=spec.reg_lambda,
        n_jobs=-1,
        random_state=random_state,
    )
    if spec.architecture == "hurdle":
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=_occurrence_weight(spec.occurrence_weight, occurred),
            **common,
        )
        classifier.fit(prepared["x_train"], occurred)
        regressor_kwargs = dict(common)
        if spec.size_objective == "tweedie":
            regressor_kwargs.update(objective="reg:tweedie", tweedie_variance_power=1.35)
            size_target = target[positive]
        else:
            regressor_kwargs.update(objective="reg:squarederror")
            size_target = np.log1p(target[positive])
        regressor = XGBRegressor(**regressor_kwargs)
        regressor.fit(prepared["x_train"][positive], size_target)
        probability = classifier.predict_proba(prepared["x_test"])[:, 1]
        size = regressor.predict(prepared["x_test"])
        if spec.size_objective == "log":
            size = np.expm1(size)
        raw = None
    else:
        regressor = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.35, **common)
        regressor.fit(prepared["x_train"], target)
        raw = regressor.predict(prepared["x_test"])
        probability = np.ones(len(raw), dtype=float)
        size = raw
    return {
        "test_samples": prepared["test_samples"].copy(),
        "probability": np.nan_to_num(probability, nan=0.0).clip(0.0, 1.0),
        "size": np.nan_to_num(size, nan=0.0, posinf=0.0).clip(0.0),
        "raw": None if raw is None else np.nan_to_num(raw, nan=0.0, posinf=0.0).clip(0.0),
        "feature_count": len(prepared["feature_columns"]),
    }


def recipe_grid(architecture: str) -> list[dict[str, Any]]:
    recipes = []
    if architecture == "direct":
        for scale in (0.75, 1.0, 1.25, 1.5):
            recipes.append({"mode": "direct", "power": 1.0, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
        return recipes
    for scale in (0.75, 1.0, 1.25, 1.5):
        recipes.append({"mode": "expected", "power": 1.0, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
    for power in (0.5, 0.75, 1.25):
        for scale in (1.0, 1.25):
            recipes.append({"mode": "soft", "power": power, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
    for threshold in (0.20, 0.35, 0.50, 0.65):
        recipes.append({"mode": "hard", "power": 1.0, "threshold": threshold, "scale": 1.0, "baseline": "none", "blend": 0.0})
    for baseline in ("sba", "tsb", "recent6", "seasonal"):
        for blend in (0.25, 0.50):
            recipes.append({"mode": "soft", "power": 0.75, "threshold": 0.0, "scale": 1.0, "baseline": baseline, "blend": blend})
    return recipes


def recipe_name(recipe: dict[str, Any]) -> str:
    return (
        f"{recipe['mode']}__p{recipe['power']:.2f}__t{recipe['threshold']:.2f}__"
        f"s{recipe['scale']:.2f}__{recipe['baseline']}__b{recipe['blend']:.2f}"
    )


def compose(model_key: str, components: dict[str, Any], recipe: dict[str, Any]) -> pd.DataFrame:
    spec = MODEL_SPECS[model_key]
    probability = components["probability"]
    size = components["size"]
    if recipe["mode"] == "direct":
        forecast = components["raw"]
    elif recipe["mode"] == "expected":
        forecast = probability * size
    elif recipe["mode"] == "soft":
        forecast = np.power(probability, recipe["power"]) * size
    elif recipe["mode"] == "hard":
        forecast = np.where(probability >= recipe["threshold"], probability * size, 0.0)
    else:
        raise ValueError(f"Unknown recipe mode: {recipe['mode']}")
    forecast = forecast * recipe["scale"]
    samples = components["test_samples"]
    if recipe["baseline"] != "none":
        baseline = samples[f"baseline_{recipe['baseline']}"] .to_numpy(float)
        forecast = (1.0 - recipe["blend"]) * forecast + recipe["blend"] * baseline
    cap = np.maximum(
        1.0,
        np.maximum(samples["hybrid_positive_max"].to_numpy(float) * 3.0, samples["baseline_recent6"].to_numpy(float) * 6.0),
    )
    frame = samples[["sku_id", "block_start", "block_number", "target", "block_naive_scale"]].copy()
    frame["forecast"] = np.minimum(np.nan_to_num(forecast, nan=0.0, posinf=0.0).clip(0.0), cap)
    frame["occurrence_probability"] = probability
    frame["positive_size_forecast"] = size
    frame["model_key"] = model_key
    frame["model"] = spec.label
    frame["recipe"] = recipe_name(recipe)
    frame["feature_count"] = components["feature_count"]
    return frame


def baseline_forecasts(prepared: dict[str, Any]) -> pd.DataFrame:
    samples = prepared["test_samples"]
    frames = []
    for key, label in BASELINE_LABELS.items():
        baseline_name = key.replace("block_", "")
        frame = samples[["sku_id", "block_start", "block_number", "target", "block_naive_scale"]].copy()
        frame["forecast"] = samples[f"baseline_{baseline_name}"].astype(float).clip(lower=0).to_numpy()
        frame["occurrence_probability"] = np.nan
        frame["positive_size_forecast"] = np.nan
        frame["model_key"] = key
        frame["model"] = label
        frame["recipe"] = "baseline"
        frame["feature_count"] = 1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_forecast(forecast: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = forecast.copy()
    frame["absolute_error"] = (frame["target"] - frame["forecast"]).abs()
    frame["bias"] = frame["forecast"] - frame["target"]
    sku = frame.groupby("sku_id", as_index=False).agg(
        block_count=("block_number", "count"),
        actual_total=("target", "sum"),
        forecast_total=("forecast", "sum"),
        absolute_error=("absolute_error", "sum"),
        bias=("bias", "sum"),
        mean_block_absolute_error=("absolute_error", "mean"),
        block_naive_scale=("block_naive_scale", "first"),
        actual_positive_blocks=("target", lambda x: int((x > 0).sum())),
        forecast_positive_blocks=("forecast", lambda x: int((x > 0).sum())),
    )
    sku["block_wmape_percent"] = np.where(
        sku.actual_total.gt(0), 100.0 * sku.absolute_error / sku.actual_total, np.nan
    )
    sku["block_mase"] = np.where(
        sku.block_naive_scale.gt(0), sku.mean_block_absolute_error / sku.block_naive_scale, np.nan
    )
    valid = sku.loc[sku.block_wmape_percent.notna()]
    actual = float(sku.actual_total.sum())
    forecast_total = float(sku.forecast_total.sum())
    summary = {
        "all_sku_count": int(len(sku)),
        "valid_positive_sku_count": int(len(valid)),
        "under_50_skus": int(valid.block_wmape_percent.lt(50).sum()),
        "under_70_skus": int(valid.block_wmape_percent.lt(70).sum()),
        "under_100_skus": int(valid.block_wmape_percent.lt(100).sum()),
        "under_50_share": float(valid.block_wmape_percent.lt(50).mean()) if len(valid) else 0.0,
        "under_70_share": float(valid.block_wmape_percent.lt(70).mean()) if len(valid) else 0.0,
        "median_sku_block_wmape": float(valid.block_wmape_percent.median()) if len(valid) else np.inf,
        "p75_sku_block_wmape": float(valid.block_wmape_percent.quantile(0.75)) if len(valid) else np.inf,
        "median_sku_block_mase": float(sku.block_mase.median()) if sku.block_mase.notna().any() else np.nan,
        "portfolio_block_wmape": float(100.0 * sku.absolute_error.sum() / actual) if actual > 0 else np.inf,
        "actual_total": actual,
        "forecast_total": forecast_total,
        "bias_pct": float(100.0 * (forecast_total - actual) / actual) if actual > 0 else np.inf,
    }
    return sku, summary


def choose_recipe(model_key: str, components: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    architecture = MODEL_SPECS[model_key].architecture
    recipes = recipe_grid(architecture)
    rows = []
    for recipe in recipes:
        _, summary = score_forecast(compose(model_key, components, recipe))
        rows.append({"model_key": model_key, "model": MODEL_SPECS[model_key].label, "recipe": recipe_name(recipe), **recipe, **summary})
    scores = pd.DataFrame(rows).sort_values(
        ["under_50_skus", "under_70_skus", "under_100_skus", "median_sku_block_wmape", "p75_sku_block_wmape", "bias_pct"],
        ascending=[False, False, False, True, True, True],
    )
    winner_name = scores.iloc[0].recipe
    winner = next(recipe for recipe in recipes if recipe_name(recipe) == winner_name)
    scores["selected_recipe"] = scores.recipe.eq(winner_name)
    return winner, scores
