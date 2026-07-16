from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import lumpy_block_hybrid as bh
import lumpy_internal_hybrid as ih


CALIBRATION_SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
WINDOW_OPTIONS = (18, 24, 30, 36, 48)


@dataclass(frozen=True)
class StructuralSpec:
    key: str
    architecture: str
    max_depth: int
    n_estimators: int
    learning_rate: float
    min_child_weight: float
    reg_lambda: float
    occurrence_weight: str = "sqrt"
    size_objective: str = "log"
    tweedie_power: float = 1.35


STRUCTURAL_SPECS = (
    StructuralSpec("direct_d1_short_p120", "direct", 1, 250, 0.050, 4, 5.0, "none", "tweedie", 1.20),
    StructuralSpec("direct_d2_fast_p135", "direct", 2, 250, 0.050, 8, 8.0, "none", "tweedie", 1.35),
    StructuralSpec("direct_d2_base_p120", "direct", 2, 400, 0.030, 8, 10.0, "none", "tweedie", 1.20),
    StructuralSpec("direct_d2_base_p135", "direct", 2, 400, 0.030, 8, 10.0, "none", "tweedie", 1.35),
    StructuralSpec("direct_d2_long_p150", "direct", 2, 650, 0.020, 12, 15.0, "none", "tweedie", 1.50),
    StructuralSpec("direct_d3_regularized", "direct", 3, 350, 0.025, 16, 20.0, "none", "tweedie", 1.35),
    StructuralSpec("hurdle_d1_sqrt_log", "hurdle", 1, 450, 0.025, 8, 12.0, "sqrt", "log"),
    StructuralSpec("hurdle_d2_none_log", "hurdle", 2, 350, 0.035, 8, 8.0, "none", "log"),
    StructuralSpec("hurdle_d2_sqrt_log", "hurdle", 2, 350, 0.035, 8, 8.0, "sqrt", "log"),
    StructuralSpec("hurdle_d2_full_log", "hurdle", 2, 350, 0.035, 8, 8.0, "full", "log"),
    StructuralSpec("hurdle_d2_sqrt_tweedie", "hurdle", 2, 350, 0.035, 8, 8.0, "sqrt", "tweedie", 1.35),
    StructuralSpec("hurdle_d2_long_sqrt_log", "hurdle", 2, 600, 0.020, 12, 12.0, "sqrt", "log"),
    StructuralSpec("hurdle_d3_sqrt_log", "hurdle", 3, 300, 0.030, 12, 12.0, "sqrt", "log"),
)


def structural_trial_table() -> pd.DataFrame:
    rows = []
    for window in WINDOW_OPTIONS:
        for spec in STRUCTURAL_SPECS:
            rows.append({**asdict(spec), "history_window": window, "trial_id": f"{spec.key}__w{window}"})
    return pd.DataFrame(rows)


def recipe_grid(architecture: str) -> list[dict[str, Any]]:
    if architecture == "direct":
        return [
            {"mode": "direct", "power": 1.0, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0}
            for scale in CALIBRATION_SCALES
        ]
    recipes = []
    for scale in CALIBRATION_SCALES:
        recipes.append({"mode": "expected", "power": 1.0, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
    for power in (0.50, 0.75, 1.25):
        for scale in (0.75, 1.0, 1.25, 1.5):
            recipes.append({"mode": "soft", "power": power, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
    for threshold in (0.20, 0.35, 0.50, 0.65):
        recipes.append({"mode": "hard", "power": 1.0, "threshold": threshold, "scale": 1.0, "baseline": "none", "blend": 0.0})
    for baseline in ("sba", "tsb", "recent6"):
        for blend in (0.25, 0.50):
            recipes.append({"mode": "soft", "power": 0.75, "threshold": 0.0, "scale": 1.0, "baseline": baseline, "blend": blend})
    return recipes


def recipe_name(recipe: dict[str, Any]) -> str:
    return (
        f"{recipe['mode']}__p{recipe['power']:.2f}__t{recipe['threshold']:.2f}__"
        f"s{recipe['scale']:.2f}__{recipe['baseline']}__b{recipe['blend']:.2f}"
    )


def _occurrence_weight(mode: str, occurred: np.ndarray) -> float:
    positives = max(1, int(occurred.sum()))
    negatives = max(1, int(len(occurred) - positives))
    ratio = negatives / positives
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": float(ratio)}[mode]


def fit_structural_components(
    prepared: dict[str, Any], spec: StructuralSpec, trial_id: str, random_state: int = 42
) -> pd.DataFrame:
    samples = prepared["train_samples"]
    target = samples.target.astype(float).clip(lower=0).to_numpy()
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
    if spec.architecture == "direct":
        model = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=spec.tweedie_power, **common)
        model.fit(prepared["x_train"], target)
        raw = np.maximum(0.0, model.predict(prepared["x_test"]))
        probability = np.ones(len(raw), dtype=float)
        size = raw.copy()
    else:
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=_occurrence_weight(spec.occurrence_weight, occurred),
            **common,
        )
        classifier.fit(prepared["x_train"], occurred)
        regressor_kwargs = dict(common)
        if spec.size_objective == "tweedie":
            regressor_kwargs.update(objective="reg:tweedie", tweedie_variance_power=spec.tweedie_power)
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
        size = np.maximum(0.0, size)
        raw = np.full(len(size), np.nan)
    test = prepared["test_samples"]
    cap = np.maximum(
        1.0,
        np.maximum(test.hybrid_positive_max.to_numpy(float) * 3.0, test.baseline_recent6.to_numpy(float) * 6.0),
    )
    frame = test[["sku_id", "block_start", "block_number", "target", "block_naive_scale"]].copy()
    frame["trial_id"] = trial_id
    frame["architecture"] = spec.architecture
    frame["probability"] = np.nan_to_num(probability).clip(0.0, 1.0)
    frame["size"] = np.nan_to_num(size, posinf=0.0).clip(0.0)
    frame["raw"] = raw
    frame["sba"] = test.baseline_sba.to_numpy(float)
    frame["tsb"] = test.baseline_tsb.to_numpy(float)
    frame["recent6"] = test.baseline_recent6.to_numpy(float)
    frame["cap"] = cap
    return frame


def compose_components(components: pd.DataFrame, recipe: dict[str, Any]) -> pd.DataFrame:
    frame = components.copy()
    if recipe["mode"] == "direct":
        raw = frame.raw.to_numpy(float)
    elif recipe["mode"] == "expected":
        raw = frame.probability.to_numpy(float) * frame["size"].to_numpy(float)
    elif recipe["mode"] == "soft":
        raw = np.power(frame.probability.to_numpy(float), recipe["power"]) * frame["size"].to_numpy(float)
    elif recipe["mode"] == "hard":
        probability = frame.probability.to_numpy(float)
        raw = np.where(probability >= recipe["threshold"], probability * frame["size"].to_numpy(float), 0.0)
    else:
        raise ValueError(recipe["mode"])
    raw = raw * recipe["scale"]
    if recipe["baseline"] != "none":
        baseline = frame[recipe["baseline"]].to_numpy(float)
        raw = (1.0 - recipe["blend"]) * raw + recipe["blend"] * baseline
    frame["forecast"] = np.minimum(np.maximum(0.0, np.nan_to_num(raw, posinf=0.0)), frame.cap)
    frame["recipe"] = recipe_name(recipe)
    frame["candidate_id"] = frame.trial_id + "__" + frame.recipe
    return frame


def classical_component_rows(
    train: pd.DataFrame,
    prepared: dict[str, Any],
    history_window: int,
    date_column: str = "month",
    target_column: str = "demand",
) -> pd.DataFrame:
    groups = {
        sku: rows.sort_values(date_column)[target_column].astype(float).clip(lower=0).to_numpy()[-history_window:]
        for sku, rows in train.groupby("sku_id", sort=False)
    }
    test = prepared["test_samples"]
    configs = []
    for alpha in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        configs.append((f"sba_a{alpha:.2f}", "sba", alpha, np.nan, np.nan))
    for alpha in (0.10, 0.20, 0.40):
        for beta in (0.05, 0.10, 0.20, 0.40):
            configs.append((f"tsb_a{alpha:.2f}_b{beta:.2f}", "tsb", alpha, beta, np.nan))
    for recent in (3, 6, 9, 12, 18):
        configs.append((f"recent_{recent}", "recent", np.nan, np.nan, recent))
    output = []
    for key, method, alpha, beta, recent in configs:
        frame = test[["sku_id", "block_start", "block_number", "target", "block_naive_scale"]].copy()
        forecasts = []
        for row in frame.itertuples(index=False):
            values = groups.get(row.sku_id, np.array([], dtype=float))
            if method == "sba":
                rate = ih.croston_rate(values, alpha=float(alpha), sba=True)
            elif method == "tsb":
                rate = ih.tsb_components(values, alpha=float(alpha), beta=float(beta))[2]
            else:
                rate = float(values[-int(recent):].mean()) if len(values) else 0.0
            forecasts.append(rate * 3.0)
        frame["trial_id"] = f"{key}__w{history_window}"
        frame["architecture"] = "classical"
        frame["base_forecast"] = forecasts
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def compose_classical(components: pd.DataFrame, scale: float) -> pd.DataFrame:
    frame = components.copy()
    frame["forecast"] = np.maximum(0.0, frame.base_forecast.astype(float) * scale)
    frame["recipe"] = f"classical__s{scale:.2f}"
    frame["candidate_id"] = frame.trial_id + "__" + frame.recipe
    return frame


def score_forecast(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = frame.copy()
    scored["absolute_error"] = (scored.target - scored.forecast).abs()
    sku = scored.groupby("sku_id", as_index=False).agg(
        actual_total=("target", "sum"), forecast_total=("forecast", "sum"),
        absolute_error=("absolute_error", "sum"), mean_absolute_error=("absolute_error", "mean"),
        block_naive_scale=("block_naive_scale", "first"),
    )
    sku["wmape"] = np.where(sku.actual_total.gt(0), 100 * sku.absolute_error / sku.actual_total, np.nan)
    sku["mase"] = np.where(sku.block_naive_scale.gt(0), sku.mean_absolute_error / sku.block_naive_scale, np.nan)
    valid = sku.loc[sku.wmape.notna()]
    actual = float(sku.actual_total.sum()); forecast = float(sku.forecast_total.sum())
    summary = {
        "all_skus": int(sku.sku_id.nunique()), "positive_skus": int(len(valid)),
        "under_50": int(valid.wmape.lt(50).sum()), "under_70": int(valid.wmape.lt(70).sum()),
        "under_100": int(valid.wmape.lt(100).sum()),
        "median_wmape": float(valid.wmape.median()), "p75_wmape": float(valid.wmape.quantile(.75)),
        "median_mase": float(sku.mase.median()) if sku.mase.notna().any() else np.nan,
        "portfolio_wmape": float(100 * sku.absolute_error.sum() / actual) if actual else np.nan,
        "actual_total": actual, "forecast_total": forecast,
        "bias_pct": float(100 * (forecast - actual) / actual) if actual else np.nan,
    }
    return sku, summary


def rank_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["abs_bias"] = ranked.bias_pct.abs()
    return ranked.sort_values(
        ["under_70", "under_50", "under_100", "median_wmape", "portfolio_wmape", "abs_bias", "candidate_id"],
        ascending=[False, False, False, True, True, True, True],
    ).reset_index(drop=True)


def add_optimization_segment(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    volatility = np.where(result.positive_cv2.ge(0.75), "volatile", "stable")
    size = np.where(result.positive_median.le(1.0), "single", "multi")
    result["optimization_segment"] = result.tournament_cohort + "__" + size + "__" + volatility
    counts = result.optimization_segment.value_counts()
    small = result.optimization_segment.map(counts).lt(20)
    result.loc[small, "optimization_segment"] = result.loc[small, "tournament_cohort"] + "__other"
    return result


def candidate_forecast(
    structural: pd.DataFrame,
    classical: pd.DataFrame,
    candidate_id: str,
) -> pd.DataFrame:
    trial_id, recipe_name_value = candidate_id.split("__", 1)
    # Trial ids themselves contain '__w'; split using a known recipe marker instead.
    matches = []
    for trial in pd.concat([structural[["trial_id"]], classical[["trial_id"]]]).trial_id.unique():
        prefix = str(trial) + "__"
        if candidate_id.startswith(prefix):
            matches.append(str(trial))
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve candidate: {candidate_id}")
    trial = matches[0]
    recipe_text = candidate_id[len(trial) + 2:]
    if recipe_text.startswith("classical"):
        scale = float(recipe_text.rsplit("s", 1)[1])
        return compose_classical(classical.loc[classical.trial_id.eq(trial)], scale)
    row = structural.loc[structural.trial_id.eq(trial)]
    architecture = row.architecture.iloc[0]
    recipe = next(recipe for recipe in recipe_grid(architecture) if recipe_name(recipe) == recipe_text)
    return compose_components(row, recipe)
