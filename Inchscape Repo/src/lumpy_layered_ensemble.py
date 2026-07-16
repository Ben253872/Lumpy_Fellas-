from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import lumpy_ab_optimization as opt


DEFAULT_STRATEGIES = (
    "mean",
    "median",
    "trimmed_mean",
    "positive_linear",
    "positive_elastic",
    "random_forest",
    "hist_poisson",
    "xgb_tweedie",
    "router_hard",
    "router_soft",
    "router_linear_blend",
    "linear_residual_hgb",
)


@dataclass(frozen=True)
class StrategyResult:
    forecast: np.ndarray
    detail: str


def select_diverse_candidates(
    tuning_summary: pd.DataFrame,
    cohort: str,
    incumbents: Iterable[str] = (),
    per_architecture: int = 4,
) -> list[str]:
    """Select strong candidates while retaining model and window diversity."""
    table = tuning_summary.loc[tuning_summary.cohort.eq(cohort)].copy()
    if table.empty:
        raise ValueError(f"No tuning candidates for {cohort}")
    selected: list[str] = []
    for architecture in ("classical", "direct", "hurdle"):
        part = table.loc[table.architecture.eq(architecture)].copy()
        if part.empty:
            continue
        part = opt.rank_summary(part)
        # One recipe per structural trial prevents calibration variants of one
        # model from filling the entire expert pool.
        part = part.drop_duplicates("trial_id")
        selected.extend(part.head(per_architecture).candidate_id.astype(str))
    available = set(table.candidate_id.astype(str))
    selected.extend(str(value) for value in incumbents if str(value) in available)
    return list(dict.fromkeys(selected))


def build_wide_candidate_frame(
    candidate_ids: list[str],
    structural: pd.DataFrame,
    classical: pd.DataFrame,
    sku_ids: Iterable,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Materialise cached candidates as one leakage-safe row per SKU/block."""
    allowed = set(sku_ids)
    base: pd.DataFrame | None = None
    catalogue = []
    keys = ["sku_id", "block_start", "block_number", "target", "block_naive_scale", "origin_id"]
    merge_keys = ["sku_id", "block_start", "origin_id"]
    candidate_columns = []
    for position, candidate_id in enumerate(candidate_ids):
        forecast = opt.candidate_forecast(structural, classical, candidate_id)
        forecast = forecast.loc[forecast.sku_id.isin(allowed)].copy()
        if "origin_id" not in forecast:
            forecast["origin_id"] = 0
        name = f"candidate_{position:02d}"
        candidate_columns.append(name)
        catalogue.append({"candidate_column": name, "candidate_id": candidate_id})
        piece = forecast[keys + ["forecast"]].rename(columns={"forecast": name})
        if base is None:
            base = piece
        else:
            base = base.merge(piece[merge_keys + [name]], on=merge_keys, how="inner", validate="one_to_one")
    if base is None:
        raise ValueError("At least one candidate is required")
    values = base[candidate_columns].to_numpy(float)
    base["expert_mean"] = values.mean(axis=1)
    base["expert_median"] = np.median(values, axis=1)
    base["expert_min"] = values.min(axis=1)
    base["expert_max"] = values.max(axis=1)
    base["expert_std"] = values.std(axis=1)
    base["expert_cv"] = base.expert_std / np.maximum(base.expert_mean, 1e-6)
    months = pd.to_datetime(base.block_start).dt.month
    base["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12)
    base["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12)
    return base, candidate_columns, pd.DataFrame(catalogue)


def meta_feature_columns(candidate_columns: list[str]) -> list[str]:
    return candidate_columns + [
        "expert_mean", "expert_median", "expert_min", "expert_max",
        "expert_std", "expert_cv", "month_sin", "month_cos", "block_number",
    ]


def _xy(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]):
    x_train = train[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    x_test = test[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    medians = x_train.median(numeric_only=True).fillna(0.0)
    return x_train.fillna(medians).to_numpy(float), x_test.fillna(medians).to_numpy(float)


def _clip_forecast(raw: np.ndarray, test: pd.DataFrame, candidate_columns: list[str]) -> np.ndarray:
    candidates = test[candidate_columns].to_numpy(float)
    cap = np.maximum(1.0, 3.0 * np.nanmax(candidates, axis=1))
    return np.minimum(np.maximum(0.0, np.nan_to_num(raw, nan=0.0, posinf=0.0)), cap)


def _router_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_columns: list[str],
    feature_columns: list[str],
    soft: bool,
    random_state: int,
) -> np.ndarray:
    train_candidates = train[candidate_columns].to_numpy(float)
    labels = np.abs(train_candidates - train.target.to_numpy(float)[:, None]).argmin(axis=1)
    if np.unique(labels).size == 1:
        return test[candidate_columns[labels[0]]].to_numpy(float)
    x_train, x_test = _xy(train, test, feature_columns)
    model = RandomForestClassifier(
        n_estimators=350, max_depth=5, min_samples_leaf=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
    )
    model.fit(x_train, labels)
    candidates = test[candidate_columns].to_numpy(float)
    if not soft:
        return candidates[np.arange(len(test)), model.predict(x_test).astype(int)]
    probabilities = model.predict_proba(x_test)
    weights = np.zeros((len(test), len(candidate_columns)), dtype=float)
    weights[:, model.classes_.astype(int)] = probabilities
    # Shrink uncertain routing toward equal weighting.
    weights = 0.75 * weights + 0.25 / len(candidate_columns)
    return np.sum(weights * candidates, axis=1)


def fit_predict_strategy(
    strategy: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_columns: list[str],
    random_state: int = 42,
) -> StrategyResult:
    features = meta_feature_columns(candidate_columns)
    candidates = test[candidate_columns].to_numpy(float)
    if strategy == "mean":
        raw = candidates.mean(axis=1)
    elif strategy == "median":
        raw = np.median(candidates, axis=1)
    elif strategy == "trimmed_mean":
        ordered = np.sort(candidates, axis=1)
        raw = ordered[:, 1:-1].mean(axis=1) if ordered.shape[1] > 2 else ordered.mean(axis=1)
    else:
        x_train, x_test = _xy(train, test, features)
        y_train = train.target.astype(float).clip(lower=0).to_numpy()
        if strategy == "positive_linear":
            model = LinearRegression(positive=True)
            model.fit(x_train, y_train); raw = model.predict(x_test)
        elif strategy == "positive_elastic":
            model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.15, positive=True, max_iter=10000))
            model.fit(x_train, y_train); raw = model.predict(x_test)
        elif strategy == "random_forest":
            model = RandomForestRegressor(n_estimators=350, max_depth=7, min_samples_leaf=15, max_features=0.75, n_jobs=-1, random_state=random_state)
            model.fit(x_train, y_train); raw = model.predict(x_test)
        elif strategy == "hist_poisson":
            model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.04, max_iter=250, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=5.0, random_state=random_state)
            model.fit(x_train, y_train); raw = model.predict(x_test)
        elif strategy == "xgb_tweedie":
            model = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.35, n_estimators=350, learning_rate=0.025, max_depth=2, min_child_weight=15, subsample=0.85, colsample_bytree=0.8, reg_lambda=15.0, n_jobs=-1, random_state=random_state)
            model.fit(x_train, y_train); raw = model.predict(x_test)
        elif strategy in {"router_hard", "router_soft", "router_linear_blend"}:
            routed = _router_predictions(train, test, candidate_columns, features, strategy != "router_hard", random_state)
            if strategy == "router_linear_blend":
                linear = LinearRegression(positive=True).fit(x_train, y_train).predict(x_test)
                raw = 0.5 * routed + 0.5 * linear
            else:
                raw = routed
        elif strategy == "linear_residual_hgb":
            linear = LinearRegression(positive=True).fit(x_train, y_train)
            train_base = linear.predict(x_train); test_base = linear.predict(x_test)
            residual = y_train - train_base
            correction = HistGradientBoostingRegressor(learning_rate=0.035, max_iter=180, max_leaf_nodes=9, min_samples_leaf=25, l2_regularization=10.0, random_state=random_state)
            correction.fit(x_train, residual)
            raw = test_base + 0.5 * correction.predict(x_test)
        else:
            raise ValueError(f"Unknown layered strategy: {strategy}")
    return StrategyResult(_clip_forecast(np.asarray(raw), test, candidate_columns), strategy)


def leave_one_origin_out(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    strategy: str,
    origins: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    predictions = []
    origin_values = list(origins)
    for held_out in origin_values:
        train = frame.loc[frame.origin_id.isin([value for value in origin_values if value != held_out])]
        test = frame.loc[frame.origin_id.eq(held_out)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Insufficient rows for held-out origin {held_out}")
        test["forecast"] = fit_predict_strategy(strategy, train, test, candidate_columns, random_state).forecast
        test["strategy"] = strategy
        test["held_out_origin"] = held_out
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def score_strategies(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strategy, frame in predictions.items():
        _, summary = opt.score_forecast(frame)
        rows.append({"candidate_id": strategy, "strategy": strategy, **summary})
    return opt.rank_summary(pd.DataFrame(rows))


def retention_decision(challenger: dict, incumbent: dict) -> tuple[bool, str]:
    """Business ordering: below 70, below 50, then median WMAPE."""
    challenger_key = (challenger["under_70"], challenger["under_50"], -challenger["median_wmape"])
    incumbent_key = (incumbent["under_70"], incumbent["under_50"], -incumbent["median_wmape"])
    accepted = challenger_key > incumbent_key
    return accepted, "layered_challenger" if accepted else "retained_incumbent"


def confidence_agent(
    development_predictions: pd.DataFrame,
    final_frame: pd.DataFrame,
    candidate_columns: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate P(SKU WMAPE < 70) without filtering any SKU from evaluation."""
    dev = development_predictions.copy()
    sku_error, _ = opt.score_forecast(dev)
    labels = sku_error.set_index("sku_id").wmape.lt(70).astype(int)
    feature_names = ["expert_mean", "expert_median", "expert_std", "expert_cv"] + candidate_columns
    train_features = dev.groupby("sku_id")[feature_names].mean().join(labels.rename("success"), how="inner")
    final_features = final_frame.groupby("sku_id")[feature_names].mean()
    if train_features.success.nunique() < 2:
        probability = np.repeat(float(train_features.success.mean()), len(final_features))
    else:
        model = RandomForestClassifier(n_estimators=350, max_depth=4, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=random_state)
        model.fit(train_features[feature_names], train_features.success)
        probability = model.predict_proba(final_features[feature_names])[:, list(model.classes_).index(1)]
    result = final_features.reset_index()[["sku_id"]]
    result["probability_below_70"] = probability
    result["confidence_band"] = pd.cut(result.probability_below_70, [-np.inf, 0.35, 0.65, np.inf], labels=["low", "medium", "high"])
    return result
