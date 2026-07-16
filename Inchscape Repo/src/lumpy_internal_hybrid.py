from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as exc:  # pragma: no cover - notebook reports the environment error
    raise ImportError("Notebook 10 requires xgboost. Install it and restart the kernel.") from exc


@dataclass(frozen=True)
class HybridSpec:
    label: str
    max_depth: int
    n_estimators: int
    learning_rate: float
    min_child_weight: float
    reg_lambda: float
    occurrence_weight: str
    size_objective: str


MODEL_SPECS: dict[str, HybridSpec] = {
    "xgb_d1_sqrt_log": HybridSpec(
        "XGB depth1 sqrt-weight log-size", 1, 450, 0.025, 8, 12.0, "sqrt", "log"
    ),
    "xgb_d2_none_log": HybridSpec(
        "XGB depth2 unweighted log-size", 2, 350, 0.035, 8, 8.0, "none", "log"
    ),
    "xgb_d2_sqrt_log": HybridSpec(
        "XGB depth2 sqrt-weight log-size", 2, 350, 0.035, 8, 8.0, "sqrt", "log"
    ),
    "xgb_d2_full_log": HybridSpec(
        "XGB depth2 full-weight log-size", 2, 350, 0.035, 8, 8.0, "full", "log"
    ),
    "xgb_d2_sqrt_tweedie": HybridSpec(
        "XGB depth2 sqrt-weight Tweedie-size", 2, 350, 0.035, 8, 8.0, "sqrt", "tweedie"
    ),
    "xgb_d2_long_sqrt_log": HybridSpec(
        "XGB depth2 long sqrt-weight log-size", 2, 600, 0.020, 12, 12.0, "sqrt", "log"
    ),
    "xgb_d3_sqrt_log": HybridSpec(
        "XGB depth3 sqrt-weight log-size", 3, 300, 0.030, 12, 12.0, "sqrt", "log"
    ),
}


def croston_rate(values: np.ndarray, alpha: float = 0.2, sba: bool = False) -> float:
    demand = np.clip(np.asarray(values, dtype=float), 0.0, None)
    positions = np.flatnonzero(demand > 0)
    if len(positions) == 0:
        return 0.0
    if len(positions) == 1:
        rate = float(demand[positions[0]] / max(len(demand), 1))
        return rate * (1.0 - alpha / 2.0) if sba else rate
    size = float(demand[positions[0]])
    interval = float(np.diff(positions).mean())
    previous = int(positions[0])
    for position in positions[1:]:
        size = alpha * float(demand[position]) + (1.0 - alpha) * size
        interval = alpha * max(1.0, float(position - previous)) + (1.0 - alpha) * interval
        previous = int(position)
    rate = size / max(interval, 1e-6)
    return float(max(0.0, rate * (1.0 - alpha / 2.0) if sba else rate))


def tsb_components(values: np.ndarray, alpha: float = 0.2, beta: float = 0.2) -> tuple[float, float, float]:
    demand = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if len(demand) == 0:
        return 0.0, 0.0, 0.0
    positive = demand[demand > 0]
    probability = float((demand > 0).mean())
    size = float(positive[0]) if len(positive) else 0.0
    for value in demand:
        occurred = float(value > 0)
        probability = beta * occurred + (1.0 - beta) * probability
        if occurred:
            size = alpha * float(value) + (1.0 - alpha) * size
    return float(probability), float(size), float(probability * size)


def summarize_history(values: np.ndarray, elapsed_months: int = 0) -> dict[str, float]:
    demand = np.clip(np.asarray(values, dtype=float), 0.0, None)
    positive = demand[demand > 0]
    positions = np.flatnonzero(demand > 0)
    intervals = np.diff(positions) if len(positions) else np.array([], dtype=float)
    if len(positions):
        months_since_positive = len(demand) - 1 - positions[-1] + elapsed_months
    else:
        months_since_positive = len(demand) + elapsed_months
    mean_interval = float(intervals.mean()) if len(intervals) else float(max(len(demand), 1))
    std_positive = float(positive.std(ddof=0)) if len(positive) else 0.0
    positive_mean = float(positive.mean()) if len(positive) else 0.0
    cv2 = (std_positive / positive_mean) ** 2 if positive_mean > 0 else 0.0
    tsb_probability, tsb_size, tsb_rate = tsb_components(demand)
    return {
        "hybrid_history_months": float(len(demand)),
        "hybrid_positive_months": float(len(positive)),
        "hybrid_positive_rate": float((demand > 0).mean()) if len(demand) else 0.0,
        "hybrid_zero_share": float((demand == 0).mean()) if len(demand) else 1.0,
        "hybrid_months_since_positive": float(months_since_positive),
        "hybrid_mean_interval": mean_interval,
        "hybrid_max_interval": float(intervals.max()) if len(intervals) else float(max(len(demand), 1)),
        "hybrid_interval_cv": float(intervals.std(ddof=0) / mean_interval) if len(intervals) and mean_interval > 0 else 0.0,
        "hybrid_positive_mean": positive_mean,
        "hybrid_positive_median": float(np.median(positive)) if len(positive) else 0.0,
        "hybrid_positive_std": std_positive,
        "hybrid_positive_cv2": float(cv2),
        "hybrid_positive_max": float(positive.max()) if len(positive) else 0.0,
        "hybrid_last_demand": float(demand[-1]) if len(demand) else 0.0,
        "hybrid_last_positive_demand": float(positive[-1]) if len(positive) else 0.0,
        "hybrid_sba_alpha_01": croston_rate(demand, 0.1, True),
        "hybrid_sba_alpha_02": croston_rate(demand, 0.2, True),
        "hybrid_sba_alpha_05": croston_rate(demand, 0.5, True),
        "hybrid_tsb_probability": tsb_probability,
        "hybrid_tsb_positive_size": tsb_size,
        "hybrid_tsb_rate": tsb_rate,
        "hybrid_recent_3m_mean": float(demand[-3:].mean()) if len(demand) else 0.0,
        "hybrid_recent_6m_mean": float(demand[-6:].mean()) if len(demand) else 0.0,
        "hybrid_recent_12m_mean": float(demand[-12:].mean()) if len(demand) else 0.0,
    }


def _train_projection_features(train: pd.DataFrame, config: Any, sku_column: str, date_column: str, target_column: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, sku_train in train.sort_values([sku_column, date_column]).groupby(sku_column, sort=False):
        values = sku_train[target_column].astype(float).to_numpy()
        rows = []
        for position, row_index in enumerate(sku_train.index):
            known_end = position - config.gap_months + 1
            rows.append((row_index, summarize_history(values[:max(known_end, 0)])))
        if rows:
            indexes, summaries = zip(*rows)
            frames.append(pd.DataFrame(list(summaries), index=indexes))
    return pd.concat(frames).sort_index() if frames else pd.DataFrame(index=train.index)


def _test_projection_features(train: pd.DataFrame, test: pd.DataFrame, sku_column: str, date_column: str, target_column: str) -> pd.DataFrame:
    history = {
        sku: rows.sort_values(date_column)[target_column].astype(float).to_numpy()
        for sku, rows in train.groupby(sku_column, sort=False)
    }
    train_ends = train.groupby(sku_column)[date_column].max()
    fallback = train.sort_values(date_column)[target_column].astype(float).to_numpy()
    fallback_end = pd.to_datetime(train[date_column]).max()
    rows = []
    for row in test[[sku_column, date_column]].itertuples(index=True):
        sku = getattr(row, sku_column)
        month = getattr(row, date_column)
        values = history.get(sku, fallback)
        train_end = train_ends.get(sku, fallback_end)
        elapsed = max(0, (month.year - train_end.year) * 12 + month.month - train_end.month)
        rows.append((row.Index, summarize_history(values, elapsed)))
    if not rows:
        return pd.DataFrame(index=test.index)
    indexes, summaries = zip(*rows)
    return pd.DataFrame(list(summaries), index=indexes).sort_index()


def build_design_matrix(lf: Any, train: pd.DataFrame, test: pd.DataFrame, config: Any) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    train_x, test_x, feature_columns = lf._rf_design_matrix(train, test, config)
    train_hybrid = _train_projection_features(train, config, lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN)
    test_hybrid = _test_projection_features(train, test, lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN)
    train_x = pd.concat([train_x.reset_index(drop=True), train_hybrid.reset_index(drop=True)], axis=1)
    test_x = pd.concat([test_x.reset_index(drop=True), test_hybrid.reset_index(drop=True)], axis=1)
    feature_columns = feature_columns + list(train_hybrid.columns)
    external_columns = [column for column in feature_columns if column.startswith("external_")]
    if external_columns:
        raise AssertionError(f"Notebook 10 must be internal-only; found {len(external_columns)} external features")
    usable = [column for column in train_x.columns if train_x[column].notna().any()]
    train_x = train_x[usable]
    test_x = test_x.reindex(columns=usable)
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(train_x), imputer.transform(test_x), usable, test_hybrid.reset_index(drop=True)


def _scale_pos_weight(mode: str, occurred: np.ndarray) -> float:
    positives = max(1, int(occurred.sum()))
    negatives = max(1, int(len(occurred) - positives))
    ratio = negatives / positives
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return float(np.sqrt(ratio))
    if mode == "full":
        return float(ratio)
    raise ValueError(f"Unknown occurrence weighting: {mode}")


def _caps(train: pd.DataFrame, sku_column: str, target_column: str) -> tuple[pd.Series, float]:
    global_positive = train.loc[train[target_column].gt(0), target_column].astype(float)
    global_cap = max(1.0, float(global_positive.quantile(0.99) * 2.0) if len(global_positive) else 1.0)
    values = {}
    for sku, rows in train.groupby(sku_column, sort=False):
        positive = rows.loc[rows[target_column].gt(0), target_column].astype(float)
        if positive.empty:
            values[sku] = 1.0
        else:
            local = max(1.0, float(positive.max() * 1.5), float(positive.quantile(0.95) * 2.0), float(positive.mean() * 4.0))
            values[sku] = min(local, max(global_cap, float(positive.max() * 3.0)))
    return pd.Series(values, dtype=float), global_cap


def fit_components(lf: Any, model_key: str, train: pd.DataFrame, test: pd.DataFrame, config: Any, random_state: int = 42) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    x_train, x_test, feature_columns, test_history = build_design_matrix(lf, train, test, config)
    y = train[lf.TARGET_COLUMN].astype(float).clip(lower=0).to_numpy()
    occurred = (y > 0).astype(int)
    positive = y > 0
    if occurred.min() == occurred.max() or positive.sum() < 2:
        probability = np.repeat(float(occurred.mean()), len(test))
        amount = np.repeat(float(y[positive].mean()) if positive.any() else 0.0, len(test))
        status = "fallback_history_average"
    else:
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=spec.n_estimators,
            learning_rate=spec.learning_rate,
            max_depth=spec.max_depth,
            min_child_weight=spec.min_child_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=spec.reg_lambda,
            scale_pos_weight=_scale_pos_weight(spec.occurrence_weight, occurred),
            n_jobs=-1,
            random_state=random_state,
        )
        regressor_kwargs = {
            "objective": "reg:tweedie" if spec.size_objective == "tweedie" else "reg:squarederror",
            "n_estimators": spec.n_estimators,
            "learning_rate": spec.learning_rate,
            "max_depth": spec.max_depth,
            "min_child_weight": spec.min_child_weight,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": spec.reg_lambda,
            "n_jobs": -1,
            "random_state": random_state,
        }
        if spec.size_objective == "tweedie":
            regressor_kwargs["tweedie_variance_power"] = 1.35
        regressor = XGBRegressor(
            **regressor_kwargs,
        )
        classifier.fit(x_train, occurred)
        size_target = y[positive] if spec.size_objective == "tweedie" else np.log1p(y[positive])
        regressor.fit(x_train[positive], size_target)
        probability = classifier.predict_proba(x_test)[:, 1]
        size_prediction = regressor.predict(x_test)
        amount = size_prediction if spec.size_objective == "tweedie" else np.expm1(size_prediction)
        status = "fit_xgboost"
    caps, global_cap = _caps(train, lf.SKU_COLUMN, lf.TARGET_COLUMN)
    return {
        "test": test.copy(),
        "probability": np.nan_to_num(probability, nan=0.0).clip(0.0, 1.0),
        "positive_amount": np.nan_to_num(amount, nan=0.0, posinf=0.0).clip(0.0),
        "sba": test_history["hybrid_sba_alpha_02"].to_numpy(),
        "tsb": test_history["hybrid_tsb_rate"].to_numpy(),
        "recent6": test_history["hybrid_recent_6m_mean"].to_numpy(),
        "caps": caps,
        "global_cap": global_cap,
        "feature_count": len(feature_columns),
        "status": status,
    }


def recipe_grid() -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    for scale in (0.75, 1.0, 1.25, 1.5):
        recipes.append({"mode": "expected", "power": 1.0, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
    for power in (0.5, 0.75, 1.25):
        for scale in (1.0, 1.25):
            recipes.append({"mode": "soft", "power": power, "threshold": 0.0, "scale": scale, "baseline": "none", "blend": 0.0})
    for threshold in (0.30, 0.50, 0.70, 0.85):
        for scale in (1.0, 1.25, 1.5):
            recipes.append({"mode": "hard", "power": 1.0, "threshold": threshold, "scale": scale, "baseline": "none", "blend": 0.0})
    for baseline in ("sba", "tsb", "recent6"):
        for blend in (0.25, 0.50):
            recipes.append({"mode": "soft", "power": 0.75, "threshold": 0.0, "scale": 1.0, "baseline": baseline, "blend": blend})
    return recipes


def recipe_name(recipe: dict[str, Any]) -> str:
    return (
        f"{recipe['mode']}__p{recipe['power']:.2f}__t{recipe['threshold']:.2f}__"
        f"s{recipe['scale']:.2f}__{recipe['baseline']}__b{recipe['blend']:.2f}"
    )


def compose(lf: Any, model_key: str, components: dict[str, Any], recipe: dict[str, Any]) -> pd.DataFrame:
    probability = components["probability"]
    amount = components["positive_amount"]
    if recipe["mode"] == "expected":
        raw = probability * amount
    elif recipe["mode"] == "soft":
        raw = np.power(probability, recipe["power"]) * amount
    elif recipe["mode"] == "hard":
        raw = np.where(probability >= recipe["threshold"], probability * amount, 0.0)
    else:
        raise ValueError(f"Unknown recipe mode: {recipe['mode']}")
    raw = raw * recipe["scale"]
    if recipe["baseline"] != "none":
        baseline = components[recipe["baseline"]]
        raw = (1.0 - recipe["blend"]) * raw + recipe["blend"] * baseline
    test = components["test"]
    cap_values = test[lf.SKU_COLUMN].map(components["caps"]).fillna(components["global_cap"]).to_numpy()
    forecast = np.minimum(np.nan_to_num(raw, nan=0.0, posinf=0.0).clip(0.0), cap_values)
    frame = test[[lf.SKU_COLUMN, lf.MONTH_COLUMN, lf.TARGET_COLUMN]].copy()
    frame["forecast"] = forecast
    frame["occurrence_probability"] = probability
    frame["positive_amount_forecast"] = amount
    frame["model_key"] = model_key
    frame["model"] = MODEL_SPECS[model_key].label
    frame["recipe"] = recipe_name(recipe)
    frame["recipe_mode"] = recipe["mode"]
    frame["probability_power"] = recipe["power"]
    frame["probability_threshold"] = recipe["threshold"]
    frame["calibration_scale"] = recipe["scale"]
    frame["baseline"] = recipe["baseline"]
    frame["baseline_blend"] = recipe["blend"]
    frame["feature_count"] = components["feature_count"]
    frame["fit_status"] = components["status"]
    return frame


def score_forecast(lf: Any, forecast: pd.DataFrame, fold_column: str | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = lf.add_error_columns(forecast)
    groups = [lf.SKU_COLUMN, "model_key"] + ([fold_column] if fold_column else [])
    scored = scored.sort_values(groups + [lf.MONTH_COLUMN]).reset_index(drop=True)
    scored["rolling_3m_abs_error"] = scored.groupby(groups)["absolute_error"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    scored["rolling_3m_actual"] = scored.groupby(groups)[lf.TARGET_COLUMN].transform(lambda x: x.rolling(3, min_periods=1).sum())
    scored["rolling_3m_wmape_percent"] = np.where(
        scored["rolling_3m_actual"].gt(0),
        100.0 * scored["rolling_3m_abs_error"] / scored["rolling_3m_actual"],
        np.nan,
    )
    sku = scored.groupby(lf.SKU_COLUMN, as_index=False).agg(
        actual_total=(lf.TARGET_COLUMN, "sum"),
        forecast_total=("forecast", "sum"),
        absolute_error=("absolute_error", "sum"),
        bias=("bias", "sum"),
        mean_rolling_3m_wmape_percent=("rolling_3m_wmape_percent", "mean"),
        actual_positive_months=(lf.TARGET_COLUMN, lambda x: int((x > 0).sum())),
        forecast_positive_months=("forecast", lambda x: int((x > 0).sum())),
    )
    valid = sku.loc[sku.actual_total.gt(0) & sku.mean_rolling_3m_wmape_percent.notna()]
    actual = float(sku.actual_total.sum())
    forecast_total = float(sku.forecast_total.sum())
    absolute_error = float(sku.absolute_error.sum())
    bias_pct = 100.0 * (forecast_total - actual) / actual if actual > 0 else np.inf
    median = float(valid.mean_rolling_3m_wmape_percent.median()) if len(valid) else np.inf
    p75 = float(valid.mean_rolling_3m_wmape_percent.quantile(0.75)) if len(valid) else np.inf
    portfolio = 100.0 * absolute_error / actual if actual > 0 else np.inf
    bias_penalty = abs(bias_pct) * (1.5 if bias_pct < 0 else 1.0)
    objective = 0.30 * median + 0.15 * p75 + 0.40 * portfolio + 0.15 * bias_penalty
    summary = {
        "valid_sku_count": int(len(valid)),
        "median_sku_rolling_wmape": median,
        "p75_sku_rolling_wmape": p75,
        "under_70_skus": int(valid.mean_rolling_3m_wmape_percent.lt(70).sum()),
        "under_100_skus": int(valid.mean_rolling_3m_wmape_percent.lt(100).sum()),
        "portfolio_horizon_wmape": portfolio,
        "actual_total": actual,
        "forecast_total": forecast_total,
        "bias_pct": bias_pct,
        "selection_objective": objective,
    }
    return sku, summary


def make_inner_split(train: pd.DataFrame, config: Any, date_column: str, validation_months: int = 6):
    months = pd.Series(pd.to_datetime(train[date_column].dropna().unique())).sort_values().tolist()
    if len(months) < config.gap_months + validation_months + 12:
        return None
    validation_end = months[-1]
    validation_start = validation_end - pd.DateOffset(months=validation_months - 1)
    inner_train_end = validation_start - pd.DateOffset(months=config.gap_months + 1)
    inner_train = train.loc[train[date_column].le(inner_train_end)].copy()
    validation = train.loc[train[date_column].between(validation_start, validation_end)].copy()
    if inner_train[date_column].nunique() < 12 or validation.empty:
        return None
    return inner_train, validation, validation_start, validation_end, inner_train_end


def tune_recipe(lf: Any, model_key: str, train: pd.DataFrame, config: Any, validation_months: int = 6) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inner = make_inner_split(train, config, lf.MONTH_COLUMN, validation_months)
    default = {"mode": "soft", "power": 0.75, "threshold": 0.0, "scale": 1.0, "baseline": "sba", "blend": 0.25}
    if inner is None:
        return default, []
    inner_train, validation, validation_start, validation_end, inner_train_end = inner
    components = fit_components(lf, model_key, inner_train, validation, config, config.random_state)
    rows = []
    recipes = recipe_grid()
    for recipe in recipes:
        forecast = compose(lf, model_key, components, recipe)
        _, summary = score_forecast(lf, forecast)
        rows.append({
            "model_key": model_key,
            "model": MODEL_SPECS[model_key].label,
            "recipe": recipe_name(recipe),
            **recipe,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "inner_train_end": inner_train_end,
            **summary,
        })
    scores = pd.DataFrame(rows).sort_values(
        ["selection_objective", "portfolio_horizon_wmape", "median_sku_rolling_wmape", "recipe"]
    )
    winner_name = scores.iloc[0].recipe
    winner = next(recipe for recipe in recipes if recipe_name(recipe) == winner_name)
    return winner, rows
