from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BLOCK_MONTHS = 3
STATIC_METADATA_COLUMNS = [
    "FAMILY_DESCRIPTION",
    "SUBFAMILY_DESCRIPTION",
    "MATERIAL_DESCRIPTION",
    "Brand",
    "Channel",
    "REGION",
]

NUMERIC_ROUTER_FEATURES = [
    "history_months",
    "history_total_demand",
    "positive_months",
    "positive_months_last12",
    "positive_blocks_last18",
    "months_since_positive",
    "positive_mean",
    "positive_median",
    "positive_cv2",
    "average_demand_interval",
    "recent_3m_total",
    "recent_6m_total",
    "recent_12m_total",
    "recent_vs_prior_6m_ratio",
    "stock_observation_share",
    "potential_stock_constraint_share",
    "trailing_12m_units",
    "trailing_12m_value",
]

CATEGORICAL_ROUTER_FEATURES = [
    "lifecycle_tier",
    "frequency_tier",
    "recency_tier",
    "size_tier",
    "abc_units_tier",
    "abc_value_tier",
    "SUBFAMILY_DESCRIPTION",
]


def _last_non_null(series: pd.Series):
    values = series.dropna()
    return values.iloc[-1] if len(values) else pd.NA


def extract_static_metadata(sales: pd.DataFrame, sku_column: str = "sku_id") -> pd.DataFrame:
    columns = [column for column in STATIC_METADATA_COLUMNS if column in sales.columns]
    if not columns:
        return pd.DataFrame({sku_column: sales[sku_column].drop_duplicates()})
    return (
        sales.sort_values([sku_column, "month"])
        .groupby(sku_column, as_index=False)[columns]
        .agg(_last_non_null)
    )


def _positive_block_count(values: np.ndarray, months: int = 18) -> int:
    trailing = np.asarray(values[-months:], dtype=float)
    if len(trailing) == 0:
        return 0
    padding = (-len(trailing)) % BLOCK_MONTHS
    if padding:
        trailing = np.pad(trailing, (padding, 0), constant_values=0.0)
    return int((trailing.reshape(-1, BLOCK_MONTHS).sum(axis=1) > 0).sum())


def _abc_tiers(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = values.sum()
    if total <= 0:
        return pd.Series("unavailable", index=values.index, dtype="object")
    ordered = values.sort_values(ascending=False)
    cumulative_before = ordered.cumsum().shift(fill_value=0.0) / total
    tier = pd.Series("C", index=ordered.index, dtype="object")
    tier.loc[cumulative_before.lt(0.80)] = "A"
    tier.loc[cumulative_before.between(0.80, 0.95, inclusive="left")] = "B"
    return tier.reindex(values.index)


def history_feature_table(
    train: pd.DataFrame,
    universe_skus: Iterable,
    metadata: pd.DataFrame,
    sku_column: str = "sku_id",
    date_column: str = "month",
    target_column: str = "demand",
) -> pd.DataFrame:
    groups = {sku: rows.sort_values(date_column) for sku, rows in train.groupby(sku_column, sort=False)}
    rows = []
    for sku in universe_skus:
        group = groups.get(sku)
        if group is None or group.empty:
            rows.append(
                {
                    sku_column: sku,
                    "history_months": 0,
                    "history_total_demand": 0.0,
                    "positive_months": 0,
                    "positive_months_last12": 0,
                    "positive_blocks_last18": 0,
                    "months_since_positive": np.nan,
                    "positive_mean": 0.0,
                    "positive_median": 0.0,
                    "positive_cv2": 0.0,
                    "average_demand_interval": np.nan,
                    "recent_3m_total": 0.0,
                    "recent_6m_total": 0.0,
                    "recent_12m_total": 0.0,
                    "recent_vs_prior_6m_ratio": 1.0,
                    "stock_observation_share": 0.0,
                    "potential_stock_constraint_share": np.nan,
                    "trailing_12m_units": 0.0,
                    "trailing_12m_value": 0.0,
                }
            )
            continue
        demand = group[target_column].astype(float).clip(lower=0).to_numpy()
        positive = demand[demand > 0]
        positions = np.flatnonzero(demand > 0)
        intervals = np.diff(positions)
        recent6 = float(demand[-6:].sum())
        prior6 = float(demand[-12:-6].sum()) if len(demand) > 6 else 0.0
        stock_share = 0.0
        constraint_share = np.nan
        if "STOCK_END_MONTH" in group.columns:
            observed_stock = group.loc[group["STOCK_END_MONTH"].notna()]
            stock_share = float(len(observed_stock) / len(group)) if len(group) else 0.0
            observed_zero = observed_stock.loc[observed_stock[target_column].eq(0)]
            if len(observed_zero):
                constraint_share = float(observed_zero["STOCK_END_MONTH"].le(0).mean())
        trailing_value = 0.0
        if "REVENUE" in group.columns:
            trailing_value = float(pd.to_numeric(group["REVENUE"].iloc[-12:], errors="coerce").fillna(0.0).sum())
        rows.append(
            {
                sku_column: sku,
                "history_months": int(group[date_column].nunique()),
                "history_total_demand": float(demand.sum()),
                "positive_months": int(len(positive)),
                "positive_months_last12": int((demand[-12:] > 0).sum()),
                "positive_blocks_last18": _positive_block_count(demand),
                "months_since_positive": int(len(demand) - 1 - positions[-1]) if len(positions) else len(demand),
                "positive_mean": float(positive.mean()) if len(positive) else 0.0,
                "positive_median": float(np.median(positive)) if len(positive) else 0.0,
                "positive_cv2": float((positive.std(ddof=0) / positive.mean()) ** 2) if len(positive) > 1 and positive.mean() > 0 else 0.0,
                "average_demand_interval": float(intervals.mean()) if len(intervals) else float(max(len(demand), 1)),
                "recent_3m_total": float(demand[-3:].sum()),
                "recent_6m_total": recent6,
                "recent_12m_total": float(demand[-12:].sum()),
                "recent_vs_prior_6m_ratio": float((recent6 + 1.0) / (prior6 + 1.0)),
                "stock_observation_share": stock_share,
                "potential_stock_constraint_share": constraint_share,
                "trailing_12m_units": float(demand[-12:].sum()),
                "trailing_12m_value": trailing_value,
            }
        )
    features = pd.DataFrame(rows).merge(metadata, on=sku_column, how="left")
    features["lifecycle_tier"] = np.select(
        [
            features.history_months.eq(0),
            features.history_months.lt(12),
            features.history_months.lt(24),
            features.months_since_positive.ge(12),
        ],
        ["cold_start", "new", "developing", "dormant"],
        default="established",
    )
    features["frequency_tier"] = pd.cut(
        features.positive_blocks_last18,
        bins=[-1, 1, 3, 6],
        labels=["rare_0_1", "occasional_2_3", "recurring_4_6"],
    ).astype("object")
    features["recency_tier"] = pd.cut(
        features.months_since_positive.fillna(999),
        bins=[-1, 2, 5, 11, np.inf],
        labels=["recent_0_2", "recent_3_5", "stale_6_11", "dormant_12_plus"],
    ).astype("object")
    features.loc[features.lifecycle_tier.eq("cold_start"), "recency_tier"] = "no_history"
    features["size_tier"] = np.select(
        [features.positive_median.le(1), features.positive_cv2.ge(1.0)],
        ["mostly_single_unit", "volatile_positive_size"],
        default="multi_unit_stable",
    )
    features.loc[features.lifecycle_tier.eq("cold_start"), "size_tier"] = "no_history"
    features["potential_stock_status"] = np.select(
        [
            features.stock_observation_share.lt(0.25),
            features.potential_stock_constraint_share.ge(0.25),
        ],
        ["stock_data_limited", "potentially_stock_constrained"],
        default="stock_generally_available",
    )
    features["abc_units_tier"] = _abc_tiers(features.trailing_12m_units)
    features["abc_value_tier"] = _abc_tiers(features.trailing_12m_value)
    for column in STATIC_METADATA_COLUMNS:
        if column in features.columns:
            features[column] = features[column].fillna("unknown").astype(str)
    return features


def score_sku_models(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = forecasts.copy()
    frame["absolute_error"] = (frame["target"] - frame["forecast"]).abs()
    scores = frame.groupby(["sku_id", "model_key", "model"], as_index=False).agg(
        actual_total=("target", "sum"),
        forecast_total=("forecast", "sum"),
        absolute_error=("absolute_error", "sum"),
    )
    scores["block_wmape_percent"] = np.where(
        scores.actual_total.gt(0), 100.0 * scores.absolute_error / scores.actual_total, np.nan
    )
    return scores


def best_model_targets(forecasts: pd.DataFrame) -> pd.DataFrame:
    scores = score_sku_models(forecasts)
    scores["selection_score"] = scores.block_wmape_percent.fillna(np.inf)
    best = (
        scores.sort_values(["sku_id", "selection_score", "absolute_error", "model_key"])
        .groupby("sku_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["best_under_50"] = best.block_wmape_percent.lt(50).astype(int)
    best["best_under_70"] = best.block_wmape_percent.lt(70).astype(int)
    return best.rename(columns={"model_key": "best_model_key", "model": "best_model"})


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _router_pipeline(random_state: int, min_samples_leaf: int = 5) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_ROUTER_FEATURES), ("categorical", categorical, CATEGORICAL_ROUTER_FEATURES)],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=12,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def fit_router(training: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    usable = training.loc[training.lifecycle_tier.ne("cold_start")].copy()
    model_router = _router_pipeline(random_state, min_samples_leaf=4)
    under50 = _router_pipeline(random_state + 1, min_samples_leaf=8)
    under70 = _router_pipeline(random_state + 2, min_samples_leaf=8)
    model_router.fit(usable, usable.best_model_key)
    under50.fit(usable, usable.best_under_50)
    under70.fit(usable, usable.best_under_70)
    return {"model_router": model_router, "under50": under50, "under70": under70}


def _positive_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    classes = model.named_steps["model"].classes_
    probability = model.predict_proba(frame)
    if 1 not in classes:
        return np.repeat(float(classes[0] == 1), len(frame))
    return probability[:, int(np.flatnonzero(classes == 1)[0])]


def predict_router(models: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    result = features[["sku_id"]].copy()
    result["router_model_key"] = models["model_router"].predict(features)
    result["probability_below_50"] = _positive_probability(models["under50"], features)
    result["probability_below_70"] = _positive_probability(models["under70"], features)
    return result


def route_forecasts(forecasts: pd.DataFrame, routing: pd.DataFrame, fallback_model_key: str) -> pd.DataFrame:
    merged = forecasts.merge(routing, on="sku_id", how="left")
    merged["router_model_key"] = merged.router_model_key.fillna(fallback_model_key)
    selected = merged.loc[merged.model_key.eq(merged.router_model_key)].copy()
    missing = set(merged.sku_id.unique()) - set(selected.sku_id.unique())
    if missing:
        fallback = merged.loc[merged.sku_id.isin(missing) & merged.model_key.eq(fallback_model_key)].copy()
        selected = pd.concat([selected, fallback], ignore_index=True)
    return selected


def _profile_table(train: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, group in train.sort_values("month").groupby("sku_id", sort=False):
        demand = group.demand.astype(float).clip(lower=0)
        row = {"sku_id": sku, "overall_monthly_rate": float(demand.mean())}
        for month_number in range(1, 13):
            values = demand.loc[pd.to_datetime(group.month).dt.month.eq(month_number)]
            row[f"month_{month_number:02d}_rate"] = float(values.mean()) if len(values) else float(demand.mean())
        rows.append(row)
    return pd.DataFrame(rows).merge(metadata, on="sku_id", how="left")


def _description_text(frame: pd.DataFrame) -> pd.Series:
    subfamily = frame.get("SUBFAMILY_DESCRIPTION", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    material = frame.get("MATERIAL_DESCRIPTION", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    return ("subfamily_" + subfamily.str.replace(" ", "_", regex=False) + " " + material).str.lower()


def cold_start_candidate_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_skus: Iterable,
    metadata: pd.DataFrame,
    test_start: pd.Timestamp,
    horizon_months: int,
) -> pd.DataFrame:
    profiles = _profile_table(train, metadata)
    targets = metadata.loc[metadata.sku_id.isin(list(target_skus))].drop_duplicates("sku_id").copy()
    if profiles.empty or targets.empty:
        return pd.DataFrame()
    combined_text = pd.concat([_description_text(profiles), _description_text(targets)], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(combined_text)
    peer_matrix = matrix[: len(profiles)]
    target_matrix = matrix[len(profiles):]
    similarities = cosine_similarity(target_matrix, peer_matrix)
    profile_index = {sku: idx for idx, sku in enumerate(profiles.sku_id)}
    profile_subfamilies = profiles["SUBFAMILY_DESCRIPTION"].fillna("unknown").astype(str).to_numpy()
    monthly_rate_matrix = profiles[[f"month_{month:02d}_rate" for month in range(1, 13)]].to_numpy(float)
    overall_rates = profiles.overall_monthly_rate.to_numpy(float)
    expected_months = pd.date_range(pd.Timestamp(test_start), periods=horizon_months, freq="MS")
    test_lookup = {
        sku: group.set_index("month").demand.astype(float).reindex(expected_months, fill_value=0.0)
        for sku, group in test.groupby("sku_id", sort=False)
    }
    methods = ("peer_knn5", "peer_knn15", "subfamily_mean", "empirical_bayes5", "empirical_bayes15")
    output = []
    for target_position, target in targets.reset_index(drop=True).iterrows():
        sku = target.sku_id
        similarity = similarities[target_position].copy()
        if sku in profile_index:
            similarity[profile_index[sku]] = -1.0
        target_subfamily = str(target.get("SUBFAMILY_DESCRIPTION", "unknown"))
        same_subfamily = profile_subfamilies == target_subfamily
        if same_subfamily.any():
            similarity = np.where(same_subfamily, similarity, -1.0)
        eligible = np.flatnonzero(similarity >= 0)
        if len(eligible) == 0:
            eligible = np.arange(len(profiles))
            similarity = np.ones(len(profiles), dtype=float)
            if sku in profile_index:
                eligible = eligible[eligible != profile_index[sku]]
        ranked = eligible[np.argsort(similarity[eligible])[::-1]]
        subfamily_indices = np.flatnonzero(same_subfamily) if same_subfamily.any() else np.arange(len(profiles))
        if sku in profile_index:
            subfamily_indices = subfamily_indices[subfamily_indices != profile_index[sku]]

        def peer_prediction(k: int, block_months: pd.DatetimeIndex) -> float:
            chosen = ranked[: min(k, len(ranked))]
            if len(chosen) == 0:
                return 0.0
            weights = np.clip(similarity[chosen], 0.0, None) + 0.05
            month_indices = np.array([month.month - 1 for month in block_months], dtype=int)
            seasonal = monthly_rate_matrix[np.ix_(chosen, month_indices)].sum(axis=1)
            overall = overall_rates[chosen] * BLOCK_MONTHS
            values = 0.5 * seasonal + 0.5 * overall
            return float(np.average(values, weights=weights))

        actual_series = test_lookup.get(sku, pd.Series(0.0, index=expected_months))
        for block_number in range(1, horizon_months // BLOCK_MONTHS + 1):
            start_position = (block_number - 1) * BLOCK_MONTHS
            block_months = expected_months[start_position:start_position + BLOCK_MONTHS]
            actual = float(actual_series.reindex(block_months, fill_value=0.0).sum())
            month_indices = np.array([month.month - 1 for month in block_months], dtype=int)
            if len(subfamily_indices):
                seasonal_values = monthly_rate_matrix[np.ix_(subfamily_indices, month_indices)].sum(axis=1)
                subfamily_values = 0.5 * seasonal_values + 0.5 * overall_rates[subfamily_indices] * BLOCK_MONTHS
                subfamily_forecast = float(subfamily_values.mean())
            else:
                subfamily_forecast = 0.0
            knn5 = peer_prediction(5, block_months)
            knn15 = peer_prediction(15, block_months)
            values = {
                "peer_knn5": knn5,
                "peer_knn15": knn15,
                "subfamily_mean": subfamily_forecast,
                "empirical_bayes5": 0.65 * knn5 + 0.35 * subfamily_forecast,
                "empirical_bayes15": 0.65 * knn15 + 0.35 * subfamily_forecast,
            }
            for method in methods:
                output.append(
                    {
                        "sku_id": sku,
                        "block_start": block_months[0],
                        "block_number": block_number,
                        "target": actual,
                        "forecast": max(0.0, float(values[method])),
                        "model_key": f"cold_{method}",
                        "model": f"Cold start {method.replace('_', ' ')}",
                        "cold_start_method": method,
                        "product_master_assumption": True,
                    }
                )
    return pd.DataFrame(output)


def score_cold_methods(forecasts: pd.DataFrame) -> pd.DataFrame:
    scores = score_sku_models(forecasts)
    rows = []
    for (model_key, model), group in scores.groupby(["model_key", "model"], sort=False):
        valid = group.loc[group.block_wmape_percent.notna()]
        rows.append(
            {
                "model_key": model_key,
                "model": model,
                "sku_count": group.sku_id.nunique(),
                "valid_positive_skus": len(valid),
                "under_50_skus": int(valid.block_wmape_percent.lt(50).sum()),
                "under_70_skus": int(valid.block_wmape_percent.lt(70).sum()),
                "under_100_skus": int(valid.block_wmape_percent.lt(100).sum()),
                "median_block_wmape": float(valid.block_wmape_percent.median()) if len(valid) else np.nan,
                "portfolio_block_wmape": float(100 * group.absolute_error.sum() / group.actual_total.sum()) if group.actual_total.sum() > 0 else np.nan,
                "actual_total": float(group.actual_total.sum()),
                "forecast_total": float(group.forecast_total.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["under_50_skus", "under_70_skus", "under_100_skus", "median_block_wmape"],
        ascending=[False, False, False, True],
    )
