from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def candidate_group_sets(available_groups: set[str]) -> list[tuple[str, tuple[str, ...]]]:
    optional = [group for group in ("commercial", "stock", "product") if group in available_groups]
    candidates = [("demand_only", ("demand",))]
    candidates.extend((f"demand_plus_{group}", ("demand", group)) for group in optional)
    if len(optional) > 1:
        for left_index, left in enumerate(optional):
            for right in optional[left_index + 1 :]:
                candidates.append((f"demand_plus_{left}_{right}", ("demand", left, right)))
        candidates.append(("all_eligible", tuple(["demand", *optional])))
    return candidates


def build_model(feature_columns: list[str], categorical_columns: list[str], random_state: int = 42) -> Pipeline:
    categorical = [column for column in categorical_columns if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7, random_state=random_state, n_jobs=-1)),
    ])


def allocate_total_to_blocks(total_forecast: pd.DataFrame, block_profile: pd.DataFrame) -> pd.DataFrame:
    profile = block_profile[["sku_id", "fold_id", "block_number", "target", "forecast"]].copy()
    profile["profile_sum"] = profile.groupby(["sku_id", "fold_id"]).forecast.transform("sum")
    block_count = profile.groupby(["sku_id", "fold_id"]).block_number.transform("count").clip(lower=1)
    profile["share"] = np.where(profile.profile_sum.gt(0), profile.forecast / profile.profile_sum, 1.0 / block_count)
    merged = profile.merge(total_forecast, on=["sku_id", "fold_id"], how="inner")
    merged["candidate_forecast"] = merged.total_forecast.clip(lower=0) * merged.share
    return merged


def sku_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.groupby(["segment", "candidate", "fold_id", "sku_id"], as_index=False).agg(
        actual=("target", "sum"), absolute_error=("candidate_forecast", lambda values: 0.0),
    )
    errors = predictions.assign(abs_error=(predictions.target - predictions.candidate_forecast).abs()).groupby(
        ["segment", "candidate", "fold_id", "sku_id"], as_index=False
    ).abs_error.sum()
    result = result.drop(columns="absolute_error").merge(errors, on=["segment", "candidate", "fold_id", "sku_id"])
    result["wmape"] = np.where(result.actual.gt(0), result.abs_error / result.actual, np.nan)
    return result


def summarize_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    positive = metrics.loc[metrics.actual.gt(0)].copy()
    summary = positive.groupby(["segment", "candidate"], as_index=False).agg(
        evaluated_sku_folds=("sku_id", "size"),
        historical_folds=("fold_id", "nunique"),
        median_wmape=("wmape", "median"),
        portfolio_actual=("actual", "sum"),
        portfolio_abs_error=("abs_error", "sum"),
    )
    summary["portfolio_wmape"] = summary.portfolio_abs_error / summary.portfolio_actual
    for threshold in (0.5, 0.7, 1.0):
        rates = positive.assign(hit=positive.wmape.le(threshold)).groupby(["segment", "candidate"]).hit.mean()
        summary[f"share_below_{int(threshold * 100)}"] = summary.set_index(["segment", "candidate"]).index.map(rates)
    return summary
