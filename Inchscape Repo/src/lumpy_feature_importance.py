from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def commercial_features(train: pd.DataFrame) -> pd.DataFrame:
    aggregations = {}
    for column, operation in {"REVENUE": "sum", "COST": "sum", "TOTAL_PROFIT": "sum", "UNIT_PRICE": "median", "UNIT_COST": "median"}.items():
        if column in train.columns:
            aggregations[column] = operation
    if not aggregations:
        return train[["sku_id"]].drop_duplicates()
    result = train.groupby("sku_id", as_index=False).agg(aggregations)
    return result.rename(columns={column: f"historical_{column.lower()}" for column in aggregations})


def external_snapshot(train: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    monthly = train.groupby("month")[list(columns)].median(numeric_only=True).sort_index()
    result = {}
    for column in columns:
        values = monthly[column].dropna() if column in monthly else pd.Series(dtype=float)
        result[f"external_last__{column}"] = float(values.iloc[-1]) if len(values) else np.nan
    return result


def rolling_permutation_importance(
    examples: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    segment: str,
    ordered_folds: list[int],
    random_state: int = 42,
) -> pd.DataFrame:
    subset = examples.loc[examples.segment.eq(segment)].copy() if segment != "all_actionable" else examples.copy()
    rows = []
    for position in range(2, len(ordered_folds)):
        train_folds = ordered_folds[:position]
        validation_fold = ordered_folds[position]
        train = subset.loc[subset.fold_id.isin(train_folds)]
        validation = subset.loc[subset.fold_id.eq(validation_fold)]
        if len(train) < 30 or len(validation) < 8:
            continue
        numeric = [column for column in feature_columns if column not in categorical_columns]
        categorical = [column for column in categorical_columns if column in feature_columns]
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), categorical),
        ])
        model = Pipeline([("features", preprocessor), ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7, random_state=random_state, n_jobs=-1))])
        model.fit(train[feature_columns], np.log1p(train.target_18m))
        importance = permutation_importance(model, validation[feature_columns], np.log1p(validation.target_18m), scoring="neg_mean_absolute_error", n_repeats=5, random_state=random_state, n_jobs=-1)
        for feature, mean, std in zip(feature_columns, importance.importances_mean, importance.importances_std):
            rows.append({"segment": segment, "validation_fold": validation_fold, "train_cases": len(train), "validation_cases": len(validation), "feature": feature, "importance_mean": float(mean), "importance_std": float(std)})
    return pd.DataFrame(rows)


def summarize_importance(
    rows: pd.DataFrame,
    feature_groups: dict[str, str],
    materiality_threshold: float = 0.001,
) -> pd.DataFrame:
    summary = rows.groupby(["segment", "feature"], as_index=False).agg(mean_importance=("importance_mean", "mean"), median_importance=("importance_mean", "median"), importance_std_across_folds=("importance_mean", "std"), evaluated_folds=("validation_fold", "nunique"), positive_folds=("importance_mean", lambda values: int((values > 0).sum())))
    summary["positive_fold_share"] = summary.positive_folds / summary.evaluated_folds
    summary["feature_group"] = summary.feature.map(feature_groups).fillna("unclassified")
    summary["stable_helpful"] = (
        summary.mean_importance.gt(materiality_threshold)
        & summary.positive_fold_share.ge(0.6)
        & summary.evaluated_folds.ge(2)
    )
    summary["materiality_threshold"] = materiality_threshold
    return summary.sort_values(["segment", "stable_helpful", "mean_importance"], ascending=[True, False, False])
