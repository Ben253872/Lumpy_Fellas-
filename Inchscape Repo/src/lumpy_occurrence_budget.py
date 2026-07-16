from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: str
    model: object | None = None
    constant: float | None = None

    def predict(self, probability: np.ndarray) -> np.ndarray:
        values = np.asarray(probability, dtype=float).clip(1e-6, 1.0 - 1e-6)
        if self.method == "raw":
            return values
        if self.constant is not None:
            return np.repeat(self.constant, len(values))
        if self.method == "sigmoid":
            logit = np.log(values / (1.0 - values)).reshape(-1, 1)
            return self.model.predict_proba(logit)[:, 1].clip(0.0, 1.0)
        if self.method == "isotonic":
            return self.model.predict(values).clip(0.0, 1.0)
        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_probability_calibrator(
    probability: np.ndarray,
    occurred: np.ndarray,
    method: str,
) -> ProbabilityCalibrator:
    probability = np.asarray(probability, dtype=float).clip(1e-6, 1.0 - 1e-6)
    occurred = np.asarray(occurred, dtype=int)
    if method == "raw":
        return ProbabilityCalibrator(method="raw")
    if len(np.unique(occurred)) < 2:
        return ProbabilityCalibrator(method=method, constant=float(occurred.mean()))
    if method == "sigmoid":
        logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(logit, occurred)
        return ProbabilityCalibrator(method=method, model=model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probability, occurred)
        return ProbabilityCalibrator(method=method, model=model)
    raise ValueError(f"Unknown calibration method: {method}")


def cross_fitted_calibration(
    validation: pd.DataFrame,
    final: pd.DataFrame,
    method: str,
    n_splits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return (
            validation.probability.to_numpy(float).clip(0.0, 1.0),
            final.probability.to_numpy(float).clip(0.0, 1.0),
        )
    groups = validation.sku_id.to_numpy()
    unique_groups = pd.unique(groups)
    splits = min(n_splits, len(unique_groups))
    if splits < 2:
        fitted = fit_probability_calibrator(
            validation.probability.to_numpy(float),
            validation.target.gt(0).to_numpy(int),
            method,
        )
        return fitted.predict(validation.probability), fitted.predict(final.probability)
    oof = np.zeros(len(validation), dtype=float)
    splitter = GroupKFold(n_splits=splits)
    for train_idx, holdout_idx in splitter.split(validation, groups=groups):
        fitted = fit_probability_calibrator(
            validation.probability.iloc[train_idx].to_numpy(float),
            validation.target.iloc[train_idx].gt(0).to_numpy(int),
            method,
        )
        oof[holdout_idx] = fitted.predict(validation.probability.iloc[holdout_idx].to_numpy(float))
    fitted = fit_probability_calibrator(
        validation.probability.to_numpy(float),
        validation.target.gt(0).to_numpy(int),
        method,
    )
    return oof.clip(0.0, 1.0), fitted.predict(final.probability).clip(0.0, 1.0)


def historical_size_table(
    train: pd.DataFrame,
    sku_ids: set,
    history_months: int = 24,
) -> pd.DataFrame:
    rows = []
    for sku_id, group in train.loc[train.sku_id.isin(sku_ids)].groupby("sku_id", sort=False):
        values = group.sort_values("month").demand.astype(float).clip(lower=0).to_numpy()[-history_months:]
        positive = values[values > 0]
        rows.append(
            {
                "sku_id": sku_id,
                "historical_positive_median": float(np.median(positive)) if len(positive) else 0.0,
                "historical_positive_mean": float(positive.mean()) if len(positive) else 0.0,
                "event_budget": int(np.clip(sum(values[i : i + 3].sum() > 0 for i in range(max(0, len(values) - 18), len(values), 3)), 1, 3)),
            }
        )
    return pd.DataFrame(rows)


def positive_size(frame: pd.DataFrame, source: str) -> np.ndarray:
    ml = frame["size"].to_numpy(float).clip(0.0)
    median = frame["historical_positive_median"].to_numpy(float).clip(0.0)
    mean = frame["historical_positive_mean"].to_numpy(float).clip(0.0)
    if source == "ml":
        return ml
    if source == "median":
        return median
    if source == "mean":
        return mean
    if source == "blend50_median":
        return 0.5 * ml + 0.5 * median
    if source == "blend50_mean":
        return 0.5 * ml + 0.5 * mean
    raise ValueError(f"Unknown positive-size source: {source}")


def compose_event_forecast(
    frame: pd.DataFrame,
    probability: np.ndarray,
    size: np.ndarray,
    mode: str,
    event_count: int | str = "adaptive",
    scale: float = 1.0,
) -> np.ndarray:
    working = frame[["sku_id", "block_number"]].copy()
    working["probability"] = np.asarray(probability, dtype=float).clip(0.0, 1.0)
    working["size"] = np.asarray(size, dtype=float).clip(0.0)
    if event_count == "adaptive":
        working["budget"] = frame.event_budget.astype(int).clip(1, 3).to_numpy()
    else:
        working["budget"] = int(event_count)
    if mode == "expected":
        values = working.probability * working["size"]
    else:
        working["rank"] = working.groupby("sku_id").probability.rank(method="first", ascending=False)
        selected = working["rank"].le(working.budget)
        if mode == "top_expected":
            values = np.where(selected, working.probability * working["size"], 0.0)
        elif mode == "top_full":
            values = np.where(selected, working["size"], 0.0)
        elif mode == "normalised":
            probability_sum = working.groupby("sku_id").probability.transform("sum").replace(0.0, np.nan)
            adjusted = (working.probability * working.budget / probability_sum).fillna(0.0).clip(0.0, 1.0)
            values = adjusted * working["size"]
        else:
            raise ValueError(f"Unknown event mode: {mode}")
    cap = frame["cap"].to_numpy(float) if "cap" in frame else np.repeat(np.inf, len(frame))
    return np.minimum(np.maximum(0.0, np.asarray(values, dtype=float) * float(scale)), cap)


def event_recipe_grid() -> pd.DataFrame:
    rows = []
    for scale in (0.5, 0.75, 1.0, 1.25):
        rows.append({"mode": "expected", "event_count": "adaptive", "scale": scale})
    for mode in ("top_expected", "top_full"):
        for event_count in (2, 3, "adaptive"):
            for scale in (0.5, 0.75, 1.0):
                rows.append({"mode": mode, "event_count": event_count, "scale": scale})
    for event_count in (2, 3, "adaptive"):
        for scale in (0.5, 0.75, 1.0):
            rows.append({"mode": "normalised", "event_count": event_count, "scale": scale})
    result = pd.DataFrame(rows)
    result["recipe_id"] = result.apply(
        lambda row: f"{row['mode']}__k{row['event_count']}__s{row['scale']:.2f}", axis=1
    )
    return result
