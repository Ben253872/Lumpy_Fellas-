from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


_RECIPE = re.compile(
    r"^cal_(?P<calibration>[^_]+)__size_(?P<size_source>.+?)__"
    r"(?P<mode>expected|top_expected|top_full|normalised)__"
    r"k(?P<event_count>adaptive|\d+)__s(?P<scale>\d+\.\d+)$"
)


def parse_candidate_id(candidate_id: str, trial_ids: list[str]) -> dict[str, object]:
    matches = [trial_id for trial_id in trial_ids if candidate_id.startswith(trial_id + "__")]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve candidate trial: {candidate_id}")
    trial_id = matches[0]
    match = _RECIPE.match(candidate_id[len(trial_id) + 2 :])
    if match is None:
        raise ValueError(f"Could not parse candidate recipe: {candidate_id}")
    values = match.groupdict()
    event_count = values["event_count"]
    return {
        "candidate_id": candidate_id,
        "trial_id": trial_id,
        "calibration": values["calibration"],
        "size_source": values["size_source"],
        "mode": values["mode"],
        "event_count": event_count if event_count == "adaptive" else int(event_count),
        "scale": float(values["scale"]),
    }


def sku_candidate_errors(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = forecasts.copy()
    frame["absolute_error"] = (frame.target - frame.forecast).abs()
    result = frame.groupby(["sku_id", "candidate_id"], as_index=False).agg(
        actual_total=("target", "sum"),
        forecast_total=("forecast", "sum"),
        absolute_error=("absolute_error", "sum"),
    )
    result["wmape"] = np.where(
        result.actual_total.gt(0), 100.0 * result.absolute_error / result.actual_total, np.nan
    )
    return result


def selected_forecasts(forecasts: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    chosen = forecasts.merge(
        selections[["sku_id", "selected_candidate_id"]], on="sku_id", how="inner"
    )
    chosen = chosen.loc[chosen.candidate_id.eq(chosen.selected_candidate_id)].copy()
    if chosen.sku_id.nunique() != selections.sku_id.nunique():
        raise ValueError("At least one routed SKU has no matching candidate forecast.")
    return chosen


def neighbour_selections(
    errors: pd.DataFrame,
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    numeric_columns: list[str],
    neighbours: int,
    aggregation: str = "median",
    exclude_same_sku: bool = True,
) -> pd.DataFrame:
    train = train_features[["sku_id", *numeric_columns]].drop_duplicates("sku_id").reset_index(drop=True)
    score = score_features[["sku_id", *numeric_columns]].drop_duplicates("sku_id").reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_matrix = scaler.fit_transform(imputer.fit_transform(train[numeric_columns]))
    score_matrix = scaler.transform(imputer.transform(score[numeric_columns]))
    output = []
    error_matrix = errors.pivot(index="sku_id", columns="candidate_id", values="wmape")
    for index, row in score.iterrows():
        distances = np.sqrt(np.square(train_matrix - score_matrix[index]).sum(axis=1))
        order = np.argsort(distances)
        neighbour_ids = []
        for position in order:
            sku_id = train.sku_id.iloc[position]
            if exclude_same_sku and sku_id == row.sku_id:
                continue
            if sku_id in error_matrix.index:
                neighbour_ids.append(sku_id)
            if len(neighbour_ids) >= neighbours:
                break
        peer = error_matrix.loc[neighbour_ids]
        expected = peer.median(axis=0) if aggregation == "median" else peer.mean(axis=0)
        output.append(
            {
                "sku_id": row.sku_id,
                "selected_candidate_id": expected.sort_values().index[0],
            }
        )
    return pd.DataFrame(output)


def own_history_selections(errors: pd.DataFrame) -> pd.DataFrame:
    return (
        errors.sort_values(["sku_id", "wmape", "absolute_error", "candidate_id"])
        .groupby("sku_id", as_index=False)
        .head(1)[["sku_id", "candidate_id"]]
        .rename(columns={"candidate_id": "selected_candidate_id"})
    )
