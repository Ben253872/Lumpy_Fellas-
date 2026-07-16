from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def launch_targets(
    sales: pd.DataFrame,
    sku_ids: Iterable,
    horizon_months: int = 18,
    gap_months: int = 3,
    fixed_test_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    sales = sales.copy()
    sales["month"] = pd.to_datetime(sales.month)
    first = sales.groupby("sku_id").month.min()
    rows = []
    for sku in sku_ids:
        start = pd.Timestamp(fixed_test_start) if fixed_test_start is not None else first[sku] + pd.DateOffset(months=gap_months)
        series = sales.loc[sales.sku_id.eq(sku)].groupby("month").demand.sum()
        for block in range(horizon_months // 3):
            months = pd.date_range(start + pd.DateOffset(months=3 * block), periods=3, freq="MS")
            rows.append({"sku_id": sku, "block_number": block + 1, "block_start": months[0], "target": float(series.reindex(months, fill_value=0.0).sum())})
    return pd.DataFrame(rows)


def _text(metadata: pd.DataFrame) -> pd.Series:
    columns = ["FAMILY_DESCRIPTION", "SUBFAMILY_DESCRIPTION", "MATERIAL_DESCRIPTION"]
    parts = [metadata.get(column, pd.Series("unknown", index=metadata.index)).fillna("unknown").astype(str) for column in columns]
    return (parts[0] + " " + parts[1] + " " + parts[2]).str.lower()


def analogue_hurdle_candidates(
    peer_targets: pd.DataFrame,
    peer_metadata: pd.DataFrame,
    target_targets: pd.DataFrame,
    target_metadata: pd.DataFrame,
    neighbour_counts: Iterable[int] = (3, 5, 10, 20),
    thresholds: Iterable[float] = (0.0, 0.25, 0.5, 0.75),
    scales: Iterable[float] = (0.5, 0.75, 1.0, 1.25),
    size_statistics: Iterable[str] = ("mean",),
    leave_self_out: bool = False,
) -> pd.DataFrame:
    peers = peer_metadata.drop_duplicates("sku_id").reset_index(drop=True)
    targets = target_metadata.drop_duplicates("sku_id").reset_index(drop=True)
    text = pd.concat([_text(peers), _text(targets)], ignore_index=True)
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(text)
    similarity = cosine_similarity(matrix[len(peers):], matrix[:len(peers)])
    peer_position = {sku: index for index, sku in enumerate(peers.sku_id)}
    peer_blocks = peer_targets.pivot(index="sku_id", columns="block_number", values="target").reindex(peers.sku_id).fillna(0.0)
    output = []
    for target_index, target in targets.iterrows():
        scores = similarity[target_index].copy()
        if leave_self_out and target.sku_id in peer_position:
            scores[peer_position[target.sku_id]] = -1.0
        order = np.argsort(scores)[::-1]
        order = order[scores[order] >= 0]
        actual = target_targets.loc[target_targets.sku_id.eq(target.sku_id)].set_index("block_number")
        for k in neighbour_counts:
            chosen = order[: min(int(k), len(order))]
            if len(chosen) == 0:
                continue
            weights = np.clip(scores[chosen], 0.0, None) + 0.05
            values = peer_blocks.iloc[chosen].to_numpy(float)
            event_probability = np.average(values > 0, axis=0, weights=weights)
            for size_statistic in size_statistics:
                positive_size = np.array([
                    _positive_size(block, weights, size_statistic) for block in values.T
                ])
                for threshold in thresholds:
                    base = event_probability * positive_size if threshold == 0 else np.where(event_probability >= threshold, positive_size, 0.0)
                    mode = "expected" if threshold == 0 else "gated"
                    for scale in scales:
                        size_label = "" if size_statistic == "mean" else f"__size_{size_statistic}"
                        candidate_id = f"launch_knn{k}__{mode}__threshold_{threshold:.2f}{size_label}__scale_{scale:.2f}"
                        for block_number in range(1, 7):
                            row = actual.loc[block_number]
                            output.append({"sku_id": target.sku_id, "block_number": block_number, "block_start": row.block_start, "target": float(row.target), "forecast": float(max(0.0, base[block_number - 1] * scale)), "event_probability": float(event_probability[block_number - 1]), "positive_size_estimate": float(positive_size[block_number - 1]), "size_statistic": size_statistic, "candidate_id": candidate_id})
    return pd.DataFrame(output)


def _positive_size(values: np.ndarray, weights: np.ndarray, statistic: str) -> float:
    mask = values > 0
    if not np.any(mask):
        return 0.0
    positive = values[mask].astype(float)
    positive_weights = weights[mask].astype(float)
    if statistic == "mean":
        return float(np.average(positive, weights=positive_weights))
    quantiles = {"median": 0.5, "p75": 0.75, "p90": 0.9}
    if statistic not in quantiles:
        raise ValueError(f"Unknown positive-size statistic: {statistic}")
    order = np.argsort(positive)
    positive = positive[order]
    positive_weights = positive_weights[order]
    cumulative = np.cumsum(positive_weights) / positive_weights.sum()
    return float(positive[np.searchsorted(cumulative, quantiles[statistic], side="left")])
