from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import lumpy_ab_optimization as opt


@dataclass(frozen=True)
class OnlineExpertConfig:
    temperature: float
    decay: float
    shrink: float
    neighbor_weight: float
    top_k: int | None
    memory_weight: float
    shift_threshold: float = 0.75
    shift_decay: float = 0.25
    neighbors: int = 12

    @property
    def config_id(self) -> str:
        top = "all" if self.top_k is None else str(self.top_k)
        return (
            f"temp{self.temperature:.2f}__decay{self.decay:.2f}__shrink{self.shrink:.1f}__"
            f"peer{self.neighbor_weight:.2f}__top{top}__memory{self.memory_weight:.2f}"
        )


def config_grid() -> list[OnlineExpertConfig]:
    return [
        OnlineExpertConfig(*values)
        for values in product(
            (1.0, 3.0, 6.0),
            (0.50, 0.80, 1.00),
            (2.0, 6.0),
            (0.0, 0.35),
            (2, 4, None),
            (0.50, 0.75, 1.00),
        )
    ]


def _normalised_losses(frame: pd.DataFrame, candidate_columns: list[str]) -> np.ndarray:
    target = frame.target.to_numpy(float)
    scale = np.maximum.reduce([
        np.abs(target),
        frame.block_naive_scale.fillna(0).to_numpy(float),
        np.ones(len(frame)),
    ])
    return np.abs(frame[candidate_columns].to_numpy(float) - target[:, None]) / scale[:, None]


def _adaptive_row_weights(history: pd.DataFrame, config: OnlineExpertConfig) -> np.ndarray:
    latest = int(history.origin_id.max())
    age = latest - history.origin_id.to_numpy(int)
    weights = np.power(config.decay, age).astype(float)
    for sku, group in history.sort_values("origin_id").groupby("sku_id", sort=False):
        values = group.target.to_numpy(float)
        if len(values) < 4:
            continue
        recent = float(values[-2:].mean())
        previous = float(values[:-2].mean())
        shift = abs(recent - previous) / max(abs(previous), 1.0)
        if shift > config.shift_threshold:
            idx = group.index.to_numpy()
            # Index lookup is positional because callers reset the history frame.
            weights[idx] = np.power(config.shift_decay, latest - group.origin_id.to_numpy(int))
    return weights


def _aggregate_loss(
    history: pd.DataFrame,
    losses: np.ndarray,
    row_weights: np.ndarray,
    candidate_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    weighted = losses * row_weights[:, None]
    loss_frame = pd.DataFrame(weighted, columns=candidate_columns)
    loss_frame["sku_id"] = history.sku_id.to_numpy()
    loss_frame["row_weight"] = row_weights
    sums = loss_frame.groupby("sku_id")[candidate_columns].sum()
    denominators = loss_frame.groupby("sku_id").row_weight.sum().replace(0, 1)
    own = sums.div(denominators, axis=0)
    global_loss = weighted.sum(axis=0) / max(row_weights.sum(), 1e-9)
    counts = history.groupby("sku_id").origin_id.nunique()
    return own, global_loss, counts


def _peer_losses(
    history: pd.DataFrame,
    own_loss: pd.DataFrame,
    neighbors: int,
) -> pd.DataFrame:
    ordered = history.sort_values(["sku_id", "origin_id"])
    signatures = ordered.groupby("sku_id").agg(
        target_mean=("target", "mean"),
        target_std=("target", "std"),
        zero_rate=("target", lambda values: float(np.mean(np.asarray(values) <= 0))),
        target_last=("target", "last"),
        naive_scale=("block_naive_scale", "mean"),
        consensus_mean=("expert_mean", "mean"),
        disagreement=("expert_std", "mean"),
    ).fillna(0.0)
    signatures = signatures.reindex(own_loss.index)
    values = signatures.to_numpy(float)
    std = values.std(axis=0)
    values = (values - values.mean(axis=0)) / np.where(std > 1e-9, std, 1.0)
    count = min(max(2, neighbors + 1), len(signatures))
    model = NearestNeighbors(n_neighbors=count).fit(values)
    indices = model.kneighbors(values, return_distance=False)
    peer = np.empty_like(own_loss.to_numpy(float))
    own_values = own_loss.to_numpy(float)
    for row, neighbor_indices in enumerate(indices):
        peers = neighbor_indices[neighbor_indices != row][:neighbors]
        peer[row] = own_values[peers].mean(axis=0) if len(peers) else own_values[row]
    return pd.DataFrame(peer, index=own_loss.index, columns=own_loss.columns)


def expert_weights(
    history: pd.DataFrame,
    candidate_columns: list[str],
    config: OnlineExpertConfig,
) -> pd.DataFrame:
    history = history.sort_values(["origin_id", "sku_id"]).reset_index(drop=True)
    losses = _normalised_losses(history, candidate_columns)
    row_weights = _adaptive_row_weights(history, config)
    own, global_loss, counts = _aggregate_loss(history, losses, row_weights, candidate_columns)
    evidence = counts.reindex(own.index).to_numpy(float)
    own_share = evidence / (evidence + config.shrink)
    combined = own.to_numpy(float) * own_share[:, None] + global_loss[None, :] * (1.0 - own_share[:, None])
    if config.neighbor_weight > 0:
        peer = _peer_losses(history, own, config.neighbors).to_numpy(float)
        combined = (1.0 - config.neighbor_weight) * combined + config.neighbor_weight * peer
    logits = -config.temperature * (combined - combined.min(axis=1, keepdims=True))
    logits = np.clip(logits, -50, 50)
    weights = np.exp(logits)
    if config.top_k is not None and config.top_k < len(candidate_columns):
        keep = np.argpartition(weights, -config.top_k, axis=1)[:, -config.top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        mask[np.arange(len(weights))[:, None], keep] = True
        weights = np.where(mask, weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    result = pd.DataFrame(weights, index=own.index, columns=candidate_columns)
    result.index.name = "sku_id"
    return result


def forecast_with_memory(
    history: pd.DataFrame,
    target_frame: pd.DataFrame,
    candidate_columns: list[str],
    config: OnlineExpertConfig,
    champion_column: str = "current_champion",
) -> pd.DataFrame:
    weights = expert_weights(history, candidate_columns, config)
    target = target_frame.copy()
    aligned = weights.reindex(target.sku_id).to_numpy(float)
    memory = np.sum(aligned * target[candidate_columns].to_numpy(float), axis=1)
    raw = config.memory_weight * memory + (1.0 - config.memory_weight) * target[champion_column].to_numpy(float)
    cap = np.maximum(1.0, 3.0 * target[candidate_columns].max(axis=1).to_numpy(float))
    target["forecast"] = np.minimum(np.maximum(0.0, np.nan_to_num(raw, posinf=0.0)), cap)
    target["config_id"] = config.config_id
    return target


def prequential_forecast(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    config: OnlineExpertConfig,
    forecast_origins: Iterable[int],
) -> pd.DataFrame:
    output = []
    for origin in forecast_origins:
        history = frame.loc[frame.origin_id.lt(origin)]
        target = frame.loc[frame.origin_id.eq(origin)]
        if history.empty or target.empty:
            raise ValueError(f"Origin {origin} has no chronology-safe history or target")
        predicted = forecast_with_memory(history, target, candidate_columns, config)
        predicted["history_through_origin"] = int(origin) - 1
        output.append(predicted)
    return pd.concat(output, ignore_index=True)


def score_config(frame: pd.DataFrame, config: OnlineExpertConfig) -> dict:
    _, summary = opt.score_forecast(frame)
    return {"candidate_id": config.config_id, "config_id": config.config_id, **asdict(config), **summary}


def rank_configs(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    return opt.rank_summary(frame)
