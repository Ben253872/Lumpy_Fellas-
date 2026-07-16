from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def age_bucket(age: int) -> str:
    if age <= 2:
        return "00_02"
    if age <= 5:
        return "03_05"
    if age <= 11:
        return "06_11"
    if age <= 17:
        return "12_17"
    return "18_plus"


@dataclass
class HazardModel:
    hazard: dict[tuple[object, str], float]
    group_for_sku: dict[object, object]
    initial_age: dict[object, int]

    def probability(self, sku_id, age: int) -> float:
        group = self.group_for_sku.get(sku_id, "all")
        return float(np.clip(self.hazard.get((sku_id, age_bucket(age)), self.hazard.get((group, age_bucket(age)), self.hazard.get(("all", age_bucket(age)), 0.0))), 0.0, 1.0))


def fit_hazard_model(
    train: pd.DataFrame,
    sku_ids: set,
    group_column: str,
    group_smoothing: float = 20.0,
    individual_smoothing: float = 5.0,
) -> HazardModel:
    history = train.loc[train.sku_id.isin(sku_ids)].sort_values(["sku_id", "month"]).copy()
    if group_column == "__global__" or group_column not in history:
        group_for_sku = {sku_id: "all" for sku_id in sku_ids}
    else:
        metadata = history.groupby("sku_id")[group_column].agg(lambda values: values.dropna().iloc[-1] if values.notna().any() else "unknown")
        group_for_sku = {sku_id: metadata.get(sku_id, "unknown") for sku_id in sku_ids}
    exposure_rows = []
    initial_age = {}
    for sku_id, rows in history.groupby("sku_id", sort=False):
        demand = rows.demand.astype(float).clip(lower=0).to_numpy()
        seen_event = False
        age = 0
        for value in demand:
            occurred = value > 0
            if seen_event:
                age += 1
                exposure_rows.append({"sku_id": sku_id, "group": group_for_sku[sku_id], "bucket": age_bucket(age), "occurred": float(occurred)})
            if occurred:
                seen_event = True
                age = 0
        initial_age[sku_id] = age if seen_event else max(len(demand), 18)
    exposure = pd.DataFrame(exposure_rows)
    buckets = ["00_02", "03_05", "06_11", "12_17", "18_plus"]
    if exposure.empty:
        return HazardModel({("all", bucket): 0.0 for bucket in buckets}, group_for_sku, initial_age)
    global_stats = exposure.groupby("bucket").occurred.agg(["sum", "count"])
    global_rate = float(exposure.occurred.mean())
    hazard = {}
    for bucket in buckets:
        stats = global_stats.loc[bucket] if bucket in global_stats.index else {"sum": 0.0, "count": 0.0}
        hazard[("all", bucket)] = (float(stats["sum"]) + group_smoothing * global_rate) / (float(stats["count"]) + group_smoothing)
    group_stats = exposure.groupby(["group", "bucket"]).occurred.agg(["sum", "count"])
    for group_value in set(group_for_sku.values()):
        for bucket in buckets:
            if (group_value, bucket) in group_stats.index:
                stats = group_stats.loc[(group_value, bucket)]
                prior = hazard[("all", bucket)]
                value = (float(stats["sum"]) + group_smoothing * prior) / (float(stats["count"]) + group_smoothing)
            else:
                value = hazard[("all", bucket)]
            hazard[(group_value, bucket)] = value
    sku_stats = exposure.groupby(["sku_id", "bucket"]).occurred.agg(["sum", "count"])
    for sku_id in sku_ids:
        group_value = group_for_sku.get(sku_id, "all")
        for bucket in buckets:
            if (sku_id, bucket) in sku_stats.index:
                stats = sku_stats.loc[(sku_id, bucket)]
                prior = hazard[(group_value, bucket)]
                value = (float(stats["sum"]) + individual_smoothing * prior) / (float(stats["count"]) + individual_smoothing)
            else:
                value = hazard[(group_value, bucket)]
            hazard[(sku_id, bucket)] = value
    return HazardModel(hazard, group_for_sku, initial_age)


def forecast_blocks(
    model: HazardModel,
    sku_ids: set,
    gap_months: int = 3,
    block_count: int = 6,
) -> pd.DataFrame:
    rows = []
    for sku_id in sorted(sku_ids):
        state = {int(model.initial_age.get(sku_id, 18)): 1.0}
        for _ in range(gap_months):
            next_state = {}
            for age, mass in state.items():
                event = mass * model.probability(sku_id, age + 1)
                next_state[0] = next_state.get(0, 0.0) + event
                next_state[age + 1] = next_state.get(age + 1, 0.0) + mass - event
            state = next_state
        for block_number in range(1, block_count + 1):
            expected_events = 0.0
            no_event_state = state.copy()
            no_event_mass = sum(no_event_state.values())
            for _ in range(3):
                next_state = {}
                next_no_event = {}
                for age, mass in state.items():
                    hazard = model.probability(sku_id, age + 1)
                    event = mass * hazard
                    expected_events += event
                    next_state[0] = next_state.get(0, 0.0) + event
                    next_state[age + 1] = next_state.get(age + 1, 0.0) + mass - event
                for age, mass in no_event_state.items():
                    hazard = model.probability(sku_id, age + 1)
                    surviving = mass * (1.0 - hazard)
                    next_no_event[age + 1] = next_no_event.get(age + 1, 0.0) + surviving
                state = next_state
                no_event_state = next_no_event
            block_no_event = sum(no_event_state.values()) / no_event_mass if no_event_mass > 0 else 1.0
            rows.append({"sku_id": sku_id, "block_number": block_number, "expected_events": expected_events, "event_probability": 1.0 - block_no_event})
    return pd.DataFrame(rows)


def compose_forecast(
    blocks: pd.DataFrame,
    sizes: pd.DataFrame,
    mode: str,
    size_source: str,
    scale: float = 1.0,
    horizon_threshold: float = 0.0,
) -> pd.DataFrame:
    frame = blocks.merge(sizes, on="sku_id", how="left")
    size = frame[size_source].fillna(0.0).to_numpy(float)
    frame["rank"] = frame.groupby("sku_id").event_probability.rank(method="first", ascending=False)
    if mode == "expected":
        forecast = frame.expected_events.to_numpy(float) * size
    elif mode == "top1_expected":
        forecast = np.where(frame["rank"].le(1), frame.event_probability * size, 0.0)
    elif mode == "top1_full":
        forecast = np.where(frame["rank"].le(1), size, 0.0)
    elif mode == "top2_expected":
        forecast = np.where(frame["rank"].le(2), frame.event_probability * size, 0.0)
    elif mode == "top2_full":
        forecast = np.where(frame["rank"].le(2), size, 0.0)
    elif mode == "horizon_gate_full":
        horizon_probability = frame.groupby("sku_id").event_probability.transform(lambda values: 1.0 - float(np.prod(1.0 - values)))
        forecast = np.where(frame["rank"].le(1) & horizon_probability.ge(horizon_threshold), size, 0.0)
    else:
        raise ValueError(f"Unknown hazard forecast mode: {mode}")
    frame["forecast"] = np.maximum(0.0, np.asarray(forecast, dtype=float) * float(scale))
    return frame
