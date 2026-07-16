from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd


BLOCK_MONTHS = 3


@dataclass(frozen=True)
class AllocationConfig:
    group_column: str
    level_method: str
    level_window: int
    share_window: int
    prior_strength: float
    gate_multiplier: float | None
    scale: float

    @property
    def config_id(self) -> str:
        group = self.group_column.replace("_", "").lower()
        gate = "none" if self.gate_multiplier is None else f"{self.gate_multiplier:.2f}"
        return (
            f"group-{group}__level-{self.level_method}{self.level_window}__"
            f"share{self.share_window}__prior{self.prior_strength:.0f}__gate{gate}__scale{self.scale:.2f}"
        )


def config_grid() -> list[AllocationConfig]:
    level_specs = (("recent", 12), ("recent", 24), ("seasonal", 12))
    return [
        AllocationConfig(group, method, level_window, share_window, prior, gate, scale)
        for group, (method, level_window), share_window, prior, gate, scale in product(
            ("__cohort__", "FAMILY_DESCRIPTION", "SUBFAMILY_DESCRIPTION"),
            level_specs,
            (12, 24),
            (2.0, 10.0),
            (None, 1.0, 1.25),
            (0.75, 1.0, 1.25),
        )
    ]


def _complete_monthly(train: pd.DataFrame, sku_ids: list, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    months = pd.date_range(start, end, freq="MS")
    index = pd.MultiIndex.from_product([sku_ids, months], names=["sku_id", "month"])
    demand = train.groupby(["sku_id", "month"]).demand.sum().reindex(index, fill_value=0.0).rename("demand").reset_index()
    return demand


def _metadata_table(train: pd.DataFrame, sku_ids: list, group_column: str) -> pd.DataFrame:
    if group_column == "__cohort__" or group_column not in train.columns:
        return pd.DataFrame({"sku_id": sku_ids, "allocation_group": "all"})
    metadata = (
        train.sort_values(["sku_id", "month"])
        .groupby("sku_id", as_index=False)[group_column]
        .agg(lambda values: values.dropna().iloc[-1] if values.notna().any() else "unknown")
        .rename(columns={group_column: "allocation_group"})
    )
    metadata["allocation_group"] = metadata.allocation_group.fillna("unknown").astype(str)
    return pd.DataFrame({"sku_id": sku_ids}).merge(metadata, on="sku_id", how="left").fillna({"allocation_group": "unknown"})


def _level_forecast(group_history: pd.DataFrame, test_months: pd.DatetimeIndex, config: AllocationConfig) -> pd.Series:
    monthly = group_history.groupby("month").demand.sum().sort_index()
    recent = monthly.tail(config.level_window)
    recent_level = float(recent.mean()) if len(recent) else 0.0
    if config.level_method == "recent":
        values = np.repeat(recent_level, len(test_months))
    else:
        seasonal = monthly.reset_index().assign(month_number=lambda frame: frame.month.dt.month).groupby("month_number").demand.mean()
        values = np.asarray([0.5 * float(seasonal.get(month.month, recent_level)) + 0.5 * recent_level for month in test_months])
    return pd.Series(np.maximum(0.0, values * config.scale), index=test_months)


def _sku_shares(group_history: pd.DataFrame, sku_ids: list, config: AllocationConfig) -> pd.Series:
    long_units = group_history.groupby("sku_id").demand.sum().reindex(sku_ids, fill_value=0.0)
    recent_start = group_history.month.max() - pd.DateOffset(months=config.share_window - 1)
    recent_units = group_history.loc[group_history.month.ge(recent_start)].groupby("sku_id").demand.sum().reindex(sku_ids, fill_value=0.0)
    uniform = pd.Series(1.0 / max(len(sku_ids), 1), index=sku_ids)
    long_share = long_units / long_units.sum() if long_units.sum() > 0 else uniform
    recent_share = recent_units / recent_units.sum() if recent_units.sum() > 0 else long_share
    reliability = float(recent_units.sum() / (recent_units.sum() + config.prior_strength))
    share = reliability * recent_share + (1.0 - reliability) * long_share
    return share / share.sum() if share.sum() > 0 else uniform


def _activity_scores(group_history: pd.DataFrame, sku_ids: list) -> tuple[pd.Series, int]:
    pivot = group_history.pivot_table(index="month", columns="sku_id", values="demand", aggfunc="sum", fill_value=0.0).reindex(columns=sku_ids, fill_value=0.0)
    block_count = len(pivot) // BLOCK_MONTHS
    if block_count == 0:
        return pd.Series(1.0, index=sku_ids), len(sku_ids)
    trimmed = pivot.iloc[-block_count * BLOCK_MONTHS:]
    block_ids = np.arange(len(trimmed)) // BLOCK_MONTHS
    active = trimmed.groupby(block_ids).sum().gt(0)
    rate = active.mean(axis=0).reindex(sku_ids, fill_value=0.0)
    months_since = {}
    for sku in sku_ids:
        positive = np.flatnonzero(pivot[sku].to_numpy(float) > 0)
        months_since[sku] = len(pivot) - 1 - positive[-1] if len(positive) else len(pivot)
    recency = pd.Series({sku: np.exp(-months_since[sku] / 12.0) for sku in sku_ids})
    score = 0.75 * rate + 0.25 * recency
    expected_active = max(1, int(round(active.sum(axis=1).mean())))
    return score, expected_active


def forecast_allocation(train: pd.DataFrame, test: pd.DataFrame, sku_ids: set, config: AllocationConfig) -> pd.DataFrame:
    sku_list = sorted(sku_ids)
    train = train.loc[train.sku_id.isin(sku_ids)].copy()
    test = test.loc[test.sku_id.isin(sku_ids)].copy()
    train["month"] = pd.to_datetime(train.month); test["month"] = pd.to_datetime(test.month)
    history = _complete_monthly(train, sku_list, train.month.min(), train.month.max())
    metadata = _metadata_table(train, sku_list, config.group_column)
    history = history.merge(metadata, on="sku_id", how="left")
    test_months = pd.DatetimeIndex(sorted(test.month.unique()))
    monthly_parts = []
    for allocation_group, members in metadata.groupby("allocation_group", sort=False):
        member_ids = members.sku_id.tolist(); group_history = history.loc[history.sku_id.isin(member_ids)]
        level = _level_forecast(group_history, test_months, config)
        shares = _sku_shares(group_history, member_ids, config)
        activity, expected_active = _activity_scores(group_history, member_ids)
        for block_index in range(len(test_months) // BLOCK_MONTHS):
            block_months = test_months[block_index * BLOCK_MONTHS:(block_index + 1) * BLOCK_MONTHS]
            block_total = float(level.reindex(block_months).sum())
            block_shares = shares.copy()
            if config.gate_multiplier is not None:
                active_count = min(len(member_ids), max(1, int(round(expected_active * config.gate_multiplier))))
                keep = (activity * np.sqrt(shares.clip(lower=0))).nlargest(active_count).index
                block_shares.loc[~block_shares.index.isin(keep)] = 0.0
                block_shares = block_shares / block_shares.sum() if block_shares.sum() > 0 else shares
            for sku, share in block_shares.items():
                monthly_parts.append({"sku_id": sku, "block_start": block_months[0], "block_number": block_index + 1, "forecast": block_total * float(share)})
    forecast = pd.DataFrame(monthly_parts)
    actual = test.sort_values(["sku_id", "month"]).copy()
    actual["block_number"] = actual.groupby("sku_id").cumcount() // BLOCK_MONTHS + 1
    actual = actual.groupby(["sku_id", "block_number"], as_index=False).agg(block_start=("month", "min"), target=("demand", "sum"))
    result = actual.merge(forecast, on=["sku_id", "block_number", "block_start"], how="left")
    result["forecast"] = result.forecast.fillna(0.0).clip(lower=0.0)
    scales = train.groupby("sku_id").demand.apply(lambda values: float(np.mean(np.abs(np.diff(values.to_numpy(float))))) if len(values) > 1 else 0.0)
    result["block_naive_scale"] = result.sku_id.map(scales).fillna(0.0)
    result["candidate_id"] = config.config_id
    return result


def config_record(config: AllocationConfig) -> dict:
    return {"config_id": config.config_id, **asdict(config)}
