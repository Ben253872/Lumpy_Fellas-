"""Lean lumpy-demand forecasting workflow.

The default model path keeps cheap controls and intermittent-demand references:
Zero Forecast, SBA/Croston, Recent Mean 6m, and TSB. Legacy Aggregate
Allocation and Hurdle Random Forest remain available as explicit opt-in
diagnostics, not default runtime work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from models.benchmarks import sba, tsb


SKU_COLUMN = "sku_id"
MONTH_COLUMN = "month"
TARGET_COLUMN = "demand"
LUMPY_DEMAND_TYPE = "lumpy"
COLLISION_AUDIT_COLUMNS = {
    "row_is_collision",
    "is_collision",
    "collision_flag",
    "collision_flag_clean",
    "collision_sku_selected",
    "collision_flag_observed",
}


DEFAULT_SELECTED_EXTERNAL_MATCHES = (
    "lag_1yr_national_annual",
    "working_days",
    "weekend_days",
    "days_in_month",
    "public_holidays",
    "min_temperature",
    "max_wind_gust",
)

DEFAULT_MODEL_NAMES = (
    "zero",
    "sba_croston",
    "recent_mean_6",
    "tsb",
)

OPTIONAL_MODEL_NAMES = (
    "aggregate_allocation",
    "hurdle_random_forest",
)

KNOWN_AHEAD_EXTERNAL_FEATURES = {
    "days_in_month",
    "weekend_days",
    "working_days",
    "public_holidays",
}

CALENDAR_FEATURE_COLUMNS = {
    "month_number",
    "quarter",
    "month_sin",
    "month_cos",
}

LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "total_demand",
    "mean_monthly_demand",
    "max_monthly_demand",
    "months_with_demand",
    "zero_demand_months",
    "zero_month_share",
    "cumulative_demand_share",
    "cumulative_sku_share",
    "average_demand_interval",
    "squared_coefficient_of_variation",
    "REVENUE",
    "COST",
    "UNIT_PRICE",
    "TOTAL_PROFIT",
    "UNIT_COST",
    "STOCK_END_MONTH",
    "STOCK_START_MONTH",
    "NEW_ENTRY_STOCK",
    "value",
}

CATEGORICAL_FEATURE_COLUMNS = (
    "Country",
    "Brand",
    "Channel",
    "REGION",
    "FAMILY_DESCRIPTION",
    "SUBFAMILY_DESCRIPTION",
)


@dataclass(frozen=True)
class LumpyConfig:
    variant: str = "all_sku_history"
    train_months: int = 48
    gap_months: int = 3
    test_months: int = 18
    step_months: int = 3
    min_train_months: int = 18
    max_folds: int | None = None
    external_mode: str = "selected"
    selected_external_matches: tuple[str, ...] = DEFAULT_SELECTED_EXTERNAL_MATCHES
    random_state: int = 42
    history_lag_months: tuple[int, ...] = (1, 3, 6, 12)
    history_rolling_windows: tuple[int, ...] = (3, 6, 12)
    external_rolling_windows: tuple[int, ...] = (3, 6, 12)


def find_project_root(start: Path | None = None) -> Path:
    """Find the repo root from a notebook, src module, or project folder."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [
        current,
        current.parent,
        current.parent.parent,
        current / "Inchscape Repo",
        current.parent / "Inchscape Repo",
    ]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the Inchscape repo root.")


def lumpy_paths(root: Path | None = None, config: LumpyConfig | None = None) -> dict[str, Path]:
    root = find_project_root(root)
    config = config or LumpyConfig()
    output = root / "results" / "lumpy_outputs"
    return {
        "root": root,
        "lumpy_sales": root / "data" / "processed" / config.variant / "collision_sales_lumpy.csv",
        "external_features": root / "data" / "external" / "monthly_external_features.csv",
        "output": output,
        "tables": output / "tables",
        "figures": output / "figures",
    }


def to_month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def ensure_output_dirs(root: Path | None = None) -> dict[str, Path]:
    paths = lumpy_paths(root)
    paths["output"].mkdir(parents=True, exist_ok=True)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    return paths


def load_lumpy_inputs(root: Path | None = None, config: LumpyConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load lumpy demand and monthly external features from repo-local files."""
    config = config or LumpyConfig()
    paths = lumpy_paths(root, config)
    sales = pd.read_csv(paths["lumpy_sales"])
    external = pd.read_csv(paths["external_features"])

    sales[MONTH_COLUMN] = to_month_start(sales[MONTH_COLUMN])
    external["date"] = to_month_start(external["date"])
    external[MONTH_COLUMN] = external["date"]
    sales[TARGET_COLUMN] = pd.to_numeric(sales[TARGET_COLUMN], errors="coerce").fillna(0.0)
    # all_sku_history already means: SKUs with any collision flag, all rows kept.
    # Row-level collision flags are audit fields, not modelling features.
    sales = sales.drop(columns=[column for column in COLLISION_AUDIT_COLUMNS if column in sales.columns])

    if "demand_type" not in sales.columns:
        sales["demand_type"] = LUMPY_DEMAND_TYPE

    return sales, external


def complete_monthly_grid(sales: pd.DataFrame) -> pd.DataFrame:
    """Ensure each SKU has one row per month from its first observation onward."""
    sales = sales.copy()
    sales[MONTH_COLUMN] = to_month_start(sales[MONTH_COLUMN])
    max_month = sales[MONTH_COLUMN].max()
    completed = []
    descriptor_columns = [
        column
        for column in sales.columns
        if column not in {MONTH_COLUMN, TARGET_COLUMN} | COLLISION_AUDIT_COLUMNS
    ]
    for sku, sku_data in sales.groupby(SKU_COLUMN, sort=False):
        sku_data = sku_data.sort_values(MONTH_COLUMN)
        months = pd.date_range(sku_data[MONTH_COLUMN].min(), max_month, freq="MS")
        base = pd.DataFrame({SKU_COLUMN: sku, MONTH_COLUMN: months})
        merged = base.merge(sku_data, on=[SKU_COLUMN, MONTH_COLUMN], how="left")
        merged[TARGET_COLUMN] = pd.to_numeric(merged[TARGET_COLUMN], errors="coerce").fillna(0.0)
        for column in descriptor_columns:
            if column in merged.columns:
                merged[column] = merged[column].ffill().bfill()
        completed.append(merged)
    return pd.concat(completed, ignore_index=True)


def _known_ahead_external_column(column: str) -> bool:
    return column in KNOWN_AHEAD_EXTERNAL_FEATURES or column.startswith("lag_")


def select_external_columns(external: pd.DataFrame, config: LumpyConfig) -> list[str]:
    numeric_columns = [
        column
        for column in external.select_dtypes(include="number").columns
        if column not in {"year", "month"}
    ]
    mode = config.external_mode.lower()
    if mode == "off":
        return []
    if mode == "calendar_only":
        return [column for column in numeric_columns if _known_ahead_external_column(column)]
    if mode == "all":
        return numeric_columns
    if mode != "selected":
        raise ValueError("external_mode must be selected, calendar_only, all, or off")
    lowered_matches = tuple(match.lower() for match in config.selected_external_matches)
    return [
        column
        for column in numeric_columns
        if any(match in column.lower() for match in lowered_matches)
    ]


def build_external_feature_frame(external: pd.DataFrame, config: LumpyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create external feature handoff columns at monthly grain.

    Known-ahead columns are model-ready. Historical-only columns are kept as
    source columns here, then lagged/rolled inside each fold so the model never
    sees values from the operational gap or forecast window.
    """
    selected_columns = select_external_columns(external, config)
    feature_frame = external[["date"] + selected_columns].drop_duplicates("date").sort_values("date")
    engineered = pd.DataFrame({MONTH_COLUMN: feature_frame["date"]})
    inventory_rows = []

    for column in selected_columns:
        if _known_ahead_external_column(column):
            feature_name = f"external_known__{column}"
            usage = "known_ahead"
        else:
            feature_name = f"external_source__{column}"
            usage = "model_lag_source"
        engineered[feature_name] = feature_frame[column]
        inventory_rows.append(
            {
                "source_column": column,
                "feature": feature_name,
                "usage": usage,
            }
        )

    return engineered, pd.DataFrame(inventory_rows)


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["month_number"] = data[MONTH_COLUMN].dt.month
    data["quarter"] = data[MONTH_COLUMN].dt.quarter
    data["month_sin"] = np.sin(2 * np.pi * data["month_number"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month_number"] / 12)
    return data


def build_lumpy_model_frame(
    sales: pd.DataFrame,
    external: pd.DataFrame,
    config: LumpyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or LumpyConfig()
    base = complete_monthly_grid(sales)
    external_features, external_inventory = build_external_feature_frame(external, config)
    model_data = base.merge(external_features, on=MONTH_COLUMN, how="left")
    model_data = add_calendar_features(model_data)
    return model_data.sort_values([MONTH_COLUMN, SKU_COLUMN]).reset_index(drop=True), external_inventory


def make_backtest_splits(data: pd.DataFrame, config: LumpyConfig | None = None) -> pd.DataFrame:
    config = config or LumpyConfig()
    months = pd.Series(pd.to_datetime(data[MONTH_COLUMN].unique())).sort_values().tolist()
    if not months:
        return pd.DataFrame()

    first_month = months[0]
    test_end = months[-1]
    rows = []
    fold_id = 1

    while True:
        test_start = test_end - pd.DateOffset(months=config.test_months - 1)
        gap_end = test_start - pd.DateOffset(months=1)
        train_end = gap_end - pd.DateOffset(months=config.gap_months)
        desired_train_start = train_end - pd.DateOffset(months=config.train_months - 1)
        train_start = max(desired_train_start, first_month)
        observed_train_months = len([month for month in months if train_start <= month <= train_end])

        if observed_train_months < config.min_train_months:
            break
        if test_start < first_month:
            break

        rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "gap_months": config.gap_months,
                "test_start": test_start,
                "test_end": test_end,
                "train_months": observed_train_months,
                "test_months": config.test_months,
                "window_label": f"{config.test_months}_month_test_lag_{config.gap_months}m",
            }
        )
        fold_id += 1
        if config.max_folds is not None and len(rows) >= config.max_folds:
            break
        test_end = test_end - pd.DateOffset(months=config.step_months)
        if test_end < first_month:
            break

    return pd.DataFrame(rows).sort_values("test_start").reset_index(drop=True)


def make_backtest_splits_for_lags(
    data: pd.DataFrame,
    config: LumpyConfig | None = None,
    gap_month_options: tuple[int, ...] = (3, 6, 8),
    test_month_options: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    config = config or LumpyConfig()
    test_month_options = test_month_options or (config.test_months,)
    frames = []
    for test_months in test_month_options:
        for gap_months in gap_month_options:
            split_config = replace(
                config,
                test_months=int(test_months),
                gap_months=int(gap_months),
            )
            frame = make_backtest_splits(data, split_config)
            if not frame.empty:
                frame["is_required_18m_3m_benchmark"] = (
                    frame["test_months"].eq(18) & frame["gap_months"].eq(3)
                )
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["test_months", "gap_months", "test_start"]).reset_index(drop=True)
    combined["global_fold_id"] = np.arange(1, len(combined) + 1)
    return combined


def split_train_test(data: pd.DataFrame, split: pd.Series | dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = dict(split)
    train = data.loc[
        data[MONTH_COLUMN].between(split["train_start"], split["train_end"])
    ].copy()
    test = data.loc[
        data[MONTH_COLUMN].between(split["test_start"], split["test_end"])
    ].copy()
    return train, test


def _forecast_frame(test: pd.DataFrame, forecast: np.ndarray | pd.Series, model: str) -> pd.DataFrame:
    return test[[SKU_COLUMN, MONTH_COLUMN, TARGET_COLUMN]].assign(
        forecast=np.maximum(0.0, np.asarray(forecast, dtype=float)),
        model=model,
    )


def forecast_zero(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return _forecast_frame(test, np.zeros(len(test)), "Zero Forecast")


def forecast_recent_mean(train: pd.DataFrame, test: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    recent = (
        train.sort_values(MONTH_COLUMN)
        .groupby(SKU_COLUMN)[TARGET_COLUMN]
        .apply(lambda values: values.tail(window).mean())
    )
    forecast = test[SKU_COLUMN].map(recent).fillna(0.0)
    return _forecast_frame(test, forecast, f"Recent Mean {window}m")


def _benchmark_forecast(train: pd.DataFrame, test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    months = sorted(pd.to_datetime(test[MONTH_COLUMN].unique()))
    model_func = sba if model_name == "SBA Croston" else tsb
    predicted = model_func(train[[SKU_COLUMN, MONTH_COLUMN, TARGET_COLUMN]], months)
    predicted = predicted.rename(columns={"demand": "forecast"})
    forecast = test[[SKU_COLUMN, MONTH_COLUMN, TARGET_COLUMN]].merge(
        predicted,
        on=[SKU_COLUMN, MONTH_COLUMN],
        how="left",
    )
    forecast["forecast"] = forecast["forecast"].fillna(0.0).clip(lower=0.0)
    forecast["model"] = model_name
    return forecast


def forecast_sba(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return _benchmark_forecast(train, test, "SBA Croston")


def forecast_tsb(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return _benchmark_forecast(train, test, "TSB")


def forecast_aggregate_allocation(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Forecast monthly lumpy demand total, then allocate by recent SKU share."""
    monthly_total = train.groupby(MONTH_COLUMN)[TARGET_COLUMN].sum().sort_index()
    recent_total = monthly_total.tail(6).mean() if not monthly_total.empty else 0.0
    seasonal_total = (
        monthly_total.reset_index()
        .assign(month_number=lambda frame: frame[MONTH_COLUMN].dt.month)
        .groupby("month_number")[TARGET_COLUMN]
        .mean()
    )

    recent_start = train[MONTH_COLUMN].max() - pd.DateOffset(months=11)
    recent_sku = train.loc[train[MONTH_COLUMN].ge(recent_start)].groupby(SKU_COLUMN)[TARGET_COLUMN].sum()
    all_sku = train.groupby(SKU_COLUMN)[TARGET_COLUMN].sum()
    weights = (0.75 * recent_sku.reindex(all_sku.index).fillna(0.0)) + (0.25 * all_sku)
    test_skus = pd.Index(test[SKU_COLUMN].unique())
    weights = weights.reindex(test_skus).fillna(0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0 / len(test_skus), index=test_skus)
    else:
        weights = weights / weights.sum()

    forecast = test[[SKU_COLUMN, MONTH_COLUMN, TARGET_COLUMN]].copy()
    forecast_total = {
        month: float(seasonal_total.get(pd.Timestamp(month).month, recent_total))
        for month in forecast[MONTH_COLUMN].unique()
    }
    forecast["forecast"] = forecast[MONTH_COLUMN].map(forecast_total) * forecast[SKU_COLUMN].map(weights).fillna(0.0)
    forecast["forecast"] = forecast["forecast"].fillna(0.0).clip(lower=0.0)
    forecast["model"] = "Aggregate Allocation"
    return forecast


def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def _summarize_known_demand(values: np.ndarray, config: LumpyConfig) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0, None)
    positive = values[values > 0]
    summary: dict[str, float] = {
        "internal_known_months": float(len(values)),
        "internal_known_total_demand": float(values.sum()) if len(values) else 0.0,
        "internal_known_mean_demand": float(values.mean()) if len(values) else 0.0,
        "internal_known_positive_rate": float((values > 0).mean()) if len(values) else 0.0,
        "internal_known_zero_share": float((values == 0).mean()) if len(values) else 1.0,
        "internal_known_mean_positive_demand": float(positive.mean()) if len(positive) else 0.0,
        "internal_last_known_demand": float(values[-1]) if len(values) else 0.0,
    }

    if len(values) and len(np.flatnonzero(values > 0)):
        positive_positions = np.flatnonzero(values > 0)
        summary["internal_last_positive_demand"] = float(values[positive_positions[-1]])
        summary["internal_months_since_positive"] = float(len(values) - 1 - positive_positions[-1])
    else:
        summary["internal_last_positive_demand"] = 0.0
        summary["internal_months_since_positive"] = float(config.gap_months)

    for lag in config.history_lag_months:
        summary[f"internal_lag_{lag}m_demand"] = float(values[-lag]) if len(values) >= lag else 0.0

    for window in config.history_rolling_windows:
        recent = values[-window:] if len(values) else values
        summary[f"internal_roll_mean_{window}m"] = float(recent.mean()) if len(recent) else 0.0
        summary[f"internal_roll_total_{window}m"] = float(recent.sum()) if len(recent) else 0.0
        summary[f"internal_roll_positive_months_{window}m"] = float((recent > 0).sum()) if len(recent) else 0.0
        summary[f"internal_roll_zero_share_{window}m"] = float((recent == 0).mean()) if len(recent) else 1.0

    return summary


def _train_internal_history_features(train: pd.DataFrame, config: LumpyConfig) -> pd.DataFrame:
    frames = []
    for sku, sku_train in train.sort_values([SKU_COLUMN, MONTH_COLUMN]).groupby(SKU_COLUMN, sort=False):
        values = sku_train[TARGET_COLUMN].astype(float).to_numpy()
        rows = []
        for position, row_index in enumerate(sku_train.index):
            known_end_position = position - config.gap_months + 1
            known_values = values[:max(known_end_position, 0)]
            rows.append(_summarize_known_demand(known_values, config) | {SKU_COLUMN: sku})
        frames.append(pd.DataFrame(rows, index=sku_train.index))
    if not frames:
        return pd.DataFrame(index=train.index)
    return pd.concat(frames).sort_index()


def _test_internal_history_features(train: pd.DataFrame, test: pd.DataFrame, config: LumpyConfig) -> pd.DataFrame:
    history_by_sku = {
        sku: sku_train.sort_values(MONTH_COLUMN)[TARGET_COLUMN].astype(float).to_numpy()
        for sku, sku_train in train.groupby(SKU_COLUMN, sort=False)
    }
    train_end_by_sku = train.groupby(SKU_COLUMN)[MONTH_COLUMN].max() if len(train) else pd.Series(dtype="datetime64[ns]")
    fallback_history = train.sort_values(MONTH_COLUMN)[TARGET_COLUMN].astype(float).to_numpy()
    fallback_train_end = pd.to_datetime(train[MONTH_COLUMN]).max() if len(train) else pd.NaT

    rows = []
    for row in test[[SKU_COLUMN, MONTH_COLUMN]].itertuples(index=True):
        known_values = history_by_sku.get(row.sku_id, fallback_history)
        summary = _summarize_known_demand(known_values, config)
        train_end = train_end_by_sku.get(row.sku_id, fallback_train_end)
        if pd.notna(train_end) and pd.notna(row.month):
            elapsed_months = max(0, _months_between(pd.Timestamp(row.month), pd.Timestamp(train_end)))
            summary["internal_months_since_positive"] += elapsed_months
        summary[SKU_COLUMN] = row.sku_id
        rows.append((row.Index, summary))

    if not rows:
        return pd.DataFrame(index=test.index)
    indexes, summaries = zip(*rows)
    return pd.DataFrame(list(summaries), index=indexes).sort_index()


def _add_internal_history_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: LumpyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_features = _train_internal_history_features(train, config)
    test_features = _test_internal_history_features(train, test, config)
    feature_columns = [column for column in train_features.columns if column != SKU_COLUMN]

    train_out = train.join(train_features[feature_columns])
    test_out = test.join(test_features[feature_columns])
    for frame in (train_out, test_out):
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return train_out, test_out, feature_columns


def _external_source_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("external_source__")]


def _summarize_external_values(values: np.ndarray, source_column: str, config: LumpyConfig) -> dict[str, float]:
    source_name = source_column.replace("external_source__", "")
    values = np.asarray(values, dtype=float)
    summary: dict[str, float] = {}

    for lag in config.history_lag_months:
        summary[f"external_hist_lag_{lag}m__{source_name}"] = float(values[-lag]) if len(values) >= lag else np.nan

    for window in config.external_rolling_windows:
        recent = values[-window:] if len(values) else values
        summary[f"external_hist_roll_mean_{window}m__{source_name}"] = float(np.nanmean(recent)) if len(recent) else np.nan

    return summary


def _external_history_by_month(train: pd.DataFrame, test: pd.DataFrame, config: LumpyConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source_columns = _external_source_columns(train) or _external_source_columns(test)
    if not source_columns:
        return train.copy(), test.copy(), []

    monthly_train = (
        train[[MONTH_COLUMN] + source_columns]
        .drop_duplicates(MONTH_COLUMN)
        .sort_values(MONTH_COLUMN)
        .reset_index(drop=True)
    )
    train_rows = []
    for position, row in monthly_train.iterrows():
        known_end_position = position - config.gap_months + 1
        feature_row = {MONTH_COLUMN: row[MONTH_COLUMN]}
        for source_column in source_columns:
            known_values = monthly_train[source_column].iloc[:max(known_end_position, 0)].astype(float).to_numpy()
            feature_row.update(_summarize_external_values(known_values, source_column, config))
        train_rows.append(feature_row)

    test_rows = []
    for month in sorted(pd.to_datetime(test[MONTH_COLUMN].dropna().unique())):
        feature_row = {MONTH_COLUMN: month}
        for source_column in source_columns:
            known_values = monthly_train[source_column].astype(float).to_numpy()
            feature_row.update(_summarize_external_values(known_values, source_column, config))
        test_rows.append(feature_row)

    train_feature_frame = pd.DataFrame(train_rows)
    test_feature_frame = pd.DataFrame(test_rows)
    feature_columns = [column for column in train_feature_frame.columns if column != MONTH_COLUMN]

    train_out = train.merge(train_feature_frame, on=MONTH_COLUMN, how="left")
    test_out = test.merge(test_feature_frame, on=MONTH_COLUMN, how="left")
    for frame in (train_out, test_out):
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return train_out, test_out, feature_columns


def _sku_history_stats(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_month = train[MONTH_COLUMN].max()
    for sku, sku_data in train.groupby(SKU_COLUMN, sort=False):
        sku_data = sku_data.sort_values(MONTH_COLUMN)
        demand = sku_data[TARGET_COLUMN].astype(float)
        positive = demand.gt(0)
        if positive.any():
            last_positive_month = sku_data.loc[positive, MONTH_COLUMN].max()
            months_since_positive = _months_between(pd.Timestamp(max_month), pd.Timestamp(last_positive_month))
            positive_mean = demand.loc[positive].mean()
        else:
            months_since_positive = 999
            positive_mean = 0.0
        rows.append(
            {
                SKU_COLUMN: sku,
                "sku_train_mean": demand.mean(),
                "sku_recent_3m_mean": demand.tail(3).mean(),
                "sku_recent_6m_mean": demand.tail(6).mean(),
                "sku_recent_12m_mean": demand.tail(12).mean(),
                "sku_positive_rate": positive.mean(),
                "sku_positive_mean": positive_mean,
                "sku_months_since_positive": months_since_positive,
            }
        )
    return pd.DataFrame(rows)


def _feature_family(column: str) -> str:
    if column.startswith("internal_") or column.startswith("sku_"):
        return "internal_history"
    if column.startswith("external_known__"):
        return "external_known_ahead"
    if column.startswith("external_hist_"):
        return "external_model_lagged"
    if column in CALENDAR_FEATURE_COLUMNS:
        return "calendar"
    if any(column.startswith(f"{prefix}_") for prefix in CATEGORICAL_FEATURE_COLUMNS):
        return "categorical"
    return "other"


def _rf_design_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: LumpyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x, test_x, _ = _add_internal_history_features(train.copy(), test.copy(), config)
    train_x, test_x, _ = _external_history_by_month(train_x, test_x, config)

    stats = _sku_history_stats(train)
    train_x = train_x.merge(stats, on=SKU_COLUMN, how="left")
    test_x = test_x.merge(stats, on=SKU_COLUMN, how="left")

    categorical_columns = [
        column
        for column in CATEGORICAL_FEATURE_COLUMNS
        if column in train_x.columns or column in test_x.columns
    ]
    combined = pd.concat(
        [
            train_x.assign(_is_train=True),
            test_x.assign(_is_train=False),
        ],
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=categorical_columns, dummy_na=True, dtype=float)

    excluded = {SKU_COLUMN, MONTH_COLUMN, TARGET_COLUMN, "demand_type", "_is_train"} | COLLISION_AUDIT_COLUMNS | LEAKAGE_COLUMNS
    feature_columns = []
    for column in combined.columns:
        if column in excluded or column.startswith("external_source__"):
            continue
        if not pd.api.types.is_numeric_dtype(combined[column]):
            continue
        family = _feature_family(column)
        if family in {"internal_history", "external_known_ahead", "external_model_lagged", "calendar", "categorical"}:
            feature_columns.append(column)

    feature_columns = sorted(set(feature_columns))
    return (
        combined.loc[combined["_is_train"], feature_columns].copy(),
        combined.loc[~combined["_is_train"], feature_columns].copy(),
        feature_columns,
    )


def build_lumpy_feature_inventory(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    config: LumpyConfig | None = None,
) -> pd.DataFrame:
    """Preview model feature families for the latest split without fitting a model."""
    config = config or LumpyConfig()
    if splits.empty:
        return pd.DataFrame(columns=["feature", "feature_family", "non_null_share_preview_train"])

    benchmark_mask = (
        splits.get("test_months", pd.Series(dtype=int)).eq(config.test_months)
        & splits.get("gap_months", pd.Series(dtype=int)).eq(config.gap_months)
    )
    preview_splits = splits.loc[benchmark_mask] if benchmark_mask.any() else splits
    split = preview_splits.iloc[-1]
    split_config = replace(
        config,
        test_months=int(split.get("test_months", config.test_months)),
        gap_months=int(split.get("gap_months", config.gap_months)),
    )
    train, test = split_train_test(data, split)
    train_x, _, feature_columns = _rf_design_matrix(train, test, split_config)
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_family": [_feature_family(column) for column in feature_columns],
            "non_null_share_preview_train": [train_x[column].notna().mean() for column in feature_columns],
        }
    ).sort_values(["feature_family", "feature"]).reset_index(drop=True)


def forecast_hurdle_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: LumpyConfig | None = None,
    random_state: int = 42,
    occurrence_threshold: float = 0.2,
) -> pd.DataFrame:
    """Two-stage occurrence/positive-size Random Forest."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
    except ModuleNotFoundError:
        fallback = forecast_recent_mean(train, test, window=6)
        fallback["model"] = "Hurdle Random Forest"
        fallback["rf_feature_count"] = 0
        fallback["rf_status"] = "fallback_recent_mean_sklearn_missing"
        return fallback

    config = config or LumpyConfig(random_state=random_state)
    train_x, test_x, feature_columns = _rf_design_matrix(train, test, config)
    y = train[TARGET_COLUMN].astype(float)
    occurred = y.gt(0).astype(int)
    fallback = forecast_recent_mean(train, test, window=6)
    fallback["model"] = "Hurdle Random Forest"

    if occurred.nunique() < 2 or y.gt(0).sum() < 2 or not feature_columns:
        fallback["rf_feature_count"] = len(feature_columns)
        fallback["rf_status"] = "fallback_recent_mean_insufficient_classes"
        return fallback

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_x)
    x_test = imputer.transform(test_x)

    classifier = RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(x_train, occurred)
    probability = classifier.predict_proba(x_test)[:, 1]

    positive_mask = y.gt(0)
    regressor = RandomForestRegressor(
        n_estimators=240,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    regressor.fit(x_train[positive_mask.to_numpy()], y.loc[positive_mask])
    positive_amount = np.maximum(0.0, regressor.predict(x_test))
    forecast_values = probability * positive_amount
    forecast_values = np.where(probability >= occurrence_threshold, forecast_values, 0.0)

    forecast = _forecast_frame(test, forecast_values, "Hurdle Random Forest")
    forecast["occurrence_probability"] = probability
    forecast["positive_amount_forecast"] = positive_amount
    forecast["rf_feature_count"] = len(feature_columns)
    forecast["rf_status"] = "fit"
    return forecast


def run_model(name: str, train: pd.DataFrame, test: pd.DataFrame, config: LumpyConfig) -> pd.DataFrame:
    if name == "zero":
        return forecast_zero(train, test)
    if name == "recent_mean_6":
        return forecast_recent_mean(train, test, window=6)
    if name == "sba_croston":
        return forecast_sba(train, test)
    if name == "tsb":
        return forecast_tsb(train, test)
    if name == "aggregate_allocation":
        return forecast_aggregate_allocation(train, test)
    if name == "hurdle_random_forest":
        return forecast_hurdle_random_forest(train, test, config=config, random_state=config.random_state)
    raise ValueError(f"Unknown lumpy model: {name}")


def run_lumpy_backtest(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    config: LumpyConfig | None = None,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
) -> pd.DataFrame:
    config = config or LumpyConfig()
    forecasts = []
    for _, split in splits.iterrows():
        train, test = split_train_test(data, split)
        if train.empty or test.empty:
            continue
        split_config = replace(config, gap_months=int(split.get("gap_months", config.gap_months)))
        for model_name in model_names:
            forecast = run_model(model_name, train, test, split_config)
            forecast = add_sku_segments(forecast, train)
            for column in splits.columns:
                forecast[column] = split[column]
            forecasts.append(forecast)
    if not forecasts:
        return pd.DataFrame()
    return pd.concat(forecasts, ignore_index=True)


def add_error_columns(forecasts: pd.DataFrame) -> pd.DataFrame:
    scored = forecasts.copy()
    scored["forecast"] = pd.to_numeric(scored["forecast"], errors="coerce").fillna(0.0).clip(lower=0.0)
    scored[TARGET_COLUMN] = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce").fillna(0.0)
    scored["absolute_error"] = (scored[TARGET_COLUMN] - scored["forecast"]).abs()
    scored["bias"] = scored["forecast"] - scored[TARGET_COLUMN]
    return scored


def summarize_by_model(forecasts: pd.DataFrame) -> pd.DataFrame:
    scored = add_error_columns(forecasts)
    group_columns = _present_columns(scored, ["window_label", "test_months", "gap_months"]) + ["model"]
    summary = (
        scored.groupby(group_columns, as_index=False)
        .agg(
            rows=(TARGET_COLUMN, "size"),
            sku_count=(SKU_COLUMN, "nunique"),
            actual_total=(TARGET_COLUMN, "sum"),
            forecast_total=("forecast", "sum"),
            absolute_error=("absolute_error", "sum"),
            bias=("bias", "sum"),
            positive_actual_rows=(TARGET_COLUMN, lambda values: int((values > 0).sum())),
            positive_forecast_rows=("forecast", lambda values: int((values > 0).sum())),
        )
    )
    summary["wmape_percent"] = np.where(
        summary["actual_total"].gt(0),
        100 * summary["absolute_error"] / summary["actual_total"],
        np.nan,
    )
    summary["bias_percent"] = np.where(
        summary["actual_total"].gt(0),
        100 * summary["bias"] / summary["actual_total"],
        np.nan,
    )
    return summary.sort_values(["window_label", "wmape_percent", "model"]).reset_index(drop=True)


def summarize_monthly_totals(forecasts: pd.DataFrame) -> pd.DataFrame:
    scored = add_error_columns(forecasts)
    group_columns = _present_columns(
        scored,
        ["window_label", "test_months", "gap_months", "fold_id", "global_fold_id"],
    ) + ["model"]
    monthly = (
        scored.groupby(group_columns + [MONTH_COLUMN], as_index=False)
        .agg(actual=(TARGET_COLUMN, "sum"), forecast=("forecast", "sum"))
        .sort_values(group_columns + [MONTH_COLUMN])
    )
    monthly["absolute_error"] = (monthly["actual"] - monthly["forecast"]).abs()
    monthly["monthly_wmape_percent"] = np.where(
        monthly["actual"].gt(0),
        100 * monthly["absolute_error"] / monthly["actual"],
        np.nan,
    )
    monthly["rolling_3m_abs_error"] = (
        monthly.groupby(group_columns)["absolute_error"]
        .transform(lambda series: series.rolling(3, min_periods=1).sum())
    )
    monthly["rolling_3m_actual"] = (
        monthly.groupby(group_columns)["actual"]
        .transform(lambda series: series.rolling(3, min_periods=1).sum())
    )
    monthly["rolling_3m_wmape_percent"] = np.where(
        monthly["rolling_3m_actual"].gt(0),
        100 * monthly["rolling_3m_abs_error"] / monthly["rolling_3m_actual"],
        np.nan,
    )
    monthly["rolling_3m_average_monthly_wmape_percent"] = (
        monthly.groupby(group_columns)["monthly_wmape_percent"]
        .transform(lambda series: series.rolling(3, min_periods=1).mean())
    )
    return monthly.reset_index(drop=True)




def _present_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data.columns]


def _wmape_percent_from_sums(absolute_error: pd.Series, actual: pd.Series) -> pd.Series:
    return np.where(actual.gt(0), 100 * absolute_error / actual.replace(0, np.nan), np.nan)


PREDICTABLE_FORECAST_POPULATION = "known_sku_with_train_demand"
COLD_START_MISSING_POPULATION = "cold_start_missing_from_train"
KNOWN_NO_TRAIN_DEMAND_POPULATION = "known_sku_no_train_demand"


def create_sku_segments(train: pd.DataFrame) -> pd.DataFrame:
    if train.empty:
        return pd.DataFrame(columns=[SKU_COLUMN, "sku_segment"])

    data = train.copy()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").fillna(0.0)

    def positive_mean(values: pd.Series) -> float:
        positives = values[values > 0]
        return float(positives.mean()) if len(positives) else 0.0

    stats = (
        data.groupby(SKU_COLUMN, as_index=False)
        .agg(
            segment_train_months=(TARGET_COLUMN, "size"),
            segment_train_total_demand=(TARGET_COLUMN, "sum"),
            segment_train_positive_months=(TARGET_COLUMN, lambda values: int((values > 0).sum())),
            segment_train_zero_share=(TARGET_COLUMN, lambda values: float((values == 0).mean())),
            segment_train_mean_positive_demand=(TARGET_COLUMN, positive_mean),
        )
    )
    stats["sku_segment"] = "no_train_demand"
    positive_mask = stats["segment_train_total_demand"].gt(0)
    positive_count = int(positive_mask.sum())
    if positive_count >= 5:
        positive_rank = stats.loc[positive_mask, "segment_train_total_demand"].rank(pct=True, method="first")
        stats.loc[positive_mask & positive_rank.ge(0.80), "sku_segment"] = "high_volume_lumpy"
        stats.loc[positive_mask & positive_rank.lt(0.80), "sku_segment"] = "mid_low_volume_lumpy"
    else:
        stats.loc[positive_mask, "sku_segment"] = "positive_lumpy"

    rare_mask = positive_mask & stats["segment_train_zero_share"].ge(0.85)
    stats.loc[rare_mask, "sku_segment"] = "rare_lumpy"
    return stats


def forecast_population_from_segment(segment: str | float | None) -> str:
    if pd.isna(segment) or segment == "missing_from_train":
        return COLD_START_MISSING_POPULATION
    if segment == "no_train_demand":
        return KNOWN_NO_TRAIN_DEMAND_POPULATION
    return PREDICTABLE_FORECAST_POPULATION


def add_sku_segments(forecast: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    segments = create_sku_segments(train)[[SKU_COLUMN, "sku_segment"]]
    enriched = forecast.merge(segments, on=SKU_COLUMN, how="left")
    enriched["sku_segment"] = enriched["sku_segment"].fillna("missing_from_train")
    enriched["forecast_population"] = enriched["sku_segment"].map(forecast_population_from_segment)
    enriched["is_predictable_population"] = enriched["forecast_population"].eq(PREDICTABLE_FORECAST_POPULATION)
    return enriched


def enrich_forecasts_with_segments(forecasts: pd.DataFrame, model_data: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()
    if {"sku_segment", "forecast_population", "is_predictable_population"}.issubset(forecasts.columns):
        return forecasts

    rows = []
    group_columns = _present_columns(forecasts, ["global_fold_id"])
    if not group_columns:
        group_columns = _present_columns(forecasts, ["window_label", "fold_id"])
    if not group_columns:
        group_columns = ["model"]

    data = model_data.copy()
    data[MONTH_COLUMN] = pd.to_datetime(data[MONTH_COLUMN])
    for _, group in forecasts.groupby(group_columns, dropna=False, sort=False):
        split = group.iloc[0]
        if "train_start" in group.columns and "train_end" in group.columns:
            train = data.loc[
                data[MONTH_COLUMN].between(pd.Timestamp(split["train_start"]), pd.Timestamp(split["train_end"]))
            ].copy()
        else:
            train = data.iloc[0:0].copy()
        existing = [column for column in ["sku_segment", "forecast_population", "is_predictable_population"] if column in group.columns]
        rows.append(add_sku_segments(group.drop(columns=existing), train))
    return pd.concat(rows, ignore_index=True) if rows else forecasts.copy()


def summarize_fold_models(forecasts: pd.DataFrame) -> pd.DataFrame:
    scored = add_error_columns(forecasts)
    group_columns = _present_columns(
        scored,
        ["window_label", "test_months", "gap_months", "fold_id", "global_fold_id"],
    ) + ["model"]
    summary = (
        scored.groupby(group_columns, as_index=False)
        .agg(
            rows=(TARGET_COLUMN, "size"),
            sku_count=(SKU_COLUMN, "nunique"),
            actual_total=(TARGET_COLUMN, "sum"),
            forecast_total=("forecast", "sum"),
            absolute_error=("absolute_error", "sum"),
            bias=("bias", "sum"),
            positive_actual_rows=(TARGET_COLUMN, lambda values: int((values > 0).sum())),
            positive_forecast_rows=("forecast", lambda values: int((values > 0).sum())),
        )
    )
    summary["wmape_percent"] = _wmape_percent_from_sums(summary["absolute_error"], summary["actual_total"])
    summary["bias_percent"] = np.where(
        summary["actual_total"].gt(0),
        100 * summary["bias"] / summary["actual_total"].replace(0, np.nan),
        np.nan,
    )
    return summary.sort_values(group_columns).reset_index(drop=True)


def summarize_monthly_total_tables(monthly_totals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if monthly_totals.empty:
        return pd.DataFrame(), pd.DataFrame()
    monthly = monthly_totals.copy()
    fold_group_columns = _present_columns(
        monthly,
        ["window_label", "test_months", "gap_months", "fold_id", "global_fold_id"],
    ) + ["model"]
    fold_summary = (
        monthly.groupby(fold_group_columns, as_index=False)
        .agg(
            months=(MONTH_COLUMN, "nunique"),
            actual_total=("actual", "sum"),
            forecast_total=("forecast", "sum"),
            absolute_error=("absolute_error", "sum"),
            mean_monthly_wmape_percent=("monthly_wmape_percent", "mean"),
            mean_rolling_3m_wmape_percent=("rolling_3m_wmape_percent", "mean"),
        )
    )
    fold_summary["monthly_total_wmape_percent"] = _wmape_percent_from_sums(
        fold_summary["absolute_error"], fold_summary["actual_total"]
    )

    model_group_columns = [column for column in fold_group_columns if column not in {"fold_id", "global_fold_id"}]
    model_summary = (
        fold_summary.groupby(model_group_columns, as_index=False)
        .agg(
            folds=("fold_id", "nunique") if "fold_id" in fold_summary.columns else ("months", "size"),
            months=("months", "sum"),
            actual_total=("actual_total", "sum"),
            forecast_total=("forecast_total", "sum"),
            absolute_error=("absolute_error", "sum"),
            mean_monthly_wmape_percent=("mean_monthly_wmape_percent", "mean"),
            mean_rolling_3m_wmape_percent=("mean_rolling_3m_wmape_percent", "mean"),
        )
    )
    model_summary["monthly_total_wmape_percent"] = _wmape_percent_from_sums(
        model_summary["absolute_error"], model_summary["actual_total"]
    )
    fold_sort = model_group_columns + ["fold_id"] if "fold_id" in fold_summary.columns else model_group_columns
    return (
        fold_summary.sort_values(fold_sort).reset_index(drop=True),
        model_summary.sort_values(model_group_columns + ["monthly_total_wmape_percent"]).reset_index(drop=True),
    )


def build_sku_horizon_results(forecasts: pd.DataFrame) -> pd.DataFrame:
    scored = add_error_columns(forecasts)
    group_columns = _present_columns(
        scored,
        ["window_label", "test_months", "gap_months", "fold_id", "global_fold_id", "model", "sku_segment", "forecast_population"],
    )
    horizon = (
        scored.groupby(group_columns + [SKU_COLUMN], as_index=False)
        .agg(actual_horizon=(TARGET_COLUMN, "sum"), forecast_horizon=("forecast", "sum"))
    )
    horizon["absolute_error"] = (horizon["actual_horizon"] - horizon["forecast_horizon"]).abs()
    horizon["bias"] = horizon["forecast_horizon"] - horizon["actual_horizon"]
    horizon["actual_positive_sku"] = horizon["actual_horizon"].gt(0)
    horizon["forecast_positive_sku"] = horizon["forecast_horizon"].gt(0)
    horizon["sku_horizon_wmape_percent"] = np.where(
        horizon["actual_horizon"].gt(0),
        100 * horizon["absolute_error"] / horizon["actual_horizon"],
        np.nan,
    )
    return horizon


def _summarize_sku_horizon_group(group: pd.DataFrame) -> pd.Series:
    actual_total = group["actual_horizon"].sum()
    absolute_error = group["absolute_error"].sum()
    positive_mask = group["actual_positive_sku"]
    zero_actual_mask = ~positive_mask
    false_positive_zero = zero_actual_mask & group["forecast_positive_sku"]
    missed_positive = positive_mask & ~group["forecast_positive_sku"]
    positive_wmape = group.loc[positive_mask, "sku_horizon_wmape_percent"]
    return pd.Series(
        {
            "sku_count": group[SKU_COLUMN].nunique(),
            "positive_actual_skus": int(positive_mask.sum()),
            "zero_actual_skus": int(zero_actual_mask.sum()),
            "forecast_positive_skus": int(group["forecast_positive_sku"].sum()),
            "false_positive_zero_actual_skus": int(false_positive_zero.sum()),
            "missed_positive_skus": int(missed_positive.sum()),
            "actual_total": actual_total,
            "forecast_total": group["forecast_horizon"].sum(),
            "absolute_error": absolute_error,
            "bias": group["bias"].sum(),
            "horizon_sku_wmape_percent": 100 * absolute_error / actual_total if actual_total > 0 else np.nan,
            "median_positive_sku_wmape_percent": positive_wmape.median() if len(positive_wmape) else np.nan,
            "positive_skus_below_70_percent": int(positive_wmape.lt(70).sum()) if len(positive_wmape) else 0,
            "positive_skus_below_100_percent": int(positive_wmape.lt(100).sum()) if len(positive_wmape) else 0,
            "false_positive_zero_actual_sku_share_percent": (
                100 * false_positive_zero.sum() / zero_actual_mask.sum() if zero_actual_mask.sum() else np.nan
            ),
        }
    )


def summarize_sku_horizon_results(
    sku_horizon_results: pd.DataFrame,
    extra_group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra_group_columns = extra_group_columns or []
    base_columns = _present_columns(
        sku_horizon_results,
        ["window_label", "test_months", "gap_months", "fold_id", "global_fold_id", "model"] + extra_group_columns,
    )
    fold_summary = (
        sku_horizon_results.groupby(base_columns, dropna=False)
        .apply(_summarize_sku_horizon_group)
        .reset_index()
    )
    model_columns = [column for column in base_columns if column not in {"fold_id", "global_fold_id"}]
    model_summary = (
        sku_horizon_results.groupby(model_columns, dropna=False)
        .apply(_summarize_sku_horizon_group)
        .reset_index()
    )
    return (
        fold_summary.sort_values(base_columns).reset_index(drop=True),
        model_summary.sort_values(model_columns + ["horizon_sku_wmape_percent"]).reset_index(drop=True),
    )


def summarize_population_rows(forecasts: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_error_columns(forecasts)
    fold_group_columns = _present_columns(
        scored,
        ["window_label", "test_months", "gap_months", "fold_id", "global_fold_id", "model", group_column],
    )
    fold_summary = (
        scored.groupby(fold_group_columns, as_index=False, dropna=False)
        .agg(
            rows=(TARGET_COLUMN, "size"),
            sku_count=(SKU_COLUMN, "nunique"),
            actual_total=(TARGET_COLUMN, "sum"),
            forecast_total=("forecast", "sum"),
            absolute_error=("absolute_error", "sum"),
            bias=("bias", "sum"),
            positive_actual_rows=(TARGET_COLUMN, lambda values: int((values > 0).sum())),
            positive_forecast_rows=("forecast", lambda values: int((values > 0).sum())),
        )
    )
    fold_summary["wmape_percent"] = _wmape_percent_from_sums(fold_summary["absolute_error"], fold_summary["actual_total"])
    fold_summary["selection_score"] = np.where(
        fold_summary["actual_total"].gt(0),
        fold_summary["wmape_percent"],
        fold_summary["absolute_error"],
    )
    model_group_columns = [column for column in fold_group_columns if column not in {"fold_id", "global_fold_id"}]
    model_summary = (
        fold_summary.groupby(model_group_columns, as_index=False, dropna=False)
        .agg(
            folds=("fold_id", "nunique") if "fold_id" in fold_summary.columns else ("rows", "size"),
            rows=("rows", "sum"),
            sku_count=("sku_count", "max"),
            actual_total=("actual_total", "sum"),
            forecast_total=("forecast_total", "sum"),
            absolute_error=("absolute_error", "sum"),
            bias=("bias", "sum"),
            positive_actual_rows=("positive_actual_rows", "sum"),
            positive_forecast_rows=("positive_forecast_rows", "sum"),
        )
    )
    model_summary["wmape_percent"] = _wmape_percent_from_sums(model_summary["absolute_error"], model_summary["actual_total"])
    model_summary["selection_score"] = np.where(
        model_summary["actual_total"].gt(0),
        model_summary["wmape_percent"],
        model_summary["absolute_error"],
    )
    return (
        fold_summary.sort_values(fold_group_columns).reset_index(drop=True),
        model_summary.sort_values(model_group_columns + ["selection_score"]).reset_index(drop=True),
    )


def _best_by_group(summary: pd.DataFrame, group_columns: list[str], score_column: str) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    present_group_columns = _present_columns(summary, group_columns)
    return (
        summary.sort_values(present_group_columns + [score_column, "model"])
        .groupby(present_group_columns, as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def build_metric_suite_model_summary(
    model_summary: pd.DataFrame,
    monthly_total_model_summary: pd.DataFrame,
    sku_horizon_model_summary: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = _present_columns(model_summary, ["window_label", "test_months", "gap_months", "model"])
    metric = model_summary[key_columns + ["wmape_percent", "actual_total", "forecast_total", "absolute_error"]].rename(
        columns={
            "wmape_percent": "sku_month_wmape_percent",
            "actual_total": "sku_month_actual_total",
            "forecast_total": "sku_month_forecast_total",
            "absolute_error": "sku_month_absolute_error",
        }
    )
    if not monthly_total_model_summary.empty:
        metric = metric.merge(
            monthly_total_model_summary[key_columns + ["monthly_total_wmape_percent"]],
            on=key_columns,
            how="left",
        )
    if not sku_horizon_model_summary.empty:
        metric = metric.merge(
            sku_horizon_model_summary[key_columns + ["horizon_sku_wmape_percent", "median_positive_sku_wmape_percent"]],
            on=key_columns,
            how="left",
        )
    score_columns = [
        column
        for column in ["sku_month_wmape_percent", "monthly_total_wmape_percent", "horizon_sku_wmape_percent"]
        if column in metric.columns
    ]
    metric["phase1_decision_score_percent"] = metric[score_columns].mean(axis=1, skipna=True)
    return metric.sort_values(_present_columns(metric, ["window_label", "phase1_decision_score_percent", "model"])).reset_index(drop=True)


def run_lag_comparison_agent(metric_suite_model_summary: pd.DataFrame) -> pd.DataFrame:
    if metric_suite_model_summary.empty or "gap_months" not in metric_suite_model_summary.columns:
        return pd.DataFrame()
    score_column = "horizon_sku_wmape_percent" if "horizon_sku_wmape_percent" in metric_suite_model_summary.columns else "phase1_decision_score_percent"
    rows = []
    for test_months, window in metric_suite_model_summary.groupby("test_months", dropna=False):
        ranked = window.dropna(subset=[score_column]).sort_values([score_column, "model"])
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        requested = ranked.loc[ranked["gap_months"].eq(3)]
        requested_best = requested.iloc[0] if not requested.empty else None
        requested_score = requested_best[score_column] if requested_best is not None else np.nan
        delta = best[score_column] - requested_score if pd.notna(requested_score) else np.nan
        if best["gap_months"] == 3:
            recommendation = "Keep the required 3-month lag as the lead benchmark for this window."
        elif pd.notna(delta) and delta < -5:
            recommendation = "A longer lead-time lag is materially better in this diagnostic view; review operational fit before promoting it."
        else:
            recommendation = "Alternative lag is close to the 3-month benchmark; keep 3-month as required and use this as sensitivity context."
        rows.append(
            {
                "agent": "Lag Comparison Agent",
                "test_months": test_months,
                "score_column": score_column,
                "required_lag_months": 3,
                "required_lag_best_model": requested_best["model"] if requested_best is not None else pd.NA,
                "required_lag_score_percent": requested_score,
                "best_lag_months": best["gap_months"],
                "best_window_label": best["window_label"],
                "best_model": best["model"],
                "best_score_percent": best[score_column],
                "delta_vs_required_lag_percent": delta,
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def run_monthly_total_phase1_agent(monthly_total_model_summary: pd.DataFrame) -> pd.DataFrame:
    if monthly_total_model_summary.empty:
        return pd.DataFrame()
    rows = []
    for window_label, window in monthly_total_model_summary.groupby("window_label", dropna=False):
        best = window.sort_values(["monthly_total_wmape_percent", "model"]).iloc[0]
        rows.append(
            {
                "agent": "Monthly Total Agent",
                "window_label": window_label,
                "test_months": best.get("test_months", pd.NA),
                "gap_months": best.get("gap_months", pd.NA),
                "best_model": best["model"],
                "monthly_total_wmape_percent": best["monthly_total_wmape_percent"],
                "recommendation": "Monthly total is the cleaner lens; use SKU allocation summaries before rejecting the run." if best["monthly_total_wmape_percent"] < 100 else "Monthly total is also weak; avoid over-claiming precision.",
            }
        )
    return pd.DataFrame(rows)


def run_sku_horizon_phase1_agent(sku_horizon_model_summary: pd.DataFrame) -> pd.DataFrame:
    if sku_horizon_model_summary.empty:
        return pd.DataFrame()
    rows = []
    for window_label, window in sku_horizon_model_summary.groupby("window_label", dropna=False):
        best = window.sort_values(["horizon_sku_wmape_percent", "model"]).iloc[0]
        rows.append(
            {
                "agent": "SKU Horizon Agent",
                "window_label": window_label,
                "test_months": best.get("test_months", pd.NA),
                "gap_months": best.get("gap_months", pd.NA),
                "best_model": best["model"],
                "horizon_sku_wmape_percent": best["horizon_sku_wmape_percent"],
                "positive_actual_skus": best["positive_actual_skus"],
                "positive_skus_below_100_percent": best["positive_skus_below_100_percent"],
                "recommendation": "Judge SKU quantity over the horizon separately from exact month timing.",
            }
        )
    return pd.DataFrame(rows)


def run_metric_decision_phase1_agent(metric_suite_model_summary: pd.DataFrame) -> pd.DataFrame:
    if metric_suite_model_summary.empty:
        return pd.DataFrame()
    rows = []
    for window_label, window in metric_suite_model_summary.groupby("window_label", dropna=False):
        best = window.sort_values(["phase1_decision_score_percent", "model"]).iloc[0]
        rows.append(
            {
                "agent": "Metric Decision Agent",
                "window_label": window_label,
                "test_months": best.get("test_months", pd.NA),
                "gap_months": best.get("gap_months", pd.NA),
                "best_model": best["model"],
                "phase1_decision_score_percent": best["phase1_decision_score_percent"],
                "sku_month_wmape_percent": best.get("sku_month_wmape_percent", np.nan),
                "monthly_total_wmape_percent": best.get("monthly_total_wmape_percent", np.nan),
                "horizon_sku_wmape_percent": best.get("horizon_sku_wmape_percent", np.nan),
                "recommendation": "Use this as a phase-1 readout only; hybrid allocation models are not restored yet.",
            }
        )
    return pd.DataFrame(rows)


def run_best_summary_agent(summary: pd.DataFrame, group_column: str, agent_name: str) -> pd.DataFrame:
    if summary.empty or group_column not in summary.columns:
        return pd.DataFrame()
    rows = []
    group_columns = _present_columns(summary, ["window_label", "test_months", "gap_months", group_column])
    for keys, group in summary.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys))
        best = group.sort_values(["selection_score", "model"]).iloc[0]
        rows.append(
            {
                "agent": agent_name,
                **key_map,
                "best_model": best["model"],
                "actual_total": best["actual_total"],
                "wmape_percent": best["wmape_percent"],
                "selection_score": best["selection_score"],
                "recommendation": "Use the best model for this slice as diagnostic context; phase-2 blends are not restored yet.",
            }
        )
    return pd.DataFrame(rows)


def build_lumpy_phase1_tables(
    model_data: pd.DataFrame,
    splits: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    enriched_forecasts = enrich_forecasts_with_segments(forecasts, model_data)
    model_summary = summarize_by_model(enriched_forecasts)
    fold_model_summary = summarize_fold_models(enriched_forecasts)
    monthly_totals = summarize_monthly_totals(enriched_forecasts)
    monthly_total_fold_summary, monthly_total_model_summary = summarize_monthly_total_tables(monthly_totals)
    sku_horizon_results = build_sku_horizon_results(enriched_forecasts)
    sku_horizon_fold_summary, sku_horizon_model_summary = summarize_sku_horizon_results(sku_horizon_results)
    _, sku_horizon_population_summary = summarize_sku_horizon_results(sku_horizon_results, ["forecast_population"])
    _, sku_horizon_segment_summary = summarize_sku_horizon_results(sku_horizon_results, ["sku_segment"])
    population_fold_summary, population_model_summary = summarize_population_rows(enriched_forecasts, "forecast_population")
    segment_fold_summary, segment_model_summary = summarize_population_rows(enriched_forecasts, "sku_segment")
    best_model = best_model_by_window(model_summary)
    best_model_by_population = _best_by_group(
        population_model_summary,
        ["window_label", "test_months", "gap_months", "forecast_population"],
        "selection_score",
    )
    best_model_by_segment = _best_by_group(
        segment_model_summary,
        ["window_label", "test_months", "gap_months", "sku_segment"],
        "selection_score",
    )
    known_history_model_summary = population_model_summary.loc[
        population_model_summary["forecast_population"].eq(PREDICTABLE_FORECAST_POPULATION)
    ].copy()
    best_known_history_by_window = _best_by_group(
        known_history_model_summary,
        ["window_label", "test_months", "gap_months"],
        "selection_score",
    )
    best_sku_horizon_model_by_window = _best_by_group(
        sku_horizon_model_summary,
        ["window_label", "test_months", "gap_months"],
        "horizon_sku_wmape_percent",
    )
    metric_suite_model_summary = build_metric_suite_model_summary(
        model_summary,
        monthly_total_model_summary,
        sku_horizon_model_summary,
    )
    metric_suite_population_summary = pd.DataFrame()

    lag_comparison_agent_report = run_lag_comparison_agent(metric_suite_model_summary)
    monthly_total_agent_report = run_monthly_total_phase1_agent(monthly_total_model_summary)
    sku_horizon_agent_report = run_sku_horizon_phase1_agent(sku_horizon_model_summary)
    metric_decision_agent_report = run_metric_decision_phase1_agent(metric_suite_model_summary)
    population_strategy_agent_report = run_best_summary_agent(
        population_model_summary,
        "forecast_population",
        "Population Strategy Agent",
    )
    segment_strategy_agent_recommendations = run_best_summary_agent(
        segment_model_summary,
        "sku_segment",
        "Segment Strategy Agent",
    )

    return {
        "forecasts": enriched_forecasts,
        "model_summary": model_summary,
        "fold_model_summary": fold_model_summary,
        "monthly_totals": monthly_totals,
        "monthly_total_fold_summary": monthly_total_fold_summary,
        "monthly_total_model_summary": monthly_total_model_summary,
        "sku_horizon_results": sku_horizon_results,
        "sku_horizon_fold_summary": sku_horizon_fold_summary,
        "sku_horizon_model_summary": sku_horizon_model_summary,
        "sku_horizon_population_summary": sku_horizon_population_summary,
        "sku_horizon_segment_summary": sku_horizon_segment_summary,
        "best_sku_horizon_model_by_window": best_sku_horizon_model_by_window,
        "population_fold_summary": population_fold_summary,
        "population_model_summary": population_model_summary,
        "best_model_by_population": best_model_by_population,
        "segment_fold_summary": segment_fold_summary,
        "segment_model_summary": segment_model_summary,
        "best_model_by_segment": best_model_by_segment,
        "known_history_model_summary": known_history_model_summary,
        "best_known_history_by_window": best_known_history_by_window,
        "metric_suite_model_summary": metric_suite_model_summary,
        "metric_suite_population_summary": metric_suite_population_summary,
        "best_model": best_model,
        "lag_comparison_agent_report": lag_comparison_agent_report,
        "monthly_total_agent_report": monthly_total_agent_report,
        "sku_horizon_agent_report": sku_horizon_agent_report,
        "metric_decision_agent_report": metric_decision_agent_report,
        "population_strategy_agent_report": population_strategy_agent_report,
        "segment_strategy_agent_recommendations": segment_strategy_agent_recommendations,
    }

def best_model_by_window(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return model_summary.copy()
    return (
        model_summary.sort_values(["window_label", "wmape_percent", "model"])
        .groupby("window_label", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def write_lumpy_outputs(
    root: Path | None,
    model_data: pd.DataFrame,
    splits: pd.DataFrame,
    forecasts: pd.DataFrame,
    external_inventory: pd.DataFrame,
    feature_inventory: pd.DataFrame | None = None,
    phase1_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    phase1_tables = phase1_tables or build_lumpy_phase1_tables(model_data, splits, forecasts)

    outputs = {
        "model_data": paths["tables"] / "lumpy_model_data.csv",
        "splits": paths["tables"] / "lumpy_backtest_splits.csv",
        "forecasts": paths["tables"] / "lumpy_backtest_forecasts.csv",
        "model_summary": paths["tables"] / "lumpy_model_summary.csv",
        "fold_model_summary": paths["tables"] / "lumpy_fold_model_summary.csv",
        "monthly_totals": paths["tables"] / "lumpy_monthly_total_results.csv",
        "monthly_total_fold_summary": paths["tables"] / "lumpy_monthly_total_fold_summary.csv",
        "monthly_total_model_summary": paths["tables"] / "lumpy_monthly_total_model_summary.csv",
        "sku_horizon_results": paths["tables"] / "lumpy_sku_horizon_results.csv",
        "sku_horizon_fold_summary": paths["tables"] / "lumpy_sku_horizon_fold_summary.csv",
        "sku_horizon_model_summary": paths["tables"] / "lumpy_sku_horizon_model_summary.csv",
        "sku_horizon_population_summary": paths["tables"] / "lumpy_sku_horizon_population_summary.csv",
        "sku_horizon_segment_summary": paths["tables"] / "lumpy_sku_horizon_segment_summary.csv",
        "best_sku_horizon_model_by_window": paths["tables"] / "lumpy_best_sku_horizon_model_by_window.csv",
        "population_fold_summary": paths["tables"] / "lumpy_population_fold_summary.csv",
        "population_model_summary": paths["tables"] / "lumpy_population_model_summary.csv",
        "best_model_by_population": paths["tables"] / "lumpy_best_model_by_population.csv",
        "segment_fold_summary": paths["tables"] / "lumpy_segment_fold_summary.csv",
        "segment_model_summary": paths["tables"] / "lumpy_segment_model_summary.csv",
        "best_model_by_segment": paths["tables"] / "lumpy_best_model_by_segment.csv",
        "known_history_model_summary": paths["tables"] / "lumpy_known_history_model_summary.csv",
        "best_known_history_by_window": paths["tables"] / "lumpy_best_known_history_by_window.csv",
        "metric_suite_model_summary": paths["tables"] / "lumpy_metric_suite_model_summary.csv",
        "metric_suite_population_summary": paths["tables"] / "lumpy_metric_suite_population_summary.csv",
        "best_model": paths["tables"] / "lumpy_best_model_by_window.csv",
        "external_inventory": paths["tables"] / "lumpy_selected_external_features.csv",
        "feature_inventory": paths["tables"] / "lumpy_feature_inventory.csv",
        "lag_comparison_agent_report": paths["tables"] / "lumpy_agent_lag_comparison_report.csv",
        "monthly_total_agent_report": paths["tables"] / "lumpy_agent_monthly_total_report.csv",
        "sku_horizon_agent_report": paths["tables"] / "lumpy_agent_sku_horizon_report.csv",
        "metric_decision_agent_report": paths["tables"] / "lumpy_agent_metric_decision_report.csv",
        "population_strategy_agent_report": paths["tables"] / "lumpy_agent_population_strategy_report.csv",
        "segment_strategy_agent_recommendations": paths["tables"] / "lumpy_agent_segment_strategy_recommendations.csv",
    }
    model_data.to_csv(outputs["model_data"], index=False)
    splits.to_csv(outputs["splits"], index=False)
    phase1_tables["forecasts"].to_csv(outputs["forecasts"], index=False)
    for key, output_path in outputs.items():
        if key in {"model_data", "splits", "forecasts", "external_inventory", "feature_inventory"}:
            continue
        table = phase1_tables.get(key)
        if table is not None:
            table.to_csv(output_path, index=False)
    external_inventory.to_csv(outputs["external_inventory"], index=False)
    if feature_inventory is not None:
        feature_inventory.to_csv(outputs["feature_inventory"], index=False)
    return outputs
