from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
WEEKEND_DAYS = {"sabado", "domingo"}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}
PEAK_HOURS = {7, 8, 9, 17, 18, 19, 20}

BASE_VALUE_COLUMNS = [
    "collisions",
    "fatalities",
    "serious_injuries",
    "moderate_injuries",
    "minor_injuries",
    "total_injuries",
]


def ascii_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(character for character in normalized if not unicodedata.combining(character)).lower().strip()


def read_sheet(path: Path, year: int) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=str(year), header=None)


def parse_month_totals(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        if not str(sheet_name).isdigit():
            continue
        year = int(sheet_name)
        sheet = pd.read_excel(path, sheet_name=sheet_name, header=None)
        for _, row in sheet.iterrows():
            month = MONTHS.get(ascii_text(row.iloc[0]))
            if month is None:
                continue
            values = pd.to_numeric(row.iloc[1:7], errors="coerce").to_numpy(dtype=float)
            if np.isnan(values).all():
                continue
            rows.append({"year": year, "month": month, **dict(zip(BASE_VALUE_COLUMNS, values))})
    result = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    result["date"] = pd.to_datetime(dict(year=result["year"], month=result["month"], day=1))
    return result[["date", "year", "month"] + BASE_VALUE_COLUMNS]


def parse_month_day(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, float | int | str]] = []
    total_rows: list[dict[str, float | int]] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        if not str(sheet_name).isdigit():
            continue
        year = int(sheet_name)
        sheet = pd.read_excel(path, sheet_name=sheet_name, header=None)
        current_month: int | None = None
        for _, row in sheet.iterrows():
            first = ascii_text(row.iloc[0])
            if first in MONTHS:
                current_month = MONTHS[first]
            if first.startswith("total "):
                month_name = first.replace("total ", "", 1).strip()
                month = MONTHS.get(month_name)
                values = pd.to_numeric(row.iloc[2:8], errors="coerce").to_numpy(dtype=float)
                if month is not None and not np.isnan(values).all():
                    total_rows.append({"year": year, "month": month, **dict(zip(BASE_VALUE_COLUMNS, values))})
                continue
            day = ascii_text(row.iloc[1])
            if current_month is None or day not in {"lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"}:
                continue
            values = pd.to_numeric(row.iloc[2:8], errors="coerce").to_numpy(dtype=float)
            detail_rows.append({"year": year, "month": current_month, "day": day, **dict(zip(BASE_VALUE_COLUMNS, values))})

    detail = pd.DataFrame(detail_rows)
    totals = pd.DataFrame(total_rows)
    weekend = (
        detail.assign(is_weekend=detail["day"].isin(WEEKEND_DAYS))
        .groupby(["year", "month", "is_weekend"], as_index=False)[BASE_VALUE_COLUMNS]
        .sum()
    )
    all_days = weekend.groupby(["year", "month"], as_index=False)[BASE_VALUE_COLUMNS].sum()
    weekend_only = weekend.loc[weekend["is_weekend"]].drop(columns="is_weekend")
    derived = all_days[["year", "month"]].merge(
        weekend_only[["year", "month", "collisions", "fatalities", "total_injuries"]],
        on=["year", "month"],
        how="left",
        suffixes=("", "_weekend"),
    )
    derived = derived.rename(
        columns={
            "collisions": "weekend_collisions",
            "fatalities": "weekend_fatalities",
            "total_injuries": "weekend_total_injuries",
        }
    )
    base_totals = all_days[["year", "month", "collisions", "fatalities", "total_injuries"]]
    derived = derived.merge(base_totals, on=["year", "month"], how="left", suffixes=("", "_all"))
    derived["weekend_collision_share"] = derived["weekend_collisions"] / derived["collisions"].replace(0, np.nan)
    derived["weekend_fatality_share"] = derived["weekend_fatalities"] / derived["fatalities"].replace(0, np.nan)
    derived["weekend_injury_share"] = derived["weekend_total_injuries"] / derived["total_injuries"].replace(0, np.nan)
    keep = [
        "year", "month", "weekend_collisions", "weekend_fatalities", "weekend_total_injuries",
        "weekend_collision_share", "weekend_fatality_share", "weekend_injury_share",
    ]
    return detail, derived[keep]


def parse_month_hour(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, float | int]] = []
    total_rows: list[dict[str, float | int]] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        if not str(sheet_name).isdigit():
            continue
        year = int(sheet_name)
        sheet = pd.read_excel(path, sheet_name=sheet_name, header=None)
        current_month: int | None = None
        for _, row in sheet.iterrows():
            first = ascii_text(row.iloc[0])
            if first in MONTHS:
                current_month = MONTHS[first]
            if first.startswith("total "):
                month_name = first.replace("total ", "", 1).strip()
                month = MONTHS.get(month_name)
                values = pd.to_numeric(row.iloc[2:8], errors="coerce").to_numpy(dtype=float)
                if month is not None and not np.isnan(values).all():
                    total_rows.append({"year": year, "month": month, **dict(zip(BASE_VALUE_COLUMNS, values))})
                continue
            hour = pd.to_numeric(pd.Series([row.iloc[1]]), errors="coerce").iloc[0]
            if current_month is None or pd.isna(hour) or not 0 <= int(hour) <= 23:
                continue
            values = pd.to_numeric(row.iloc[2:8], errors="coerce").to_numpy(dtype=float)
            detail_rows.append({"year": year, "month": current_month, "hour": int(hour), **dict(zip(BASE_VALUE_COLUMNS, values))})

    detail = pd.DataFrame(detail_rows)
    totals = pd.DataFrame(total_rows)
    all_hours = detail.groupby(["year", "month"], as_index=False)[BASE_VALUE_COLUMNS].sum()
    night = detail.loc[detail["hour"].isin(NIGHT_HOURS)].groupby(["year", "month"], as_index=False)[BASE_VALUE_COLUMNS].sum()
    peak = detail.loc[detail["hour"].isin(PEAK_HOURS)].groupby(["year", "month"], as_index=False)[BASE_VALUE_COLUMNS].sum()
    derived = all_hours[["year", "month", "collisions", "fatalities", "total_injuries"]].copy()
    derived = derived.merge(
        night[["year", "month", "collisions", "fatalities", "total_injuries"]],
        on=["year", "month"], how="left", suffixes=("", "_night"),
    ).merge(
        peak[["year", "month", "collisions", "fatalities", "total_injuries"]],
        on=["year", "month"], how="left", suffixes=("", "_peak"),
    )
    derived = derived.rename(columns={
        "collisions_night": "night_collisions", "fatalities_night": "night_fatalities",
        "total_injuries_night": "night_total_injuries", "collisions_peak": "peak_collisions",
        "fatalities_peak": "peak_fatalities", "total_injuries_peak": "peak_total_injuries",
    })
    derived["night_collision_share"] = derived["night_collisions"] / derived["collisions"].replace(0, np.nan)
    derived["night_fatality_share"] = derived["night_fatalities"] / derived["fatalities"].replace(0, np.nan)
    derived["peak_collision_share"] = derived["peak_collisions"] / derived["collisions"].replace(0, np.nan)
    keep = [
        "year", "month", "night_collisions", "night_fatalities", "night_total_injuries",
        "peak_collisions", "peak_fatalities", "peak_total_injuries", "night_collision_share",
        "night_fatality_share", "peak_collision_share",
    ]
    return detail, derived[keep]


def build_observed_features(month_path: Path, day_path: Path, hour_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    totals = parse_month_totals(month_path)
    day_detail, day = parse_month_day(day_path)
    hour_detail, hour = parse_month_hour(hour_path)
    observed = totals.merge(day, on=["year", "month"], how="left").merge(hour, on=["year", "month"], how="left")
    observed["injuries_per_collision"] = observed["total_injuries"] / observed["collisions"].replace(0, np.nan)
    observed["fatalities_per_collision"] = observed["fatalities"] / observed["collisions"].replace(0, np.nan)

    day_reconciled = (
        day_detail.groupby(["year", "month"], as_index=False)[BASE_VALUE_COLUMNS].sum()
        .merge(totals[["year", "month"] + BASE_VALUE_COLUMNS], on=["year", "month"], suffixes=("_detail", "_base"))
    )
    hour_reconciled = (
        hour_detail.groupby(["year", "month"], as_index=False)[BASE_VALUE_COLUMNS].sum()
        .merge(totals[["year", "month"] + BASE_VALUE_COLUMNS], on=["year", "month"], suffixes=("_detail", "_base"))
    )
    day_difference_cells = float(sum(
        (day_reconciled[f"{column}_detail"] - day_reconciled[f"{column}_base"]).abs().gt(1e-9).sum()
        for column in BASE_VALUE_COLUMNS
    ))
    hour_difference_cells = float(sum(
        (hour_reconciled[f"{column}_detail"] - hour_reconciled[f"{column}_base"]).abs().gt(1e-9).sum()
        for column in BASE_VALUE_COLUMNS
    ))
    day_max_difference = float(max(
        (day_reconciled[f"{column}_detail"] - day_reconciled[f"{column}_base"]).abs().max()
        for column in BASE_VALUE_COLUMNS
    ))
    hour_max_difference = float(max(
        (hour_reconciled[f"{column}_detail"] - hour_reconciled[f"{column}_base"]).abs().max()
        for column in BASE_VALUE_COLUMNS
    ))
    checks = {
        "observed_rows": float(len(observed)),
        "observed_duplicate_months": float(observed.duplicated(["year", "month"]).sum()),
        "observed_missing_months": float(observed.set_index("date").reindex(pd.date_range("2000-01-01", "2025-12-01", freq="MS")).isna().all(axis=1).sum()),
        "month_day_reconciliation_difference_cells": day_difference_cells,
        "month_hour_reconciliation_difference_cells": hour_difference_cells,
        "month_day_reconciliation_max_abs_difference": day_max_difference,
        "month_hour_reconciliation_max_abs_difference": hour_max_difference,
    }
    return observed.sort_values("date").reset_index(drop=True), checks


def build_forecast_safe_features(observed: pd.DataFrame, start_date: str, end_date: str, release_month: int = 7) -> pd.DataFrame:
    months = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="MS")})
    months["year"] = months["date"].dt.year
    months["month"] = months["date"].dt.month
    months["conaset_available_asof_year"] = np.where(
        months["month"] >= release_month,
        months["year"] - 1,
        months["year"] - 2,
    ).astype(int)

    candidate_columns = [
        "collisions", "fatalities", "total_injuries", "serious_injuries",
        "weekend_collision_share", "weekend_fatality_share", "night_collision_share",
        "night_fatality_share", "peak_collision_share", "injuries_per_collision",
        "fatalities_per_collision",
    ]
    records: list[dict[str, float | int | pd.Timestamp]] = []
    for row in months.itertuples(index=False):
        eligible = observed.loc[
            observed["year"].le(row.conaset_available_asof_year)
            & observed["month"].eq(row.month)
        ].sort_values("year")
        record: dict[str, float | int | pd.Timestamp] = {
            "date": row.date,
            "year": row.year,
            "month": row.month,
            "conaset_available_asof_year": row.conaset_available_asof_year,
        }
        recent5 = eligible.tail(5)
        recent3 = eligible.tail(3)
        for column in candidate_columns:
            record[f"conaset_available_expected5y_{column}"] = recent5[column].mean()
            record[f"conaset_available_expected3y_{column}"] = recent3[column].mean()
            record[f"conaset_available_last_{column}"] = eligible[column].iloc[-1] if len(eligible) else np.nan
        if len(eligible) >= 2:
            previous = eligible["collisions"].iloc[-2]
            record["conaset_available_collision_yoy"] = (
                eligible["collisions"].iloc[-1] / previous - 1 if previous else np.nan
            )
        else:
            record["conaset_available_collision_yoy"] = np.nan
        records.append(record)
    return pd.DataFrame(records)


def update_registry(registry_path: Path, source_dir: Path) -> pd.DataFrame:
    columns = [
        "source_id", "source_name", "file_path", "file_type", "domain", "country", "region",
        "granularity", "start_period", "end_period", "forecast_signal_category",
        "possible_target_relationship", "forecasting_usefulness", "usefulness_reason",
        "known_limitations", "join_keys", "target_model_use", "status",
    ]
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
    else:
        registry = pd.DataFrame(columns=columns)
    registry = registry.loc[~registry["source_id"].astype(str).str.startswith("cl_road_safety_annual")].copy()
    registry = registry.loc[~registry["source_id"].astype(str).str.startswith("cl_conaset_month")].copy()
    rows = [
        {
            "source_id": "cl_conaset_month_occurrence_2000_2025",
            "source_name": "CONASET traffic collisions and casualties by month",
            "file_path": str(source_dir / "CONASET_month_occurrence_2000_2025.xlsx"),
            "file_type": "xlsx", "domain": "road_safety", "country": "Chile", "region": "national",
            "granularity": "monthly", "start_period": "2000-01", "end_period": "2025-12",
            "forecast_signal_category": "collision_frequency",
            "possible_target_relationship": "monthly collision volume drives collision-part repair demand",
            "forecasting_usefulness": "high",
            "usefulness_reason": "True monthly collision and casualty counts replace repeated annual context.",
            "known_limitations": "Official workbook is refreshed annually; forecast handoff uses publication-aware historical expectations.",
            "join_keys": "country, year, month", "target_model_use": "publication-safe monthly regressor", "status": "active",
        },
        {
            "source_id": "cl_conaset_month_day_occurrence_2000_2025",
            "source_name": "CONASET traffic collisions by month and day of week",
            "file_path": str(source_dir / "CONASET_month_day_occurrence_2000_2025.xlsx"),
            "file_type": "xlsx", "domain": "road_safety", "country": "Chile", "region": "national",
            "granularity": "month_day_to_monthly", "start_period": "2000-01", "end_period": "2025-12",
            "forecast_signal_category": "collision_timing_mix",
            "possible_target_relationship": "weekend collision mix changes repair severity and demand timing",
            "forecasting_usefulness": "high", "usefulness_reason": "Provides monthly weekend collision and casualty shares.",
            "known_limitations": "National aggregate and annual refresh.", "join_keys": "country, year, month",
            "target_model_use": "publication-safe monthly regressor", "status": "active",
        },
        {
            "source_id": "cl_conaset_month_hour_occurrence_2000_2025",
            "source_name": "CONASET traffic collisions by month and hour",
            "file_path": str(source_dir / "CONASET_month_hour_occurrence_2000_2025.xlsx"),
            "file_type": "xlsx", "domain": "road_safety", "country": "Chile", "region": "national",
            "granularity": "month_hour_to_monthly", "start_period": "2000-01", "end_period": "2025-12",
            "forecast_signal_category": "collision_timing_mix",
            "possible_target_relationship": "night and peak-hour collision mix proxies repair severity",
            "forecasting_usefulness": "high", "usefulness_reason": "Provides monthly night and peak-hour collision shares.",
            "known_limitations": "National aggregate and annual refresh.", "join_keys": "country, year, month",
            "target_model_use": "publication-safe monthly regressor", "status": "active",
        },
    ]
    registry = pd.concat([registry, pd.DataFrame(rows)], ignore_index=True)
    return registry[columns]


def merge_with_existing_handoff(existing_path: Path, safe: pd.DataFrame) -> pd.DataFrame:
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing["date"] = pd.to_datetime(existing["date"])
    else:
        existing = safe[["date", "year", "month"]].copy()
    obsolete = [
        column for column in existing.columns
        if column.startswith("lag_1yr_national_annual_") or column.startswith("conaset_available_")
    ]
    existing = existing.drop(columns=obsolete, errors="ignore")
    safe_columns = [column for column in safe.columns if column not in {"year", "month"}]
    merged = existing.merge(safe[safe_columns], on="date", how="outer")
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    front = ["date", "year", "month"]
    return merged[front + [column for column in merged.columns if column not in front]].sort_values("date")


def build_feature_inventory(existing_path: Path, safe: pd.DataFrame) -> pd.DataFrame:
    if existing_path.exists():
        inventory = pd.read_csv(existing_path)
        inventory = inventory.loc[~inventory["source_id"].astype(str).str.startswith("cl_road_safety_annual")]
        inventory = inventory.loc[~inventory["source_id"].astype(str).str.startswith("cl_conaset_month")]
    else:
        inventory = pd.DataFrame()
    rows = []
    for column in [column for column in safe.columns if column.startswith("conaset_available_")]:
        numeric = pd.to_numeric(safe[column], errors="coerce")
        rows.append({
            "source_id": "cl_conaset_monthly_combined_2000_2025",
            "source_name": "CONASET monthly collision forecast-safe feature bridge",
            "table_name": "monthly_conaset_forecast_safe_features",
            "original_column_name": column,
            "standard_feature_name": column,
            "pandas_dtype": str(numeric.dtype),
            "non_null_count": int(numeric.notna().sum()),
            "numeric_count": int(numeric.notna().sum()),
            "min_numeric": numeric.min(), "max_numeric": numeric.max(),
            "granularity": "monthly", "country": "Chile", "join_keys": "country, year, month",
            "forecast_signal_category": "collision_frequency_and_timing_mix",
            "target_model_use": "publication-safe monthly regressor",
        })
    return pd.concat([inventory, pd.DataFrame(rows)], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-safe monthly CONASET features.")
    parser.add_argument("--external-dir", type=Path, default=Path("lumpy_fellas_reapply_backup_2026-07-14/data/external"))
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-01")
    args = parser.parse_args()

    external_dir = args.external_dir
    source_dir = external_dir / "External source files"
    cache_dir = external_dir / "api_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    month_path = source_dir / "CONASET_month_occurrence_2000_2025.xlsx"
    day_path = source_dir / "CONASET_month_day_occurrence_2000_2025.xlsx"
    hour_path = source_dir / "CONASET_month_hour_occurrence_2000_2025.xlsx"
    observed, checks = build_observed_features(month_path, day_path, hour_path)
    safe = build_forecast_safe_features(observed, args.start_date, args.end_date)

    observed_path = cache_dir / "monthly_conaset_observed_features_2000_2025.csv"
    safe_path = cache_dir / "monthly_conaset_forecast_safe_features_2021_2026.csv"
    observed.to_csv(observed_path, index=False)
    safe.to_csv(safe_path, index=False)

    handoff_path = external_dir / "monthly_external_features.csv"
    handoff = merge_with_existing_handoff(handoff_path, safe)
    handoff.to_csv(handoff_path, index=False)

    registry_path = external_dir / "external_source_registry.csv"
    registry = update_registry(registry_path, source_dir)
    registry.to_csv(registry_path, index=False)

    inventory_path = external_dir / "feature_inventory_all_sources.csv"
    inventory = build_feature_inventory(inventory_path, safe)
    inventory.to_csv(inventory_path, index=False)

    materialised_path = external_dir / "materialised_feature_sources.csv"
    feature_columns = [column for column in safe.columns if column.startswith("conaset_available_")]
    materialised = pd.DataFrame([
        {
            "source_id": "cl_conaset_monthly_combined_2000_2025",
            "default_handoff_scope": "national_monthly_publication_safe",
            "optional_regional_scope": "not_available",
            "monthly_rows_default": len(safe),
            "monthly_rows_regional_optional": 0,
            "feature_columns": json.dumps(feature_columns),
        }
    ])
    if materialised_path.exists():
        old = pd.read_csv(materialised_path)
        old = old.loc[~old["source_id"].astype(str).str.startswith("cl_road_safety_annual")]
        old = old.loc[~old["source_id"].astype(str).str.startswith("cl_conaset_month")]
        materialised = pd.concat([old, materialised], ignore_index=True)
    materialised.to_csv(materialised_path, index=False)

    check_rows = []
    for key, value in checks.items():
        if "duplicate" in key or "missing" in key:
            passed = value == 0
        elif "max_abs_difference" in key:
            passed = value <= 1
        else:
            passed = True
        check_rows.append({"check": key, "value": value, "passed": passed})
    check_rows.extend([
        {"check": "safe_feature_rows", "value": len(safe), "passed": len(safe) == len(pd.date_range(args.start_date, args.end_date, freq="MS"))},
        {"check": "safe_feature_null_cells", "value": int(safe[feature_columns].isna().sum().sum()), "passed": not safe[feature_columns].isna().any().any()},
        {"check": "handoff_duplicate_dates", "value": int(handoff.duplicated("date").sum()), "passed": not handoff.duplicated("date").any()},
        {"check": "annual_collision_columns_remaining", "value": sum(column.startswith("lag_1yr_national_annual_") for column in handoff.columns), "passed": not any(column.startswith("lag_1yr_national_annual_") for column in handoff.columns)},
    ])
    quality = pd.DataFrame(check_rows)
    quality_path = external_dir / "monthly_conaset_quality_report.csv"
    quality.to_csv(quality_path, index=False)
    if not quality["passed"].all():
        raise AssertionError(f"CONASET quality checks failed:\n{quality.loc[~quality['passed']].to_string(index=False)}")

    print(quality.to_string(index=False))
    print(f"Observed features: {observed_path}")
    print(f"Forecast-safe features: {safe_path}")
    print(f"Updated model handoff: {handoff_path}")


if __name__ == "__main__":
    main()
