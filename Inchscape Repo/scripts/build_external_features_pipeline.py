"""Build the complete monthly external feature handoff.

The pipeline combines six source families:
- CONASET monthly collision signals
- ANAC Suzuki monthly model registrations
- MOP monthly vehicle passages (with INE transport context)
- CMF insured vehicle repair activity
- Open-Meteo monthly weather
- Chile monthly calendar features

The four collision and vehicle-activity sources are publication-safe. The old
annual Chile road-safety source is deliberately excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_monthly_conaset_signals import build_forecast_safe_features, build_observed_features
from build_monthly_vehicle_activity_signals import main as build_vehicle_activity_features


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    SCRIPT_ROOT
    if (SCRIPT_ROOT / "data" / "external").exists()
    else SCRIPT_ROOT / "lumpy_fellas_reapply_backup_2026-07-14"
)
EXTERNAL = PROJECT_ROOT / "data" / "external"
SOURCE = EXTERNAL / "External source files"
CACHE = EXTERNAL / "api_cache"

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2026-04-01")

REGISTRY_COLUMNS = [
    "source_id", "source_name", "file_path", "file_type", "domain", "country",
    "region", "granularity", "start_period", "end_period",
    "forecast_signal_category", "possible_target_relationship",
    "forecasting_usefulness", "usefulness_reason", "known_limitations",
    "join_keys", "target_model_use", "status",
]

METADATA_COLUMNS = {
    "country", "region", "source_id", "feature_scope", "feature_semantics",
    "aggregation_rule", "weather_location_weight", "weather_days",
}


def source_registry() -> pd.DataFrame:
    rows = [
        {
            "source_id": "cl_conaset_monthly_collision_signals",
            "source_name": "CONASET monthly collision, casualty, day and hour signals",
            "file_path": str(SOURCE / "CONASET_month_occurrence_2000_2025.xlsx"),
            "file_type": "xlsx_bundle", "domain": "road_safety", "country": "Chile",
            "region": "national", "granularity": "monthly", "start_period": "2000-01",
            "end_period": "2025-12", "forecast_signal_category": "collision_frequency",
            "possible_target_relationship": "Monthly collision volume and timing mix drive collision-part repair demand.",
            "forecasting_usefulness": "high",
            "usefulness_reason": "True monthly counts plus weekend, night and peak-hour shares replace repeated annual context.",
            "known_limitations": "Three workbooks are combined and refreshed annually; the handoff uses publication-safe expectations.",
            "join_keys": "date", "target_model_use": "monthly external regressor", "status": "materialised",
        },
        {
            "source_id": "cl_anac_suzuki_monthly_model_sales",
            "source_name": "ANAC Suzuki monthly model registrations and fleet-age proxy",
            "file_path": str(SOURCE / "ANAC monthly market reports"),
            "file_type": "pdf_bundle", "domain": "vehicle_market", "country": "Chile",
            "region": "national", "granularity": "monthly", "start_period": "2021-01",
            "end_period": "2026-06", "forecast_signal_category": "vehicle_model_population",
            "possible_target_relationship": "Suzuki model sales and fleet age help allocate demand to model-linked parts.",
            "forecasting_usefulness": "medium",
            "usefulness_reason": "Provides monthly model units and an age-weighted installed-fleet proxy.",
            "known_limitations": "ANAC top-20 coverage is partial and SKU-to-model matching is incomplete.",
            "join_keys": "date", "target_model_use": "monthly regressor and SKU allocation context", "status": "materialised",
        },
        {
            "source_id": "cl_mop_ine_monthly_vehicle_passages",
            "source_name": "MOP Vialidad monthly vehicle passages with INE transport context",
            "file_path": str(SOURCE / "MOP monthly traffic"),
            "file_type": "xls_bundle", "domain": "road_traffic", "country": "Chile",
            "region": "selected_stations", "granularity": "monthly", "start_period": "2021-01",
            "end_period": "2026-04", "forecast_signal_category": "vehicle_exposure",
            "possible_target_relationship": "Light and heavy vehicle passages proxy kilometres driven and parts exposure.",
            "forecasting_usefulness": "high",
            "usefulness_reason": "Observed monthly traffic is the strongest incremental vehicle-activity signal in the check.",
            "known_limitations": "Selected stations are used; missing and invalid months are covered by the QA report.",
            "join_keys": "date", "target_model_use": "monthly exposure regressor", "status": "materialised",
        },
        {
            "source_id": "cl_cmf_insured_vehicle_repairs",
            "source_name": "CMF insured vehicle damage and repair activity",
            "file_path": str(CACHE / "cmf_vehicle_repair_semesters.csv"),
            "file_type": "csv", "domain": "insured_vehicle_repairs", "country": "Chile",
            "region": "national", "granularity": "semiannual_to_monthly_step",
            "start_period": "2023-H2", "end_period": "2025-H2",
            "forecast_signal_category": "repair_activity",
            "possible_target_relationship": "Damaged insured vehicles and repair delay connect accidents to repair workload.",
            "forecasting_usefulness": "medium",
            "usefulness_reason": "Published insured-damage totals provide direct repair-activity context.",
            "known_limitations": "Official releases are semiannual, so monthly values are release-safe steps; detailed dashboard fields are not yet exported.",
            "join_keys": "date", "target_model_use": "monthly step regressor", "status": "materialised",
        },
        {
            "source_id": "cl_weather_open_meteo_daily",
            "source_name": "Open-Meteo monthly weather for Chile locations",
            "file_path": str(CACHE / "monthly_weather_features_2021-01-01_2026-04-30.csv"),
            "file_type": "csv", "domain": "weather", "country": "Chile",
            "region": "national_proxy", "granularity": "daily_to_monthly",
            "start_period": "2021-01", "end_period": "2026-04",
            "forecast_signal_category": "weather_collision_risk",
            "possible_target_relationship": "Rain, wind and temperature may affect collision frequency.",
            "forecasting_usefulness": "medium",
            "usefulness_reason": "Monthly weather matches the forecasting grain and is available before demand is observed.",
            "known_limitations": "The national value is an equal-weight average of selected locations.",
            "join_keys": "date", "target_model_use": "monthly external regressor", "status": "materialised",
        },
        {
            "source_id": "cl_public_holidays_nager_date",
            "source_name": "Chile monthly calendar features",
            "file_path": str(CACHE / "monthly_calendar_features_2021-01-01_2026-04-30.csv"),
            "file_type": "csv", "domain": "calendar", "country": "Chile",
            "region": "national", "granularity": "date_to_monthly",
            "start_period": "2021-01", "end_period": "2026-12",
            "forecast_signal_category": "calendar_effect",
            "possible_target_relationship": "Working days and holidays affect driving and repair-shop activity.",
            "forecasting_usefulness": "high",
            "usefulness_reason": "Calendar fields are known in advance and join directly to month.",
            "known_limitations": "Regional holiday exceptions are not included.",
            "join_keys": "date", "target_model_use": "monthly known-ahead regressor", "status": "materialised",
        },
    ]
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def add_date(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "date" not in data.columns:
        data["date"] = pd.to_datetime(
            data["year"].astype(int).astype(str)
            + "-" + data["month"].astype(int).astype(str).str.zfill(2) + "-01"
        )
    else:
        data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    return data


def model_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def build_conaset() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed, checks = build_observed_features(
        SOURCE / "CONASET_month_occurrence_2000_2025.xlsx",
        SOURCE / "CONASET_month_day_occurrence_2000_2025.xlsx",
        SOURCE / "CONASET_month_hour_occurrence_2000_2025.xlsx",
    )
    safe = build_forecast_safe_features(observed, start_date=START_DATE, end_date=END_DATE)
    safe = add_date(safe)
    observed.to_csv(CACHE / "monthly_conaset_observed_features_2000_2025.csv", index=False)
    safe.to_csv(CACHE / "monthly_conaset_forecast_safe_features_2021_2026.csv", index=False)
    quality = pd.DataFrame([{"check": key, "value": value, "passed": True} for key, value in checks.items()])
    quality.to_csv(EXTERNAL / "monthly_conaset_quality_report.csv", index=False)
    return safe, quality


def load_cached_monthly(filename: str) -> pd.DataFrame:
    path = CACHE / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly cache: {path}")
    return add_date(pd.read_csv(path))


def merge_national(conaset: pd.DataFrame, weather: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    tables = [
        conaset[model_columns(conaset)],
        weather[model_columns(weather)],
        calendar[model_columns(calendar)],
    ]
    combined = tables[0]
    for table in tables[1:]:
        value_columns = [column for column in table.columns if column not in {"date", "year", "month"}]
        combined = combined.merge(table[["date"] + value_columns], on="date", how="outer")
    combined = add_date(combined)
    combined = combined.loc[combined["date"].between(START_DATE, END_DATE)].copy()
    feature_columns = [column for column in combined.columns if column not in {"date", "year", "month"}]
    return combined[["date", "year", "month"] + feature_columns].sort_values("date").reset_index(drop=True)


def inventory_for(frame: pd.DataFrame, registry: pd.DataFrame, source_id: str, prefix: str) -> pd.DataFrame:
    profile = registry.set_index("source_id").loc[source_id]
    rows = []
    for column in [name for name in frame.columns if name.startswith(prefix)]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        rows.append({
            "source_id": source_id,
            "source_name": profile.source_name,
            "table_name": "monthly_external_features",
            "original_column_name": column,
            "standard_feature_name": column,
            "pandas_dtype": str(frame[column].dtype),
            "non_null_count": int(frame[column].notna().sum()),
            "numeric_count": int(numeric.notna().sum()),
            "min_numeric": numeric.min() if numeric.notna().any() else np.nan,
            "max_numeric": numeric.max() if numeric.notna().any() else np.nan,
            "granularity": profile.granularity,
            "country": profile.country,
            "join_keys": profile.join_keys,
            "forecast_signal_category": profile.forecast_signal_category,
            "target_model_use": profile.target_model_use,
        })
    return pd.DataFrame(rows)


def write_base_metadata(registry: pd.DataFrame, handoff: pd.DataFrame) -> None:
    registry.to_csv(EXTERNAL / "external_source_registry.csv", index=False)
    inventory = inventory_for(handoff, registry, "cl_conaset_monthly_collision_signals", "conaset_available_")
    for source_id, columns in [
        ("cl_weather_open_meteo_daily", [c for c in handoff if c.startswith(("avg_", "max_", "min_", "weather_"))]),
        ("cl_public_holidays_nager_date", [c for c in handoff if c in {"days_in_month", "weekend_days", "public_holidays", "working_days"}]),
    ]:
        profile = registry.set_index("source_id").loc[source_id]
        for column in columns:
            numeric = pd.to_numeric(handoff[column], errors="coerce")
            inventory.loc[len(inventory)] = {
                "source_id": source_id, "source_name": profile.source_name,
                "table_name": "monthly_external_features", "original_column_name": column,
                "standard_feature_name": column, "pandas_dtype": str(handoff[column].dtype),
                "non_null_count": int(handoff[column].notna().sum()), "numeric_count": int(numeric.notna().sum()),
                "min_numeric": numeric.min(), "max_numeric": numeric.max(), "granularity": profile.granularity,
                "country": profile.country, "join_keys": profile.join_keys,
                "forecast_signal_category": profile.forecast_signal_category, "target_model_use": profile.target_model_use,
            }
    inventory.to_csv(EXTERNAL / "feature_inventory_all_sources.csv", index=False)
    materialised_rows = []
    for source_id, prefix in [
        ("cl_conaset_monthly_collision_signals", "conaset_available_"),
        ("cl_weather_open_meteo_daily", None),
        ("cl_public_holidays_nager_date", None),
    ]:
        if source_id == "cl_weather_open_meteo_daily":
            columns = [c for c in handoff if c.startswith(("avg_", "max_", "min_", "weather_"))]
        elif source_id == "cl_public_holidays_nager_date":
            columns = [c for c in handoff if c in {"days_in_month", "weekend_days", "public_holidays", "working_days"}]
        else:
            columns = [c for c in handoff if c.startswith(prefix)]
        materialised_rows.append({
            "source_id": source_id, "default_handoff_scope": "national_monthly",
            "optional_regional_scope": "not_exported",
            "monthly_rows_default": len(handoff), "monthly_rows_regional_optional": 0,
            "feature_columns": str(columns),
        })
    pd.DataFrame(materialised_rows).to_csv(EXTERNAL / "materialised_feature_sources.csv", index=False)


def final_quality() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    handoff = add_date(pd.read_csv(EXTERNAL / "monthly_external_features.csv"))
    registry = pd.read_csv(EXTERNAL / "external_source_registry.csv")
    inventory = pd.read_csv(EXTERNAL / "feature_inventory_all_sources.csv")
    materialised = pd.read_csv(EXTERNAL / "materialised_feature_sources.csv")
    expected_ids = {
        "cl_conaset_monthly_collision_signals", "cl_anac_suzuki_monthly_model_sales",
        "cl_mop_ine_monthly_vehicle_passages", "cl_cmf_insured_vehicle_repairs",
        "cl_weather_open_meteo_daily", "cl_public_holidays_nager_date",
    }
    prefix_counts = {
        "CONASET feature columns": sum(column.startswith("conaset_available_") for column in handoff),
        "MOP feature columns": sum(column.startswith("mop_available_") for column in handoff),
        "ANAC feature columns": sum(column.startswith("anac_available_") for column in handoff),
        "CMF feature columns": sum(column.startswith("cmf_available_") for column in handoff),
    }
    checks = [
        ("Monthly rows", len(handoff), len(handoff) == 64),
        ("Duplicate months", int(handoff.date.duplicated().sum()), not handoff.date.duplicated().any()),
        ("First month", handoff.date.min().date().isoformat(), handoff.date.min() == START_DATE),
        ("Last month", handoff.date.max().date().isoformat(), handoff.date.max() == END_DATE),
        ("Sources in registry", len(expected_ids & set(registry.source_id)), set(registry.source_id) == expected_ids),
        ("Sources in inventory", len(expected_ids & set(inventory.source_id)), expected_ids.issubset(set(inventory.source_id))),
        ("Sources in source summary", len(expected_ids & set(materialised.source_id)), expected_ids.issubset(set(materialised.source_id))),
        ("Materialised registry sources", int(registry.status.eq("materialised").sum()), registry.status.eq("materialised").all()),
    ]
    checks.extend((name, value, value > 0) for name, value in prefix_counts.items())
    quality = pd.DataFrame(checks, columns=["check", "value", "passed"])
    quality.to_csv(EXTERNAL / "external_features_pipeline_quality_report.csv", index=False)
    if not quality.passed.all():
        failed = quality.loc[~quality.passed, "check"].tolist()
        raise AssertionError(f"External feature pipeline failed: {failed}")
    return handoff, registry, inventory, quality


def run_pipeline() -> dict[str, pd.DataFrame]:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    registry = source_registry()
    conaset, conaset_quality = build_conaset()
    weather = load_cached_monthly("monthly_weather_features_2021-01-01_2026-04-30.csv")
    calendar = load_cached_monthly("monthly_calendar_features_2021-01-01_2026-04-30.csv")
    base_handoff = merge_national(conaset, weather, calendar)
    base_handoff.to_csv(EXTERNAL / "monthly_external_features.csv", index=False)
    write_base_metadata(registry, base_handoff)
    build_vehicle_activity_features()
    handoff, registry, inventory, quality = final_quality()
    vehicle_quality = pd.read_csv(EXTERNAL / "monthly_vehicle_activity_quality_report.csv")
    materialised_sources = pd.read_csv(EXTERNAL / "materialised_feature_sources.csv")

    # Keep review tables in memory and retain only the model-ready feature CSV.
    review_exports = [
        "external_features_pipeline_quality_report.csv",
        "external_source_registry.csv",
        "feature_inventory_all_sources.csv",
        "materialised_feature_sources.csv",
        "monthly_conaset_quality_report.csv",
        "monthly_regional_external_features.csv",
        "monthly_suzuki_model_mix.csv",
        "monthly_vehicle_activity_quality_report.csv",
        "suzuki_sku_model_mapping.csv",
    ]
    for filename in review_exports:
        (EXTERNAL / filename).unlink(missing_ok=True)

    return {
        "registry": registry,
        "handoff": handoff,
        "inventory": inventory,
        "quality": quality,
        "conaset_quality": conaset_quality,
        "vehicle_quality": vehicle_quality,
        "materialised_sources": materialised_sources,
    }


def main() -> None:
    outputs = run_pipeline()
    print(outputs["quality"].to_string(index=False))
    print(f"\nWrote {len(outputs['handoff'])} monthly rows and {len(outputs['handoff'].columns)} columns")
    print(f"Output: {EXTERNAL / 'monthly_external_features.csv'}")


if __name__ == "__main__":
    main()
