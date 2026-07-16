"""Build publication-safe ANAC, MOP/INE and CMF vehicle activity signals.

The output is monthly even when a source is not.  CMF is semiannual, so its
features are step functions that change only after the official release date.
No future semester or source month is backfilled into an earlier forecast date.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT if (ROOT / "data" / "external").exists() else ROOT / "lumpy_fellas_reapply_backup_2026-07-14"
EXTERNAL = BASE / "data" / "external"
SOURCE = EXTERNAL / "External source files"
MOP_DIR = SOURCE / "MOP monthly traffic"
ANAC_DIR = SOURCE / "ANAC monthly market reports"
CACHE = EXTERNAL / "api_cache"

MOP_INDEX_URL = "https://vialidad.mop.gob.cl/peaje-y-pasadas-vehiculares/"
ANAC_INDEX_URL = "https://www.anac.cl/category/estudio-de-mercado/"
CMF_DASHBOARD_URL = "https://www.cmfchile.cl/portal/estadisticas/626/w4-propertyvalue-46411.html"

MONTH_NAMES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
SUZUKI_MODELS = [
    "GRAND VITARA", "BALENO HB", "NUEVO ALTO", "S-PRESSO", "S-CROSS",
    "SWIFT", "CELERIO", "FRONX", "JIMNY", "VITARA", "DZIRE", "ERTIGA",
    "IGNIS", "XL7", "ALTO",
]


def ascii_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().upper()


def parse_int(value: str) -> int:
    return int(re.sub(r"[^0-9]", "", value))


def filename_month(path: Path) -> pd.Timestamp:
    year = int(path.name[:4])
    normalized = ascii_text(path.name)
    for name, month in MONTH_NAMES.items():
        if ascii_text(name) in normalized:
            return pd.Timestamp(year, month, 1)
    match = re.match(r"20\d{2}_(\d{2})", path.name)
    if match:
        return pd.Timestamp(year, int(match.group(1)), 1)
    raise ValueError(f"Cannot determine month from {path.name}")


def total_row(sheet: pd.DataFrame) -> list[float]:
    for _, row in sheet.iterrows():
        first = next((v for v in row.tolist() if pd.notna(v)), None)
        if ascii_text(first).rstrip(".") == "TOTAL":
            return [float(v) for v in row.tolist() if isinstance(v, (int, float, np.number)) and pd.notna(v)]
    raise ValueError("TOTAL row not found")


def build_mop_observed() -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for path in sorted(MOP_DIR.glob("*.xls")):
        date = filename_month(path)
        excel = pd.ExcelFile(path)
        selected = [excel.sheet_names[i] for i in (0, 1, 4, 7) if i < len(excel.sheet_names)]
        light = heavy = total = 0.0
        used = 0
        for sheet_name in selected:
            values = total_row(pd.read_excel(path, sheet_name=sheet_name, header=None))
            if len(values) < 3:
                continue
            sheet_total = values[-1]
            # First two categories are cars/light vehicles; penultimate is motorcycles.
            sheet_light = values[0] + values[1] + (values[-2] if len(values) >= 4 else 0)
            total += sheet_total
            light += sheet_light
            heavy += max(sheet_total - sheet_light, 0)
            used += 1
        rows.append({
            "date": date,
            "mop_observed_total_passages": total,
            "mop_observed_light_passages": light,
            "mop_observed_heavy_passages": heavy,
            "mop_observed_light_share": light / total if total else np.nan,
            "mop_observed_plazas_used": used,
            "source_file": path.name,
        })
    raw = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    raw["mop_source_month_invalid"] = raw["mop_observed_total_passages"].le(0).astype(int)
    invalid = raw["mop_source_month_invalid"].eq(1)
    raw.loc[invalid, [c for c in raw if c.startswith("mop_observed_")]] = np.nan
    full_index = pd.date_range(raw.date.min(), raw.date.max(), freq="MS")
    observed = raw.set_index("date").reindex(full_index).rename_axis("date").reset_index()
    observed["mop_source_month_missing"] = observed["source_file"].isna().astype(int)
    observed["mop_source_month_invalid"] = observed["mop_source_month_invalid"].fillna(0).astype(int)
    numeric = [c for c in observed if c.startswith("mop_observed_")]
    observed[numeric] = observed[numeric].interpolate(method="linear", limit=2, limit_direction="both")
    observed["mop_observed_yoy"] = observed["mop_observed_total_passages"].pct_change(12)
    observed["mop_observed_rolling3"] = observed["mop_observed_total_passages"].rolling(3, min_periods=1).mean()
    return observed


def find_market_total(pages: list[str], year: int) -> float:
    for text in pages:
        if "RESULTADOS EN" not in ascii_text(text) or "LIVIANOS Y MEDIANOS" not in ascii_text(text):
            continue
        match = re.search(rf"\n{year}\s+[^\n]*\n([\d.]+)\s*\n", text)
        if match:
            value = parse_int(match.group(1))
            if 5_000 <= value <= 100_000:
                return float(value)
        for candidate in re.findall(r"(?m)^\s*([\d.]{4,})\s*$", text):
            value = parse_int(candidate)
            if 5_000 <= value <= 100_000:
                return float(value)
    for text in pages[:3]:
        normalized = ascii_text(text)
        candidates = re.findall(r"(?:COMERCIALIZAR|COMERCIALIZARON|VENDIERON)[A-Z ]{0,80}?(\d[\d.]*) UNIDADES", normalized)
        for candidate in candidates:
            value = parse_int(candidate)
            if 5_000 <= value <= 100_000:
                return float(value)
    return np.nan


def parse_anac_report(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    date = filename_month(path)
    pdf = PdfReader(str(path))
    # ANAC's narrative/market summary is at the front and model rankings are
    # consistently in the middle. Pypdf extracts the embedded Power BI text
    # directly, avoiding slow page rendering.
    summary_pages = [pdf.pages[i].extract_text() or "" for i in range(1, min(5, len(pdf.pages)))]
    ranking_pages = [pdf.pages[i].extract_text() or "" for i in range(5, min(17, len(pdf.pages)))]
    market_total = find_market_total(summary_pages, date.year)
    model_rows: list[dict[str, object]] = []
    model_pattern = "|".join(re.escape(model) for model in sorted(SUZUKI_MODELS, key=len, reverse=True))
    regex = re.compile(rf"(?P<model>{model_pattern})\s+SUZUKI\s+(?P<month>[\d.]+)\s+(?P<cum>[\d.]+)", re.I)
    for text in ranking_pages:
        normalized = ascii_text(text)
        if "RANKING MODELOS MAS VENDIDOS" not in normalized or "PASAJEROS Y SUV" not in normalized:
            continue
        for match in regex.finditer(normalized):
            model_rows.append({
                "date": date,
                "model": ascii_text(match.group("model")),
                "monthly_units": parse_int(match.group("month")),
                "reported_cumulative_units": parse_int(match.group("cum")),
                "source_file": path.name,
            })
    models = pd.DataFrame(model_rows).drop_duplicates(["date", "model"]) if model_rows else pd.DataFrame()
    suzuki_units = float(models.monthly_units.sum()) if not models.empty else 0.0
    summary = {
        "date": date,
        "anac_observed_market_units": market_total,
        "anac_observed_suzuki_top20_units": suzuki_units,
        "anac_observed_suzuki_top20_model_count": int(len(models)),
        "anac_observed_suzuki_top20_market_share": suzuki_units / market_total if market_total else np.nan,
        "source_file": path.name,
    }
    return summary, model_rows


def build_anac_observed() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    models: list[dict[str, object]] = []
    for path in sorted(ANAC_DIR.glob("*.pdf")):
        summary, rows = parse_anac_report(path)
        summaries.append(summary)
        models.extend(rows)
    observed = pd.DataFrame(summaries).sort_values("date").drop_duplicates("date", keep="last")
    model_mix = pd.DataFrame(models).sort_values(["date", "model"]).drop_duplicates(["date", "model"], keep="last")
    if model_mix.empty:
        return observed, model_mix
    dates = pd.date_range(observed.date.min(), observed.date.max(), freq="MS")
    grid = pd.MultiIndex.from_product([dates, sorted(model_mix.model.unique())], names=["date", "model"]).to_frame(index=False)
    grid = grid.merge(model_mix, on=["date", "model"], how="left")
    grid["monthly_units"] = grid["monthly_units"].fillna(0)
    by_model = grid.groupby("model", group_keys=False)["monthly_units"]
    grid["fleet_proxy_age_0_12m"] = by_model.transform(lambda s: s.rolling(12, min_periods=1).sum()) * 0.98
    grid["fleet_proxy_age_13_36m"] = by_model.transform(lambda s: s.shift(12).rolling(24, min_periods=1).sum()) * 0.94
    grid["fleet_proxy_age_37_60m"] = by_model.transform(lambda s: s.shift(36).rolling(24, min_periods=1).sum()) * 0.85
    monthly_suzuki = grid.groupby("date")["monthly_units"].transform("sum")
    grid["share_of_listed_suzuki_units"] = np.where(monthly_suzuki > 0, grid.monthly_units / monthly_suzuki, 0)
    market = observed.set_index("date")["anac_observed_market_units"]
    grid["share_of_total_market"] = grid.monthly_units / grid.date.map(market)
    fleet = grid.groupby("date")[["fleet_proxy_age_0_12m", "fleet_proxy_age_13_36m", "fleet_proxy_age_37_60m"]].sum().reset_index()
    fleet.columns = ["date", "anac_observed_fleet_proxy_age_0_12m", "anac_observed_fleet_proxy_age_13_36m", "anac_observed_fleet_proxy_age_37_60m"]
    observed = observed.merge(fleet, on="date", how="left")
    return observed, grid


def load_or_build_anac() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_path = CACHE / "monthly_anac_suzuki_observed_2021_2026.csv"
    mix_path = EXTERNAL / "monthly_suzuki_model_mix.csv"
    source_files = list(ANAC_DIR.glob("*.pdf"))
    cache_is_current = (
        observed_path.exists()
        and mix_path.exists()
        and source_files
        and min(observed_path.stat().st_mtime, mix_path.stat().st_mtime)
        >= max(path.stat().st_mtime for path in source_files)
    )
    if cache_is_current:
        observed = pd.read_csv(observed_path, parse_dates=["date"])
        model_mix = pd.read_csv(mix_path, parse_dates=["date"])
        print("Using cached ANAC report extraction")
        return observed, model_mix
    return build_anac_observed()


def cmf_semesters() -> pd.DataFrame:
    # Only values stated in official CMF releases are populated. Missing historical
    # insured-fleet values are intentionally left null.
    rows = [
        ("2022H1", "2023-06-29", np.nan, np.nan, 70.0),
        ("2022H2", "2023-06-29", 100324, np.nan, 76.0),
        ("2023H1", "2023-10-10", 95921, 2002125, 74.0),
        ("2023H2", "2024-04-10", 91840, 2179187, 63.0),
        ("2024H1", "2024-10-30", 83540, 2157069, 60.0),
        ("2024H2", "2025-04-08", 80664, 2128128, 55.0),
        ("2025H1", "2025-10-20", 82697, 2194573, 55.0),
        ("2025H2", "2026-04-15", 89084, 2230000, 52.3),
    ]
    frame = pd.DataFrame(rows, columns=["period", "release_date", "damaged_insured_vehicles", "insured_vehicles", "repair_delay_days"])
    frame["release_date"] = pd.to_datetime(frame.release_date)
    frame["claim_rate"] = frame.damaged_insured_vehicles / frame.insured_vehicles
    return frame


def publication_safe_monthly(observed: pd.DataFrame, prefix: str, lag_months: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    source = observed.copy().sort_values("date").set_index("date")
    numeric = [c for c in source if c.startswith(prefix + "_observed_")]
    out = pd.DataFrame({"date": dates})
    for col in numeric:
        short = col.replace(prefix + "_observed_", "")
        out[f"{prefix}_available_last_{short}"] = (out.date - pd.offsets.MonthBegin(lag_months)).map(source[col])
        history = source[col]
        expected = []
        for target in out.date:
            cutoff = target - pd.offsets.MonthBegin(lag_months)
            values = history[(history.index < cutoff) & (history.index.month == target.month)].tail(3)
            expected.append(values.mean() if len(values) else np.nan)
        out[f"{prefix}_available_expected3y_{short}"] = expected
    return out


def build_cmf_monthly(dates: pd.DatetimeIndex) -> pd.DataFrame:
    semesters = cmf_semesters().sort_values("release_date")
    rows = []
    for date in dates:
        available = semesters[semesters.release_date <= date]
        if available.empty:
            rows.append({"date": date})
            continue
        latest = available.iloc[-1]
        rows.append({
            "date": date,
            "cmf_available_damaged_insured_vehicles": latest.damaged_insured_vehicles,
            "cmf_available_insured_vehicles": latest.insured_vehicles,
            "cmf_available_claim_rate": latest.claim_rate,
            "cmf_available_repair_delay_days": latest.repair_delay_days,
            "cmf_available_report_age_months": (date.year - latest.release_date.year) * 12 + date.month - latest.release_date.month,
        })
    return pd.DataFrame(rows)


def build_sku_model_map(model_mix: pd.DataFrame) -> pd.DataFrame:
    demand_files = sorted((ROOT / "data" / "collision_demand_databases").glob("collision_*_demand.csv"))
    if not demand_files:
        demand_files = sorted((ROOT / "data" / "processed" / "all_sku_history").glob("collision_sales_*.csv"))
    if not demand_files:
        raise FileNotFoundError("Could not locate collision demand CSVs for the Suzuki SKU-model map.")
    parts = []
    for path in demand_files:
        frame = pd.read_csv(path, usecols=["sku_id", "MATERIAL_DESCRIPTION", "SUBFAMILY_DESCRIPTION"])
        parts.append(frame.drop_duplicates("sku_id"))
    skus = pd.concat(parts, ignore_index=True).drop_duplicates("sku_id")
    text = (skus.MATERIAL_DESCRIPTION.fillna("") + " " + skus.SUBFAMILY_DESCRIPTION.fillna("")).map(ascii_text)
    available_models = sorted(model_mix.model.unique(), key=len, reverse=True)
    skus["suzuki_model"] = [next((m for m in available_models if m in value), np.nan) for value in text]
    skus["mapping_method"] = np.where(skus.suzuki_model.notna(), "description_keyword_exact", "unmapped")
    return skus[["sku_id", "suzuki_model", "mapping_method"]]


def merge_handoff(features: pd.DataFrame) -> pd.DataFrame:
    path = EXTERNAL / "monthly_external_features.csv"
    existing = pd.read_csv(path)
    existing["date"] = pd.to_datetime(existing.date)
    drop_prefixes = ("mop_available_", "anac_available_", "cmf_available_")
    existing = existing[[c for c in existing if not c.startswith(drop_prefixes)]]
    merged = existing.merge(features, on="date", how="left")
    new_cols = [c for c in features if c != "date"]
    merged[new_cols] = merged[new_cols].ffill().bfill()
    merged.to_csv(path, index=False)
    return merged


def update_metadata(feature_columns: list[str]) -> None:
    registry_path = EXTERNAL / "external_source_registry.csv"
    registry = pd.read_csv(registry_path)
    remove_ids = {
        "cl_road_safety_annual_1972_2024",
        "cl_conaset_month_occurrence_2000_2025",
        "cl_conaset_month_day_occurrence_2000_2025",
        "cl_conaset_month_hour_occurrence_2000_2025",
        "anac_suzuki_monthly_model_sales",
        "mop_monthly_vehicle_passages",
        "cmf_vehicle_repairs",
        "cl_conaset_monthly_collision_signals",
        "cl_anac_suzuki_monthly_model_sales",
        "cl_mop_ine_monthly_vehicle_passages",
        "cl_cmf_insured_vehicle_repairs",
    }
    if "source_id" in registry:
        registry = registry[~registry.source_id.isin(remove_ids)]
    additions = pd.DataFrame([
        {
            "source_id": "cl_conaset_monthly_collision_signals",
            "source_name": "CONASET monthly collision, casualty, day and hour signals",
            "file_path": str(SOURCE / "CONASET_month_occurrence_2000_2025.xlsx"),
            "file_type": "xlsx", "domain": "road_safety", "country": "Chile", "region": "national",
            "granularity": "monthly", "start_period": "2000-01", "end_period": "2025-12",
            "forecast_signal_category": "collision_frequency",
            "possible_target_relationship": "monthly collision volume and timing mix drive collision-part repair demand",
            "forecasting_usefulness": "high",
            "usefulness_reason": "True monthly counts plus weekend, night and peak-hour shares replace repeated annual context.",
            "known_limitations": "Three official workbooks are combined and refreshed annually; use the publication-aware bridge.",
            "join_keys": "country, year, month", "target_model_use": "publication-safe monthly regressor", "status": "active",
        },
        {
            "source_id": "cl_anac_suzuki_monthly_model_sales",
            "source_name": "ANAC Suzuki monthly model registrations and fleet-age proxy",
            "file_path": str(ANAC_DIR), "file_type": "pdf_bundle", "domain": "vehicle_market",
            "country": "Chile", "region": "national", "granularity": "monthly",
            "start_period": "2021-01", "end_period": "2026-06", "forecast_signal_category": "vehicle_model_population",
            "possible_target_relationship": "Suzuki model sales and fleet age allocate demand to model-linked parts",
            "forecasting_usefulness": "medium", "usefulness_reason": "Monthly model units and an age-weighted installed-fleet proxy.",
            "known_limitations": "Top-20 coverage is partial and SKU-model matching is incomplete.",
            "join_keys": "country, year, month", "target_model_use": "publication-safe monthly regressor and SKU allocation context", "status": "materialised",
        },
        {
            "source_id": "cl_mop_ine_monthly_vehicle_passages",
            "source_name": "MOP Vialidad monthly vehicle passages with INE transport context",
            "file_path": str(MOP_DIR), "file_type": "xls_bundle", "domain": "road_traffic",
            "country": "Chile", "region": "selected_stations", "granularity": "monthly",
            "start_period": "2021-01", "end_period": "2026-04", "forecast_signal_category": "vehicle_exposure",
            "possible_target_relationship": "light and heavy vehicle passages proxy kilometres driven and parts exposure",
            "forecasting_usefulness": "high", "usefulness_reason": "Observed monthly traffic is the strongest incremental vehicle-activity signal in the check.",
            "known_limitations": "Selected stations; missing and invalid source months are handled by QA rules.",
            "join_keys": "country, year, month", "target_model_use": "publication-safe monthly exposure regressor", "status": "materialised",
        },
        {
            "source_id": "cl_cmf_insured_vehicle_repairs",
            "source_name": "CMF insured vehicle damage and repair activity",
            "file_path": str(CACHE / "cmf_vehicle_repair_semesters.csv"), "file_type": "csv", "domain": "insured_vehicle_repairs",
            "country": "Chile", "region": "national", "granularity": "semiannual_to_monthly_step",
            "start_period": "2023-H2", "end_period": "2025-H2", "forecast_signal_category": "repair_activity",
            "possible_target_relationship": "damaged insured vehicles and repair delay connect accidents to repair workload",
            "forecasting_usefulness": "medium", "usefulness_reason": "Published insured-damage aggregates provide direct repair-activity context.",
            "known_limitations": "Semiannual releases represented as release-safe monthly steps; granular dashboard fields are not yet exported.",
            "join_keys": "country, year, month", "target_model_use": "publication-safe monthly step regressor", "status": "materialised",
        },
    ])
    for col in registry.columns:
        if col not in additions:
            additions[col] = ""
    pd.concat([registry, additions[registry.columns]], ignore_index=True).to_csv(registry_path, index=False)

    inventory_path = EXTERNAL / "feature_inventory_all_sources.csv"
    inventory = pd.read_csv(inventory_path)
    if "source_id" in inventory.columns:
        inventory["source_id"] = inventory["source_id"].replace({
            "cl_conaset_monthly_combined_2000_2025": "cl_conaset_monthly_collision_signals"
        })
    if "source_name" in inventory.columns:
        conaset_mask = inventory.get("source_id", pd.Series(index=inventory.index, dtype=str)).eq(
            "cl_conaset_monthly_collision_signals"
        )
        inventory.loc[conaset_mask, "source_name"] = "CONASET monthly collision, casualty, day and hour signals"
    feature_name_column = "feature" if "feature" in inventory.columns else "standard_feature_name"
    source_details = {
        "mop": ("cl_mop_ine_monthly_vehicle_passages", "MOP Vialidad monthly vehicle passages with INE transport context", "vehicle_exposure"),
        "anac": ("cl_anac_suzuki_monthly_model_sales", "ANAC Suzuki monthly model registrations and fleet-age proxy", "vehicle_model_population"),
        "cmf": ("cl_cmf_insured_vehicle_repairs", "CMF insured vehicle damage and repair activity", "repair_activity"),
    }
    inventory = inventory[
        ~inventory[feature_name_column].astype(str).str.startswith(("mop_available_", "anac_available_", "cmf_available_"))
    ]
    new_rows = []
    for feature in feature_columns:
        family = feature.split("_", 1)[0]
        source_id, source_name, category = source_details[family]
        row = {column: "" for column in inventory.columns}
        row.update({
            "source_id": source_id,
            "source_name": source_name,
            "table_name": "monthly_external_features",
            "original_column_name": feature,
            "standard_feature_name": feature,
            "feature": feature,
            "pandas_dtype": "float64",
            "granularity": "monthly",
            "country": "Chile",
            "join_keys": "date, year, month",
            "forecast_signal_category": category,
            "target_model_use": "publication-safe monthly external regressor",
        })
        new_rows.append(row)
    inventory = pd.concat([inventory, pd.DataFrame(new_rows)[inventory.columns]], ignore_index=True)
    inventory.to_csv(inventory_path, index=False)

    materialised_path = EXTERNAL / "materialised_feature_sources.csv"
    materialised = pd.read_csv(materialised_path)
    obsolete_source_ids = {
        "cl_road_safety_annual_1972_2024",
        "cl_conaset_monthly_combined_2000_2025",
        "cl_conaset_monthly_collision_signals",
        "cl_anac_suzuki_monthly_model_sales",
        "cl_mop_ine_monthly_vehicle_passages",
        "cl_cmf_insured_vehicle_repairs",
        "anac_suzuki_monthly_model_sales",
        "mop_monthly_vehicle_passages",
        "cmf_vehicle_repairs",
    }
    materialised = materialised[~materialised.source_id.isin(obsolete_source_ids)]
    handoff_columns = pd.read_csv(EXTERNAL / "monthly_external_features.csv", nrows=1).columns.tolist()
    family_prefixes = {
        "cl_conaset_monthly_collision_signals": "conaset_available_",
        "cl_anac_suzuki_monthly_model_sales": "anac_available_",
        "cl_mop_ine_monthly_vehicle_passages": "mop_available_",
        "cl_cmf_insured_vehicle_repairs": "cmf_available_",
    }
    source_rows = []
    for source_id, prefix in family_prefixes.items():
        columns = [column for column in handoff_columns if column.startswith(prefix)]
        row = {column: "" for column in materialised.columns}
        row.update({
            "source_id": source_id,
            "default_handoff_scope": "national_monthly_publication_safe",
            "optional_regional_scope": "not_available",
            "monthly_rows_default": len(pd.read_csv(EXTERNAL / "monthly_external_features.csv", usecols=["date"])),
            "monthly_rows_regional_optional": 0,
            "feature_columns": str(columns),
        })
        source_rows.append(row)
    materialised = pd.concat([materialised, pd.DataFrame(source_rows)[materialised.columns]], ignore_index=True)
    materialised.to_csv(materialised_path, index=False)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    mop = build_mop_observed()
    anac, model_mix = load_or_build_anac()
    dates = pd.date_range("2021-01-01", "2026-12-01", freq="MS")
    mop_safe = publication_safe_monthly(mop, "mop", 2, dates)
    anac_safe = publication_safe_monthly(anac, "anac", 2, dates)
    cmf_safe = build_cmf_monthly(dates)
    features = mop_safe.merge(anac_safe, on="date", how="outer").merge(cmf_safe, on="date", how="outer")
    handoff = merge_handoff(features)

    mop.to_csv(CACHE / "monthly_mop_traffic_observed_2021_2026.csv", index=False)
    anac.to_csv(CACHE / "monthly_anac_suzuki_observed_2021_2026.csv", index=False)
    model_mix.to_csv(EXTERNAL / "monthly_suzuki_model_mix.csv", index=False)
    cmf_semesters().to_csv(CACHE / "cmf_vehicle_repair_semesters.csv", index=False)
    features.to_csv(CACHE / "monthly_vehicle_activity_forecast_safe_features_2021_2026.csv", index=False)
    build_sku_model_map(model_mix).to_csv(EXTERNAL / "suzuki_sku_model_mapping.csv", index=False)
    update_metadata([c for c in features if c != "date"])

    quality = pd.DataFrame([
        ("mop_source_rows", len(mop), len(mop) >= 60),
        ("mop_missing_source_months", int(mop.mop_source_month_missing.sum()), int(mop.mop_source_month_missing.sum()) <= 6),
        ("mop_invalid_source_months", int(mop.mop_source_month_invalid.sum()), int(mop.mop_source_month_invalid.sum()) <= 2),
        ("anac_report_months", len(anac), len(anac) >= 60),
        ("anac_months_with_suzuki_models", int((anac.anac_observed_suzuki_top20_model_count > 0).sum()), int((anac.anac_observed_suzuki_top20_model_count > 0).sum()) >= 40),
        ("anac_months_with_market_total", int(anac.anac_observed_market_units.notna().sum()), int(anac.anac_observed_market_units.notna().sum()) >= 30),
        ("suzuki_model_mix_rows", len(model_mix), len(model_mix) > 100),
        ("forecast_safe_rows", len(features), len(features) == 72),
        ("handoff_duplicate_dates", int(handoff.date.duplicated().sum()), handoff.date.duplicated().sum() == 0),
        ("new_feature_columns", len(features.columns) - 1, len(features.columns) > 10),
    ], columns=["check", "value", "passed"])
    quality.to_csv(EXTERNAL / "monthly_vehicle_activity_quality_report.csv", index=False)
    print(quality.to_string(index=False))
    if not quality.passed.all():
        raise RuntimeError("Vehicle activity QA failed")


if __name__ == "__main__":
    main()
