from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_BLOCK_STARTS = pd.to_datetime(["2024-11-01", "2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01", "2026-02-01"])


def contract_checks(forecasts: pd.DataFrame, expected_skus: int = 690, expected_actual: float = 9111.0) -> pd.DataFrame:
    checks = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    add("sku_coverage", forecasts.sku_id.nunique() == expected_skus, f"observed={forecasts.sku_id.nunique()} expected={expected_skus}")
    add("row_coverage", len(forecasts) == expected_skus * 6, f"observed={len(forecasts)} expected={expected_skus*6}")
    counts = forecasts.groupby("sku_id").block_number.nunique()
    add("six_blocks_per_sku", counts.eq(6).all(), f"nonconforming={int((~counts.eq(6)).sum())}")
    duplicates = forecasts.duplicated(["sku_id", "block_number"]).sum()
    add("no_route_overlap", duplicates == 0, f"duplicate_rows={duplicates}")
    starts = pd.DatetimeIndex(pd.to_datetime(forecasts.block_start).drop_duplicates().sort_values())
    add("official_block_dates", starts.equals(pd.DatetimeIndex(EXPECTED_BLOCK_STARTS)), f"observed={list(starts.strftime('%Y-%m-%d'))}")
    add("target_reconciliation", abs(float(forecasts.target.sum()) - expected_actual) < 1e-9, f"observed={forecasts.target.sum()} expected={expected_actual}")
    add("nonnegative_forecasts", forecasts.forecast.ge(0).all(), f"negative_rows={int(forecasts.forecast.lt(0).sum())}")
    return pd.DataFrame(checks)


def evidence_checks(project_root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.itertuples():
        evidence = project_root / item.evidence_path
        rows.append({"segment": item.segment, "source_notebook": item.source_notebook, "evidence_path": item.evidence_path, "evidence_exists": evidence.exists(), "selected_before_final": bool(item.selected_before_final), "cutoff_safe_population": bool(item.cutoff_safe_population), "official_contract_compatible": bool(item.official_contract_compatible), "external_features_used": bool(item.external_features_used), "passed": evidence.exists() and bool(item.selected_before_final) and bool(item.cutoff_safe_population) and bool(item.official_contract_compatible)})
    return pd.DataFrame(rows)
