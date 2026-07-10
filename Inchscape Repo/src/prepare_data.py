"""Build clean, model-ready collision-demand CSV files from the raw sales extract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "chile_suzuki_historical_sales.csv"
CHUNK_SIZE = 100_000
KEEP_COLUMNS = ["ts_id", "Date", "value", "collision_flag", "Country", "Brand", "Channel", "REGION"]
VARIANTS = {"collision_flag_only", "all_sku_history"}


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Standardise the fields used by the analytical data set."""
    chunk = chunk.rename(columns={"ts_id": "sku_id", "Date": "month", "value": "demand"}).copy()
    chunk["sku_id"] = chunk["sku_id"].astype(str)
    chunk["month"] = pd.to_datetime(chunk["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    chunk["demand"] = pd.to_numeric(chunk["demand"], errors="coerce").fillna(0.0)
    flag = chunk["collision_flag"].fillna("").astype(str).str.strip().str.upper()
    chunk["is_collision"] = flag.str.contains("COLLISION") & ~flag.str.contains("NON")
    return chunk


def find_collision_skus() -> set[str]:
    """Return every SKU that is collision-flagged in at least one source row."""
    sku_ids: set[str] = set()
    for chunk in pd.read_csv(RAW_FILE, usecols=["ts_id", "collision_flag"], chunksize=CHUNK_SIZE):
        flag = chunk["collision_flag"].fillna("").astype(str).str.strip().str.upper()
        mask = flag.str.contains("COLLISION") & ~flag.str.contains("NON")
        sku_ids.update(chunk.loc[mask, "ts_id"].astype(str))
    return sku_ids


def make_sku_profile(variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ADI/CV² classifications in a memory-conscious first pass."""
    totals: dict[str, list[float]] = {}
    quality = {"rows": 0, "invalid_dates": 0, "negative_demand_rows": 0, "duplicate_sku_month_rows": 0}
    seen_pairs: set[tuple[str, pd.Timestamp]] = set()

    collision_skus = find_collision_skus() if variant == "all_sku_history" else set()
    for raw in pd.read_csv(RAW_FILE, usecols=KEEP_COLUMNS, chunksize=CHUNK_SIZE):
        data = clean_chunk(raw)
        data = data.loc[data["sku_id"].isin(collision_skus)] if variant == "all_sku_history" else data.loc[data["is_collision"]]
        quality["rows"] += len(data)
        quality["invalid_dates"] += int(data["month"].isna().sum())
        quality["negative_demand_rows"] += int((data["demand"] < 0).sum())
        valid = data.dropna(subset=["month"])
        pairs = list(zip(valid["sku_id"], valid["month"]))
        quality["duplicate_sku_month_rows"] += sum(pair in seen_pairs for pair in pairs)
        seen_pairs.update(pairs)

        grouped = valid.groupby("sku_id")["demand"].agg(
            months="size", positive_months=lambda s: (s > 0).sum(), total_demand="sum",
            positive_sum=lambda s: s[s > 0].sum(), positive_sum_sq=lambda s: (s[s > 0] ** 2).sum(),
        )
        for sku, row in grouped.iterrows():
            values = totals.setdefault(str(sku), [0, 0, 0.0, 0.0, 0.0])
            values[0] += int(row.months)
            values[1] += int(row.positive_months)
            values[2] += float(row.total_demand)
            values[3] += float(row.positive_sum)
            values[4] += float(row.positive_sum_sq)

    profile = pd.DataFrame.from_dict(
        totals, orient="index", columns=["total_months", "months_with_demand", "total_demand", "positive_demand_sum", "positive_demand_sum_sq"]
    ).rename_axis("sku_id").reset_index()
    profile["average_demand_interval"] = profile["total_months"] / profile["months_with_demand"].replace(0, np.nan)
    n = profile["months_with_demand"]
    sample_variance = (profile["positive_demand_sum_sq"] - (profile["positive_demand_sum"] ** 2 / n)) / (n - 1)
    profile["squared_coefficient_of_variation"] = sample_variance / ((profile["positive_demand_sum"] / n) ** 2)
    profile.loc[n <= 1, "squared_coefficient_of_variation"] = np.nan

    adi, cv2 = profile["average_demand_interval"], profile["squared_coefficient_of_variation"]
    profile["demand_type"] = np.select(
        [n.eq(0), adi.le(1.32) & (cv2.le(0.49) | cv2.isna()), adi.gt(1.32) & (cv2.le(0.49) | cv2.isna()), adi.le(1.32) & cv2.gt(0.49)],
        ["No demand observed", "Smooth", "Intermittent", "Erratic"], default="Lumpy",
    )
    profile["zero_month_share"] = 1 - profile["months_with_demand"] / profile["total_months"]
    quality["selection_variant"] = variant
    quality["collision_skus_identified"] = len(collision_skus) if collision_skus else profile["sku_id"].nunique()
    quality_df = pd.DataFrame([{"metric": key, "value": value} for key, value in quality.items()])
    return profile, quality_df


def write_segment_files(profile: pd.DataFrame, output_dir: Path, variant: str) -> None:
    """Make a second pass and write one lightweight CSV for each demand segment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classifications = profile.set_index("sku_id")["demand_type"]
    names = ["Smooth", "Intermittent", "Erratic", "Lumpy", "No demand observed"]
    paths = {name: output_dir / f"collision_sales_{name.lower().replace(' ', '_')}.csv" for name in names}
    for path in paths.values():
        path.unlink(missing_ok=True)

    first_write = {name: True for name in names}
    collision_skus = find_collision_skus() if variant == "all_sku_history" else set()
    for raw in pd.read_csv(RAW_FILE, usecols=KEEP_COLUMNS, chunksize=CHUNK_SIZE):
        data = clean_chunk(raw)
        data = data.loc[data["sku_id"].isin(collision_skus)] if variant == "all_sku_history" else data.loc[data["is_collision"]]
        data["demand_type"] = data["sku_id"].map(classifications)
        data = data.rename(columns={"is_collision": "row_is_collision"})
        data = data[["sku_id", "month", "demand", "demand_type", "row_is_collision", "collision_flag", "Country", "Brand", "Channel", "REGION"]]
        for name, part in data.groupby("demand_type", dropna=False):
            if pd.isna(name):
                continue
            part.to_csv(paths[str(name)], index=False, mode="w" if first_write[str(name)] else "a", header=first_write[str(name)])
            first_write[str(name)] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="collision_flag_only")
    args = parser.parse_args()
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw sales file not found: {RAW_FILE}")
    output_dir = ROOT / "data" / "processed" / args.variant
    table_dir = ROOT / "results" / args.variant / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    profile, quality = make_sku_profile(args.variant)
    profile.sort_values("sku_id").to_csv(table_dir / "sku_demand_profile.csv", index=False)
    quality.to_csv(table_dir / "data_quality_summary.csv", index=False)
    write_segment_files(profile, output_dir, args.variant)
    print(f"Created demand-segment CSVs in {output_dir}")
    print(f"Classified {len(profile):,} collision SKUs.")


if __name__ == "__main__":
    main()
