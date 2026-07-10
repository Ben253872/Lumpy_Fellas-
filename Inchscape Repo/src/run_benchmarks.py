"""Run the original baseline models and save all model outputs as CSV files."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.benchmarks import croston, naive, rolling_origin_validation, sba, seasonal_naive, simple_moving_average, tsb, wmape  # noqa: E402


MODELS = {
    "naive": naive,
    "seasonal_naive_12": seasonal_naive,
    "sma_3": lambda train, months: simple_moving_average(train, months, 3),
    "sma_6": lambda train, months: simple_moving_average(train, months, 6),
    "croston": croston,
    "sba": sba,
    "tsb": tsb,
}
SHORT_HISTORY_TRAIN_MONTHS = 18
STANDARD_TRAIN_MONTHS = 36


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["collision_flag_only", "all_sku_history"], default="collision_flag_only")
    args = parser.parse_args()
    processed_dir = ROOT / "data" / "processed" / args.variant
    output = ROOT / "results" / args.variant / "tables"
    output.mkdir(parents=True, exist_ok=True)
    metrics, forecasts = [], []
    for demand_type in ("smooth", "intermittent", "erratic", "lumpy"):
        data = pd.read_csv(processed_dir / f"collision_sales_{demand_type}.csv", parse_dates=["month"])
        initial_train_months = STANDARD_TRAIN_MONTHS if data["month"].nunique() > STANDARD_TRAIN_MONTHS else SHORT_HISTORY_TRAIN_MONTHS
        for model_name, model in MODELS.items():
            # The supplied collision subset has 28 months (Jan 2024–Apr 2026),
            # so the original 36-month starting window cannot produce a test fold.
            result = rolling_origin_validation(data, model, initial_train_months=initial_train_months)
            result["demand_type"] = demand_type.title()
            result["model"] = model_name
            forecasts.append(result)
            metrics.append({"variant": args.variant, "demand_type": demand_type.title(), "model": model_name, "initial_train_months": initial_train_months, "wmape_percent": wmape(result["month"], result["demand"], result["forecast"]), "forecast_rows": len(result)})
            print(f"{demand_type.title():12} {model_name:20} complete")
    pd.concat(forecasts, ignore_index=True).to_csv(output / "benchmark_forecasts.csv", index=False)
    pd.DataFrame(metrics).sort_values(["demand_type", "wmape_percent"]).to_csv(output / "benchmark_metrics.csv", index=False)


if __name__ == "__main__":
    main()
