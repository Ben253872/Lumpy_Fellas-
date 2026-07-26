"""Integrate Chronos foundation-model results into erratic-demand comparison tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tables"
CHRONOS_PREFIX = "chronos"

CHRONOS_RUNS = [
    (
        "chronos_t5_tiny",
        RESULTS_DIR / "rolling_evaluation_erratic_chronos_t5_tiny_summary.csv",
        RESULTS_DIR / "rolling_evaluation_erratic_chronos_t5_tiny_monthly.csv",
    ),
    (
        "chronos_2_erratic_features",
        RESULTS_DIR / "rolling_evaluation_erratic_chronos_erratic_features_summary.csv",
        RESULTS_DIR / "rolling_evaluation_erratic_chronos_erratic_features_monthly.csv",
    ),
    (
        "chronos_2_erratic_features_plus_external",
        RESULTS_DIR / "rolling_evaluation_erratic_chronos_erratic_features_plus_external_summary.csv",
        RESULTS_DIR / "rolling_evaluation_erratic_chronos_erratic_features_plus_external_monthly.csv",
    ),
]


def _is_chronos_row(model_value: object) -> bool:
    return str(model_value).startswith(CHRONOS_PREFIX)


def _load_chronos_summary_row(model_name: str, summary_path: Path) -> pd.DataFrame | None:
    if not summary_path.exists():
        print(f"Skipping missing summary file: {summary_path.name}")
        return None

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        print(f"Skipping empty summary file: {summary_path.name}")
        return None

    summary_row = summary_df.iloc[0]
    return pd.DataFrame(
        {
            "model": [model_name],
            "tuning_type": ["Foundation Model"],
            "mean_wmape": [summary_row["mean_wmape"]],
            "std_wmape": [summary_row["std_wmape"]],
            "min_wmape": [summary_row["min_wmape"]],
            "max_wmape": [summary_row["max_wmape"]],
            "final_rolling_3m": [summary_row["final_rolling_3m"]],
            "num_months": [int(summary_row["num_months"])],
        }
    )


base_results_path = RESULTS_DIR / "erratic_demand_all_models_complete.csv"
base_results_df = pd.read_csv(base_results_path)
base_results_df = base_results_df.loc[~base_results_df["model"].apply(_is_chronos_row)].copy()

chronos_summary_rows: list[pd.DataFrame] = []
for model_name, summary_path, _ in CHRONOS_RUNS:
    loaded_row = _load_chronos_summary_row(model_name, summary_path)
    if loaded_row is not None:
        chronos_summary_rows.append(loaded_row)

if chronos_summary_rows:
    combined = pd.concat([base_results_df] + chronos_summary_rows, ignore_index=True)
else:
    combined = base_results_df.copy()

combined = combined.sort_values("mean_wmape").reset_index(drop=True)
combined["rank"] = range(1, len(combined) + 1)
combined = combined[["rank", "model", "tuning_type", "mean_wmape", "std_wmape", "min_wmape", "max_wmape", "final_rolling_3m", "num_months"]]
combined.to_csv(base_results_path, index=False)

print("Updated: erratic_demand_all_models_complete.csv")
print()
print(combined.to_string(index=False))

monthly_path = RESULTS_DIR / "erratic_all_models_monthly.csv"
monthly_df = pd.read_csv(monthly_path)
monthly_df = monthly_df.loc[~monthly_df["model"].apply(_is_chronos_row)].copy()

chronos_monthly_rows: list[pd.DataFrame] = []
for model_name, _, monthly_file in CHRONOS_RUNS:
    if not monthly_file.exists():
        print(f"Skipping missing monthly file: {monthly_file.name}")
        continue

    model_monthly = pd.read_csv(monthly_file)
    if model_monthly.empty:
        print(f"Skipping empty monthly file: {monthly_file.name}")
        continue

    model_monthly = model_monthly.copy()
    model_monthly["model"] = model_name
    if "tuning" not in model_monthly.columns:
        model_monthly["tuning"] = "Foundation Model"
    chronos_monthly_rows.append(model_monthly[["model", "eval_month", "month_num", "wmape", "evaluation_rows", "unique_skus", "tuning"]])

if chronos_monthly_rows:
    combined_monthly = pd.concat([monthly_df] + chronos_monthly_rows, ignore_index=True)
else:
    combined_monthly = monthly_df.copy()

combined_monthly = combined_monthly.sort_values(["model", "month_num"]).reset_index(drop=True)
combined_monthly.to_csv(monthly_path, index=False)

print()
print("Updated: erratic_all_models_monthly.csv")
print(f"Total rows: {len(combined_monthly)}")
print()
print("Chronos variants included:")
for model_name, _, _ in CHRONOS_RUNS:
    print(f"- {model_name}")
