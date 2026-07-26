"""
Comprehensive erratic demand comparison: all models with tuned vs untuned hyperparameters.
Chronos will be added once installation completes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load baseline erratic results (untuned)
baseline_file = Path("results/tables/rolling_evaluation_erratic_summary.csv")
baseline_df = pd.read_csv(baseline_file)
baseline_df["tuning"] = "Untuned (Baseline)"
baseline_df["hyperparameters"] = "Default"

# Load tuned erratic results
tuned_file = Path("results/tables/rolling_evaluation_erratic_tuned_summary.csv")
tuned_df = pd.read_csv(tuned_file)
tuned_df["tuning"] = "Tuned"
tuned_df["hyperparameters"] = "Grid Search CV (Training Set)"

# Combine
all_results = pd.concat([baseline_df, tuned_df], ignore_index=True)

# Reorder columns for clarity
cols = ["model", "tuning", "mean_wmape", "std_wmape", "min_wmape", "max_wmape", 
        "final_rolling_3m", "num_months", "hyperparameters"]
all_results = all_results[cols]

# Sort by model and tuning
all_results = all_results.sort_values(["model", "tuning"]).reset_index(drop=True)

print("="*120)
print("ERRATIC DEMAND: MODEL COMPARISON (BASELINE vs TUNED)")
print("="*120)
print(all_results.to_string(index=False))

print(f"\n{'='*120}")
print("IMPROVEMENT SUMMARY (Tuned - Baseline)")
print(f"{'='*120}")

for model in baseline_df["model"].unique():
    baseline = baseline_df[baseline_df["model"] == model].iloc[0]
    tuned_rows = tuned_df[tuned_df["model"] == model]
    
    if len(tuned_rows) == 0:
        print(f"\n{model.upper()}: Not included in tuning (baseline only)")
        continue
    
    tuned = tuned_rows.iloc[0]
    
    improvement = tuned["mean_wmape"] - baseline["mean_wmape"]
    pct_improvement = (improvement / baseline["mean_wmape"]) * 100
    
    print(f"\n{model.upper()}:")
    print(f"  Mean WMAPE:  {baseline['mean_wmape']:7.2f}% → {tuned['mean_wmape']:7.2f}% (Δ {improvement:+7.2f}%, {pct_improvement:+.1f}%)")
    print(f"  Best Month:  {baseline['min_wmape']:7.2f}% → {tuned['min_wmape']:7.2f}% (Δ {tuned['min_wmape']-baseline['min_wmape']:+7.2f}%)")
    print(f"  Worst Month: {baseline['max_wmape']:7.2f}% → {tuned['max_wmape']:7.2f}% (Δ {tuned['max_wmape']-baseline['max_wmape']:+7.2f}%)")

# Save comprehensive results
output_file = Path("results/tables/erratic_all_models_comparison.csv")
all_results.to_csv(output_file, index=False)
print(f"\n{'='*120}")
print(f"Results saved to: {output_file}")
print(f"{'='*120}")

# Also create a monthly detailed comparison
print("\nLoading monthly detail files for comprehensive comparison...")

baseline_monthly = pd.read_csv(Path("results/tables/rolling_evaluation_erratic_monthly.csv"))
baseline_monthly["tuning"] = "Baseline"

tuned_monthly = pd.read_csv(Path("results/tables/rolling_evaluation_erratic_tuned_monthly.csv"))
tuned_monthly["tuning"] = "Tuned"

monthly_all = pd.concat([baseline_monthly, tuned_monthly], ignore_index=True)
monthly_all = monthly_all.sort_values(["model", "eval_month", "tuning"]).reset_index(drop=True)

monthly_file = Path("results/tables/erratic_all_models_monthly.csv")
monthly_all.to_csv(monthly_file, index=False)
print(f"Monthly details saved to: {monthly_file}")

print("\n✅ Comprehensive erratic comparison complete!")
print("\nNext: Run Chronos evaluation once installation is complete.")
