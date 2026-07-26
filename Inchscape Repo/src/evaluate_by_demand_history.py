"""
Evaluate models on intermittent SKUs split by demand history.

Split 1: SKUs with non-zero demand in last 12 months of training (2025)
Split 2: SKUs with zero demand in last 12 months of training (sparse/inactive)

This matches Laszlo's methodology of focusing on "forecastable" (recent demand) SKUs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_PATH = Path("data") / "processed" / "all_sku_history"
RESULTS_PATH = Path("results")
TABLES_PATH = RESULTS_PATH / "tables"
MODELS_PATH = RESULTS_PATH / "models" / "advanced"

# Load intermittent demand data
print("Loading intermittent demand data...")
intermittent_df = pd.read_csv(DATA_PATH / "collision_sales_intermittent.csv")
intermittent_df["month"] = pd.to_datetime(intermittent_df["month"])

# Split train/test at Dec 2025
train_data = intermittent_df[intermittent_df["month"] <= "2025-12-01"].copy()
march_2026_data = intermittent_df[intermittent_df["month"] == "2026-03-01"].copy()

print(f"Total intermittent SKUs: {intermittent_df['sku_id'].nunique()}")
print(f"Train data: {len(train_data)} rows, {train_data['month'].min()} to {train_data['month'].max()}")
print(f"March 2026 data: {len(march_2026_data)} rows")

# Identify SKUs with non-zero demand in last 12 months (2025)
last_12_months = train_data[train_data["month"] >= "2025-01-01"].copy()
skus_with_2025_demand = set(last_12_months[last_12_months["demand"] > 0]["sku_id"].unique())

skus_without_2025_demand = set(march_2026_data["sku_id"].unique()) - skus_with_2025_demand

print(f"\nSplit by 2025 demand:")
print(f"  SKUs with non-zero 2025 demand (forecastable): {len(skus_with_2025_demand)}")
print(f"  SKUs with zero 2025 demand (sparse/inactive): {len(skus_without_2025_demand)}")
print(f"  Total: {len(skus_with_2025_demand) + len(skus_without_2025_demand)}")

# WMAPE calculation function
def wmape(actual, forecast):
    """Pooled WMAPE across all rows"""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return np.nan
    return np.abs(actual - forecast).sum() / denom * 100

# Load official evaluation results
print("\nLoading official evaluation results...")
sku_details = pd.read_csv(TABLES_PATH / "official_evaluation_sku_details.csv")
forecast_pivot = pd.read_csv(TABLES_PATH / "official_forecast_january_2026.csv")

# Filter to March 2026 forecasts
forecast_pivot["Date"] = pd.to_datetime(forecast_pivot["Date"])
march_forecasts = forecast_pivot[forecast_pivot["Date"] == "2026-03-01"].copy()

# Get all model names (columns in forecast pivot excluding ts_id and Date)
models = [col for col in forecast_pivot.columns if col not in ["ts_id", "Date"]]

# Prepare results
results_forecastable = []
results_sparse = []

print("\nEvaluating by subset...")
print("-" * 70)

for model in models:
    print(f"\n{model.upper()}")
    
    # Get actuals and forecasts for March 2026
    march_data = march_2026_data[["sku_id", "demand"]].copy()
    march_data.rename(columns={"demand": "actual"}, inplace=True)
    march_data["sku_id"] = march_data["sku_id"].astype(str)
    
    # Match with forecasts
    march_forecasts["ts_id"] = march_forecasts["ts_id"].astype(str)
    march_data = march_data.merge(
        march_forecasts[["ts_id", model]].rename(columns={"ts_id": "sku_id", model: "forecast"}),
        on="sku_id",
        how="left"
    )
    march_data["forecast"] = march_data["forecast"].fillna(0)
    
    # Split 1: Forecastable (2025 demand)
    forecastable_subset = march_data[march_data["sku_id"].isin(skus_with_2025_demand)].copy()
    wmape_forecastable = wmape(forecastable_subset["actual"], forecastable_subset["forecast"])
    
    results_forecastable.append({
        "model": model,
        "wmape_percent": wmape_forecastable,
        "sku_count": len(forecastable_subset["sku_id"].unique()),
        "evaluation_rows": len(forecastable_subset),
        "total_actual_demand": forecastable_subset["actual"].sum(),
    })
    
    print(f"  Forecastable (2025 demand):  {wmape_forecastable:.2f}% ({len(forecastable_subset)} rows, {len(forecastable_subset['sku_id'].unique())} SKUs)")
    
    # Split 2: Sparse (no 2025 demand)
    sparse_subset = march_data[march_data["sku_id"].isin(skus_without_2025_demand)].copy()
    wmape_sparse = wmape(sparse_subset["actual"], sparse_subset["forecast"])
    
    results_sparse.append({
        "model": model,
        "wmape_percent": wmape_sparse,
        "sku_count": len(sparse_subset["sku_id"].unique()),
        "evaluation_rows": len(sparse_subset),
        "total_actual_demand": sparse_subset["actual"].sum(),
    })
    
    print(f"  Sparse (no 2025 demand):     {wmape_sparse:.2f}% ({len(sparse_subset)} rows, {len(sparse_subset['sku_id'].unique())} SKUs)")

# Save results
df_forecastable = pd.DataFrame(results_forecastable)
df_sparse = pd.DataFrame(results_sparse)

forecastable_file = TABLES_PATH / "official_evaluation_intermittent_forecastable_2025.csv"
sparse_file = TABLES_PATH / "official_evaluation_intermittent_sparse_2025.csv"

df_forecastable.to_csv(forecastable_file, index=False)
df_sparse.to_csv(sparse_file, index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)
print(f"Forecastable SKUs: {forecastable_file}")
print(df_forecastable.to_string(index=False))
print(f"\nSparse SKUs: {sparse_file}")
print(df_sparse.to_string(index=False))

print("\n" + "=" * 70)
print("COMPARISON: Forecastable vs Sparse Performance")
print("=" * 70)
comparison = pd.merge(
    df_forecastable.rename(columns={"wmape_percent": "forecastable_wmape", "evaluation_rows": "forecastable_rows"}),
    df_sparse.rename(columns={"wmape_percent": "sparse_wmape", "evaluation_rows": "sparse_rows"}),
    on="model"
)
comparison["wmape_gap"] = comparison["sparse_wmape"] - comparison["forecastable_wmape"]
print(comparison[["model", "forecastable_wmape", "sparse_wmape", "wmape_gap"]].to_string(index=False))

print("\nKey insights:")
print(f"  • Laszlo's approach: Top 500 forecastable SKUs → 93.30% WMAPE")
print(f"  • Your forecastable: {len(skus_with_2025_demand)} SKUs with 2025 demand")
print(f"  • Your sparse: {len(skus_without_2025_demand)} SKUs with zero 2025 demand (the hard ones)")
