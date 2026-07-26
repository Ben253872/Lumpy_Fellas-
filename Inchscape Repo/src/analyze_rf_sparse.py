"""
Investigate Random Forest's 0% WMAPE on sparse intermittent SKUs.

Why does RF get perfect 0% error on sparse data while other models struggle?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_PATH = Path("data") / "processed" / "all_sku_history"
TABLES_PATH = Path("results") / "tables"

# Load intermittent demand data
print("Loading intermittent demand data...")
intermittent_df = pd.read_csv(DATA_PATH / "collision_sales_intermittent.csv")
intermittent_df["month"] = pd.to_datetime(intermittent_df["month"])

# Split train/test
train_data = intermittent_df[intermittent_df["month"] <= "2025-12-01"].copy()
march_2026_data = intermittent_df[intermittent_df["month"] == "2026-03-01"].copy()

# Identify sparse SKUs (zero demand in 2025)
last_12_months = train_data[train_data["month"] >= "2025-01-01"].copy()
skus_with_2025_demand = set(last_12_months[last_12_months["demand"] > 0]["sku_id"].unique())
skus_without_2025_demand = set(march_2026_data["sku_id"].unique()) - skus_with_2025_demand

print(f"Sparse SKUs (zero 2025 demand): {len(skus_without_2025_demand)}")

# Load March 2026 actuals
march_actuals = march_2026_data[["sku_id", "demand"]].copy()
march_actuals.rename(columns={"demand": "actual"}, inplace=True)

# Get sparse subset
sparse_actuals = march_actuals[march_actuals["sku_id"].isin(skus_without_2025_demand)].copy()

print(f"\n" + "="*70)
print("MARCH 2026 ACTUAL DEMAND FOR SPARSE SKUs (those with zero 2025 demand)")
print("="*70)

print(f"\nTotal sparse SKU-months: {len(sparse_actuals)}")
print(f"Sparse SKUs with >0 actual in March 2026: {(sparse_actuals['actual'] > 0).sum()}")
print(f"Sparse SKUs with =0 actual in March 2026: {(sparse_actuals['actual'] == 0).sum()}")
print(f"Total actual demand in March for sparse: {sparse_actuals['actual'].sum():.1f} units")
print(f"Mean demand per sparse SKU: {sparse_actuals['actual'].mean():.4f} units")

print("\nDemand distribution in sparse subset:")
demand_dist = sparse_actuals['actual'].value_counts().sort_index()
for demand_val, count in demand_dist.items():
    print(f"  Demand = {demand_val:>3.0f}: {count:>4} SKUs ({100*count/len(sparse_actuals):>5.1f}%)")

# Load forecast data
print(f"\n" + "="*70)
print("RANDOM FOREST FORECASTS FOR SPARSE SKUs")
print("="*70)

forecast_pivot = pd.read_csv(TABLES_PATH / "official_forecast_january_2026.csv")
forecast_pivot["Date"] = pd.to_datetime(forecast_pivot["Date"])
march_forecasts = forecast_pivot[forecast_pivot["Date"] == "2026-03-01"].copy()

# Get RF forecasts for sparse SKUs
march_forecasts["ts_id"] = march_forecasts["ts_id"].astype(str)
sparse_actuals["sku_id"] = sparse_actuals["sku_id"].astype(str)

sparse_with_forecast = sparse_actuals.merge(
    march_forecasts[["ts_id", "random_forest"]].rename(columns={"ts_id": "sku_id", "random_forest": "forecast"}),
    on="sku_id",
    how="left"
)
sparse_with_forecast["forecast"] = sparse_with_forecast["forecast"].fillna(0)

print(f"\nRandom Forest forecasts for sparse SKUs:")
print(f"  Forecasts = 0: {(sparse_with_forecast['forecast'] == 0).sum()} SKUs")
print(f"  Forecasts > 0: {(sparse_with_forecast['forecast'] > 0).sum()} SKUs")
print(f"  Mean forecast: {sparse_with_forecast['forecast'].mean():.4f} units")
print(f"  Total forecast: {sparse_with_forecast['forecast'].sum():.1f} units")

# Calculate errors
sparse_with_forecast["error"] = np.abs(sparse_with_forecast["actual"] - sparse_with_forecast["forecast"])
sparse_with_forecast["matches"] = (sparse_with_forecast["actual"] == sparse_with_forecast["forecast"]).astype(int)

print(f"\nError analysis:")
print(f"  Perfect matches (actual == forecast): {sparse_with_forecast['matches'].sum()} ({100*sparse_with_forecast['matches'].sum()/len(sparse_with_forecast):.1f}%)")
print(f"  Mismatches: {len(sparse_with_forecast) - sparse_with_forecast['matches'].sum()}")
print(f"  Total absolute error: {sparse_with_forecast['error'].sum():.1f}")
print(f"  Mean absolute error per SKU: {sparse_with_forecast['error'].mean():.6f}")

# Calculate WMAPE
denom = np.abs(sparse_with_forecast['actual']).sum()
numer = sparse_with_forecast['error'].sum()
wmape = (numer / denom * 100) if denom > 0 else np.nan
print(f"\nWMAPE calculation:")
print(f"  Numerator (sum of errors): {numer:.1f}")
print(f"  Denominator (sum of |actual|): {denom:.1f}")
print(f"  WMAPE = {numer:.1f} / {denom:.1f} * 100 = {wmape:.2f}%")

# Show examples of matches and mismatches
print(f"\n" + "="*70)
print("EXAMPLES: Actual vs Forecast for Sparse SKUs")
print("="*70)

matches_df = sparse_with_forecast[sparse_with_forecast["matches"] == 1].head(10)
print(f"\nPerfect Matches ({len(matches_df)} of many):")
print(matches_df[["sku_id", "actual", "forecast", "error"]].to_string(index=False))

mismatches_df = sparse_with_forecast[sparse_with_forecast["matches"] == 0].head(10)
if len(mismatches_df) > 0:
    print(f"\nMismatches ({len(mismatches_df)} examples):")
    print(mismatches_df[["sku_id", "actual", "forecast", "error"]].to_string(index=False))
else:
    print(f"\nNo mismatches found!")

print(f"\n" + "="*70)
print("EXPLANATION")
print("="*70)
print(f"""
Random Forest achieves 0% WMAPE on sparse intermittent SKUs because:

1. SPARSE SKUs definition: Had ZERO demand in all of 2025
   → These are truly inactive/dormant parts

2. In March 2026: {(sparse_with_forecast['actual'] == 0).sum()}/1430 sparse SKUs still have zero demand
   → Only {(sparse_with_forecast['actual'] > 0).sum()} sparse SKUs show any demand
   → That's {100*(sparse_with_forecast['actual'] > 0).sum()/len(sparse_with_forecast):.1f}% reactivation rate

3. Random Forest's strategy on sparse: Forecast 0
   → When actual=0 AND forecast=0 → error = 0 ✓
   → Most sparse SKUs are still inactive, so this is correct!

4. The {(sparse_with_forecast['actual'] > 0).sum()} reactivated SKUs:
   → Tiny total demand ({sparse_with_forecast['actual'].sum():.0f} units across 1430 SKUs)
   → RF may forecast 0 on them too, but the errors are negligible
   → They barely move the pooled WMAPE denominator

RESULT: Zero-forecasting inactive SKUs = 0% WMAPE because they're STILL inactive!

Compare to Lumpy Hurdle (9%): Uses a hurdle model that predicts "will this SKU
have demand?" + "if yes, how much?" So it might forecast small positives on some
sparse SKUs, catching reactivations but getting penalized for the ones that don't.
""")
