import pandas as pd
import numpy as np

sku = pd.read_csv("Inchscape Repo/results/tables/erratic_chronos2_external_sku_wmape.csv")

# What I reported
print("From script output (pooled): 68.906841%")
print(f"Actual pooled recalc: {(sku['abs_error_sum'].sum() / sku['abs_actual_sum'].sum() * 100):.6f}%")

# What's the mean of the per-SKU WMAPE_percent column?
print(f"\nMean of wmape_percent column: {sku['wmape_percent'].mean():.6f}%")

# Are there NaNs?
print(f"NaN count: {sku['wmape_percent'].isna().sum()}")

# What if I exclude NaNs?
non_nan = sku[sku['wmape_percent'].notna()]
print(f"Mean without NaN: {non_nan['wmape_percent'].mean():.6f}%")
print(f"Non-NaN count: {len(non_nan)}")

# Recalculate from abs_error_sum and abs_actual_sum
recalc_wmape = np.where(sku['abs_actual_sum'] > 0, sku['abs_error_sum'] / sku['abs_actual_sum'] * 100, np.nan)
print(f"\nRecalculated wmape from error/actual: {np.nanmean(recalc_wmape):.6f}%")

# Wait - maybe the column values don't match due to precision or something
print(f"\nFirst 5 rows detail:")
print(sku[['sku_id', 'wmape_percent', 'abs_error_sum', 'abs_actual_sum']].head())
for idx, row in sku.head().iterrows():
    recalc = row['abs_error_sum'] / row['abs_actual_sum'] * 100 if row['abs_actual_sum'] > 0 else np.nan
    print(f"SKU {row['sku_id']}: stored={row['wmape_percent']:.6f}, recalc={recalc:.6f}")
