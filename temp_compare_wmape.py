import pandas as pd

# Check what the aligned script actually produced
sku_wmape = pd.read_csv("Inchscape Repo/results/tables/erratic_chronos2_external_sku_wmape.csv")
print(f"SKU WMAPE file shape: {sku_wmape.shape}")
print(f"Mean wmape_percent: {sku_wmape['wmape_percent'].mean():.6f}%")
print(f"First 5:")
print(sku_wmape[['sku_id', 'wmape_percent']].head())

# Now check what's in best_model
best = pd.read_csv("Inchscape Repo/results/tables/best_model_per_sku.csv")
erratic = best[best['demand_type'] == 'erratic']
erratic_joined = erratic.merge(sku_wmape[['sku_id', 'wmape_percent']], on='sku_id', how='inner', suffixes=('_best', '_sku_file'))
print(f"\nComparison:")
print(f"Rows match: {len(erratic_joined)}")
if len(erratic_joined) > 0:
    print(f"Mean from best file: {erratic['wmape_percent'].mean():.6f}%")
    print(f"Mean from sku file: {sku_wmape['wmape_percent'].mean():.6f}%")
    print(f"Values differ: {(erratic_joined['wmape_percent_best'] != erratic_joined['wmape_percent_sku_file']).sum()} rows")
