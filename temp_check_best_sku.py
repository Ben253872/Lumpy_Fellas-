import pandas as pd

best = pd.read_csv("Inchscape Repo/results/tables/best_model_per_sku.csv")
erratic = best[best['demand_type'] == 'erratic']
print(f"Total rows: {len(best)}")
print(f"Erratic rows: {len(erratic)}")
print(f"\nErratic WMAPE stats:")
print(erratic['wmape_percent'].describe())
print(f"\nMean erratic WMAPE: {erratic['wmape_percent'].mean():.6f}%")
print(f"\nFirst 5:")
print(erratic.head())
