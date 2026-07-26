import pandas as pd

best = pd.read_csv("Inchscape Repo/results/tables/best_model_per_sku.csv")
lumpy = best[best['demand_type'] == 'lumpy'].head(5)
print("Sample lumpy SKUs:")
print(lumpy[['sku_id', 'demand_type', 'model']].to_string())
