import pandas as pd

# Check what data we have
data = pd.read_csv("Inchscape Repo/data/processed/all_sku_history/collision_sales_erratic.csv")
print(f"Erratic data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Date range: {data['month'].min()} to {data['month'].max()}")
print(f"Unique SKUs: {data['sku_id'].nunique()}")
print(f"\nFirst 3 rows:")
print(data.head(3))

# Check summary stats
print(f"\nDemand stats:")
print(data['demand'].describe())
