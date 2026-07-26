import pandas as pd

# Check erratic_demand_chronos_only
df1 = pd.read_csv("Inchscape Repo/results/tables/erratic_demand_chronos_only.csv")
print("erratic_demand_chronos_only.csv shape:", df1.shape)
print("Columns:", df1.columns.tolist())
print("First 3 rows:")
print(df1.head(3))
print("\n")

# Check if there's a multi per-sku file
df2 = pd.read_csv("Inchscape Repo/results/tables/rolling_evaluation_erratic_chronos_multi_summary.csv")
print("rolling_evaluation_erratic_chronos_multi_summary.csv shape:", df2.shape)
print("Columns:", df2.columns.tolist())
print("First 3 rows:")
print(df2.head(3))
