import pandas as pd
df = pd.read_csv("Inchscape Repo/results/tables/rolling_evaluation_erratic_chronos_erratic_features_plus_external_monthly.csv")
print("Mean WMAPE from monthly file:", df["wmape"].mean())
print("\nAll months:")
for idx, row in df.iterrows():
    print(f"{row['eval_month']}: wmape={row['wmape']:.2f}, scale={row['scale_factor']:.3f}, skus={row['unique_skus']}")
