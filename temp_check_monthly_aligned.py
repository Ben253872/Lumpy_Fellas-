import pandas as pd

monthly = pd.read_csv("Inchscape Repo/results/tables/erratic_chronos2_external_monthly_aligned.csv")
print("Monthly aligned file:")
print(f"Shape: {monthly.shape}")
print(f"Mean WMAPE: {monthly['wmape'].mean():.6f}%")
print(f"All months:")
print(monthly[['eval_month', 'wmape', 'rows']].to_string())

# Recalculate pooled from the full SKU file
sku = pd.read_csv("Inchscape Repo/results/tables/erratic_chronos2_external_sku_wmape.csv")
total_err = sku['abs_error_sum'].sum()
total_act = sku['abs_actual_sum'].sum()
pooled = total_err / total_act * 100
print(f"\nPooled from SKU file: {pooled:.6f}%")
print(f"Sum of errors: {total_err:.2f}")
print(f"Sum of actuals: {total_act:.2f}")
