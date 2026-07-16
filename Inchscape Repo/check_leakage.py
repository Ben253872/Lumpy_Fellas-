"""Check for potential data leakage in feature engineering."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.advanced import engineer_time_features, _encoded_features

# Load the data
data_path = Path("data/processed/collision_flag_only/collision_sales_lumpy.csv")
data = pd.read_csv(data_path)
data["month"] = pd.to_datetime(data["month"])

# Sort by sku and month
data = data.sort_values(["sku_id", "month"])

# Get unique months
months = sorted(data["month"].unique())
print(f"Total months in dataset: {len(months)}")
print(f"Date range: {months[0]} to {months[-1]}")

# Simulate the holdout validation split (same as the actual code)
evaluation_months = 6
split_month = months[-evaluation_months]
print(f"\nTrain/Test split at: {split_month}")
print(f"Test period: {min(months[-evaluation_months:])} to {months[-1]}")

train = data.loc[data["month"] < split_month].copy()
actual = data.loc[data["month"] >= split_month].copy()

print(f"\nTrain data: {train.shape[0]} rows")
print(f"Test data: {actual.shape[0]} rows")

# Now do feature engineering ON COMBINED data (this is what tree_forecast does)
combined = pd.concat([train, actual], ignore_index=True)
engineered = engineer_time_features(combined)

# Separate back into train and test after engineering
train_months_dt = pd.to_datetime(train["month"])
test_months_dt = pd.to_datetime(actual["month"])
engineered_train = engineered.loc[engineered["month"].isin(train_months_dt)]
engineered_test = engineered.loc[engineered["month"].isin(test_months_dt)]

print(f"Engineered train: {engineered_train.shape[0]} rows")
print(f"Engineered test: {engineered_test.shape[0]} rows")

# CRITICAL CHECK: For test data, verify that features don't look into the future
print("\n" + "="*60)
print("LEAKAGE CHECK: Analyzing a test row's features")
print("="*60)

# Pick a test row
test_row = engineered_test[engineered_test["demand"] > 0].iloc[0]
test_sku = test_row["sku_id"]
test_month = test_row["month"]

print(f"\nTest row: SKU {test_sku}, Month {test_month}")

# Get all data for this SKU, sorted
sku_full = engineered.loc[engineered["sku_id"] == test_sku].sort_values("month")

# Find the row index in the full dataset
test_row_idx = sku_full[sku_full["month"] == test_month].index[0]
print(f"Position in full dataset: {test_row_idx}")

# Show context: rows before and after
print("\nContext (showing month, demand, lag_1):")
display_start = max(0, list(sku_full.index).index(test_row_idx) - 3)
display_end = min(len(sku_full), list(sku_full.index).index(test_row_idx) + 4)

for i, (idx, row) in enumerate(sku_full.iloc[display_start:display_end].iterrows()):
    marker = " <-- TEST ROW" if idx == test_row_idx else ""
    print(f"  {row['month']}: demand={row['demand']:6.1f}, lag_1={row['lag_1']:6.1f}{marker}")

# CRITICAL: Check rolling_mean features
print("\n" + "="*60)
print("CHECKING ROLLING MEAN FEATURES")
print("="*60)

# For test row, rolling_mean_3 should be the mean of the 3 months BEFORE it (due to shift(1))
test_row_full = sku_full.loc[test_row_idx]
print(f"\nTest row features:")
print(f"  rolling_mean_3: {test_row_full.get('rolling_mean_3', np.nan)}")
print(f"  rolling_mean_6: {test_row_full.get('rolling_mean_6', np.nan)}")
print(f"  rolling_mean_12: {test_row_full.get('rolling_mean_12', np.nan)}")

# Verify rolling_mean_3 manually - it should ONLY use data from BEFORE test period
# rolling_mean_3 with shift(1) means: for month M, average of months [M-3:M-1]
print(f"\nFor {test_month}:")
print(f"  rolling_mean_3 should average: {test_month - pd.DateOffset(months=3)} to {test_month - pd.DateOffset(months=1)}")

# Get those months
check_months = [test_month - pd.DateOffset(months=i) for i in range(1, 4)]
check_rows = sku_full[sku_full["month"].isin(check_months)].sort_values("month")
if len(check_rows) > 0:
    manual_mean_3 = check_rows["demand"].mean()
    print(f"  Manual calculation: {manual_mean_3:.2f}")
    print(f"  Feature value: {test_row_full.get('rolling_mean_3', np.nan):.2f}")
    if abs(manual_mean_3 - test_row_full.get('rolling_mean_3', np.nan)) < 0.01:
        print("  ✓ MATCH - No forward-looking!")
    else:
        print("  ✗ MISMATCH - Possible leakage!")

# Check if all dates in rolling window are in training period
for i, month in enumerate(check_months):
    in_train = month < split_month
    print(f"  Month {month} in train period? {in_train}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
If all months used in rolling_mean features are BEFORE the split_month,
then there is NO leakage. Features only use historical data.

If any months are >= split_month, then there IS leakage.
""")
