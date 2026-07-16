"""Verify the data leakage issue more directly."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.advanced import engineer_time_features

# Load and prepare data
data = pd.read_csv(Path("data/processed/collision_flag_only/collision_sales_lumpy.csv"))
data["month"] = pd.to_datetime(data["month"])
data = data.sort_values(["sku_id", "month"])

months = sorted(data["month"].unique())
evaluation_months = 6
split_month = months[-evaluation_months]

train = data.loc[data["month"] < split_month].copy()
actual = data.loc[data["month"] >= split_month].copy()

print("="*70)
print("CRITICAL LEAKAGE CHECK: Rolling Features and Test Data")
print("="*70)
print(f"\nTrain period: up to {split_month}")
print(f"Test period: {split_month} to {max(actual['month'])}")
print(f"\nTrain rows: {len(train)}, Test rows: {len(actual)}")

# Feature engineering on COMBINED data (as the code does)
combined = pd.concat([train, actual], ignore_index=True)
engineered = engineer_time_features(combined)

# Separate back
test_dates = pd.to_datetime(actual["month"])
engineered_test = engineered.loc[engineered["month"].isin(test_dates)].copy()

print("\n" + "="*70)
print("ISSUE: For rolling_mean calculations in test period")
print("="*70)

# Check a specific test row
test_row = engineered_test[engineered_test["demand"] > 0].iloc[0]
test_month = test_row["month"]
test_sku = test_row["sku_id"]

print(f"\nTest row: SKU {test_sku}, Month {test_month}")
print(f"rolling_mean_3 for this row: {test_row.get('rolling_mean_3', np.nan):.2f}")
print(f"rolling_mean_6 for this row: {test_row.get('rolling_mean_6', np.nan):.2f}")

# The rolling_mean with shift(1) looks at 3 months BEFORE
# For April (if test_month is April), it would use Jan, Feb, Mar
# But those are ALL in the TEST PERIOD!

months_used = [test_month - pd.DateOffset(months=i) for i in range(1, 4)]
print(f"\nrolling_mean_3 uses months: {months_used}")
print(f"But split_month is: {split_month}")

in_train_period = [m < split_month for m in months_used]
print(f"Are these in TRAIN period? {in_train_period}")

if not all(in_train_period):
    print("\n🚨 LEAKAGE DETECTED!")
    print("   Rolling features for test data use values from TEST period itself!")
    print("   This is problematic for backtesting (though valid for real forecasting)")
else:
    print("\n✓ No leakage - all rolling windows use only training data")

# Show the actual values used
print("\n" + "="*70)
print("Actual values used in rolling_mean_3:")
print("="*70)

sku_all = engineered.loc[engineered["sku_id"] == test_sku].sort_values("month")
for month in months_used:
    val_rows = sku_all[sku_all["month"] == month]
    if len(val_rows) > 0:
        val = val_rows["demand"].values[0]
        in_train = "TRAIN" if month < split_month else "TEST"
        print(f"  {month}: {val:6.1f}  ({in_train})")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The current implementation uses a COMBINED train+test dataset for
feature engineering. This means:

1. Lags are OK (lag_1 looks at previous month only)
2. Rolling means with shift(1) look backward, BUT they include
   values from the test period itself when predicting future
   test dates

This is LEAKAGE in a backtesting context because you're using
information (rolling averages computed from test data) to predict
the test data itself.

For REAL forecasting (predicting future unknowns), this would be
fine - you'd use all available history.

But for BACKTESTING, you should:
- Fit features ONLY on training data
- Apply the same transformation to test data
""")
