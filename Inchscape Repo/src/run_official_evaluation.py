"""
Official evaluation following their specification:
- Train: Jan 2024 - Dec 2025
- Forecast: Generated Jan 2026 (18-month horizon)
- Evaluate: March 2026 WMAPE
- Output format: ts_id, Date, Forecast
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from models.benchmarks import wmape, wmape_per_sku
from models.advanced import tree_forecast, lumpy_hurdle_forecast

# Load data - all_sku_history only
data_path = Path(__file__).parent.parent / "data" / "processed" / "all_sku_history"
all_skus = []

for csv_file in data_path.glob("*.csv"):
    df = pd.read_csv(csv_file)
    df["demand_type"] = csv_file.stem.replace("collision_sales_", "")
    all_skus.append(df)

data = pd.concat(all_skus, ignore_index=True)
data["month"] = pd.to_datetime(data["month"])
data = data.sort_values(["sku_id", "month"])

# Official split: Train on Jan 2024 - Dec 2025, Forecast Jan 2026
train_end = pd.Timestamp("2025-12-31")
forecast_start = pd.Timestamp("2026-01-01")
evaluation_month = pd.Timestamp("2026-03-01")

train = data[data["month"] <= train_end].copy()
forecast_period = data[data["month"] >= forecast_start].copy()

print("="*70)
print("OFFICIAL EVALUATION")
print("="*70)
print(f"Train period: {train['month'].min()} to {train['month'].max()}")
print(f"Forecast period: {forecast_period['month'].min()} to {forecast_period['month'].max()}")
print(f"Evaluation month (WMAPE): {evaluation_month}")
print(f"Training rows: {len(train)}, Forecast rows: {len(forecast_period)}")

# Generate forecasts for each model type
models_to_eval = ["xgboost", "lightgbm", "random_forest", "lumpy_hurdle"]
results_list = []
sku_details_list = []
forecast_output = []

for model_type in models_to_eval:
    print(f"\n{'='*70}")
    print(f"Forecasting with {model_type.upper()}")
    print(f"{'='*70}")
    
    for demand_type in ["smooth", "intermittent", "erratic", "lumpy"]:
        train_subset = train[train["demand_type"] == demand_type].copy()
        forecast_subset = forecast_period[forecast_period["demand_type"] == demand_type].copy()
        
        if len(train_subset) == 0 or len(forecast_subset) == 0:
            print(f"{demand_type}: Skipped (no data)")
            continue
        
        try:
            if model_type == "lumpy_hurdle":
                preds = lumpy_hurdle_forecast(train_subset, forecast_subset)
            else:
                preds = tree_forecast(train_subset, forecast_subset, model_type=model_type)
            
            # Calculate WMAPE for March 2026 only
            march_preds = preds[preds["month"] == evaluation_month]
            march_actual = forecast_subset[forecast_subset["month"] == evaluation_month]
            
            if len(march_preds) > 0 and len(march_actual) > 0:
                # Merge predictions with actuals
                merged = march_preds.merge(march_actual[["sku_id", "month", "demand"]], 
                                          on=["sku_id", "month"], 
                                          how="left")
                
                # Calculate WMAPE using the original per-SKU formula
                # Also get individual SKU WMAPEs
                sku_wmapes, average_wmape = wmape_per_sku(
                    merged["month"],
                    merged["demand_y"],  # actual
                    merged["demand_x"],  # forecast
                    merged["sku_id"]
                )
                
                # Store average result
                results_list.append({
                    "model": model_type,
                    "demand_type": demand_type,
                    "march_2026_wmape": average_wmape,
                    "forecast_rows": len(preds),
                    "evaluation_rows": len(merged)
                })
                
                # Store individual SKU results
                for sku_id, sku_wmape in sku_wmapes.items():
                    sku_details_list.append({
                        "model": model_type,
                        "demand_type": demand_type,
                        "sku_id": sku_id,
                        "sku_wmape": sku_wmape
                    })
                
                print(f"  {demand_type:12} → WMAPE: {average_wmape:6.2f}%  ({len(merged)} rows, {len(sku_wmapes)} SKUs)")
                
                # Add to forecast output (all forecasts for Jan 2026 onwards)
                for _, row in preds.iterrows():
                    forecast_output.append({
                        "ts_id": row["sku_id"],
                        "Date": row["month"].strftime("%m/%d/%Y"),
                        "Forecast": int(np.maximum(0, row["demand"])),
                        "Model": model_type,
                        "DemandType": demand_type
                    })
        
        except Exception as e:
            print(f"  {demand_type:12} → ERROR: {str(e)[:50]}")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv(
    Path(__file__).parent.parent / "results" / "tables" / "official_evaluation_march_2026.csv",
    index=False
)

# Save individual SKU WMAPE details
sku_details_df = pd.DataFrame(sku_details_list)
sku_details_df.to_csv(
    Path(__file__).parent.parent / "results" / "tables" / "official_evaluation_sku_details.csv",
    index=False
)

# Save forecast output in official format
forecast_df = pd.DataFrame(forecast_output)
forecast_df_pivot = forecast_df.pivot_table(
    index=["ts_id", "Date"],
    columns="Model",
    values="Forecast",
    aggfunc="first"
).reset_index()

forecast_df_pivot.to_csv(
    Path(__file__).parent.parent / "results" / "tables" / "official_forecast_january_2026.csv",
    index=False
)

print("\n" + "="*70)
print("RESULTS SAVED")
print("="*70)
print(f"Evaluation results: official_evaluation_march_2026.csv")
print(f"SKU details: official_evaluation_sku_details.csv")
print(f"Forecast output: official_forecast_january_2026.csv")
print(f"\nMarch 2026 WMAPE Summary:")
if len(results_df) > 0:
    print(results_df.to_string(index=False))
