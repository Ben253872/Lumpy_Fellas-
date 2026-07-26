"""
Hyperparameter tuning for erratic demand - training set only (Jan 2021 - Aug 2024).

This script tunes hyperparameters on the training data without touching any test data.
Results are saved and will be used in the rolling evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# Load training data (erratic only)
data_path = Path(__file__).parent.parent / "data" / "processed" / "all_sku_history"
erratic_file = data_path / "collision_sales_erratic.csv"

data = pd.read_csv(erratic_file)
data["month"] = pd.to_datetime(data["month"])
data = data.sort_values(["sku_id", "month"])

# Use only training data
train_end = pd.Timestamp("2024-08-31")
train_data = data[data["month"] <= train_end].copy()

print("="*80)
print("HYPERPARAMETER TUNING - ERRATIC DEMAND (TRAINING DATA ONLY)")
print("="*80)
print(f"Training data: {train_data['month'].min()} to {train_data['month'].max()}")
print(f"Training rows: {len(train_data)}")
print(f"Unique SKUs: {train_data['sku_id'].nunique()}")

# Feature engineering for training data
def create_features(df):
    """Create lagged and rolling features."""
    df = df.sort_values(["sku_id", "month"]).copy()
    
    for sku_id in df["sku_id"].unique():
        mask = df["sku_id"] == sku_id
        
        # Lags
        df.loc[mask, "lag_1"] = df.loc[mask, "demand"].shift(1)
        df.loc[mask, "lag_3"] = df.loc[mask, "demand"].shift(3)
        df.loc[mask, "lag_6"] = df.loc[mask, "demand"].shift(6)
        df.loc[mask, "lag_12"] = df.loc[mask, "demand"].shift(12)
        
        # Rolling means
        df.loc[mask, "rolling_mean_3"] = df.loc[mask, "demand"].rolling(3, min_periods=1).mean()
        df.loc[mask, "rolling_mean_6"] = df.loc[mask, "demand"].rolling(6, min_periods=1).mean()
        df.loc[mask, "rolling_mean_12"] = df.loc[mask, "demand"].rolling(12, min_periods=1).mean()
        
        # Positive rate
        df.loc[mask, "pos_rate_12"] = (df.loc[mask, "demand"] > 0).rolling(12, min_periods=1).mean()
        
        # Expanding mean
        df.loc[mask, "expanding_mean"] = df.loc[mask, "demand"].expanding().mean()
    
    # Fill NaNs
    df = df.fillna(0)
    return df

print("\nCreating features...")
train_data = create_features(train_data)

# Prepare X, y
feature_cols = ["lag_1", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_6", 
                "rolling_mean_12", "pos_rate_12", "expanding_mean"]
X = train_data[feature_cols].values
y = train_data["demand"].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Best hyperparameters dictionary
best_params = {}

# 1. XGBoost tuning
print(f"\n{'='*80}")
print("TUNING: XGBOOST")
print(f"{'='*80}")

xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
}

xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_search = RandomizedSearchCV(
    xgb_model, 
    xgb_param_grid, 
    n_iter=12, 
    cv=3, 
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)
print("Running RandomizedSearchCV...")
xgb_search.fit(X, y)
best_params['xgboost'] = xgb_search.best_params_
print(f"Best params: {best_params['xgboost']}")
print(f"Best CV score (MAPE): {-xgb_search.best_score_:.4f}")

# 2. LightGBM tuning
print(f"\n{'='*80}")
print("TUNING: LIGHTGBM")
print(f"{'='*80}")

lgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63],
    'subsample': [0.7, 0.9],
}

lgb_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgb_search = RandomizedSearchCV(
    lgb_model, 
    lgb_param_grid, 
    n_iter=12, 
    cv=3, 
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)
print("Running RandomizedSearchCV...")
lgb_search.fit(X, y)
best_params['lightgbm'] = lgb_search.best_params_
print(f"Best params: {best_params['lightgbm']}")
print(f"Best CV score (MAPE): {-lgb_search.best_score_:.4f}")

# 3. Random Forest tuning
print(f"\n{'='*80}")
print("TUNING: RANDOM FOREST")
print(f"{'='*80}")

rf_param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'n_estimators': [100, 200],
}

rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_model, 
    rf_param_grid, 
    n_iter=12, 
    cv=3, 
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)
print("Running RandomizedSearchCV...")
rf_search.fit(X, y)
best_params['random_forest'] = rf_search.best_params_
print(f"Best params: {best_params['random_forest']}")
print(f"Best CV score (MAPE): {-rf_search.best_score_:.4f}")

# For lumpy_hurdle, use LightGBM params
print(f"\n{'='*80}")
print("TUNING: LUMPY_HURDLE")
print(f"{'='*80}")
print("Using LightGBM tuning results for Lumpy Hurdle occurrence model")
best_params['lumpy_hurdle'] = best_params['lightgbm'].copy()

# Save best parameters
output_path = Path(__file__).parent.parent / "results" / "tuning" / "best_hyperparams_erratic.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    # Convert any non-JSON-serializable values
    params_to_save = {}
    for model, params in best_params.items():
        params_to_save[model] = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in params.items()}
    json.dump(params_to_save, f, indent=2)

print(f"\n{'='*80}")
print("TUNING COMPLETE")
print(f"{'='*80}")
print(f"Best hyperparameters saved to: {output_path}")
print("\nSummary:")
for model, params in best_params.items():
    print(f"\n{model.upper()}:")
    for k, v in params.items():
        print(f"  {k}: {v}")
