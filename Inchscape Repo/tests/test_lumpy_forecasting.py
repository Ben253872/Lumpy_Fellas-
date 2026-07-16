import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_forecasting as lf


def test_complete_monthly_grid_does_not_backfill_time_varying_values():
    sales = pd.DataFrame(
        {
            "sku_id": ["A", "A"],
            "month": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "demand": [1.0, 2.0],
            "SUBFAMILY_DESCRIPTION": [None, "Body"],
            "REVENUE": [10.0, 30.0],
            "UNIT_PRICE": [100.0, 120.0],
            "STOCK_END_MONTH": [5.0, 1.0],
            "Inflation_Rate": [2.0, 3.0],
        }
    )
    completed = lf.complete_monthly_grid(sales).sort_values("month").reset_index(drop=True)
    february = completed.loc[completed.month.eq(pd.Timestamp("2024-02-01"))].iloc[0]
    assert february.demand == 0.0
    assert february.SUBFAMILY_DESCRIPTION == "Body"
    assert february.REVENUE == 0.0
    assert february.UNIT_PRICE == 100.0
    assert np.isnan(february.STOCK_END_MONTH)
    assert february.Inflation_Rate == 2.0


def test_complete_monthly_grid_preserves_missing_observed_flows():
    sales = pd.DataFrame(
        {
            "sku_id": ["A", "A"],
            "month": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "demand": [0.0, 1.0],
            "MATERIAL_DESCRIPTION": [None, "Lamp"],
            "REVENUE": [np.nan, 20.0],
            "UNIT_PRICE": [np.nan, 75.0],
            "STOCK_START_MONTH": [np.nan, 4.0],
        }
    )
    completed = lf.complete_monthly_grid(sales).sort_values("month").reset_index(drop=True)
    assert completed.MATERIAL_DESCRIPTION.eq("Lamp").all()
    assert np.isnan(completed.loc[0, "REVENUE"])
    assert completed.loc[1, "REVENUE"] == 0.0
    assert completed.loc[0:1, "UNIT_PRICE"].isna().all()
    assert completed.loc[0:1, "STOCK_START_MONTH"].isna().all()
