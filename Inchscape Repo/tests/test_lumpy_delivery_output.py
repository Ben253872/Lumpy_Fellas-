import pandas as pd

from lumpy_delivery_output import build_delivery_table, reliability_tier


def test_reliability_tier_prioritises_dormant_review():
    result = reliability_tier(pd.Series([20.0, 60.0, 80.0, 120.0]), pd.Series([True, False, False, False]))
    assert result.tolist() == ["manual_lifecycle_review", "forecast_plus_review", "manual_review_with_forecast", "exception_policy"]


def test_delivery_table_pivots_six_blocks():
    forecasts = pd.DataFrame({"sku_id": ["x"]*6, "block_number": range(1,7), "target": [1.0]*6, "forecast": [1.0]*6, "segment": ["new"]*6, "champion_source": ["test"]*6})
    assignments = pd.DataFrame({"sku_id": ["x"], "lifecycle_tier": ["new"]})
    result = build_delivery_table(forecasts, assignments)
    assert result.iloc[0].forecast_18m == 6.0
    assert "forecast_block_6" in result.columns
