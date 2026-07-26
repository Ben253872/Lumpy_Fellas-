import pandas as pd

from lumpy_dormant_reactivation import inventory_policy, recency_band


def test_recency_band_starts_at_twelve_months():
    result = recency_band(pd.Series([12, 18, 24, 40])).tolist()
    assert result == ["12_17", "18_23", "24_35", "36_plus"]


def test_inventory_policy_combines_block_probabilities():
    frame = pd.DataFrame({"sku_id": ["x", "x"], "forecast": [1.0, 1.0], "event_probability": [0.5, 0.5]})
    result = inventory_policy(frame).iloc[0]
    assert result.reactivation_probability == 0.75
    assert result.inventory_policy == "safety_stock_review"
