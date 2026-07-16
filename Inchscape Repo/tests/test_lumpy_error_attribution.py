import pandas as pd

from lumpy_error_attribution import block_error_attribution, concentration_table, sku_error_table


def test_block_attribution_separates_missed_and_false_events():
    frame = pd.DataFrame({"segment": ["x", "x"], "sku_id": ["a", "b"], "target": [5.0, 0.0], "forecast": [0.0, 2.0]})
    result = block_error_attribution(frame)
    assert set(result.error_type) == {"missed_event", "false_event"}


def test_concentration_uses_positive_skus():
    frame = pd.DataFrame({"segment": ["x", "x"], "sku_id": ["a", "b"], "target": [10.0, 0.0], "forecast": [0.0, 4.0]})
    sku = sku_error_table(frame)
    result = concentration_table(sku, shares=(1.0,)).iloc[0]
    assert result.sku_count == 1
    assert result.absolute_error_share == 1.0
