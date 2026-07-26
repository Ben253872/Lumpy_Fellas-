import pandas as pd
import pytest

from lumpy_consolidated_scorecard import consolidate, normalize_forecasts, score


def test_consolidation_requires_six_unique_blocks():
    frame = pd.DataFrame({"sku_id": ["x"] * 6, "block_number": range(1, 7), "block_start": pd.date_range("2024-01-01", periods=6, freq="3MS"), "target": [1.0] * 6, "forecast": [1.0] * 6})
    result = consolidate([normalize_forecasts(frame, "test", "source")], expected_skus=1)
    assert len(result) == 6
    with pytest.raises(ValueError):
        consolidate([result, result], expected_skus=1)


def test_score_counts_positive_skus_only_for_thresholds():
    frame = pd.DataFrame({"sku_id": ["a", "b"], "target": [10.0, 0.0], "forecast": [5.0, 0.0]})
    result = score(frame).iloc[0]
    assert result.all_skus == 2
    assert result.positive_skus == 1
    assert result.under_70 == 1
