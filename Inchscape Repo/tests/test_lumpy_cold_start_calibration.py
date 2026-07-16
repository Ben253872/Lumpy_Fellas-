import pandas as pd

from lumpy_cold_start_calibration import calibrated_candidates, candidate_summary


def test_top_block_placement_keeps_only_largest_block_per_sku():
    frame = pd.DataFrame(
        {
            "sku_id": ["a", "a", "a"],
            "block_number": [1, 2, 3],
            "target": [0.0, 2.0, 0.0],
            "forecast": [1.0, 3.0, 2.0],
            "model_key": ["peer"] * 3,
        }
    )
    result = calibrated_candidates(frame, scales=[0.5], top_blocks=[1])
    assert result.forecast.tolist() == [0.0, 1.5, 0.0]


def test_candidate_summary_uses_pooled_sku_wmape():
    frame = pd.DataFrame(
        {
            "sku_id": ["a", "a"],
            "target": [0.0, 10.0],
            "forecast": [0.0, 5.0],
            "candidate_id": ["x", "x"],
        }
    )
    summary = candidate_summary(frame).iloc[0]
    assert summary.median_wmape == 50.0
    assert summary.under_70 == 1
