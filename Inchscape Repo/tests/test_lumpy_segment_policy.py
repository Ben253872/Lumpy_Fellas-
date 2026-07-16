import numpy as np
import pandas as pd

import lumpy_segment_policy as sp


def test_known_forecasts_respect_monthly_cutoff_and_deduplicate():
    frame = pd.DataFrame(
        {
            "sku_id": [1, 1, 1],
            "model_key": ["a", "a", "a"],
            "block_start": ["2024-02-01", "2024-02-01", "2024-05-01"],
            "train_end": ["2023-10-01", "2024-01-01", "2024-01-01"],
            "fold_id": [2, 1, 1],
            "target": [3.0, 3.0, 4.0],
            "forecast": [2.0, 2.5, 5.0],
        }
    )
    known = sp.known_forecasts_at_cutoff(frame, pd.Timestamp("2024-04-01"))
    assert len(known) == 1
    assert known.iloc[0].forecast == 2.5


def test_strategy_tiers_are_mutually_exclusive():
    frame = pd.DataFrame(
        {
            "sku_id": [1, 2, 3],
            "frequency_tier": ["recurring_4_6", "occasional_2_3", "rare_0_1"],
            "recency_tier": ["recent_0_2", "recent_0_2", "stale_6_11"],
            "abc_units_tier": ["A", "B", "C"],
            "lifecycle_tier": ["established", "developing", "dormant"],
        }
    )
    result = sp.add_strategy_tier(frame)
    assert result.strategy_tier.tolist() == [
        "point_forecast_priority",
        "cautious_point_forecast",
        "inventory_policy_priority",
    ]


def test_score_forecasts_counts_positive_skus_only_for_thresholds():
    frame = pd.DataFrame(
        {
            "sku_id": [1, 2],
            "target": [10.0, 0.0],
            "forecast": [6.0, 2.0],
        }
    )
    per_sku, summary = sp.score_forecasts(frame)
    assert summary["valid_positive_sku_count"] == 1
    assert summary["under_50_skus"] == 1
    assert np.isnan(per_sku.loc[per_sku.sku_id.eq(2), "block_wmape_percent"]).all()
