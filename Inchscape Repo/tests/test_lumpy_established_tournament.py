import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_established_tournament as et


def test_variant_grid_contains_eight_model_families_and_calibration():
    variants = et.variant_grid()
    assert variants.base_model_key.nunique() == 8
    assert set(et.CALIBRATION_SCALES) == set(variants.calibration_scale if "calibration_scale" in variants else variants.scale)
    assert variants.variant_id.is_unique


def test_rank_summary_prioritizes_business_70_percent_target():
    common = {
        "under_100_skus": 5,
        "median_sku_block_wmape": 80.0,
        "portfolio_block_wmape": 90.0,
        "abs_bias_pct": 10.0,
    }
    frame = pd.DataFrame(
        [
            {"variant_id": "more_50", "under_70_skus": 4, "under_50_skus": 4, **common},
            {"variant_id": "more_70", "under_70_skus": 5, "under_50_skus": 2, **common},
        ]
    )
    assert et.rank_summary(frame).iloc[0].variant_id == "more_70"


def test_score_forecast_uses_positive_demand_denominator_for_thresholds():
    frame = pd.DataFrame(
        {
            "sku_id": [1, 2],
            "target": [10.0, 0.0],
            "forecast": [5.0, 1.0],
            "block_naive_scale": [2.0, 1.0],
        }
    )
    _, summary = et.score_forecast(frame)
    assert summary["valid_positive_sku_count"] == 1
    assert summary["under_70_skus"] == 1
