import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_quantity_calibration as quantity


def test_cap_preserves_zero_and_caps_positive():
    frame = pd.DataFrame({"sku_id": [1, 1], "forecast": [0.0, 10.0]})
    stats = pd.DataFrame({"sku_id": [1], "positive_median": [4.0], "positive_mean": [5.0], "positive_p75": [6.0], "positive_max": [8.0], "positive_count": [2]})
    result = quantity.apply_quantity_transform(frame, stats, "cap_median", 1.5)
    assert result.forecast.tolist() == [0.0, 6.0]


def test_pooled_ratio_can_be_shrunk():
    frame = pd.DataFrame({"sku_id": [1], "forecast": [10.0]})
    stats = pd.DataFrame({"sku_id": [1]})
    result = quantity.apply_quantity_transform(frame, stats, "pooled_ratio", 0.5, pooled_ratio=0.5)
    assert result.forecast.iloc[0] == 7.5


def test_transform_ids_are_unique():
    assert quantity.transform_grid().transform_id.is_unique
