import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_group_occurrence as group


def test_adjustment_preserves_probability_sum_per_sku():
    frame = pd.DataFrame({"sku_id": [1] * 3 + [2] * 3})
    probability = np.array([.1, .2, .3, .2, .2, .2])
    adjusted = group.adjust_probabilities(frame, probability, np.array([1, 2, 1, 2, 1, 1]), 1.0)
    before = pd.Series(probability).groupby(frame.sku_id).sum()
    after = pd.Series(adjusted).groupby(frame.sku_id).sum()
    np.testing.assert_allclose(before, after)


def test_group_priors_have_six_blocks_and_mean_one():
    rows = []
    for sku in (1, 2):
        for month in pd.date_range("2021-01-01", periods=24, freq="MS"):
            rows.append({"sku_id": sku, "month": month, "demand": float(month.month == 1), "family": "x"})
    starts = list(pd.date_range("2023-01-01", periods=6, freq="3MS"))
    priors = group.group_block_priors(pd.DataFrame(rows), {1, 2}, starts, "family")
    assert priors.groupby("sku_id").size().eq(6).all()
    np.testing.assert_allclose(priors.groupby("sku_id").group_factor.mean(), 1.0)
