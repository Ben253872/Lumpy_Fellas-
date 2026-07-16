import pandas as pd

import numpy as np

from lumpy_launch_hurdle import _positive_size, launch_targets


def test_launch_targets_respects_three_month_gap():
    sales = pd.DataFrame({"sku_id": ["x", "x"], "month": pd.to_datetime(["2022-01-01", "2022-04-01"]), "demand": [2.0, 3.0]})
    result = launch_targets(sales, ["x"])
    assert result.iloc[0].block_start == pd.Timestamp("2022-04-01")
    assert result.iloc[0].target == 3.0
    assert len(result) == 6


def test_weighted_positive_size_supports_upper_quantiles():
    values = np.array([1.0, 3.0, 10.0])
    weights = np.ones(3)
    assert _positive_size(values, weights, "median") == 3.0
    assert _positive_size(values, weights, "p75") == 10.0
