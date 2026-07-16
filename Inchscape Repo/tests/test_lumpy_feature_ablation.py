import pandas as pd

from lumpy_feature_ablation import allocate_total_to_blocks, candidate_group_sets, summarize_candidates


def test_candidate_sets_only_use_available_groups():
    names = [name for name, _ in candidate_group_sets({"demand", "stock"})]
    assert names == ["demand_only", "demand_plus_stock"]


def test_equal_allocation_when_profile_is_zero():
    totals = pd.DataFrame({"sku_id": ["a"], "fold_id": [1], "total_forecast": [12.0]})
    profile = pd.DataFrame({"sku_id": ["a", "a"], "fold_id": [1, 1], "block_number": [1, 2], "target": [2.0, 3.0], "forecast": [0.0, 0.0]})
    result = allocate_total_to_blocks(totals, profile)
    assert result.candidate_forecast.tolist() == [6.0, 6.0]


def test_summary_scores_positive_skus_only():
    metrics = pd.DataFrame({"segment": ["x", "x"], "candidate": ["a", "a"], "fold_id": [1, 1], "sku_id": ["p", "z"], "actual": [10.0, 0.0], "abs_error": [5.0, 3.0], "wmape": [0.5, float("nan")]})
    result = summarize_candidates(metrics).iloc[0]
    assert result.evaluated_sku_folds == 1
    assert result.share_below_50 == 1.0
