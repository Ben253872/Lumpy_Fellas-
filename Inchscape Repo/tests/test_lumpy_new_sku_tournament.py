import pandas as pd

from lumpy_new_sku_tournament import combine_candidates, pair_blends, scale_candidate_grid, score_candidates


def test_blend_uses_requested_short_history_weight():
    base = {"fold_id": [1], "sku_id": ["x"], "block_number": [1], "block_start": pd.to_datetime(["2024-01-01"]), "target": [5.0]}
    short = pd.DataFrame({**base, "forecast": [8.0], "model_key": ["s"]})
    analogue = pd.DataFrame({**base, "forecast": [2.0], "model_key": ["a"]})
    result = combine_candidates(short, analogue, weights=[0.75])
    blend = result.loc[result.candidate_id.str.startswith("blend")].iloc[0]
    assert blend.forecast == 6.5


def test_scores_fold_sku_as_separate_cases():
    frame = pd.DataFrame({"fold_id": [1, 2], "sku_id": ["x", "x"], "target": [10.0, 10.0], "forecast": [5.0, 10.0], "candidate_id": ["c", "c"]})
    result = score_candidates(frame).iloc[0]
    assert result.positive_cases == 2
    assert result.under_70 == 2


def test_pair_blend_and_scale_are_composable():
    base = pd.DataFrame({"fold_id": [1, 1], "sku_id": ["x", "x"], "block_number": [1, 1], "block_start": pd.to_datetime(["2024-01-01"] * 2), "target": [4.0, 4.0], "forecast": [2.0, 6.0], "model_key": ["sba", "tsb"]})
    blended = pair_blends(base, "sba", "tsb", left_weights=[0.5])
    scaled = scale_candidate_grid(blended, [1.25])
    assert scaled.iloc[0].forecast == 5.0
