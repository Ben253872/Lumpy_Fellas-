import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_layered_ensemble as layer


def test_diverse_selection_preserves_architectures_and_incumbent():
    rows = []
    for architecture in ("classical", "direct", "hurdle"):
        for number in range(3):
            rows.append({"cohort": "a", "architecture": architecture, "trial_id": f"{architecture}_{number}", "candidate_id": f"{architecture}_{number}__x", "under_70": 5-number, "under_50": 2, "under_100": 6, "median_wmape": 60, "portfolio_wmape": 70, "bias_pct": 0})
    table = pd.DataFrame(rows)
    result = layer.select_diverse_candidates(table, "a", ["direct_2__x"], per_architecture=1)
    assert {value.split("_")[0] for value in result} == {"classical", "direct", "hurdle"}
    assert "direct_2__x" in result


def test_leave_one_origin_out_never_predicts_from_empty_training_set():
    frame = pd.DataFrame({"sku_id": [1, 1], "block_start": pd.to_datetime(["2024-01-01", "2024-04-01"]), "block_number": [1, 1], "target": [2.0, 3.0], "block_naive_scale": [1.0, 1.0], "origin_id": [1, 2], "candidate_00": [2.0, 2.0], "candidate_01": [3.0, 3.0], "expert_mean": [2.5, 2.5], "expert_median": [2.5, 2.5], "expert_min": [2, 2], "expert_max": [3, 3], "expert_std": [.5, .5], "expert_cv": [.2, .2], "month_sin": [0, 1], "month_cos": [1, 0]})
    out = layer.leave_one_origin_out(frame, ["candidate_00", "candidate_01"], "mean", [1, 2])
    assert set(out.held_out_origin) == {1, 2}
    assert len(out) == 2


def test_router_forecasts_are_valid_candidate_values():
    train = pd.DataFrame({"target": [1., 10., 1., 10.], "candidate_00": [1., 1., 1., 1.], "candidate_01": [10., 10., 10., 10.]})
    test = train.copy()
    for frame in (train, test):
        frame["expert_mean"] = 5.5; frame["expert_median"] = 5.5; frame["expert_min"] = 1.; frame["expert_max"] = 10.; frame["expert_std"] = 4.5; frame["expert_cv"] = .8; frame["month_sin"] = 0.; frame["month_cos"] = 1.; frame["block_number"] = 1
    result = layer.fit_predict_strategy("router_hard", train, test, ["candidate_00", "candidate_01"]).forecast
    assert set(np.unique(result)).issubset({1., 10.})


def test_retention_prioritises_skus_below_70():
    challenger = {"under_70": 11, "under_50": 1, "median_wmape": 80}
    incumbent = {"under_70": 10, "under_50": 9, "median_wmape": 60}
    assert layer.retention_decision(challenger, incumbent)[0]


def test_unknown_strategy_fails_clearly():
    frame = pd.DataFrame({"candidate_00": [1.], "target": [1.], "expert_mean": [1.], "expert_median": [1.], "expert_min": [1.], "expert_max": [1.], "expert_std": [0.], "expert_cv": [0.], "month_sin": [0.], "month_cos": [1.], "block_number": [1]})
    try:
        layer.fit_predict_strategy("not_real", frame, frame, ["candidate_00"])
    except ValueError as exc:
        assert "Unknown layered strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
