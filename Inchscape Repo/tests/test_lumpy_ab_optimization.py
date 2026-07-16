import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_ab_optimization as opt


def test_structural_trials_cover_windows_and_architectures():
    trials = opt.structural_trial_table()
    assert set(trials.history_window) == set(opt.WINDOW_OPTIONS)
    assert set(trials.architecture) == {"direct", "hurdle"}
    assert trials.trial_id.is_unique


def test_business_ranking_prioritizes_below_70():
    common = {"under_100": 5, "median_wmape": 80, "portfolio_wmape": 90, "bias_pct": 0}
    rows = pd.DataFrame([
        {"candidate_id": "a", "under_70": 4, "under_50": 4, **common},
        {"candidate_id": "b", "under_70": 5, "under_50": 1, **common},
    ])
    assert opt.rank_summary(rows).iloc[0].candidate_id == "b"


def test_optimization_segments_collapse_small_groups():
    frame = pd.DataFrame({
        "tournament_cohort": ["recurring_a"] * 3,
        "positive_cv2": [0.1, 1.0, 0.2],
        "positive_median": [1, 2, 2],
    })
    result = opt.add_optimization_segment(frame)
    assert result.optimization_segment.eq("recurring_a__other").all()
