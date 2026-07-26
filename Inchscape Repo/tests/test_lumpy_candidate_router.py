import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_candidate_router as router


def test_parse_candidate_id():
    trial = "pool__hurdle_d2__w24"
    candidate = trial + "__cal_isotonic__size_blend50_mean__top_full__kadaptive__s0.75"
    parsed = router.parse_candidate_id(candidate, [trial])
    assert parsed["calibration"] == "isotonic"
    assert parsed["size_source"] == "blend50_mean"
    assert parsed["event_count"] == "adaptive"
    assert parsed["scale"] == 0.75


def test_selected_forecasts_uses_one_candidate_per_sku():
    forecasts = pd.DataFrame(
        {
            "sku_id": [1, 1, 2, 2],
            "candidate_id": ["a", "b", "a", "b"],
            "target": [1, 1, 1, 1],
            "forecast": [1, 2, 3, 1],
        }
    )
    selections = pd.DataFrame({"sku_id": [1, 2], "selected_candidate_id": ["a", "b"]})
    chosen = router.selected_forecasts(forecasts, selections)
    assert chosen.forecast.tolist() == [1, 1]


def test_own_history_picks_lowest_error():
    errors = pd.DataFrame(
        {
            "sku_id": [1, 1],
            "candidate_id": ["a", "b"],
            "wmape": [80.0, 40.0],
            "absolute_error": [8.0, 4.0],
        }
    )
    selected = router.own_history_selections(errors)
    assert selected.selected_candidate_id.iloc[0] == "b"
