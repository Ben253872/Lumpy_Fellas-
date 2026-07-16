import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_occurrence_budget as ob


def _frame():
    return pd.DataFrame(
        {
            "sku_id": ["a"] * 6 + ["b"] * 6,
            "block_number": list(range(1, 7)) * 2,
            "probability": [0.1, 0.9, 0.4, 0.8, 0.2, 0.3] * 2,
            "size": [10.0] * 12,
            "event_budget": [2] * 6 + [3] * 6,
            "cap": [100.0] * 12,
        }
    )


def test_top_event_budget_selects_expected_number_per_sku():
    frame = _frame()
    forecast = ob.compose_event_forecast(
        frame, frame.probability, frame["size"], "top_full", "adaptive"
    )
    selected = pd.Series(forecast).gt(0).groupby(frame.sku_id).sum()
    assert selected.to_dict() == {"a": 2, "b": 3}


def test_expected_forecast_is_probability_times_size():
    frame = _frame()
    forecast = ob.compose_event_forecast(frame, frame.probability, frame["size"], "expected")
    np.testing.assert_allclose(forecast, frame.probability * frame["size"])


def test_probability_calibration_handles_constant_target():
    fitted = ob.fit_probability_calibrator(np.array([0.1, 0.8]), np.array([0, 0]), "sigmoid")
    np.testing.assert_allclose(fitted.predict(np.array([0.2, 0.9])), 0.0)


def test_recipe_grid_has_unique_ids():
    recipes = ob.event_recipe_grid()
    assert recipes.recipe_id.is_unique
    assert {"expected", "top_expected", "top_full", "normalised"}.issubset(set(recipes["mode"]))
