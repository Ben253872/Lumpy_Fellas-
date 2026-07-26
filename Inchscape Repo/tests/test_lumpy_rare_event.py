import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_rare_event as rare


def frame():
    return pd.DataFrame({"sku_id": [1] * 6, "block_number": range(1, 7), "cap": [100] * 6})


def test_top_one_full_forecasts_one_block():
    data = frame(); probability = np.array([.1, .8, .2, .3, .4, .5]); size = np.repeat(4.0, 6)
    forecast = rare.compose_rare_forecast(data, probability, size, "top1_full")
    assert (forecast > 0).sum() == 1
    assert forecast[1] == 4.0


def test_horizon_gate_can_return_no_event():
    data = frame(); probability = np.repeat(.01, 6); size = np.repeat(4.0, 6)
    forecast = rare.compose_rare_forecast(data, probability, size, "horizon_gate_full", horizon_threshold=.5)
    assert np.allclose(forecast, 0.0)


def test_occurrence_diagnostics():
    data = pd.DataFrame({"target": [1, 0, 2], "forecast": [1, 1, 0]})
    result = rare.occurrence_diagnostics(data)
    assert result["true_positive_blocks"] == 1
    assert result["false_positive_blocks"] == 1
