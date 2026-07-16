import pandas as pd

from lumpy_quantile_calibration import apply_quantile_model, fit_quantile_model, horizon_table


def test_quantile_calibration_scales_forecasts_from_training_ratio():
    history = pd.DataFrame({"segment": ["x"] * 4, "sku_id": list("abcd"), "actual_total": [2.0, 4.0, 6.0, 16.0], "forecast_total": [1.0, 2.0, 3.0, 4.0]})
    model = fit_quantile_model(history)
    forecasts = pd.DataFrame({"segment": ["x"], "sku_id": ["z"], "target": [0.0], "forecast": [5.0]})
    result = apply_quantile_model(forecasts, model, strength=1.0, upper_cap=5.0)
    assert result.iloc[0].forecast == 20.0


def test_horizon_table_keeps_fold_cases_separate():
    frame = pd.DataFrame({"fold_id": [1, 2], "segment": ["x", "x"], "sku_id": ["a", "a"], "target": [1.0, 2.0], "forecast": [1.0, 2.0]})
    assert len(horizon_table(frame)) == 2
