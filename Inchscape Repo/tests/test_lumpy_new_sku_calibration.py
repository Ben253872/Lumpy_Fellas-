import pandas as pd

from lumpy_new_sku_calibration import calibrated_routes


def test_recent_demand_route_scales_high_history_sku():
    forecasts = pd.DataFrame({"fold_id": [1, 1], "sku_id": ["a", "b"], "forecast": [10.0, 10.0]})
    features = pd.DataFrame({"fold_id": [1, 1], "sku_id": ["a", "b"], "recent_6m_total": [1.0, 9.0]})
    result = calibrated_routes(forecasts, features, [], [.5], [1.0], [2.0])
    assert result.forecast.tolist() == [10.0, 20.0]


def test_route_can_use_alternative_cutoff_safe_feature():
    forecasts = pd.DataFrame({"fold_id": [1, 1], "sku_id": ["a", "b"], "forecast": [4.0, 4.0]})
    features = pd.DataFrame({"fold_id": [1, 1], "sku_id": ["a", "b"], "positive_mean": [2.0, 8.0]})
    result = calibrated_routes(forecasts, features, [], [.5], [1.0], [1.5], route_features=["positive_mean"])
    assert result.forecast.tolist() == [4.0, 6.0]
