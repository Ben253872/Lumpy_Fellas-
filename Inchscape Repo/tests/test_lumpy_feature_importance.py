import pandas as pd

from lumpy_feature_importance import commercial_features, summarize_importance


def test_commercial_features_are_historical_aggregates():
    frame = pd.DataFrame({"sku_id": ["a", "a"], "REVENUE": [2.0, 3.0], "UNIT_PRICE": [2.0, 4.0]})
    result = commercial_features(frame).iloc[0]
    assert result.historical_revenue == 5.0
    assert result.historical_unit_price == 3.0


def test_importance_requires_stability():
    rows = pd.DataFrame({"segment": ["x", "x"], "feature": ["a", "a"], "validation_fold": [1, 2], "importance_mean": [0.2, -0.1]})
    result = summarize_importance(rows, {"a": "demand"}).iloc[0]
    assert not result.stable_helpful


def test_importance_rejects_floating_point_noise():
    rows = pd.DataFrame({"segment": ["x", "x"], "feature": ["a", "a"], "validation_fold": [1, 2], "importance_mean": [1e-15, 2e-15]})
    result = summarize_importance(rows, {"a": "external"}).iloc[0]
    assert not result.stable_helpful
