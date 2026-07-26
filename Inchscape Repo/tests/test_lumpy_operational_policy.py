import pandas as pd

from lumpy_operational_policy import add_operational_policy


def test_policy_maps_reliability_and_underforecast_escalation():
    table = pd.DataFrame({"reliability_tier": ["exception_policy", "forecast_led"], "bias_direction": ["underforecast", "underforecast"]})
    result = add_operational_policy(table)
    assert result.underforecast_escalation.tolist() == [True, False]
    assert result.iloc[0].owner == "parts_inventory_control"
