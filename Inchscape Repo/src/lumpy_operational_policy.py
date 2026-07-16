from __future__ import annotations

import pandas as pd


POLICIES = {
    "forecast_led": {"planning_action": "forecast_led_replenishment", "review_cadence": "monthly", "inventory_treatment": "standard_service_level_safety_stock", "owner": "demand_planning"},
    "forecast_plus_review": {"planning_action": "forecast_with_planner_review", "review_cadence": "monthly", "inventory_treatment": "review_safety_stock_and_lead_time", "owner": "demand_planning"},
    "manual_review_with_forecast": {"planning_action": "manual_approval_before_replenishment", "review_cadence": "each_planning_cycle", "inventory_treatment": "use_forecast_as_guidance_only", "owner": "planner_and_parts_team"},
    "exception_policy": {"planning_action": "exception_or_inventory_policy", "review_cadence": "event_driven", "inventory_treatment": "avoid_point_forecast_only_decisions", "owner": "parts_inventory_control"},
    "manual_lifecycle_review": {"planning_action": "lifecycle_and_reactivation_review", "review_cadence": "quarterly_or_triggered", "inventory_treatment": "supersession_and_obsolescence_policy", "owner": "parts_lifecycle_owner"},
}


def add_operational_policy(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    policy = pd.DataFrame.from_dict(POLICIES, orient="index").rename_axis("reliability_tier").reset_index()
    result = result.merge(policy, on="reliability_tier", how="left", validate="many_to_one")
    if result.planning_action.isna().any():
        raise ValueError("Unknown reliability tier in delivery table")
    result["underforecast_escalation"] = result.bias_direction.eq("underforecast") & result.reliability_tier.isin(["manual_review_with_forecast", "exception_policy"])
    return result
