import pandas as pd

from lumpy_feature_quality import quality_table


def test_quality_flags_constant_and_target():
    frame = pd.DataFrame({"sku_id": ["a", "b"], "month": pd.to_datetime(["2024-01-01"]*2), "demand": [1.0, 2.0], "Brand": ["same", "same"], "useful": [1.0, 3.0]})
    result = quality_table(frame, pd.Timestamp("2024-07-01")).set_index("feature")
    assert result.loc["Brand", "eligibility_status"] == "constant"
    assert result.loc["demand", "eligibility_status"] == "target"
    assert result.loc["useful", "eligible_for_importance"]


def test_quality_flags_external_for_lagging():
    frame = pd.DataFrame({"sku_id": ["a", "a"], "month": pd.to_datetime(["2024-01-01", "2024-02-01"]), "Inflation_Rate": [2.0, 2.1]})
    result = quality_table(frame, pd.Timestamp("2024-07-01")).set_index("feature")
    assert result.loc["Inflation_Rate", "eligibility_status"] == "external_lag_required"


def test_static_metadata_uses_effective_sku_coverage():
    frame = pd.DataFrame({"sku_id": ["a", "a", "b", "b"], "month": pd.to_datetime(["2024-01-01", "2024-02-01"]*2), "MATERIAL_DESCRIPTION": ["one", None, "two", None]})
    result = quality_table(frame, pd.Timestamp("2024-07-01")).set_index("feature")
    assert result.loc["MATERIAL_DESCRIPTION", "raw_cutoff_missing_pct"] == 50.0
    assert result.loc["MATERIAL_DESCRIPTION", "cutoff_missing_pct"] == 0.0
    assert result.loc["MATERIAL_DESCRIPTION", "eligible_for_importance"]
