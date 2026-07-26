import pandas as pd

from lumpy_methodology_audit import EXPECTED_BLOCK_STARTS, contract_checks


def test_contract_checks_accept_complete_single_sku():
    frame = pd.DataFrame({"sku_id": ["x"]*6, "block_number": range(1,7), "block_start": EXPECTED_BLOCK_STARTS, "target": [1.0]*6, "forecast": [1.0]*6})
    result = contract_checks(frame, expected_skus=1, expected_actual=6.0)
    assert result.passed.all()


def test_contract_checks_detect_overlap():
    frame = pd.DataFrame({"sku_id": ["x"]*7, "block_number": [1,1,2,3,4,5,6], "block_start": [EXPECTED_BLOCK_STARTS[0],*EXPECTED_BLOCK_STARTS], "target": [0.0]*7, "forecast": [0.0]*7})
    result = contract_checks(frame, expected_skus=1, expected_actual=0.0).set_index("check")
    assert not result.loc["no_route_overlap", "passed"]
