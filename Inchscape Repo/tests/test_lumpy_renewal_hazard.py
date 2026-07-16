import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_renewal_hazard as hazard


def test_age_buckets():
    assert hazard.age_bucket(2) == "00_02"
    assert hazard.age_bucket(7) == "06_11"
    assert hazard.age_bucket(30) == "18_plus"


def test_constant_hazard_produces_six_blocks():
    values = {(1, bucket): 0.1 for bucket in ("00_02", "03_05", "06_11", "12_17", "18_plus")}
    values.update({("all", bucket): 0.1 for bucket in ("00_02", "03_05", "06_11", "12_17", "18_plus")})
    model = hazard.HazardModel(values, {1: "all"}, {1: 5})
    result = hazard.forecast_blocks(model, {1})
    assert len(result) == 6
    assert result.event_probability.between(0, 1).all()
    assert np.allclose(result.event_probability, 1 - 0.9**3)
