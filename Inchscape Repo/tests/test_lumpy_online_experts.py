import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_online_experts as online


def sample_frame():
    rows=[]
    for origin in range(1,5):
        for sku,target in ((1,float(origin)),(2,float(5-origin))):
            rows.append({"sku_id":sku,"origin_id":origin,"block_start":pd.Timestamp("2024-01-01")+pd.DateOffset(months=3*origin),"block_number":1,"target":target,"block_naive_scale":1.,"candidate_00":target+.1,"candidate_01":10.,"expert_mean":5.,"expert_std":2.,"current_champion":5.})
    return pd.DataFrame(rows)


def test_grid_has_expected_size_and_unique_ids():
    grid=online.config_grid()
    assert len(grid)==324
    assert len({config.config_id for config in grid})==len(grid)


def test_personalised_weights_sum_to_one():
    frame=sample_frame()
    weights=online.expert_weights(frame,["candidate_00","candidate_01"],online.config_grid()[0])
    assert np.allclose(weights.sum(axis=1),1.)
    assert (weights.candidate_00>weights.candidate_01).all()


def test_prequential_forecast_only_uses_earlier_origins():
    frame=sample_frame()
    predicted=online.prequential_forecast(frame,["candidate_00","candidate_01"],online.config_grid()[0],[3,4])
    assert (predicted.history_through_origin==predicted.origin_id-1).all()


def test_champion_blend_is_respected():
    frame=sample_frame(); config=online.OnlineExpertConfig(3,.8,2,0,2,0)
    result=online.forecast_with_memory(frame.loc[frame.origin_id.lt(4)],frame.loc[frame.origin_id.eq(4)],["candidate_00","candidate_01"],config)
    assert np.allclose(result.forecast,5.)
