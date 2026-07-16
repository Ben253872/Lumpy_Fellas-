import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import lumpy_total_demand as total


def sample():
    train=[]; test=[]
    for sku,family in ((1,"x"),(2,"y")):
        for i,month in enumerate(pd.date_range("2021-01-01",periods=24,freq="MS")): train.append({"sku_id":sku,"month":month,"demand":float((i+sku)%5==0),"FAMILY_DESCRIPTION":family})
        for month in pd.date_range("2023-01-01",periods=18,freq="MS"): test.append({"sku_id":sku,"month":month,"demand":1.,"FAMILY_DESCRIPTION":family})
    return pd.DataFrame(train),pd.DataFrame(test)


def test_target_blocks_has_six_blocks():
    _,test=sample(); blocks=total.target_blocks(test,{1,2}); assert blocks.groupby("sku_id").block_number.nunique().eq(6).all()


def test_timing_weights_sum_to_one():
    train,test=sample(); starts=total.target_blocks(test,{1,2}).block_start.drop_duplicates().tolist(); weights=total.timing_weights(train,{1,2},starts,"FAMILY_DESCRIPTION"); assert np.allclose(weights.groupby("sku_id").timing_weight.sum(),1)


def test_distribution_preserves_total():
    train,test=sample(); blocks=total.target_blocks(test,{1,2}); starts=blocks.block_start.drop_duplicates().tolist(); weights=total.timing_weights(train,{1,2},starts,"__cohort__"); forecasts=pd.DataFrame({"sku_id":[1,2],"forecast_total":[12.,18.]}); result=total.distribute_totals(forecasts,blocks,weights); assert np.allclose(result.groupby("sku_id").forecast.sum(),[12,18])


def test_empirical_forecasts_are_nonnegative():
    features=pd.DataFrame({"recent_12m_total":[0,2],"recent_6m_total":[0,1],"history_months":[12,24],"history_total_demand":[0,4]});
    for method in ("recent12_rate","recent6_rate","history_rate","bayes_rate"): assert (total.empirical_total_forecast(features,method)>=0).all()
