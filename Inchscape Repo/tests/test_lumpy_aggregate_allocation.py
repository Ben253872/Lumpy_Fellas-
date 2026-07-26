import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import lumpy_aggregate_allocation as allocation


def sample():
    months=pd.date_range("2021-01-01",periods=24,freq="MS"); rows=[]
    for sku,family in ((1,"x"),(2,"x"),(3,"y")):
        for i,month in enumerate(months): rows.append({"sku_id":sku,"month":month,"demand":float((i+sku)%7==0),"FAMILY_DESCRIPTION":family,"SUBFAMILY_DESCRIPTION":family})
    train=pd.DataFrame(rows); test=[]
    for sku,family in ((1,"x"),(2,"x"),(3,"y")):
        for month in pd.date_range("2023-01-01",periods=18,freq="MS"): test.append({"sku_id":sku,"month":month,"demand":0.,"FAMILY_DESCRIPTION":family,"SUBFAMILY_DESCRIPTION":family})
    return train,pd.DataFrame(test)


def test_grid_ids_are_unique():
    grid=allocation.config_grid(); assert len(grid)==324; assert len({item.config_id for item in grid})==324


def test_forecast_has_six_blocks_per_sku_and_nonnegative_values():
    train,test=sample(); result=allocation.forecast_allocation(train,test,{1,2,3},allocation.config_grid()[0])
    assert result.groupby("sku_id").block_number.nunique().eq(6).all()
    assert result.forecast.ge(0).all()


def test_group_block_forecast_preserves_allocated_total():
    train,test=sample(); config=allocation.AllocationConfig("__cohort__","recent",12,12,2,None,1)
    result=allocation.forecast_allocation(train,test,{1,2,3},config)
    expected=train.groupby("month").demand.sum().tail(12).mean()*3
    assert np.allclose(result.groupby("block_number").forecast.sum(),expected)
