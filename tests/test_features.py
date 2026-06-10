"""Tests for feature engineering functions."""

import math
import pandas as pd
import pytest
from src.ml.features import return_lag

def _series(values: list[float]) -> pd.Series:
    """Helper: build a Pandas Series with business-day index."""
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values)))

def test_return_lag_1d():
    s = _series([100.0, 110.0, 121.0])
    result = return_lag(s, n=1)
    #result[0] is NaN(no prior), result[1] 110/100-1=0.10, result[2]=121/110-1=0.10
    assert math.isnan(result.iloc[0])
    assert math.isclose(result.iloc[1], 0.10)
    assert math.isclose(result.iloc[2], 0.10)

def test_return_lag_5d():
    #6 prices, return_5d only valid from index 5 onward
    s = _series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    result = return_lag(s, n=5)
    #result[5]= 110/100-1 = 0.10
    assert math.isnan(result.iloc[4])
    assert math.isclose(result.iloc[5], 0.10)

def test_return_same_length_as_input():
    s = _series([100.0, 101.0, 102.0])
    result = return_lag(s, n=1)
    assert len(result) == len(s)
