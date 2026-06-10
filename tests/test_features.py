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


from src.ml.features import volatility, sma_ratio, volume_ratio

def test_volatility_zero_on_flat_prices():
    s = _series([100.0] * 25)
    result = volatility(s, window=20)
    #After 20-day warmup, std of all-zero returns is 0
    assert math.isclose(result.iloc[20], 0.0, abs_tol=1e-9)

def test_volatility_positive_on_oscillating_prices():
    #Alternating +1%/-1% pattern
    values = [100.0]
    for i in range(1, 30):
        values.append(values[-1] * (1.01 if i % 2 else 0.99))
    s = _series(values)
    result = volatility(s, window=20)
    assert result.iloc[25] > 0 # some volatility detected

def test_sma_ratio_above_one_when_short_above_long():
    #Steady uptrend -> SMA(5) > SMA(20)
    s = _series([100.0 + i for i in range(30)])
    result = sma_ratio(s, short=5, long=20)
    assert result.iloc[25] > 1.0

def test_sma_ration_at_one_when_flat():
    s = _series([100.0] * 30)
    result = sma_ratio(s, short=5, long=20)

def test_volume_ratio_one_when_volume_constant():
    v = _series([1000.0] * 25)
    result = volume_ratio(v, window=20)
    assert math.isclose(result.iloc[22], 1.0, abs_tol=1e-9)

def test_volume_ratio_spike():
    v = _series([1000.0] *20 + [5000.0, 1000.0, 1000, 0]) #spike on day 20
    result = volume_ratio(v, window=20)
    assert result.iloc[20] > 4.0  #5000 vs ~1000 avg
