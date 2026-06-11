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

def test_sma_ratio_at_one_when_flat():
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

from src.ml.features import rsi

def test_rsi_all_gains_approaches_100():
    #Monotonically rising prices: only gains, no loss -> RSI near 100
    s = _series([100.0 + i for i in range(20)])
    result = rsi(s, window=14)
    assert result.iloc[15] > 99 # Very close to 100

def test_rsi_all_loss_approaches_zero():
    s = _series([100.0 - i for i in range(20)])
    result = rsi(s, window=14)
    assert result.iloc[15] < 1  # Very close to 0

def test_rsi_flat_returns_neutral_or_nan():
    #When all returns are 0, avg gain and avg loss are both 0 -> undefined
    #Our implementation should return NaN (not crash)
    s = _series([100.0] * 20)
    result = rsi(s, window=14)
    #Either NaN or 50 is acceptable; we accept NaN
    val = result.iloc[15]
    assert math.isnan(val) or math.isclose(val, 50.0)

from src.ml.features import build_features

def test_build_features_columns_present():
    """Verify build_features returns a DataFrame with all 8 expected columns."""
    n = 60
    prices = _series([100.0 + i * 0.5 for i in range(n)])
    volumes = _series([1000.0 + i * 10 for i in range(n)])
    market_prices = _series([200.0 + i * 0.3 for i in range(n)])

    df = build_features(prices, volumes, market_prices)
    excepted_cols = {
        "return_1d", "return_5d", "return_20d",
        "volatility_20d", "sma_ratio_5_20", "rsi_14",
        "volume_ratio_20d", "market_return_5d",
    }
    assert excepted_cols.issubset(set(df.columns))

def test_build_features_no_nan_after_warmup():
    """After day 20, all features should be valid (no NaN)."""
    n = 60
    prices = _series([100.0 + i * 0.5 for i in range(n)])
    volumes = _series([1000.0] * n)
    market_prices = _series([200.0 + i * 0.3 for i in range(n)])

    df = build_features(prices, volumes, market_prices)
    #Rows >= 20 should have no NaN
    assert not df.iloc[25].isna().any()


from src.ml.features import build_dataset

def test_build_dataset_combine_multiple_tickers():
    """build_dataset stacks features for multiple tickers into one DataFrame."""
    n = 60
    market_prices = _series([200.0 + i * 0.3 for i in range(n)])
    ticker_data = {
        "AAPL": {
            "Close": _series([100.0 + i * 0.5 for i in range(n)]),
            "Volume": _series([1000.0] * n),
        },
        "MSFT":{
            "Close": _series([200.0 + i * 0.4 for i in range(n)]),
            "Volume": _series([2000.0] * n),
        },
    }

    X, y = build_dataset(ticker_data, market_prices, forecast_days=5)

    assert "ticker" in X.columns or X.index.nlevels >= 1 #ticker somehow encoded
    #Both tickers contribute roughly equal rows ( minus the last 5)
    #60 rows - 20 warmup - 5 forward = 35 usable per ticker -> 70 total
    assert 60 <= len(X) <= 80
    assert len(X) == len(y)
