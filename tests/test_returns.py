"""Tests for return calculations."""

import math
import pandas as pd
import pytest
from src.analytics.returns import(
    daily_returns,
    cumulative_return,
    annualized_return,
)

def _series(values: list[float], start_date: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start_date, periods=len(values)) #business days
    return pd.Series(values, index=idx)

def test_daily_returns_two_points():
    s = _series([100.0, 110.0])
    result = daily_returns(s)
    assert len(result) == 1
    assert math.isclose(result.iloc[0], 0.10, abs_tol=1e-9)

def test_daily_returns_drops_first_nan():
    s = _series([100.0, 110.0, 121.0])
    result = daily_returns(s)
    assert len(result) == 2
    assert math.isclose(result.iloc[0], 0.10)
    assert math.isclose(result.iloc[1], 0.10)

def test_cumulative_return_positive():
    s = _series([100.0, 120.0])
    assert math.isclose(cumulative_return(s), 0.20)

def test_cumulative_return_negative():
    s = _series([100.0, 90.0])
    assert math.isclose(cumulative_return(s), -0.10)

def test_annualized_return_full_year_equals_cumulative():
    #252 business days = ~1 year of trading
    values = [100.0 * (1.20 ** (i/251)) for i in range(252)]
    s = _series(values)
    assert math.isclose(annualized_return(s), 0.20, abs_tol=0.001)
