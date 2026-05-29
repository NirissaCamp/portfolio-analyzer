"""Tests for risk metrics."""

import math
import pandas as pd
import pytest
from src.analytics.risk import annualized_volatility, max_drawdown
from src.analytics.risk import beta

def _series(values:list[float], start_date: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start_date, periods=len(values))
    return pd.Series(values, index=idx)

def test_annualized_volatility_zero_when_flat():
    s = _series([100.0] * 10)
    assert math.isclose(annualized_volatility(s), 0.0, abs_tol=1e-9)

def test_annualized_volatility_known_input():
    #Alternating +1% / -1% daily returns
    values = [100.0]
    for i in range(1, 252):
        values.append(values[-1] * (1.01 if i%2 else 0.99))
    s = _series(values)

    # Daily return std ~= 0.01, annualized ~= 0.01 * sqrt(252) ~= 0.159
    result = annualized_volatility(s)
    assert 0.10 < result < 0.20

def test_max_drawdown_v_shape():
    # 100 -> 120 -> 60 -> 80: peak 120, though 60, drawdown = -50%
    s = _series([100.0, 120.0, 60.0, 80.0])
    assert math.isclose(max_drawdown(s), -0.50)

def test_max_drawdown_monotonic_increase_is_zero():
    s = _series([100.0, 110.0, 120.0, 130.0])
    assert math.isclose(max_drawdown(s), 0.0)

def test_beta_perfectly_correlated_one_to_one():
    benchmark = _series([100.0, 102.0, 104.0, 102.0, 100.0])
    portfolio = _series([200.0, 204.0, 208.0, 204.0, 200.0]) #same %, 2x price
    #Identical returns -> beta == 1.0
    assert math.isclose(beta(portfolio, benchmark), 1.0, abs_tol=1e-9)

def test_beta_2x_amplified():
    benchmark = _series([100.0, 102.0, 100.0, 102.0])  # +2%, -2%, +2%
    portfolio = _series([100.0, 104.0, 100.0, 104.0])  # +4%, -4%, +4%
    assert math.isclose(beta(portfolio, benchmark), 2.0, abs_tol=0.05)
