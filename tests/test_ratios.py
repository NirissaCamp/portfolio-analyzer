"""Test for Sharpe and Alpha."""

import math
import pandas as pd
from src.analytics.ratios import sharpe_ratio, alpha

def _series(values: list[float], start_date: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start_date, periods=len(values))
    return pd.Series(values, index=idx)

def test_sharpe_positive_returns_above_risk_free():
    #252 days, 20% annualized return, low volatility => positive sharpe
    values = [100.0 *(1.20 ** (i/251)) for i in range(252)]
    s = _series(values)
    result = sharpe_ratio(s)
    assert result > 0

def test_sharpe_zero_volatility_returns_zero():
    s = _series([100.0] * 252)
    assert sharpe_ratio(s) == 0.0

def test_alpha_zero_when_portfolio_matches_benchmark():
    #Identical returns =>alpha == 0 (after rounding)
    values = [100.0 * (1.10 ** (i / 251)) for i in range(252)]
    s = _series(values)
    result = alpha(s, s)
    assert abs(result) < 0.001
