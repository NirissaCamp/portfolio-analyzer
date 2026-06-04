"""Return calculations: daily, cumulative, annualized."""

import pandas as pd
from src.config import TRADING_DAYS_PER_YEAR

def daily_returns(prices: pd.Series) -> pd.Series:
    """Compute simple daily returns. First row is dropped( no prior price)."""
    return prices.pct_change().dropna()

def cumulative_return(prices: pd.Series) -> float:
    """Total return over the full series: (end/start) - 1."""
    if len(prices) < 2:
        return 0.0
    return float(prices.iloc[-1] / prices.iloc[0] - 1)

def annualized_return(prices: pd.Series) -> float:
    """Annualized the cumulative return using 252-trading-day convention."""
    n_days = len(prices)
    if n_days < 2:
        return 0.0
    total = cumulative_return(prices)
    return float((1 + total) **(TRADING_DAYS_PER_YEAR/(n_days-1)) - 1)
