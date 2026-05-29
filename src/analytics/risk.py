"""Risk metrics: volatility, drawdown, beta."""

from math import sqrt
import pandas as pd
from src.config import TRADING_DAYS_PER_YEAR

def annualized_volatility(prices: pd.Series) -> float:
    """Annualized standard deviation if daily returns."""
    returns = prices.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * sqrt(TRADING_DAYS_PER_YEAR))

def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough drop, as a negative number(or 0 if no drop)."""
    if prices.empty:
        return 0.0
    running_max = prices.cummax()
    drawdowns = (prices - running_max) / running_max
    return float(drawdowns.min())
