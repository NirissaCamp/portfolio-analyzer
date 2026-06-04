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

def beta(portfolio_prices: pd.Series, benchmark_prices: pd.Series) -> float:
    """Beta = cov(portfolio_returns, benchmark_returns) / var(benchmark_returns).
    Aligns the two series on their common dates first."""

    p_returns = portfolio_prices.pct_change().dropna()
    b_returns = benchmark_prices.pct_change().dropna()
    aligned = pd.concat([p_returns, b_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    p_col, b_col = aligned.columns
    covariance = aligned[p_col].cov(aligned[b_col])
    benchmark_var = aligned[b_col].var()
    if benchmark_var == 0:
        return 0.0
    return float(covariance / benchmark_var)
