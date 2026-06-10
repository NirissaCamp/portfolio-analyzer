"""Feature engineering for  return forecasting."""

import pandas as pd

def return_lag(prices: pd.Series, n: int) -> pd.Series:
    """Return the n-period lookback return for each date.
    return_lag(t, n) = prices[t] / prices[t-n] - 1
    First n rows are NaN (no prior price).
    """
    return prices / prices.shift(n) - 1

def volatility(prices: pd.Series, window: int=20) -> pd.Series:
    """Rolling standard deviation of daily returns over 'window' days."""
    daily_returns = prices.pct_change()
    return daily_returns.rolling(window).std(ddof=0)

def sma_ratio(prices: pd.Series, short: int=5, long: int=20) -> pd.Series:
    """Ratio of short-window SMA to long-window SMA.
    > 1 means short-term trend above long-term; < 1 the reverse.
    """
    return prices.rolling(short).mean() / prices.rolling(long).mean()

def volume_ratio(volume: pd.Series, window: int=20) -> pd.Series:
    """Today's volume divided by rolling average volume."""
    return volume / volume.rolling(window).mean()

def rsi(prices: pd.Series, window: int=14) -> pd.Series:
    """Relative Strength Index (RSI) - 14 days classic momentum oscillator.
    RSI ranges 0-100. > 70 typically "overbought"; < 30 typically "oversold".
    Uses simple moving average of gains/losses (not Wilder's smoothing).
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = - delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
