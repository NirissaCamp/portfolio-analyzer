"""Feature engineering for  return forecasting."""

import pandas as pd

def return_lag(prices: pd.Series, n: int) -> pd.Series:
    """Return the n-period lookback return for each date.
    return_lag(t, n) = prices[t] / prices[t-n] - 1
    First n rows are NaN (no prior price).
    """
    return prices / prices.shift(n) - 1
