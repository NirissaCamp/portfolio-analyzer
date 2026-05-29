"""Correlation matrix across portfolio holdings,"""

import pandas as pd

def correlation_matrix(price_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of daily returns.
    Input: {ticker: price_series}
    Output: DataFrame with tickers on both axes.
    """
    returns = {ticker: prices.pct_change().dropna() for ticker, prices in price_series.items()}
    df = pd.DataFrame(returns)
    return df.corr(method="pearson")
