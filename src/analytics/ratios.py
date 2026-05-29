"""Performance ratios: Sharpe and Alpha."""

import pandas as pd
from src.config import RISK_FREE_RATE
from src.analytics.returns import annualized_return
from src.analytics.risk import annualized_volatility, beta


def sharpe_ratio(prices: pd.Series) -> float:
    """(annualized_return - risk_free_rate) / annualized_volatility."""
    vol = annualized_volatility(prices)
    if vol == 0:
        return 0.0
    return float((annualized_return(prices) - RISK_FREE_RATE) / vol)

def alpha(portfolio_prices: pd.Series, benchmark_prices: pd.Series) -> float:
    """Excess return over what CAPM predicts given the portfolio's beta."""
    portfilio_ar = annualized_return(portfolio_prices)
    benchmark_ar = annualized_return(benchmark_prices)
    b = beta(portfolio_prices, benchmark_prices)
    camp_expected = RISK_FREE_RATE + b * (benchmark_ar - RISK_FREE_RATE)
    return float(portfilio_ar - camp_expected)
