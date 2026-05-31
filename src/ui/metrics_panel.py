"""6-card metrics panel for a single combined portfolio price series."""

import streamlit as st
import pandas as pd
from src.analytics.returns import annualized_return
from src.analytics.risk import annualized_volatility, max_drawdown, beta
from src.analytics.ratios import sharpe_ratio, alpha

def render_metrics_cards(
        portfolio_nav: pd.Series,
        benchmark_prices: pd.Series,
) -> None:
    """Render 6 metrics cards in two rows."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Annualized Return", f"{annualized_return(portfolio_nav):.2%}")
    col2.metric("Annualized Volatility", f"{annualized_volatility(portfolio_nav):.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio(portfolio_nav):.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Max Drawdown", f"{max_drawdown(portfolio_nav):.2%}")
    col5.metric("Beta vs S&P 500", f"{beta(portfolio_nav, benchmark_prices):.2f}")
    col6.metric("Alpha", f"{alpha(portfolio_nav, benchmark_prices):.2%}")
