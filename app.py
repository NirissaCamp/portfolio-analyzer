"""Portfolio Analyzer - Streamlit entry point."""

from datetime import date, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config import BENCHMARK_TICKER
from src.data import get_price_history
from src.data.models import Portfolio
from src.ui.input_form import render_portfolio_input
from src.ui.metrics_panel import render_metrics_cards

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

def _equal_weight_nav(portfolio: Portfolio, start: date, end: date) -> pd.Series | None:
    """Compute equal-weighted NAV from holdings. Returns None if no data."""
    closes = {}
    for h in portfolio.holdings:
        df = get_price_history(h.ticker, start, end)
        if df.empty:
            st.warning(f"No data for {h.ticker}")
            continue
        closes[h.ticker] = df["Close"]
    if not closes:
        return None
    combined = pd.DataFrame(closes).dropna()
    #Rebase each column to 100 at the start, then average equally
    rebased = combined.div(combined.iloc[0]) * 100
    return rebased.mean(axis=1)

def main() -> None:
    st.title("Portfolio Analyzer")

    portfolio = render_portfolio_input()
    days_back = st.slider("Lookback (days)", 30, 730, 365)

    if portfolio is None:
        st.info("Add at least one holding to analyze")
        return

    if not st.button("Analyze"):
        return

    end = date.today()
    start = end - timedelta(days=days_back)

    st.subheader(f"Results for: {portfolio.name}")

    nav = _equal_weight_nav(portfolio, start, end)
    if nav is None:
        st.error(f"Could not load any holdings - aborting")
        return

    benchmark = get_price_history(BENCHMARK_TICKER, start, end)
    if benchmark.empty:
        st.error(f"Could not load S&P 500 benchmark")
        return

    render_metrics_cards(nav, benchmark["Close"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name=portfolio.name))
    fig.update_layout(title = "Portfolio NAV (equal-weighted, rebased to 100)", height=400)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
