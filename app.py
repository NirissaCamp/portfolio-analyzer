"""Portfolio Analyzer - Streamlit entry point."""

from datetime import date, timedelta
import streamlit as st
import pandas as pd

from src.config import BENCHMARK_TICKER
from src.data import get_price_history
from src.data.models import Portfolio
from src.ui.input_form import render_portfolio_input
from src.ui.metrics_panel import render_metrics_cards
from src.ui.charts import build_nav_vs_benchmark

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

def _weight_nav(portfolio: Portfolio, start: date, end: date) -> pd.Series | None:
    """Compute  NAV time series weighted by current shares per holdings.
    NAV(t) = sum over holdings of (shares * close_price(t)).
    Then rebased to 100 at start date for display.
    """
    holdings_close = {}
    for h in portfolio.holdings:
        df = get_price_history(h.ticker, start, end)
        if df.empty:
            st.warning(f"No data for {h.ticker}")
            continue
        holdings_close[h.ticker] = df["Close"] * h.shares
    if not holdings_close:
        return None
    combined = pd.DataFrame(holdings_close).dropna()
    nav = combined.sum(axis=1)
    return nav / nav.iloc[0] * 100 # rebase to 100

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

    nav = _weight_nav(portfolio, start, end)
    if nav is None:
        st.error(f"Could not load any holdings - aborting")
        return

    benchmark = get_price_history(BENCHMARK_TICKER, start, end)
    if benchmark.empty:
        st.error(f"Could not load S&P 500 benchmark")
        return

    render_metrics_cards(nav, benchmark["Close"])

    fig = build_nav_vs_benchmark(nav, benchmark["Close"], portfolio.name)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
