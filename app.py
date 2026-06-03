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
        try:
            df = get_price_history(h.ticker, start, end)
        except Exception as e:
            st.error(f"Failed to fetch {h.ticker}: {e}")
            continue
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

    if days_back < 30:
        st.warning("Need at least 30 days for meaningful metrics")
        return

    st.subheader(f"Results for: {portfolio.name}")

    #Overview: market value, cost basis, P&L
    total_cost = sum(h.shares * h.cost_basis for h in portfolio.holdings)
    total_value = 0.0
    for h in portfolio.holdings:
        df = get_price_history(h.ticker, start, end)
        if not df.empty:
            total_value += float(df["Close"].iloc[-1]) * h.shares

    pnl_dollars = total_value - total_cost
    pnl_pct = (pnl_dollars / total_cost) if total_cost else 0.0

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Cost", f"${total_cost:,.2f}")
    ov2.metric("Market Value", f"${total_value:,.2f}")
    ov3.metric("P&L ($)", f"${pnl_dollars:,.2f}")
    ov4.metric("P&L (%)", f"{pnl_pct:.2%}")

    st.divider()

    nav = _weight_nav(portfolio, start, end)
    if nav is None:
        st.error(f"Could not load any holdings - aborting")
        return

    benchmark = get_price_history(BENCHMARK_TICKER, start, end)
    if benchmark.empty:
        st.error(f"Could not load S&P 500 benchmark")
        return

    render_metrics_cards(nav, benchmark["Close"])

    st.divider()

    fig = build_nav_vs_benchmark(nav, benchmark["Close"], portfolio.name)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    #Correlation heatmap (skip if only 1 holding)
    if len(portfolio.holdings) >= 2:
        from src.analytics.correlation import correlation_matrix
        from src.ui.charts import build_correlation_heatmap

        price_series_map = {}
        for h in portfolio.holdings:
            df = get_price_history(h.ticker, start, end)
            if not df.empty:
                price_series_map[h.ticker] = df["Close"]
        if len(price_series_map) >= 2:
            corr = correlation_matrix(price_series_map)
            st.plotly_chart(build_correlation_heatmap(corr), use_container_width=True)
            st.divider()

    #Holdings pie chart based on most recent close * shares
    holdings_value: dict[str, float] = {}
    for h in portfolio.holdings:
        df = get_price_history(h.ticker, start, end)
        if df.empty:
            continue
        latest_close = float(df["Close"].iloc[-1])
        holdings_value[h.ticker] = latest_close * h.shares

    if holdings_value:
        from src.ui.charts import build_holdings_pie
        st.plotly_chart(build_holdings_pie(holdings_value), use_container_width=True)

if __name__ == "__main__":
    main()
