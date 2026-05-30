"""Portfolio Analyzer - Streamlit entry point."""

from datetime import date, timedelta
import streamlit as st
import plotly.graph_objects as go

from src.data import get_price_history
from src.ui.input_form import render_portfolio_input

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

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

    fig = go.Figure()
    for holding in portfolio.holdings:
        with st.spinner(f"Fetching  {holding.ticker}..."):
            prices = get_price_history(holding.ticker, start, end)
        if prices.empty:
            st.warning(f"No data for {holding.ticker}")
            continue
        fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name=holding.ticker))
    fig.update_layout(title = f"Closing Prices", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
