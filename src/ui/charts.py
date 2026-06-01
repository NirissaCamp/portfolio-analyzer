"""Plotly chart builders for the dashboard."""

import pandas as pd
import plotly.graph_objects as go

def build_nav_vs_benchmark(
        portfolio_nav: pd.Series,
        benchmark_close: pd.Series,
        portfolio_label: str,
) -> go.Figure:
    """Two-line chart: portfolio NAV vs benchmark, both rebased to 100."""
    aligned = pd.concat(
        [portfolio_nav, benchmark_close],
        axis=1,
        join="inner",
        keys=[portfolio_label, "S&P 500"],
    ).dropna()

    rebased = aligned.div(aligned.iloc[0]) * 100

    fig = go.Figure()
    for col in rebased.columns:
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased[col], name=col))
    fig.update_layout(
        title="Portfolio vs S&P 500 (rebased to 100)",
        yaxis_title="Index value",
        height=420,
        hovermode="x unified",
    )
    return fig
