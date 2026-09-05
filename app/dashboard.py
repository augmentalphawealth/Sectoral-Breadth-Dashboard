# Requirements: streamlit>=1.60.0, plotly>=6.0.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = PROCESSED / "dashboard_basic_industry_history.parquet"
INDUSTRY_HISTORY_FILE = PROCESSED / "dashboard_industry_history.parquet"
STOCK_HISTORY_FILE = PROCESSED / "dashboard_stock_history.parquet"
SYNC_FILE = PROCESSED / "last_sync.txt"
SMALL_GROUP_LIMIT = 5
TOP_N_SETUPS = 20
MAX_LABELED_BUBBLES = 15
MAX_LABELED_TILES = 25

INK = "#1F2937"
PALETTE = {
    "Fresh Leader (HUNT)": "#2E7D63",
    "Extended Leader (WAIT)": "#D98E3B",
    "Speculative Coil (AVOID)": "#8B5FBF",
    "Dead (AVOID)": "#B0483C",
    "Neutral Transition": "#9AA5B1",
}

CHART_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK, size=13)


def styled_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        font=CHART_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=15, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


st.set_page_config(
    page_title="NSE Sectoral Breadth & Buy Setups",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1440px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.03em; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h3 { font-weight: 600; margin-top: 1.6rem; color: #1F2937; }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #e5e7eb; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 18px; }
    div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def as_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def fmt_int(value: object) -> str:
    number = as_number(value)
    return "—" if number is None else f"{int(round(number)):,}"


def fmt_num(value: object, decimals: int = 2) -> str:
    number = as_number(value)
    return "—" if number is None else f"{number:,.{decimals}f}"


def fmt_pct(value: object, source_is_fraction: bool) -> str:
    number = as_number(value)
    if number is None:
        return "—"
    if source_is_fraction:
        number *= 100.0
    return f"{number:,.2f}%"


def heat_color(value: object) -> str:
    score = as_number(value)
    if score is None:
        return "background-color: #f8fafc; color: #64748b;"
    if score >= 80:
        return "background-color: #14532d; color: #ffffff; font-weight: 700;"
    if score >= 70:
        return "background-color: #166534; color: #ffffff; font-weight: 700;"
    if score >= 60:
        return "background-color: #22c55e; color: #052e16; font-weight: 700;"
    if score >= 50:
        return "background-color: #86efac; color: #14532d; font-weight: 700;"
    if score >= 40:
        return "background-color: #fef3c7; color: #78350f; font-weight: 700;"
    if score >= 30:
        return "background-color: #fed7aa; color: #7c2d12; font-weight: 700;"
    return "background-color: #fecaca; color: #7f1d1d; font-weight: 700;"


def actionability_color(value: object) -> str:
    pct = as_number(value)
    if pct is None:
        return ""
    if pct >= 20:
        return "background-color: #1e3a8a; color: #ffffff; font-weight: 700;"
    if pct > 0:
        return "background-color: #bfdbfe; color: #1e3a8a; font-weight: 700;"
    return "color: #94a3b8;"


def style_with_heatmap(raw: pd.DataFrame, display: pd.DataFrame):
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    if "Leadership Score" in raw.columns and "Leadership Score" in display.columns:
        styles.loc[:, "Leadership Score"] = raw["Leadership Score"].map(heat_color)
    if "Industry Leadership Score" in raw.columns and "Industry Leadership Score" in display.columns:
        styles.loc[:, "Industry Leadership Score"] = raw["Industry Leadership Score"].map(heat_color)
    if "Strength" in raw.columns and "Strength" in display.columns:
        styles.loc[:, "Strength"] = raw["Strength"].map(heat_color)
    if "Actionability (Setup %)" in raw.columns and "Actionability (Setup %)" in display.columns:
        styles.loc[:, "Actionability (Setup %)"] = raw["Actionability (Setup %)"].map(actionability_color)
    if "Buy Setup Score" in raw.columns and "Buy Setup Score" in display.columns:
        styles.loc[:, "Buy Setup Score"] = raw["Buy Setup Score"].map(heat_color)
    return display.style.apply(lambda _: styles, axis=None)


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def ensure_group_columns(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Unclassified"
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    if "leadership_score" not in data.columns and "strength_score" in data.columns:
        data["leadership_score"] = data["strength_score"]
    if "actionability_score" not in data.columns:
        data["actionability_score"] = 0.0
    return data


def ensure_stock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["symbol", "industry", "basic_industry", "sector"]:
        if column not in data.columns:
            data[column] = "Unclassified"
        data[column] = data[column].map(clean_text)
    return data


def trading_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(frame["date"].dropna().unique())
    return sorted(pd.Timestamp(date) for date in dates)


def global_date_navigator(dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = pd.Timestamp(dates[-1])
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest

    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in dates:
        selected = latest
        st.session_state[state_key] = latest

    index = dates.index(selected) if selected in dates else len(dates) - 1

    heading, controls = st.columns([5.7, 2.3])
    with heading:
        st.subheader("Historical Date")
    with controls:
        previous, calendar, next_button = st.columns([0.4, 1.55, 0.4])
        with previous:
            if st.button("‹", key=f"{key}_previous", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = dates[index - 1]
                st.rerun()
        with calendar:
            chosen = st.date_input(
                "Historical date",
                value=selected.date(),
                min_value=pd.Timestamp(dates[0]).date(),
                max_value=latest.date(),
                key=f"{key}_calendar",
                label_visibility="collapsed",
            )
        with next_button:
            if st.button("›", key=f"{key}_next", disabled=index == len(dates) - 1, use_container_width=True):
                st.session_state[state_key] = dates[index + 1]
                st.rerun()

    requested = pd.Timestamp(chosen)
    valid_dates = [date for date in dates if date <= requested]
    resolved = valid_dates[-1] if valid_dates else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"As of {resolved.strftime('%d %b %Y')}")
    return resolved


def make_group_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    wanted = [
        group_column, "leadership_score", "actionability_score", "regime",
        "members", "nh_nl_net", "eq_ret_1d", "eq_ret_5d", "eq_ret_20d",
        "eq_ret_60d", "pct_above_50", "pct_above_200", "acc_minus_dist",
        "breakout_count", "vcp_ready_count",
    ]
    data = frame[[c for c in wanted if c in frame.columns]].copy()
    data = data.rename(columns={
        group_column: "Basic Industry" if group_column == "basic_industry" else "Industry",
        "leadership_score": "Leadership Score",
        "actionability_score": "Actionability (Setup %)",
        "regime": "Trading State",
        "members": "Constituent Stocks",
        "nh_nl_net": "Net New Highs (%)",
        "eq_ret_1d": "1D Return",
        "eq_ret_5d": "5D Return",
        "eq_ret_20d": "20D Return",
        "eq_ret_60d": "60D Return",
        "pct_above_50": "Stocks Above 50 EMA",
        "pct_above_200": "Stocks Above 200 EMA",
        "acc_minus_dist": "Accumulation − Distribution",
        "breakout_count": "Breakouts",
        "vcp_ready_count": "VCP Ready",
    })
    if "Leadership Score" in data.columns:
        data = data.sort_values("Leadership Score", ascending=False)
    data.insert(0, "Rank", range(1, len(data) + 1))
    return data.reset_index(drop=True)


def format_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["1D Return", "5D Return", "20D Return", "60D Return", "Stocks Above 50 EMA", "Stocks Above 200 EMA"]:
        if column in data.columns:
            data[column] = data[column].apply(lambda value: fmt_pct(value, True))
    for column in ["Rank", "Constituent Stocks", "Accumulation − Distribution", "Breakouts", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    for column in ["Leadership Score", "Actionability (Setup %)", "Net New Highs (%)"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_num)
    return data


def format_stock_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["1D Return", "5D Return", "20D Return", "60D Return", "Distance from 52W High", "6M Gain", "Candle Range", "Price Tightness (3D)"]:
        if column in data.columns:
            data[column] = data[column].apply(lambda value: fmt_pct(value, True))
    for column in ["Rank", "Industry Rank", "Heavy Volume Days (6M)"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    for column in ["Strength", "Close", "Buy Setup Score", "Buy Priority Score", "Vol Contraction (vs 50D)", "50D Up/Down Vol", "14D ATR", "Avg Turnover (Cr)", "Industry Leadership Score"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_num)
    return data


def overview_tab(basic_history: pd.DataFrame, industry_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("Market Breadth & 2-Axis Overview")

    basic = basic_history[basic_history["date"] == selected_date].copy()

    kpi_dates = basic_history["date"].sort_values().unique()[-20:]
    basic_kpi = basic_history[basic_history["date"].isin(kpi_dates)]

    hunt_count_series = basic_kpi[basic_kpi["regime"] == "Fresh Leader (HUNT)"].groupby("date")["basic_industry"].nunique()
    wait_count_series = basic_kpi[basic_kpi["regime"] == "Extended Leader (WAIT)"].groupby("date")["basic_industry"].nunique()
    total_bi = basic_kpi.groupby("date")["basic_industry"].nunique()
    avg_actionability = basic_kpi.groupby("date")["actionability_score"].mean()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Latest Data", selected_date.strftime("%d %b %Y"), chart_data=hunt_count_series, chart_type="area")
    with kpi2:
        st.metric("Basic Industries", fmt_int(basic["basic_industry"].nunique()), chart_data=total_bi, chart_type="line")
    with kpi3:
        st.metric("Fresh Leaders (HUNT)", f"{(basic['regime'] == 'Fresh Leader (HUNT)').sum()}", chart_data=hunt_count_series, chart_type="bar")
    with kpi4:
        st.metric("Extended Leaders (WAIT)", f"{(basic['regime'] == 'Extended Leader (WAIT)').sum()}", chart_data=wait_count_series, chart_type="bar")
    with kpi5:
        st.metric("Avg Actionability", fmt_num(basic["actionability_score"].mean(), 1), chart_data=avg_actionability, chart_type="area")

    st.markdown("### Sector Leadership Treemap")
    treemap_df = basic[basic["members"] >= SMALL_GROUP_LIMIT].copy().sort_values("members", ascending=False)
    labeled_idx = treemap_df.head(MAX_LABELED_TILES).index
    treemap_df["tile_label"] = treemap_df["basic_industry"].where(treemap_df.index.isin(labeled_idx), "")
    fig_treemap = go.Figure(go.Treemap(
        labels=treemap_df["basic_industry"],
        parents=[""] * len(treemap_df),
        values=treemap_df["members"],
        text=treemap_df["tile_label"],
        textinfo="text",
        marker=dict(colors=treemap_df["leadership_score"], colorscale="Greens", showscale=True),
        hovertemplate="%{label}<br>Members: %{value}<br>Leadership: %{marker.color:.1f}<extra></extra>",
    ))
    fig_treemap.update_layout(title="Sector Size (Members) colored by Leadership Score, top 25 tiles labeled", margin=dict(l=10, r=10, t=44, b=10))
    st.plotly_chart(styled_fig(fig_treemap, height=420), use_container_width=True)

    st.markdown("### Regime Composition (Last 60 Sessions)")
    regime_dates = basic_history["date"].sort_values().unique()[-60:]
    regime_df = basic_history[basic_history["date"].isin(regime_dates)].copy()
    regime_counts = regime_df.groupby(["date", "regime"]).size().unstack(fill_value=0).reset_index()

    fig_regime = go.Figure()
    for regime in ["Fresh Leader (HUNT)", "Extended Leader (WAIT)", "Neutral Transition", "Speculative Coil (AVOID)", "Dead (AVOID)"]:
        if regime in regime_counts.columns:
            fig_regime.add_trace(go.Scatter(
                x=regime_counts["date"], y=regime_counts[regime], stackgroup="one",
                name=regime, fillcolor=PALETTE.get(regime, "#9AA5B1"), line=dict(width=0),
            ))
    fig_regime.update_layout(title="Regime Mix Over Time", barmode="stack", yaxis_title="Count", margin=dict(l=10, r=10, t=44, b=10))
    st.plotly_chart(styled_fig(fig_regime, height=320), use_container_width=True)

    st.markdown("### Market Breadth Gauges")
    g1, g2 = st.columns(2)
    pct_above_50 = (basic["pct_above_50"].mean()) if "pct_above_50" in basic.columns else 0
    pct_above_200 = (basic["pct_above_200"].mean()) if "pct_above_200" in basic.columns else 0

    with g1:
        fig_gauge_50 = go.Figure(go.Indicator(
            mode="gauge+number", value=pct_above_50,
            title=dict(text="% Stocks Above 50 EMA", font=dict(size=14)),
            gauge=dict(axis=dict(range=[0, 100]), steps=[
                dict(range=[0, 30], color="#fecaca"), dict(range=[30, 70], color="#fef3c7"), dict(range=[70, 100], color="#86efac"),
            ]),
        ))
        st.plotly_chart(styled_fig(fig_gauge_50, height=270), use_container_width=True)
    with g2:
        fig_gauge_200 = go.Figure(go.Indicator(
            mode="gauge+number", value=pct_above_200,
            title=dict(text="% Stocks Above 200 EMA", font=dict(size=14)),
            gauge=dict(axis=dict(range=[0, 100]), steps=[
                dict(range=[0, 30], color="#fecaca"), dict(range=[30, 70], color="#fef3c7"), dict(range=[70, 100], color="#86efac"),
            ]),
        ))
        st.plotly_chart(styled_fig(fig_gauge_200, height=270), use_container_width=True)

    st.markdown("### Leadership vs Actionability Scatter")
    scatter_df = basic[basic["members"] >= SMALL_GROUP_LIMIT].copy()
    top_label_idx = scatter_df.nlargest(MAX_LABELED_BUBBLES, "leadership_score").index
    scatter_df["point_label"] = scatter_df["basic_industry"].where(scatter_df.index.isin(top_label_idx), "")
    fig_scatter = go.Figure(go.Scatter(
        x=scatter_df["leadership_score"], y=scatter_df["actionability_score"],
        mode="markers+text", text=scatter_df["point_label"], textposition="top center",
        marker=dict(
            size=scatter_df["members"] * 2,
            color=scatter_df["regime"].map(PALETTE).fillna("#9AA5B1"),
            line=dict(width=1, color="white"),
        ),
        hovertext=scatter_df["basic_industry"],
        hovertemplate="%{hovertext}<br>Leadership: %{x:.1f}<br>Actionability: %{y:.1f}%<br>Members: %{marker.size:.0f}<extra></extra>",
    ))
    fig_scatter.update_layout(
        title="Each bubble = Basic Industry (size = members, color = regime); top 15 leaders labeled",
        xaxis_title="Leadership Score", yaxis_title="Actionability (Setup %)",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, max(scatter_df["actionability_score"].max() * 1.2, 30)]),
        margin=dict(l=10, r=10, t=44, b=10),
    )
    st.plotly_chart(styled_fig(fig_scatter, height=400), use_container_width=True)

    st.markdown("### Current Basic Industry Leadership (2-Axis Matrix)")
    eligible_basic = basic[basic["members"] >= SMALL_GROUP_LIMIT] if "members" in basic.columns else basic
    raw = make_group_table(eligible_basic, "basic_industry")
    st.dataframe(style_with_heatmap(raw, format_group_table(raw)), use_container_width=True, hide_index=True, height=480)


def top_buy_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("🎯 Top Individual Buy Setups")
    st.caption(
        "Stock-level ranking only — no parent-industry gate. Every stock already passes the hard "
        "gates (liquidity ≥ 5 Cr turnover, full EMA alignment, trend, volatility/volume contraction, "
        "precision score). Industry Rank/Leadership are shown as CONTEXT so you can judge a strong "
        "stock sitting in a lagging industry yourself."
    )

    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    stock_latest = stock_history[stock_history["date"] == selected_date].copy()

    industry_context = basic_latest[["basic_industry", "leadership_score", "regime"]].copy()
    industry_context = industry_context.sort_values("leadership_score", ascending=False).reset_index(drop=True)
    industry_context.insert(0, "industry_rank", range(1, len(industry_context) + 1))
    industry_context = industry_context.rename(columns={
        "basic_industry": "Basic Industry",
        "leadership_score": "Industry Leadership Score",
        "regime": "Industry Regime",
        "industry_rank": "Industry Rank",
    })

    buy_candidates = stock_latest[stock_latest["established_buy_setup"] == 1].copy()
    ipo_candidates = stock_latest[stock_latest["ipo_buy_setup"] == 1].copy()

    m_col = st.columns(4)
    m_col[0].metric("Total Qualified Setups", fmt_int(len(buy_candidates)))
    m_col[1].metric("Total IPO Setups (>5 Cr)", fmt_int(len(ipo_candidates)))
    m_col[2].metric("Shortlist Size", f"Top {TOP_N_SETUPS}")
    m_col[3].metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top 20 Established Buy Setups (Individual Ranking)")
    if buy_candidates.empty:
        st.info("No stocks currently pass the hard gates (liquidity, EMA alignment, trend, precision score) on this date.")
    else:
        buy_candidates["buy_priority_score"] = (
            0.30 * (1 - buy_candidates["tight_3d_range"].rank(pct=True))
            + 0.25 * (1 - buy_candidates["vol_ratio_50"].rank(pct=True))
            + 0.20 * buy_candidates["gain_6m"].rank(pct=True)
            + 0.15 * buy_candidates["up_down_ratio"].rank(pct=True)
            + 0.10 * buy_candidates["stock_strength_score"].rank(pct=True)
        ) * 100

        buy_candidates = buy_candidates.merge(
            industry_context[["Basic Industry", "Industry Rank", "Industry Leadership Score"]],
            left_on="basic_industry", right_on="Basic Industry", how="left",
        )

        buy_candidates = buy_candidates.sort_values("buy_priority_score", ascending=False).head(TOP_N_SETUPS)
        buy_candidates = buy_candidates.reset_index(drop=True)
        buy_candidates.insert(0, "Rank", range(1, len(buy_candidates) + 1))

        chart_df = buy_candidates.sort_values("buy_priority_score")
        fig = go.Figure(go.Bar(
            x=chart_df["buy_priority_score"].round(1), y=chart_df["symbol"], orientation="h",
            marker=dict(color=PALETTE["Fresh Leader (HUNT)"]),
            text=chart_df["buy_priority_score"].round(1), textposition="outside",
        ))
        fig.update_layout(title="Top 20 by Priority Score", xaxis_title=None, yaxis_title=None, xaxis=dict(range=[0, 115]))
        st.plotly_chart(styled_fig(fig, height=max(240, 30 * len(chart_df))), use_container_width=True)

        buy_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy_candidates["symbol"].astype(str)

        display_buy = buy_candidates.rename(columns={
            "symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
            "buy_priority_score": "Buy Priority Score",
            "tight_3d_range": "Price Tightness (3D)", "vol_ratio_50": "Vol Contraction (vs 50D)",
            "gain_6m": "6M Gain", "nearest_ema_tag": "EMA Proximity", "momentum_badge": "Momentum",
        })
        keep_cols = [
            "Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership Score",
            "Close", "Buy Priority Score", "Price Tightness (3D)", "Vol Contraction (vs 50D)",
            "6M Gain", "EMA Proximity", "Momentum",
        ]
        display_buy = display_buy[[col for col in keep_cols if col in display_buy.columns]]

        st.dataframe(
            style_with_heatmap(display_buy, format_stock_table(display_buy)),
            use_container_width=True, hide_index=True, height=560,
            column_config={
                "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
                "Industry Leadership Score": st.column_config.ProgressColumn("Industry Leadership Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

    st.markdown("### Top 20 IPO Setups (Individual Ranking)")
    st.caption("Newly listed stocks (<150 days), gated only on liquidity (turnover > 5 Cr). No industry filter — ranked purely by Setup Score.")
    if ipo_candidates.empty:
        st.info("No newly listed stocks currently meet the liquidity and score bar.")
    else:
        if "ipo_turnover_avg" in ipo_candidates.columns:
            ipo_candidates["Avg Turnover (Cr)"] = ipo_candidates["ipo_turnover_avg"] / 10000000.0
        else:
            ipo_candidates["Avg Turnover (Cr)"] = None

        ipo_candidates = ipo_candidates.merge(
            industry_context[["Basic Industry", "Industry Rank", "Industry Leadership Score"]],
            left_on="basic_industry", right_on="Basic Industry", how="left",
        )
        ipo_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo_candidates["symbol"].astype(str)

        if "ipo_setup_score" in ipo_candidates.columns:
            ipo_candidates = ipo_candidates.sort_values("ipo_setup_score", ascending=False).head(TOP_N_SETUPS)
        else:
            ipo_candidates = ipo_candidates.head(TOP_N_SETUPS)

        ipo_candidates = ipo_candidates.reset_index(drop=True)
        ipo_candidates.insert(0, "Rank", range(1, len(ipo_candidates) + 1))

        if "ipo_setup_score" in ipo_candidates.columns:
            chart_df = ipo_candidates.sort_values("ipo_setup_score")
            fig = go.Figure(go.Bar(
                x=chart_df["ipo_setup_score"].round(1), y=chart_df["symbol"], orientation="h",
                marker=dict(color=PALETTE["Extended Leader (WAIT)"]),
                text=chart_df["ipo_setup_score"].round(1), textposition="outside",
            ))
            fig.update_layout(title="Top 20 IPO Setups by Score", xaxis_title=None, yaxis_title=None, xaxis=dict(range=[0, 115]))
            st.plotly_chart(styled_fig(fig, height=max(240, 30 * len(chart_df))), use_container_width=True)

        display_ipo = ipo_candidates.rename(columns={
            "symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
            "days_listed": "Days Listed", "ipo_phase": "Phase", "ipo_setup_score": "Setup Score",
            "vwap_premium": "Above VWAP", "retracement_from_listing_high": "Off Post-List High",
        })
        keep_ipo = [
            "Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership Score",
            "Close", "Days Listed", "Phase", "Setup Score", "Above VWAP", "Off Post-List High", "Avg Turnover (Cr)",
        ]
        display_ipo = display_ipo[[col for col in keep_ipo if col in display_ipo.columns]]

        st.dataframe(
            style_with_heatmap(display_ipo, format_stock_table(display_ipo)),
            use_container_width=True, hide_index=True, height=max(240, 40 * len(display_ipo) + 60),
            column_config={
                "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
                "Industry Leadership Score": st.column_config.ProgressColumn("Industry Leadership Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

    st.markdown("### Reference: Full Basic Industry Leaderboard (context only, not a filter)")
    st.caption("Use this to manually check whether a stock above sits in a lagging industry before you buy.")
    st.dataframe(
        industry_context, use_container_width=True, hide_index=True, height=360,
        column_config={"Industry Leadership Score": st.column_config.ProgressColumn("Industry Leadership Score", min_value=0, max_value=100, format="%.1f")},
    )


def basic_industry_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = global_date_navigator(trading_dates(basic_history), "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()

    filters = st.columns([1.45, 0.85, 0.85])
    with filters[0]:
        regimes = st.multiselect(
            "Trading State filter",
            ["Fresh Leader (HUNT)", "Extended Leader (WAIT)", "Speculative Coil (AVOID)", "Dead (AVOID)", "Neutral Transition"],
            default=["Fresh Leader (HUNT)", "Extended Leader (WAIT)", "Speculative Coil (AVOID)", "Dead (AVOID)", "Neutral Transition"],
            key="basic_regimes",
        )
    with filters[1]:
        minimum = st.number_input("Minimum stocks", min_value=1, value=SMALL_GROUP_LIMIT, step=1, key="basic_minimum")
    with filters[2]:
        ranking = st.selectbox("Ranking", ["Highest Leadership", "Lowest Leadership"], key="basic_sort")

    if regimes:
        selected = selected[selected["regime"].isin(regimes)]
    if "members" in selected.columns:
        selected = selected[selected["members"] >= minimum]
    raw_table = make_group_table(selected, "basic_industry")
    if ranking == "Lowest Leadership" and "Leadership Score" in raw_table.columns:
        raw_table = raw_table.sort_values("Leadership Score").reset_index(drop=True)
        raw_table["Rank"] = range(1, len(raw_table) + 1)
    if raw_table.empty:
        st.warning("No Basic Industries match the selected filters.")
        return

    st.dataframe(style_with_heatmap(raw_table, format_group_table(raw_table)), use_container_width=True, hide_index=True, height=410)

    st.markdown("### Selected Basic Industry Details")
    selected_group = st.selectbox("Basic Industry", raw_table["Basic Industry"].tolist(), key="basic_group_selector")

    stocks = stock_history[(stock_history["date"] == selected_date) & (stock_history["basic_industry"] == selected_group)].copy()
    if not stocks.empty:
        stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
        if "ret_20d" in stocks.columns:
            stocks = stocks.sort_values("ret_20d", ascending=False)
        stocks = stocks.reset_index(drop=True)
        stocks.insert(0, "Rank", range(1, len(stocks) + 1))
        disp_stocks = stocks.rename(columns={
            "symbol": "Symbol", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return",
            "gain_6m": "6M Gain", "stock_strength_score": "Strength",
        })
        keep_cols = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength"]
        disp_stocks = disp_stocks[[col for col in keep_cols if col in disp_stocks.columns]]
        st.dataframe(
            style_with_heatmap(disp_stocks, format_stock_table(disp_stocks)),
            use_container_width=True, hide_index=True, height=320,
            column_config={"Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗")},
        )


def industry_tab(industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = global_date_navigator(trading_dates(industry_history), "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    if "members" in selected.columns:
        selected = selected[selected["members"] >= SMALL_GROUP_LIMIT]
    raw_table = make_group_table(selected, "industry")
    if raw_table.empty:
        st.warning("No Industry data is available for this date.")
        return
    st.dataframe(style_with_heatmap(raw_table, format_group_table(raw_table)), use_container_width=True, hide_index=True, height=470)

    st.markdown("### Selected Industry Details")
    selected_group = st.selectbox("Industry", raw_table["Industry"].tolist(), key="industry_group_selector")

    stocks = stock_history[(stock_history["date"] == selected_date) & (stock_history["industry"] == selected_group)].copy()
    if not stocks.empty:
        stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
        if "ret_20d" in stocks.columns:
            stocks = stocks.sort_values("ret_20d", ascending=False)
        stocks = stocks.reset_index(drop=True)
        stocks.insert(0, "Rank", range(1, len(stocks) + 1))
        disp_stocks = stocks.rename(columns={
            "symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
            "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain",
            "stock_strength_score": "Strength", "nearest_ema_tag": "EMA Proximity", "momentum_badge": "Momentum",
        })
        keep_cols = ["Rank", "Symbol", "Chart", "Basic Industry", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "EMA Proximity", "Momentum"]
        disp_stocks = disp_stocks[[col for col in keep_cols if col in disp_stocks.columns]]
        st.dataframe(
            style_with_heatmap(disp_stocks, format_stock_table(disp_stocks)),
            use_container_width=True, hide_index=True, height=320,
            column_config={"Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗")},
        )


def methodology_tab() -> None:
    st.subheader("Methodology (2-Axis System)")
    st.markdown(
        """
**Axis 1: Leadership Score (Macro Institutional Trend, 0–100)**
* **35% Price Velocity:** Median 20-day and 60-day equal-weighted returns percentile-ranked cross-sectionally.
* **35% Structural Alignment:** Percentage of constituents in full EMA alignment (20 EMA > 50 EMA > 200 EMA).
* **30% Institutional Volume:** 50-Day Cumulative Up/Down Volume Ratio.
* *Smoothed with a 3-day EWM to eliminate single-day ranking noise.*

**Axis 2: Actionability Score (Micro Setup Density %)**
Displays the exact raw percentage of stocks in that industry passing the **2-Rule Hard Gate + 3-Metric Precision Score**:
0. **Liquidity:** 20-day average turnover ≥ 5 Crore.
1. **Trend:** Price > 50 EMA, 20 EMA > 50 EMA > 200 EMA, within -25% of the 52-week high.
2. **Power (20%):** 6-Month Gain, percentile-ranked within the eligible pool.
3. **Coil (35%):** 3-Day Range ÷ ATR-14, percentile-ranked (tighter = higher score).
4. **Dry-up (45%):** Today's volume ÷ 50-day average volume, percentile-ranked (lower = higher score).

**Top Buy Setups (Individual Ranking, no industry gate)**
A stock is eligible purely on its own merits: it must pass the hard gates (liquidity, EMA alignment,
trend, precision score) via the `established_buy_setup` flag. Within that pool, stocks are ranked by a
blended **Priority Score** (tightness 30%, vol contraction 25%, gain 20%, up/down ratio 15%, strength 10%).
The stock's parent Basic Industry Rank and Leadership Score are shown as context columns only — use them
to manually judge whether a strong stock in a lagging industry is worth the risk.

**IPO Setups (New Listings, no industry gate)**
Stocks listed <150 days, gated on liquidity only (≥ 5 Crore average turnover).
Ranked by a blended **Setup Score**: tightness 25%, dry-up 20%, VWAP premium 20%, retracement 20%, HH-HL structure 15%.

**EMA Proximity Tag**
Contextual metadata showing signed distance to the nearest of 10/20/50 EMA ("On EMA", "Testing", "Riding", "Extended", "Broken").
This is displayed but not scored — it is a trader's timing filter, not a model input.
        """
    )


def main() -> None:
    required = [BASIC_HISTORY_FILE, INDUSTRY_HISTORY_FILE, STOCK_HISTORY_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Required dashboard files are missing. Run the data workflow first.")
        st.code("\n".join(missing))
        st.stop()

    basic_history = ensure_group_columns(load_parquet(str(BASIC_HISTORY_FILE)), "basic_industry")
    industry_history = ensure_group_columns(load_parquet(str(INDUSTRY_HISTORY_FILE)), "industry")
    stock_history = ensure_stock_columns(load_parquet(str(STOCK_HISTORY_FILE)))

    st.title("NSE Sectoral Breadth & 2-Axis Setup Engine")
    sync_text = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Data as of {sync_text.replace('T', ' ').replace('Z', ' IST')}")

    all_dates = sorted(set(trading_dates(basic_history) + trading_dates(industry_history) + trading_dates(stock_history)))
    if not all_dates:
        st.error("No valid trading dates found in history files.")
        st.stop()

    tabs = st.tabs(["🎯 Top Buy Setups", "Overview", "Basic Industry", "Industry", "Methodology"])

    with tabs[0]:
        buy_setup_date = global_date_navigator(all_dates, "buys")
        top_buy_tab(basic_history, stock_history, buy_setup_date)
    with tabs[1]:
        overview_date = global_date_navigator(all_dates, "overview")
        overview_tab(basic_history, industry_history, stock_history, overview_date)
    with tabs[2]:
        basic_industry_tab(basic_history, stock_history)
    with tabs[3]:
        industry_tab(industry_history, stock_history)
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
