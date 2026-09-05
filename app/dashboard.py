# Requirements: streamlit>=1.60.0, plotly>=6.0.0
# COMPLETE REPLACEMENT for app/dashboard.py
# This version intentionally does NOT use pandas Styler anywhere.
# That removes the Streamlit/Pandas Styler KeyError permanently.

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

st.set_page_config(
    page_title="NSE Sectoral Breadth & Buy Setups",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1440px; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: .03em; }
    h1 { font-weight: 700; letter-spacing: -.02em; }
    h3 { font-weight: 650; margin-top: 1.5rem; color: #1f2937; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def num_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def fmt_int(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_num(value: object, decimals: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: object, source_is_fraction: bool = True) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        number = float(value)
        if source_is_fraction:
            number *= 100
        return f"{number:,.2f}%"
    except (TypeError, ValueError):
        return "—"


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
    if "leadership_score" not in data.columns:
        data["leadership_score"] = data["strength_score"] if "strength_score" in data.columns else 0.0
    if "actionability_score" not in data.columns:
        data["actionability_score"] = 0.0
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    return data


def ensure_stock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["symbol", "industry", "basic_industry", "sector"]:
        if column not in data.columns:
            data[column] = "Unclassified"
        data[column] = data[column].map(clean_text)
    for flag in ["established_buy_setup", "ipo_buy_setup"]:
        if flag not in data.columns:
            data[flag] = 0
        data[flag] = pd.to_numeric(data[flag], errors="coerce").fillna(0).astype(int)
    return data


def trading_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if "date" not in frame.columns:
        return []
    return sorted(pd.Timestamp(value) for value in pd.to_datetime(frame["date"].dropna().unique()))


def global_date_navigator(dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = pd.Timestamp(dates[-1])
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest

    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in dates:
        selected = latest
        st.session_state[state_key] = latest
    index = dates.index(selected)

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

    valid_dates = [value for value in dates if value <= pd.Timestamp(chosen)]
    resolved = valid_dates[-1] if valid_dates else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"As of {resolved.strftime('%d %b %Y')}")
    return resolved


def make_group_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    wanted = [
        group_column, "leadership_score", "actionability_score", "regime", "members",
        "nh_nl_net", "eq_ret_1d", "eq_ret_5d", "eq_ret_20d", "eq_ret_60d",
        "pct_above_50", "pct_above_200", "acc_minus_dist", "breakout_count", "vcp_ready_count",
    ]
    data = frame[[column for column in wanted if column in frame.columns]].copy()
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
    data = data.reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    return data


def display_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["1D Return", "5D Return", "20D Return", "60D Return", "Stocks Above 50 EMA", "Stocks Above 200 EMA"]:
        if column in data.columns:
            data[column] = data[column].map(lambda value: fmt_pct(value, True))
    for column in ["Rank", "Constituent Stocks", "Accumulation − Distribution", "Breakouts", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].map(fmt_int)
    for column in ["Leadership Score", "Actionability (Setup %)", "Net New Highs (%)"]:
        if column in data.columns:
            data[column] = data[column].map(fmt_num)
    return data


def display_stock_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["1D Return", "5D Return", "20D Return", "60D Return", "Distance from 52W High", "6M Gain", "Candle Range", "Price Tightness (3D)", "Above VWAP", "Off Post-List High"]:
        if column in data.columns:
            data[column] = data[column].map(lambda value: fmt_pct(value, True))
    for column in ["Rank", "Industry Rank", "Days Listed", "Heavy Volume Days (6M)"]:
        if column in data.columns:
            data[column] = data[column].map(fmt_int)
    for column in ["Strength", "Close", "Buy Setup Score", "Buy Priority Score", "Setup Score", "Vol Contraction (vs 50D)", "50D Up/Down Vol", "14D ATR", "Avg Turnover (Cr)", "Industry Leadership Score"]:
        if column in data.columns:
            data[column] = data[column].map(fmt_num)
    return data


def show_dataframe(frame: pd.DataFrame, height: int, link_column: bool = False) -> None:
    config = {}
    if link_column and "Chart" in frame.columns:
        config["Chart"] = st.column_config.LinkColumn("TradingView", display_text="Open ↗")
    st.dataframe(frame, use_container_width=True, hide_index=True, height=height, column_config=config)


def percentile_rank(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    values = num_series(frame, column, default=float("nan"))
    ranks = values.rank(pct=True, ascending=ascending)
    return ranks.fillna(0.5)


def overview_tab(basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    basic = basic_history[basic_history["date"] == selected_date].copy()
    if basic.empty:
        st.warning("No Basic Industry data is available for this date.")
        return

    recent_dates = basic_history["date"].sort_values().unique()[-20:]
    recent = basic_history[basic_history["date"].isin(recent_dates)]
    hunt_series = recent[recent["regime"] == "Fresh Leader (HUNT)"].groupby("date")["basic_industry"].nunique()
    wait_series = recent[recent["regime"] == "Extended Leader (WAIT)"].groupby("date")["basic_industry"].nunique()
    total_series = recent.groupby("date")["basic_industry"].nunique()
    action_series = recent.groupby("date")["actionability_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Data", selected_date.strftime("%d %b %Y"), chart_data=hunt_series, chart_type="area")
    c2.metric("Basic Industries", fmt_int(basic["basic_industry"].nunique()), chart_data=total_series, chart_type="line")
    c3.metric("Fresh Leaders", fmt_int((basic["regime"] == "Fresh Leader (HUNT)").sum()), chart_data=hunt_series, chart_type="bar")
    c4.metric("Extended Leaders", fmt_int((basic["regime"] == "Extended Leader (WAIT)").sum()), chart_data=wait_series, chart_type="bar")
    c5.metric("Avg Actionability", fmt_num(num_series(basic, "actionability_score").mean(), 1), chart_data=action_series, chart_type="area")

    st.markdown("### Sector Leadership Treemap")
    tree = basic[num_series(basic, "members") >= SMALL_GROUP_LIMIT].copy().sort_values("members", ascending=False)
    tree["tile_label"] = tree["basic_industry"].where(tree.index.isin(tree.head(MAX_LABELED_TILES).index), "")
    fig = go.Figure(go.Treemap(
        labels=tree["basic_industry"], parents=[""] * len(tree), values=num_series(tree, "members"),
        text=tree["tile_label"], textinfo="text",
        marker=dict(colors=num_series(tree, "leadership_score"), colorscale="Greens", showscale=True),
        hovertemplate="%{label}<br>Members: %{value}<br>Leadership: %{marker.color:.1f}<extra></extra>",
    ))
    fig.update_layout(title="Sector Size colored by Leadership Score — top 25 tiles labeled")
    st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    st.markdown("### Market Breadth")
    g1, g2 = st.columns(2)
    for holder, column, title in [(g1, "pct_above_50", "% Stocks Above 50 EMA"), (g2, "pct_above_200", "% Stocks Above 200 EMA")]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=num_series(basic, column).mean(),
            title=dict(text=title, font=dict(size=14)),
            gauge=dict(axis=dict(range=[0, 100]), steps=[
                dict(range=[0, 30], color="#fecaca"), dict(range=[30, 70], color="#fef3c7"), dict(range=[70, 100], color="#86efac"),
            ]),
        ))
        holder.plotly_chart(styled_fig(fig, 270), use_container_width=True)

    st.markdown("### Leadership vs Actionability")
    scatter = basic[num_series(basic, "members") >= SMALL_GROUP_LIMIT].copy()
    label_idx = scatter.nlargest(MAX_LABELED_BUBBLES, "leadership_score").index
    scatter["label"] = scatter["basic_industry"].where(scatter.index.isin(label_idx), "")
    fig = go.Figure(go.Scatter(
        x=num_series(scatter, "leadership_score"), y=num_series(scatter, "actionability_score"),
        mode="markers+text", text=scatter["label"], textposition="top center",
        marker=dict(size=(num_series(scatter, "members") * 2).clip(lower=8, upper=70), color=scatter["regime"].map(PALETTE).fillna("#9AA5B1"), line=dict(width=1, color="white")),
        hovertext=scatter["basic_industry"],
        hovertemplate="%{hovertext}<br>Leadership: %{x:.1f}<br>Actionability: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Top 15 leaders labeled; hover every point for details", xaxis_title="Leadership Score", yaxis_title="Actionability (Setup %)", xaxis=dict(range=[0, 100]))
    st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

    st.markdown("### Current Basic Industry Leadership")
    group_table = make_group_table(basic[num_series(basic, "members") >= SMALL_GROUP_LIMIT], "basic_industry")
    show_dataframe(display_group_table(group_table), 480)


def top_buy_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("🎯 Top Individual Buy Setups")
    st.caption("No industry gate. Stocks are ranked individually. Industry Rank and Industry Leadership Score are context only.")

    basic = basic_history[basic_history["date"] == selected_date].copy()
    stocks = stock_history[stock_history["date"] == selected_date].copy()
    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return

    context = basic[["basic_industry", "leadership_score", "regime"]].copy()
    context = context.sort_values("leadership_score", ascending=False).drop_duplicates("basic_industry").reset_index(drop=True)
    context.insert(0, "Industry Rank", range(1, len(context) + 1))
    context = context.rename(columns={"basic_industry": "Basic Industry", "leadership_score": "Industry Leadership Score", "regime": "Industry Regime"})

    buy = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualified Established", fmt_int(len(buy)))
    c2.metric("Qualified IPO", fmt_int(len(ipo)))
    c3.metric("Shortlist", f"Top {TOP_N_SETUPS}")
    c4.metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top 20 Established Buy Setups")
    if buy.empty:
        st.info("No established stocks pass the individual hard gates on this date.")
    else:
        buy["Buy Priority Score"] = (
            0.30 * percentile_rank(buy, "tight_3d_range", ascending=True)
            + 0.25 * percentile_rank(buy, "vol_ratio_50", ascending=True)
            + 0.20 * percentile_rank(buy, "gain_6m", ascending=False)
            + 0.15 * percentile_rank(buy, "up_down_ratio", ascending=False)
            + 0.10 * percentile_rank(buy, "stock_strength_score", ascending=False)
        ) * 100
        buy = buy.merge(context[["Basic Industry", "Industry Rank", "Industry Leadership Score"]], left_on="basic_industry", right_on="Basic Industry", how="left")
        buy = buy.sort_values("Buy Priority Score", ascending=False).head(TOP_N_SETUPS).reset_index(drop=True)
        buy.insert(0, "Rank", range(1, len(buy) + 1))
        buy["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy["symbol"].astype(str)

        chart = buy.sort_values("Buy Priority Score")
        fig = go.Figure(go.Bar(x=chart["Buy Priority Score"], y=chart["symbol"], orientation="h", marker=dict(color=PALETTE["Fresh Leader (HUNT)"]), text=chart["Buy Priority Score"].round(1), textposition="outside"))
        fig.update_layout(title="Priority Score Ranking", xaxis=dict(range=[0, 115]), yaxis_title=None)
        st.plotly_chart(styled_fig(fig, max(240, 30 * len(chart))), use_container_width=True)

        display = buy.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close", "tight_3d_range": "Price Tightness (3D)", "vol_ratio_50": "Vol Contraction (vs 50D)", "gain_6m": "6M Gain", "nearest_ema_tag": "EMA Proximity", "momentum_badge": "Momentum"})
        columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership Score", "Close", "Buy Priority Score", "Price Tightness (3D)", "Vol Contraction (vs 50D)", "6M Gain", "EMA Proximity", "Momentum"]
        display = display[[column for column in columns if column in display.columns]]
        show_dataframe(display_stock_table(display), 560, link_column=True)

    st.markdown("### Top 20 IPO Setups")
    st.caption("Newly listed stocks (<150 days), no industry filter, ranked by IPO Setup Score.")
    if ipo.empty:
        st.info("No IPO stocks pass the individual IPO gates on this date.")
    else:
        ipo["Avg Turnover (Cr)"] = num_series(ipo, "ipo_turnover_avg") / 10_000_000
        if "ipo_setup_score" not in ipo.columns:
            ipo["ipo_setup_score"] = (
                0.25 * percentile_rank(ipo, "tight_3d_range", ascending=True)
                + 0.20 * percentile_rank(ipo, "vol_ratio_50", ascending=True)
                + 0.20 * percentile_rank(ipo, "vwap_premium", ascending=False)
                + 0.20 * percentile_rank(ipo, "retracement_from_listing_high", ascending=True)
                + 0.15 * percentile_rank(ipo, "hh_hl_count", ascending=False)
            ) * 100
        else:
            ipo["ipo_setup_score"] = num_series(ipo, "ipo_setup_score")

        ipo = ipo.merge(context[["Basic Industry", "Industry Rank", "Industry Leadership Score"]], left_on="basic_industry", right_on="Basic Industry", how="left")
        ipo = ipo.sort_values("ipo_setup_score", ascending=False).head(TOP_N_SETUPS).reset_index(drop=True)
        ipo.insert(0, "Rank", range(1, len(ipo) + 1))
        ipo["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo["symbol"].astype(str)

        chart = ipo.sort_values("ipo_setup_score")
        fig = go.Figure(go.Bar(x=chart["ipo_setup_score"], y=chart["symbol"], orientation="h", marker=dict(color=PALETTE["Extended Leader (WAIT)"]), text=chart["ipo_setup_score"].round(1), textposition="outside"))
        fig.update_layout(title="IPO Setup Score Ranking", xaxis=dict(range=[0, 115]), yaxis_title=None)
        st.plotly_chart(styled_fig(fig, max(240, 30 * len(chart))), use_container_width=True)

        display = ipo.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close", "days_listed": "Days Listed", "ipo_phase": "Phase", "ipo_setup_score": "Setup Score", "vwap_premium": "Above VWAP", "retracement_from_listing_high": "Off Post-List High"})
        columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership Score", "Close", "Days Listed", "Phase", "Setup Score", "Above VWAP", "Off Post-List High", "Avg Turnover (Cr)"]
        display = display[[column for column in columns if column in display.columns]]
        show_dataframe(display_stock_table(display), max(240, 40 * len(display) + 60), link_column=True)

    st.markdown("### Full Basic Industry Leaderboard — Context Only")
    show_dataframe(context, 360)


def basic_industry_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = global_date_navigator(trading_dates(basic_history), "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()
    f1, f2, f3 = st.columns([1.45, 0.85, 0.85])
    with f1:
        regimes = st.multiselect("Trading State filter", list(PALETTE), default=list(PALETTE), key="basic_regimes")
    with f2:
        minimum = st.number_input("Minimum stocks", min_value=1, value=SMALL_GROUP_LIMIT, step=1, key="basic_minimum")
    with f3:
        ranking = st.selectbox("Ranking", ["Highest Leadership", "Lowest Leadership"], key="basic_sort")
    if regimes:
        selected = selected[selected["regime"].isin(regimes)]
    selected = selected[num_series(selected, "members") >= minimum]
    table = make_group_table(selected, "basic_industry")
    if ranking == "Lowest Leadership" and "Leadership Score" in table.columns:
        table = table.sort_values("Leadership Score").reset_index(drop=True)
        table["Rank"] = range(1, len(table) + 1)
    if table.empty:
        st.warning("No Basic Industries match the selected filters.")
        return
    show_dataframe(display_group_table(table), 410)

    st.markdown("### Selected Basic Industry Details")
    selected_group = st.selectbox("Basic Industry", table["Basic Industry"].tolist(), key="basic_group_selector")
    stocks = stock_history[(stock_history["date"] == selected_date) & (stock_history["basic_industry"] == selected_group)].copy()
    if stocks.empty:
        st.info("No stock records found for this Basic Industry.")
        return
    stocks = stocks.sort_values("ret_20d", ascending=False) if "ret_20d" in stocks.columns else stocks
    stocks = stocks.reset_index(drop=True)
    stocks.insert(0, "Rank", range(1, len(stocks) + 1))
    stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
    display = stocks.rename(columns={"symbol": "Symbol", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain", "stock_strength_score": "Strength"})
    columns = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength"]
    display = display[[column for column in columns if column in display.columns]]
    show_dataframe(display_stock_table(display), 320, link_column=True)


def industry_tab(industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = global_date_navigator(trading_dates(industry_history), "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    selected = selected[num_series(selected, "members") >= SMALL_GROUP_LIMIT]
    table = make_group_table(selected, "industry")
    if table.empty:
        st.warning("No Industry data is available for this date.")
        return
    show_dataframe(display_group_table(table), 470)

    st.markdown("### Selected Industry Details")
    selected_group = st.selectbox("Industry", table["Industry"].tolist(), key="industry_group_selector")
    stocks = stock_history[(stock_history["date"] == selected_date) & (stock_history["industry"] == selected_group)].copy()
    if stocks.empty:
        st.info("No stock records found for this Industry.")
        return
    stocks = stocks.sort_values("ret_20d", ascending=False) if "ret_20d" in stocks.columns else stocks
    stocks = stocks.reset_index(drop=True)
    stocks.insert(0, "Rank", range(1, len(stocks) + 1))
    stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
    display = stocks.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain", "stock_strength_score": "Strength", "nearest_ema_tag": "EMA Proximity", "momentum_badge": "Momentum"})
    columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "EMA Proximity", "Momentum"]
    display = display[[column for column in columns if column in display.columns]]
    show_dataframe(display_stock_table(display), 320, link_column=True)


def methodology_tab() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
**Individual stock selection:** The Top Buy Setups tab does not filter stocks by parent-industry leadership. It selects from the upstream individual setup flags, then ranks the strongest 20 established and IPO candidates. Industry Rank and Industry Leadership Score are context fields for discretionary confirmation.

**Established setup ranking:** Price tightness 30%, volume contraction 25%, six-month gain 20%, up/down volume ratio 15%, and stock strength 10%.

**Established hard gates:** The upstream `established_buy_setup` flag is expected to enforce liquidity (average turnover ≥ 5 Cr), EMA alignment, trend requirements, and the volume/volatility contraction conditions.

**IPO setup ranking:** Tightness 25%, volume dry-up 20%, VWAP premium 20%, post-listing retracement 20%, and HH-HL structure 15%. The upstream IPO flag is expected to enforce its liquidity criteria.

This is a research and ranking dashboard, not a trade recommendation. Confirm price structure, liquidity, entry level, invalidation, and risk before taking a position.
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

    all_dates = sorted(set(trading_dates(basic_history) + trading_dates(industry_history) + trading_dates(stock_history)))
    if not all_dates:
        st.error("No valid trading dates found in history files.")
        st.stop()

    st.title("NSE Sectoral Breadth & 2-Axis Setup Engine")
    sync_text = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Data as of {sync_text.replace('T', ' ').replace('Z', ' IST')}")

    tabs = st.tabs(["🎯 Top Buy Setups", "Overview", "Basic Industry", "Industry", "Methodology"])
    with tabs[0]:
        top_buy_tab(basic_history, stock_history, global_date_navigator(all_dates, "buys"))
    with tabs[1]:
        overview_tab(basic_history, global_date_navigator(all_dates, "overview"))
    with tabs[2]:
        basic_industry_tab(basic_history, stock_history)
    with tabs[3]:
        industry_tab(industry_history, stock_history)
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
