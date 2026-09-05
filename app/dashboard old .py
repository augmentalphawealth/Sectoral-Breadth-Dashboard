# COMPLETE REPLACEMENT: app/dashboard.py
# Institutional-style redesign. Removes raw Excel-style dumps and the
# overlapping scatter/treemap clutter. Fixes the 0-1 vs 0-100 scaling bug
# that was making Leadership/Actionability look tiny on some charts.

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
TOP_N_INDUSTRIES_SHOWN = 15

INK = "#0F172A"
MUTED = "#64748B"
PALETTE = {
    "Fresh Leader (HUNT)": "#15803D",
    "Extended Leader (WAIT)": "#B45309",
    "Speculative Coil (AVOID)": "#7C3AED",
    "Dead (AVOID)": "#B91C1C",
    "Neutral Transition": "#64748B",
}

st.set_page_config(page_title="NSE Sectoral Breadth & Buy Setups", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {padding-top:1.3rem;padding-bottom:2.5rem;max-width:1400px}
[data-testid="stMetricValue"] {font-size:1.6rem;font-weight:700;color:#0F172A}
[data-testid="stMetricLabel"] {font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.06em;font-weight:600}
[data-testid="stMetricDelta"] {font-size:.75rem}
h1 {font-weight:700;letter-spacing:-.02em;color:#0F172A}
h2, h3 {font-weight:650;color:#0F172A;margin-top:1.4rem;margin-bottom:.3rem}
p.section-note {color:#64748B;font-size:.86rem;margin-top:-.2rem;margin-bottom:.8rem}
[data-testid="stDataFrame"] {border:1px solid #E2E8F0;border-radius:10px;overflow:hidden}
.stTabs [data-baseweb="tab"] {font-weight:600}
hr {margin:1.6rem 0;border-color:#E2E8F0}
</style>
""", unsafe_allow_html=True)


def fig_style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=44, b=10),
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14.5, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"), yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
    )
    return fig


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def autoscale_to_100(series: pd.Series) -> pd.Series:
    """Fixes the bug where score columns arrive as 0-1 fractions instead of 0-100."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if len(finite) and finite.max() <= 1.5:
        return values * 100
    return values


def number(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def integer(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def percent(value: object, fraction: bool = True) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        value = float(value) * (100 if fraction else 1)
        return f"{value:,.2f}%"
    except (TypeError, ValueError):
        return "—"


@st.cache_data(show_spinner=False)
def load(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def prepare_groups(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Unclassified"
    if "leadership_score" not in data.columns:
        data["leadership_score"] = data["strength_score"] if "strength_score" in data.columns else 0.0
    if "actionability_score" not in data.columns:
        data["actionability_score"] = 0.0
    data["leadership_score"] = autoscale_to_100(data["leadership_score"])
    data["actionability_score"] = autoscale_to_100(data["actionability_score"])
    data[group_column] = data[group_column].map(clean)
    data["regime"] = data["regime"].map(clean)
    return data


def prepare_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for col in ["symbol", "industry", "basic_industry", "sector"]:
        if col not in data.columns:
            data[col] = "Unclassified"
        data[col] = data[col].map(clean)
    for flag in ["established_buy_setup", "ipo_buy_setup"]:
        if flag not in data.columns:
            data[flag] = 0
        data[flag] = pd.to_numeric(data[flag], errors="coerce").fillna(0).astype(int)
    return data


def dates_of(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if "date" not in frame.columns:
        return []
    return sorted(pd.Timestamp(x) for x in pd.to_datetime(frame["date"].dropna().unique()))


def date_picker(dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = dates[-1]
    state_key = f"{key}_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest
    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in dates:
        selected = latest
        st.session_state[state_key] = latest
    index = dates.index(selected)
    left, right = st.columns([5.7, 2.3])
    with left:
        st.subheader("Historical Date")
    with right:
        prev, calendar, nxt = st.columns([.4, 1.55, .4])
        with prev:
            if st.button("‹", key=f"{key}_prev", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = dates[index - 1]
                st.rerun()
        with calendar:
            chosen = st.date_input("Date", value=selected.date(), min_value=dates[0].date(), max_value=latest.date(), key=f"{key}_calendar", label_visibility="collapsed")
        with nxt:
            if st.button("›", key=f"{key}_next", disabled=index == len(dates) - 1, use_container_width=True):
                st.session_state[state_key] = dates[index + 1]
                st.rerun()
    valid = [d for d in dates if d <= pd.Timestamp(chosen)]
    resolved = valid[-1] if valid else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"As of {resolved.strftime('%d %b %Y')}")
    return resolved


def rank_pct(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    values = numeric(frame, column, float("nan"))
    return values.rank(pct=True, ascending=ascending).fillna(.5)


def show(data: pd.DataFrame, height: int, links: bool = False, progress_cols: list[str] | None = None) -> None:
    data = data.copy()
    data.columns = [str(c) for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated(keep="first")]
    config = {}
    if links and "Chart" in data.columns:
        config["Chart"] = st.column_config.LinkColumn("Chart", display_text="Open ↗")
    for col in (progress_cols or []):
        if col in data.columns:
            config[col] = st.column_config.ProgressColumn(col, min_value=0, max_value=100, format="%.0f")
    st.dataframe(data, use_container_width=True, hide_index=True, height=height, column_config=config)


# ---------------------------------------------------------------------------
# OVERVIEW — institutional layout: KPI strip, ranked bar leaderboards
# (no overlapping scatter/treemap clutter), compact curated table.
# ---------------------------------------------------------------------------
def overview_tab(basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    basic = basic_history[basic_history["date"] == selected_date].copy()
    if basic.empty:
        st.warning("No Basic Industry data is available for this date.")
        return
    basic = basic[numeric(basic, "members") >= SMALL_GROUP_LIMIT].copy()

    st.subheader("Market Breadth Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Basic Industries Tracked", integer(basic["basic_industry"].nunique()))
    c2.metric("Fresh Leaders (HUNT)", integer((basic["regime"] == "Fresh Leader (HUNT)").sum()))
    c3.metric("Extended Leaders (WAIT)", integer((basic["regime"] == "Extended Leader (WAIT)").sum()))
    c4.metric("Avg Leadership Score", number(basic["leadership_score"].mean(), 1))
    c5.metric("Avg Actionability %", number(basic["actionability_score"].mean(), 1))

    g1, g2 = st.columns(2)
    for holder, col, title in [(g1, "pct_above_50", "% Stocks Above 50 EMA"), (g2, "pct_above_200", "% Stocks Above 200 EMA")]:
        value = autoscale_to_100(numeric(basic, col)).mean() if col in basic.columns else 0
        fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number=dict(suffix="%"), title=dict(text=title, font=dict(size=13)),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=INK, thickness=.25),
                       steps=[dict(range=[0, 30], color="#FEE2E2"), dict(range=[30, 70], color="#FEF3C7"), dict(range=[70, 100], color="#DCFCE7")])))
        holder.plotly_chart(fig_style(fig, 230), use_container_width=True)

    st.markdown("---")
    st.markdown("### Sector Leadership — Top 15 by Leadership Score")
    st.markdown('<p class="section-note">Ranked bar view replaces the old overlapping bubble chart. Bar length = Leadership Score, color = current trading regime.</p>', unsafe_allow_html=True)
    top_lead = basic.nlargest(TOP_N_INDUSTRIES_SHOWN, "leadership_score").sort_values("leadership_score")
    fig = go.Figure(go.Bar(
        x=top_lead["leadership_score"], y=top_lead["basic_industry"], orientation="h",
        marker=dict(color=top_lead["regime"].map(PALETTE).fillna("#94A3B8")),
        text=top_lead["leadership_score"].round(1), textposition="outside",
        hovertemplate="%{y}<br>Leadership: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(title=None, xaxis=dict(range=[0, 105], title="Leadership Score"), yaxis_title=None)
    st.plotly_chart(fig_style(fig, 60 + 28 * len(top_lead)), use_container_width=True)

    st.markdown("### Setup Density — Top 15 by Actionability %")
    st.markdown('<p class="section-note">How many stocks in each sector currently have a live, tradable setup — independent of Leadership Score.</p>', unsafe_allow_html=True)
    top_action = basic.nlargest(TOP_N_INDUSTRIES_SHOWN, "actionability_score").sort_values("actionability_score")
    fig = go.Figure(go.Bar(
        x=top_action["actionability_score"], y=top_action["basic_industry"], orientation="h",
        marker=dict(color="#1D4ED8"),
        text=top_action["actionability_score"].round(1), textposition="outside",
        hovertemplate="%{y}<br>Actionability: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(range=[0, max(top_action["actionability_score"].max() * 1.25, 10)], title="Actionability (Setup %)"), yaxis_title=None)
    st.plotly_chart(fig_style(fig, 60 + 28 * len(top_action)), use_container_width=True)

    st.markdown("### Sector Leaderboard")
    st.markdown('<p class="section-note">Curated view — only the fields that matter for a quick scan. Full raw table removed.</p>', unsafe_allow_html=True)
    lb = basic.sort_values("leadership_score", ascending=False).head(30).copy()
    lb.insert(0, "Rank", range(1, len(lb) + 1))
    lb = lb.rename(columns={
        "basic_industry": "Basic Industry", "leadership_score": "Leadership", "actionability_score": "Actionability %",
        "regime": "Regime", "members": "Stocks", "eq_ret_20d": "20D Return",
    })
    cols = ["Rank", "Basic Industry", "Leadership", "Actionability %", "Regime", "Stocks", "20D Return"]
    lb = lb[[c for c in cols if c in lb.columns]]
    if "20D Return" in lb.columns:
        lb["20D Return"] = lb["20D Return"].map(lambda v: percent(v, True))
    if "Stocks" in lb.columns:
        lb["Stocks"] = lb["Stocks"].map(integer)
    lb["Leadership"] = lb["Leadership"].round(1)
    lb["Actionability %"] = lb["Actionability %"].round(1)
    show(lb, 420, progress_cols=["Leadership", "Actionability %"])


# ---------------------------------------------------------------------------
# TOP BUY SETUPS — individual stock ranking, no industry gate
# ---------------------------------------------------------------------------
def top_buy_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("🎯 Top Individual Buy Setups")
    st.markdown('<p class="section-note">No industry gate — every stock is judged on its own trend, liquidity, and volatility/volume-contraction gates. Industry Rank/Leadership are shown as context only.</p>', unsafe_allow_html=True)

    basic = basic_history[basic_history["date"] == selected_date].copy()
    stocks = stock_history[stock_history["date"] == selected_date].copy()
    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return

    context = basic[["basic_industry", "leadership_score", "regime"]].copy()
    context = context.sort_values("leadership_score", ascending=False).drop_duplicates("basic_industry").reset_index(drop=True)
    context.insert(0, "industry_rank", range(1, len(context) + 1))
    lookup = context.set_index("basic_industry")[["industry_rank", "leadership_score", "regime"]].rename(
        columns={"industry_rank": "Industry Rank", "leadership_score": "Industry Leadership", "regime": "Industry Regime"})

    buy = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualified Established", integer(len(buy)))
    c2.metric("Qualified IPO", integer(len(ipo)))
    c3.metric("Shortlist Size", f"Top {TOP_N_SETUPS}")
    c4.metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top 20 Established Buy Setups")
    if buy.empty:
        st.info("No established stocks pass the individual hard gates on this date.")
    else:
        buy["Priority Score"] = (0.30 * rank_pct(buy, "tight_3d_range", True) + 0.25 * rank_pct(buy, "vol_ratio_50", True)
            + 0.20 * rank_pct(buy, "gain_6m", False) + 0.15 * rank_pct(buy, "up_down_ratio", False) + 0.10 * rank_pct(buy, "stock_strength_score", False)) * 100
        buy["Industry Rank"] = buy["basic_industry"].map(lookup["Industry Rank"])
        buy["Industry Leadership"] = buy["basic_industry"].map(lookup["Industry Leadership"])
        buy["Industry Regime"] = buy["basic_industry"].map(lookup["Industry Regime"])
        buy = buy.sort_values("Priority Score", ascending=False).head(TOP_N_SETUPS).reset_index(drop=True)
        buy.insert(0, "Rank", range(1, len(buy) + 1))
        buy["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy["symbol"].astype(str)

        chart = buy.sort_values("Priority Score")
        fig = go.Figure(go.Bar(x=chart["Priority Score"], y=chart["symbol"], orientation="h", marker=dict(color=PALETTE["Fresh Leader (HUNT)"]), text=chart["Priority Score"].round(1), textposition="outside"))
        fig.update_layout(title="Priority Score Ranking", xaxis=dict(range=[0, 115]))
        st.plotly_chart(fig_style(fig, max(240, 30 * len(chart))), use_container_width=True)

        display = buy.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
            "tight_3d_range": "Tightness (3D)", "vol_ratio_50": "Vol vs 50D", "gain_6m": "6M Gain",
            "nearest_ema_tag": "EMA Zone", "momentum_badge": "Momentum"})
        columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership", "Close", "Priority Score", "Tightness (3D)", "Vol vs 50D", "6M Gain", "EMA Zone", "Momentum"]
        display = display[[c for c in columns if c in display.columns]]
        for col in ["Tightness (3D)", "6M Gain"]:
            if col in display.columns: display[col] = display[col].map(lambda v: percent(v, True))
        for col in ["Close", "Priority Score", "Vol vs 50D"]:
            if col in display.columns: display[col] = display[col].map(lambda v: number(v, 2))
        if "Industry Leadership" in display.columns: display["Industry Leadership"] = display["Industry Leadership"].round(1)
        show(display, 560, links=True, progress_cols=["Industry Leadership"])

    st.markdown("### Top 20 IPO Setups")
    st.markdown('<p class="section-note">Listed &lt;150 days, liquidity gate only. No industry filter.</p>', unsafe_allow_html=True)
    if ipo.empty:
        st.info("No IPO stocks pass the individual IPO gates on this date.")
    else:
        ipo["Avg Turnover (Cr)"] = numeric(ipo, "ipo_turnover_avg") / 10_000_000
        if "ipo_setup_score" not in ipo.columns:
            ipo["ipo_setup_score"] = (0.25 * rank_pct(ipo, "tight_3d_range", True) + 0.20 * rank_pct(ipo, "vol_ratio_50", True)
                + 0.20 * rank_pct(ipo, "vwap_premium", False) + 0.20 * rank_pct(ipo, "retracement_from_listing_high", True) + 0.15 * rank_pct(ipo, "hh_hl_count", False)) * 100
        else:
            ipo["ipo_setup_score"] = numeric(ipo, "ipo_setup_score")
        ipo["Industry Rank"] = ipo["basic_industry"].map(lookup["Industry Rank"])
        ipo["Industry Leadership"] = ipo["basic_industry"].map(lookup["Industry Leadership"])
        ipo = ipo.sort_values("ipo_setup_score", ascending=False).head(TOP_N_SETUPS).reset_index(drop=True)
        ipo.insert(0, "Rank", range(1, len(ipo) + 1))
        ipo["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo["symbol"].astype(str)

        chart = ipo.sort_values("ipo_setup_score")
        fig = go.Figure(go.Bar(x=chart["ipo_setup_score"], y=chart["symbol"], orientation="h", marker=dict(color=PALETTE["Extended Leader (WAIT)"]), text=chart["ipo_setup_score"].round(1), textposition="outside"))
        fig.update_layout(title="IPO Setup Score Ranking", xaxis=dict(range=[0, 115]))
        st.plotly_chart(fig_style(fig, max(240, 30 * len(chart))), use_container_width=True)

        display = ipo.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
            "days_listed": "Days Listed", "ipo_phase": "Phase", "ipo_setup_score": "Setup Score",
            "vwap_premium": "Above VWAP", "retracement_from_listing_high": "Off High"})
        columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership", "Close", "Days Listed", "Phase", "Setup Score", "Above VWAP", "Off High", "Avg Turnover (Cr)"]
        display = display[[c for c in columns if c in display.columns]]
        for col in ["Above VWAP", "Off High"]:
            if col in display.columns: display[col] = display[col].map(lambda v: percent(v, True))
        for col in ["Close", "Setup Score", "Avg Turnover (Cr)"]:
            if col in display.columns: display[col] = display[col].map(lambda v: number(v, 2))
        if "Days Listed" in display.columns: display["Days Listed"] = display["Days Listed"].map(integer)
        if "Industry Leadership" in display.columns: display["Industry Leadership"] = display["Industry Leadership"].round(1)
        show(display, max(240, 40 * len(display) + 60), links=True, progress_cols=["Industry Leadership"])

    with st.expander("Full Basic Industry Leaderboard (context only, not a filter)"):
        cd = context.rename(columns={"industry_rank": "Industry Rank", "basic_industry": "Basic Industry", "leadership_score": "Leadership", "regime": "Regime"})
        cd["Leadership"] = cd["Leadership"].round(1)
        show(cd[["Industry Rank", "Basic Industry", "Leadership", "Regime"]], 360, progress_cols=["Leadership"])


def basic_industry_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = date_picker(dates_of(basic_history), "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()
    f1, f2, f3 = st.columns([1.45, .85, .85])
    with f1: regimes = st.multiselect("Trading State filter", list(PALETTE), default=list(PALETTE), key="basic_regimes")
    with f2: minimum = st.number_input("Minimum stocks", min_value=1, value=SMALL_GROUP_LIMIT, step=1, key="basic_minimum")
    with f3: ranking_choice = st.selectbox("Ranking", ["Highest Leadership", "Lowest Leadership"], key="basic_sort")
    if regimes: selected = selected[selected["regime"].isin(regimes)]
    selected = selected[numeric(selected, "members") >= minimum].copy()
    if selected.empty:
        st.warning("No Basic Industries match the selected filters.")
        return
    selected = selected.sort_values("leadership_score", ascending=(ranking_choice == "Lowest Leadership")).reset_index(drop=True)
    selected.insert(0, "Rank", range(1, len(selected) + 1))
    table = selected.rename(columns={"basic_industry": "Basic Industry", "leadership_score": "Leadership", "actionability_score": "Actionability %", "regime": "Trading State", "members": "Stocks", "eq_ret_20d": "20D Return"})
    cols = ["Rank", "Basic Industry", "Leadership", "Actionability %", "Trading State", "Stocks", "20D Return"]
    table = table[[c for c in cols if c in table.columns]]
    if "20D Return" in table.columns: table["20D Return"] = table["20D Return"].map(lambda v: percent(v, True))
    if "Stocks" in table.columns: table["Stocks"] = table["Stocks"].map(integer)
    table["Leadership"] = table["Leadership"].round(1)
    table["Actionability %"] = table["Actionability %"].round(1)
    show(table, 410, progress_cols=["Leadership", "Actionability %"])

    st.markdown("### Selected Basic Industry — Constituent Stocks")
    group = st.selectbox("Basic Industry", table["Basic Industry"].tolist(), key="basic_group_selector")
    data = stock_history[(stock_history["date"] == selected_date) & (stock_history["basic_industry"] == group)].copy()
    if data.empty:
        st.info("No stock records found for this Basic Industry.")
        return
    if "ret_20d" in data.columns: data = data.sort_values("ret_20d", ascending=False)
    data = data.reset_index(drop=True); data.insert(0, "Rank", range(1, len(data) + 1)); data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    display = data.rename(columns={"symbol": "Symbol", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain", "stock_strength_score": "Strength"})
    display = display[[c for c in ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength"] if c in display.columns]]
    for col in ["20D Return", "60D Return", "6M Gain"]:
        if col in display.columns: display[col] = display[col].map(lambda v: percent(v, True))
    for col in ["Close", "Strength"]:
        if col in display.columns: display[col] = display[col].map(lambda v: number(v, 2))
    show(display, 320, links=True)


def industry_tab(industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = date_picker(dates_of(industry_history), "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    selected = selected[numeric(selected, "members") >= SMALL_GROUP_LIMIT].copy()
    if selected.empty:
        st.warning("No Industry data is available for this date.")
        return
    selected = selected.sort_values("leadership_score", ascending=False).reset_index(drop=True)
    selected.insert(0, "Rank", range(1, len(selected) + 1))
    table = selected.rename(columns={"industry": "Industry", "leadership_score": "Leadership", "actionability_score": "Actionability %", "regime": "Trading State", "members": "Stocks", "eq_ret_20d": "20D Return"})
    cols = ["Rank", "Industry", "Leadership", "Actionability %", "Trading State", "Stocks", "20D Return"]
    table = table[[c for c in cols if c in table.columns]]
    if "20D Return" in table.columns: table["20D Return"] = table["20D Return"].map(lambda v: percent(v, True))
    if "Stocks" in table.columns: table["Stocks"] = table["Stocks"].map(integer)
    table["Leadership"] = table["Leadership"].round(1)
    table["Actionability %"] = table["Actionability %"].round(1)
    show(table, 470, progress_cols=["Leadership", "Actionability %"])

    st.markdown("### Selected Industry — Constituent Stocks")
    group = st.selectbox("Industry", table["Industry"].tolist(), key="industry_group_selector")
    data = stock_history[(stock_history["date"] == selected_date) & (stock_history["industry"] == group)].copy()
    if data.empty:
        st.info("No stock records found for this Industry.")
        return
    if "ret_20d" in data.columns: data = data.sort_values("ret_20d", ascending=False)
    data = data.reset_index(drop=True); data.insert(0, "Rank", range(1, len(data) + 1)); data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    display = data.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain", "stock_strength_score": "Strength", "nearest_ema_tag": "EMA Zone", "momentum_badge": "Momentum"})
    display = display[[c for c in ["Rank", "Symbol", "Chart", "Basic Industry", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "EMA Zone", "Momentum"] if c in display.columns]]
    for col in ["20D Return", "60D Return", "6M Gain"]:
        if col in display.columns: display[col] = display[col].map(lambda v: percent(v, True))
    for col in ["Close", "Strength"]:
        if col in display.columns: display[col] = display[col].map(lambda v: number(v, 2))
    show(display, 320, links=True)


def methodology_tab() -> None:
    st.subheader("Methodology")
    st.markdown("""
**Leadership Score (0–100):** 35% price velocity, 35% EMA structural alignment, 30% institutional up/down volume — 3-day smoothed.

**Actionability % (0–100):** Share of constituents in that group currently passing the full setup rules (liquidity, trend, precision score).

**Top Buy Setups — individual only:** No industry Leadership filter. Selection is `established_buy_setup = 1` / `ipo_buy_setup = 1`, computed upstream from liquidity (≥ 5 Cr turnover), EMA alignment, trend, and volatility/volume contraction. Industry Rank and Industry Leadership are shown as context only.

**Established ranking:** Tightness 30%, volume contraction 25%, six-month gain 20%, up/down ratio 15%, strength 10%.

**IPO ranking:** Tightness 25%, dry-up 20%, VWAP premium 20%, retracement 20%, HH-HL structure 15%.

This is a research shortlist, not a trade recommendation. Confirm price structure, entry, invalidation, and position size before acting.
""")


def main() -> None:
    required = [BASIC_HISTORY_FILE, INDUSTRY_HISTORY_FILE, STOCK_HISTORY_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Required dashboard files are missing. Run the data workflow first.")
        st.code("\n".join(missing))
        st.stop()
    basic = prepare_groups(load(str(BASIC_HISTORY_FILE)), "basic_industry")
    industry = prepare_groups(load(str(INDUSTRY_HISTORY_FILE)), "industry")
    stocks = prepare_stocks(load(str(STOCK_HISTORY_FILE)))
    all_dates = sorted(set(dates_of(basic) + dates_of(industry) + dates_of(stocks)))
    if not all_dates:
        st.error("No valid trading dates found in history files.")
        st.stop()
    st.title("NSE Sectoral Breadth & 2-Axis Setup Engine")
    sync = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Data as of {sync.replace('T', ' ').replace('Z', ' IST')}")
    tabs = st.tabs(["🎯 Top Buy Setups", "Overview", "Basic Industry", "Industry", "Methodology"])
    with tabs[0]: top_buy_tab(basic, stocks, date_picker(all_dates, "buys"))
    with tabs[1]: overview_tab(basic, date_picker(all_dates, "overview"))
    with tabs[2]: basic_industry_tab(basic, stocks)
    with tabs[3]: industry_tab(industry, stocks)
    with tabs[4]: methodology_tab()


if __name__ == "__main__":
    main()
