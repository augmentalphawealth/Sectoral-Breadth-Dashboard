# COMPLETE REPLACEMENT: app/dashboard.py
# Fixes duplicate Arrow/PyArrow column-name errors by never merging a
# second Basic Industry column into stock data. No pandas Styler is used.

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

PALETTE = {
    "Fresh Leader (HUNT)": "#2E7D63",
    "Extended Leader (WAIT)": "#D98E3B",
    "Speculative Coil (AVOID)": "#8B5FBF",
    "Dead (AVOID)": "#B0483C",
    "Neutral Transition": "#9AA5B1",
}

st.set_page_config(page_title="NSE Sectoral Breadth & Buy Setups", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {padding-top:1.4rem;padding-bottom:2.5rem;max-width:1440px}
[data-testid="stMetricValue"] {font-size:1.45rem;font-weight:700}
[data-testid="stMetricLabel"] {font-size:.78rem;color:#64748b;text-transform:uppercase;letter-spacing:.03em}
h1 {font-weight:700;letter-spacing:-.02em}
[data-testid="stDataFrame"] {border:1px solid #e5e7eb;border-radius:10px;overflow:hidden}
</style>
""", unsafe_allow_html=True)


def fig_style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=48, b=10), font=dict(family="Inter, sans-serif", color="#1F2937", size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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


def number(value: object, decimals: int = 2) -> str:
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


def group_table(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    wanted = [key, "leadership_score", "actionability_score", "regime", "members", "nh_nl_net", "eq_ret_1d", "eq_ret_5d", "eq_ret_20d", "eq_ret_60d", "pct_above_50", "pct_above_200", "acc_minus_dist", "breakout_count", "vcp_ready_count"]
    data = frame[[c for c in wanted if c in frame.columns]].copy()
    data = data.rename(columns={key: "Basic Industry" if key == "basic_industry" else "Industry", "leadership_score": "Leadership Score", "actionability_score": "Actionability (Setup %)", "regime": "Trading State", "members": "Constituent Stocks", "nh_nl_net": "Net New Highs (%)", "eq_ret_1d": "1D Return", "eq_ret_5d": "5D Return", "eq_ret_20d": "20D Return", "eq_ret_60d": "60D Return", "pct_above_50": "Stocks Above 50 EMA", "pct_above_200": "Stocks Above 200 EMA", "acc_minus_dist": "Accumulation − Distribution", "breakout_count": "Breakouts", "vcp_ready_count": "VCP Ready"})
    if "Leadership Score" in data.columns:
        data = data.sort_values("Leadership Score", ascending=False)
    data = data.reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    return data


def format_group(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for col in ["1D Return", "5D Return", "20D Return", "60D Return", "Stocks Above 50 EMA", "Stocks Above 200 EMA"]:
        if col in data.columns: data[col] = data[col].map(lambda x: percent(x, True))
    for col in ["Rank", "Constituent Stocks", "Accumulation − Distribution", "Breakouts", "VCP Ready"]:
        if col in data.columns: data[col] = data[col].map(integer)
    for col in ["Leadership Score", "Actionability (Setup %)", "Net New Highs (%)"]:
        if col in data.columns: data[col] = data[col].map(number)
    return data


def format_stock(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for col in ["1D Return", "5D Return", "20D Return", "60D Return", "Distance from 52W High", "6M Gain", "Candle Range", "Price Tightness (3D)", "Above VWAP", "Off Post-List High"]:
        if col in data.columns: data[col] = data[col].map(lambda x: percent(x, True))
    for col in ["Rank", "Industry Rank", "Days Listed", "Heavy Volume Days (6M)"]:
        if col in data.columns: data[col] = data[col].map(integer)
    for col in ["Close", "Buy Priority Score", "Setup Score", "Strength", "Vol Contraction (vs 50D)", "Avg Turnover (Cr)", "Industry Leadership Score"]:
        if col in data.columns: data[col] = data[col].map(number)
    return data


def show(data: pd.DataFrame, height: int, links: bool = False) -> None:
    # Final Arrow safety net: remove duplicate labels and normalize all headers.
    data = data.copy()
    data.columns = [str(c) for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated(keep="first")]
    config = {}
    if links and "Chart" in data.columns:
        config["Chart"] = st.column_config.LinkColumn("TradingView", display_text="Open ↗")
    st.dataframe(data, use_container_width=True, hide_index=True, height=height, column_config=config)


def rank(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    values = numeric(frame, column, float("nan"))
    return values.rank(pct=True, ascending=ascending).fillna(.5)


def overview_tab(basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    basic = basic_history[basic_history["date"] == selected_date].copy()
    if basic.empty:
        st.warning("No Basic Industry data is available for this date.")
        return
    st.subheader("Market Breadth & 2-Axis Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Data", selected_date.strftime("%d %b %Y"))
    c2.metric("Basic Industries", integer(basic["basic_industry"].nunique()))
    c3.metric("Fresh Leaders", integer((basic["regime"] == "Fresh Leader (HUNT)").sum()))
    c4.metric("Extended Leaders", integer((basic["regime"] == "Extended Leader (WAIT)").sum()))
    c5.metric("Avg Actionability", number(numeric(basic, "actionability_score").mean(), 1))

    st.markdown("### Sector Leadership Treemap")
    tree = basic[numeric(basic, "members") >= SMALL_GROUP_LIMIT].sort_values("members", ascending=False).copy()
    tree["label"] = tree["basic_industry"].where(tree.index.isin(tree.head(25).index), "")
    fig = go.Figure(go.Treemap(labels=tree["basic_industry"], parents=[""] * len(tree), values=numeric(tree, "members"), text=tree["label"], textinfo="text", marker=dict(colors=numeric(tree, "leadership_score"), colorscale="Greens", showscale=True), hovertemplate="%{label}<br>Members: %{value}<br>Leadership: %{marker.color:.1f}<extra></extra>"))
    fig.update_layout(title="Sector Size colored by Leadership Score — top 25 labeled")
    st.plotly_chart(fig_style(fig, 420), use_container_width=True)

    st.markdown("### Market Breadth Gauges")
    g1, g2 = st.columns(2)
    for holder, col, title in [(g1, "pct_above_50", "% Stocks Above 50 EMA"), (g2, "pct_above_200", "% Stocks Above 200 EMA")]:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=numeric(basic, col).mean(), title=dict(text=title), gauge=dict(axis=dict(range=[0, 100]), steps=[dict(range=[0, 30], color="#fecaca"), dict(range=[30, 70], color="#fef3c7"), dict(range=[70, 100], color="#86efac")])) )
        holder.plotly_chart(fig_style(fig, 270), use_container_width=True)

    st.markdown("### Leadership vs Actionability")
    scatter = basic[numeric(basic, "members") >= SMALL_GROUP_LIMIT].copy()
    top = scatter.nlargest(15, "leadership_score").index
    scatter["label"] = scatter["basic_industry"].where(scatter.index.isin(top), "")
    fig = go.Figure(go.Scatter(x=numeric(scatter, "leadership_score"), y=numeric(scatter, "actionability_score"), mode="markers+text", text=scatter["label"], textposition="top center", marker=dict(size=numeric(scatter, "members").clip(lower=8, upper=70), color=scatter["regime"].map(PALETTE).fillna("#9AA5B1")), hovertext=scatter["basic_industry"], hovertemplate="%{hovertext}<br>Leadership: %{x:.1f}<br>Actionability: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Top 15 leaders labeled; hover all points", xaxis_title="Leadership Score", yaxis_title="Actionability (Setup %)", xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_style(fig, 400), use_container_width=True)
    show(format_group(group_table(basic[numeric(basic, "members") >= SMALL_GROUP_LIMIT], "basic_industry")), 480)


def top_buy_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("🎯 Top Individual Buy Setups")
    st.caption("No industry gate. Industry Rank and Industry Leadership Score are context only.")
    basic = basic_history[basic_history["date"] == selected_date].copy()
    stocks = stock_history[stock_history["date"] == selected_date].copy()
    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return

    # IMPORTANT: lookup uses the original lowercase key only. It never adds
    # another Basic Industry column to the stock dataframe.
    context = basic[["basic_industry", "leadership_score", "regime"]].copy()
    context = context.sort_values("leadership_score", ascending=False).drop_duplicates("basic_industry").reset_index(drop=True)
    context.insert(0, "industry_rank", range(1, len(context) + 1))
    lookup = context.set_index("basic_industry")[["industry_rank", "leadership_score", "regime"]].rename(columns={"industry_rank": "Industry Rank", "leadership_score": "Industry Leadership Score", "regime": "Industry Regime"})

    # Individual-level selection only.
    buy = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualified Established", integer(len(buy)))
    c2.metric("Qualified IPO", integer(len(ipo)))
    c3.metric("Shortlist", f"Top {TOP_N_SETUPS}")
    c4.metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top 20 Established Buy Setups")
    if buy.empty:
        st.info("No established stocks pass the individual hard gates on this date.")
    else:
        buy["Buy Priority Score"] = (0.30 * rank(buy, "tight_3d_range", True) + 0.25 * rank(buy, "vol_ratio_50", True) + 0.20 * rank(buy, "gain_6m", False) + 0.15 * rank(buy, "up_down_ratio", False) + 0.10 * rank(buy, "stock_strength_score", False)) * 100
        # Safe index-based enrichment: no duplicate Basic Industry column.
        buy["Industry Rank"] = buy["basic_industry"].map(lookup["Industry Rank"])
        buy["Industry Leadership Score"] = buy["basic_industry"].map(lookup["Industry Leadership Score"])
        buy["Industry Regime"] = buy["basic_industry"].map(lookup["Industry Regime"])
        buy = buy.sort_values("Buy Priority Score", ascending=False).head(TOP_N_SETUPS).reset_index(drop=True)
        buy.insert(0, "Rank", range(1, len(buy) + 1))
        buy["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy["symbol"].astype(str)
        chart = buy.sort_values("Buy Priority Score")
        fig = go.Figure(go.Bar(x=chart["Buy Priority Score"], y=chart["symbol"], orientation="h", marker=dict(color=PALETTE["Fresh Leader (HUNT)"]), text=chart["Buy Priority Score"].round(1), textposition="outside"))
        fig.update_layout(title="Priority Score Ranking", xaxis=dict(range=[0, 115]))
        st.plotly_chart(fig_style(fig, max(240, 30 * len(chart))), use_container_width=True)
        display = buy.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close", "tight_3d_range": "Price Tightness (3D)", "vol_ratio_50": "Vol Contraction (vs 50D)", "gain_6m": "6M Gain", "nearest_ema_tag": "EMA Proximity", "momentum_badge": "Momentum"})
        columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership Score", "Industry Regime", "Close", "Buy Priority Score", "Price Tightness (3D)", "Vol Contraction (vs 50D)", "6M Gain", "EMA Proximity", "Momentum"]
        display = display[[c for c in columns if c in display.columns]]
        show(format_stock(display), 560, True)

    st.markdown("### Top 20 IPO Setups")
    st.caption("Newly listed stocks (<150 days), no industry filter, ranked by IPO Setup Score.")
    if ipo.empty:
        st.info("No IPO stocks pass the individual IPO gates on this date.")
    else:
        ipo["Avg Turnover (Cr)"] = numeric(ipo, "ipo_turnover_avg") / 10_000_000
        if "ipo_setup_score" not in ipo.columns:
            ipo["ipo_setup_score"] = (0.25 * rank(ipo, "tight_3d_range", True) + 0.20 * rank(ipo, "vol_ratio_50", True) + 0.20 * rank(ipo, "vwap_premium", False) + 0.20 * rank(ipo, "retracement_from_listing_high", True) + 0.15 * rank(ipo, "hh_hl_count", False)) * 100
        else:
            ipo["ipo_setup_score"] = numeric(ipo, "ipo_setup_score")
        # Same safe index-based enrichment for IPO rows.
        ipo["Industry Rank"] = ipo["basic_industry"].map(lookup["Industry Rank"])
        ipo["Industry Leadership Score"] = ipo["basic_industry"].map(lookup["Industry Leadership Score"])
        ipo["Industry Regime"] = ipo["basic_industry"].map(lookup["Industry Regime"])
        ipo = ipo.sort_values("ipo_setup_score", ascending=False).head(TOP_N_SETUPS).reset_index(drop=True)
        ipo.insert(0, "Rank", range(1, len(ipo) + 1))
        ipo["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo["symbol"].astype(str)
        chart = ipo.sort_values("ipo_setup_score")
        fig = go.Figure(go.Bar(x=chart["ipo_setup_score"], y=chart["symbol"], orientation="h", marker=dict(color=PALETTE["Extended Leader (WAIT)"]), text=chart["ipo_setup_score"].round(1), textposition="outside"))
        fig.update_layout(title="IPO Setup Score Ranking", xaxis=dict(range=[0, 115]))
        st.plotly_chart(fig_style(fig, max(240, 30 * len(chart))), use_container_width=True)
        display = ipo.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close", "days_listed": "Days Listed", "ipo_phase": "Phase", "ipo_setup_score": "Setup Score", "vwap_premium": "Above VWAP", "retracement_from_listing_high": "Off Post-List High"})
        columns = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership Score", "Industry Regime", "Close", "Days Listed", "Phase", "Setup Score", "Above VWAP", "Off Post-List High", "Avg Turnover (Cr)"]
        display = display[[c for c in columns if c in display.columns]]
        show(format_stock(display), max(240, 40 * len(display) + 60), True)

    st.markdown("### Full Basic Industry Leaderboard — Context Only")
    context_display = context.rename(columns={"industry_rank": "Industry Rank", "basic_industry": "Basic Industry", "leadership_score": "Industry Leadership Score", "regime": "Industry Regime"})
    context_display = context_display[[c for c in ["Industry Rank", "Basic Industry", "Industry Leadership Score", "Industry Regime"] if c in context_display.columns]]
    show(context_display, 360)


def basic_industry_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = date_picker(dates_of(basic_history), "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()
    f1, f2, f3 = st.columns([1.45, .85, .85])
    with f1: regimes = st.multiselect("Trading State filter", list(PALETTE), default=list(PALETTE), key="basic_regimes")
    with f2: minimum = st.number_input("Minimum stocks", min_value=1, value=SMALL_GROUP_LIMIT, step=1, key="basic_minimum")
    with f3: ranking_choice = st.selectbox("Ranking", ["Highest Leadership", "Lowest Leadership"], key="basic_sort")
    if regimes: selected = selected[selected["regime"].isin(regimes)]
    selected = selected[numeric(selected, "members") >= minimum]
    table = group_table(selected, "basic_industry")
    if ranking_choice == "Lowest Leadership" and "Leadership Score" in table.columns:
        table = table.sort_values("Leadership Score").reset_index(drop=True); table["Rank"] = range(1, len(table) + 1)
    if table.empty: st.warning("No Basic Industries match the selected filters."); return
    show(format_group(table), 410)
    group = st.selectbox("Basic Industry", table["Basic Industry"].tolist(), key="basic_group_selector")
    data = stock_history[(stock_history["date"] == selected_date) & (stock_history["basic_industry"] == group)].copy()
    if data.empty: st.info("No stock records found for this Basic Industry."); return
    if "ret_20d" in data.columns: data = data.sort_values("ret_20d", ascending=False)
    data = data.reset_index(drop=True); data.insert(0, "Rank", range(1, len(data) + 1)); data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    display = data.rename(columns={"symbol":"Symbol","close":"Close","ret_20d":"20D Return","ret_60d":"60D Return","gain_6m":"6M Gain","stock_strength_score":"Strength"})
    display = display[[c for c in ["Rank","Symbol","Chart","Close","20D Return","60D Return","6M Gain","Strength"] if c in display.columns]]
    show(format_stock(display), 320, True)


def industry_tab(industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = date_picker(dates_of(industry_history), "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy(); selected = selected[numeric(selected, "members") >= SMALL_GROUP_LIMIT]
    table = group_table(selected, "industry")
    if table.empty: st.warning("No Industry data is available for this date."); return
    show(format_group(table), 470)
    group = st.selectbox("Industry", table["Industry"].tolist(), key="industry_group_selector")
    data = stock_history[(stock_history["date"] == selected_date) & (stock_history["industry"] == group)].copy()
    if data.empty: st.info("No stock records found for this Industry."); return
    if "ret_20d" in data.columns: data = data.sort_values("ret_20d", ascending=False)
    data = data.reset_index(drop=True); data.insert(0, "Rank", range(1, len(data) + 1)); data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    display = data.rename(columns={"symbol":"Symbol","basic_industry":"Basic Industry","close":"Close","ret_20d":"20D Return","ret_60d":"60D Return","gain_6m":"6M Gain","stock_strength_score":"Strength","nearest_ema_tag":"EMA Proximity","momentum_badge":"Momentum"})
    display = display[[c for c in ["Rank","Symbol","Chart","Basic Industry","Close","20D Return","60D Return","6M Gain","Strength","EMA Proximity","Momentum"] if c in display.columns]]
    show(format_stock(display), 320, True)


def methodology_tab() -> None:
    st.subheader("Methodology")
    st.markdown("""
**Top Buy Setups:** Selection is individual-stock based; there is no parent-industry gate. Industry Rank and Industry Leadership Score are context fields only.

**Established ranking:** Tightness 30%, volume contraction 25%, six-month gain 20%, up/down ratio 15%, strength 10%.

**IPO ranking:** Tightness 25%, dry-up 20%, VWAP premium 20%, retracement 20%, HH-HL structure 15%.

The upstream setup flags are expected to enforce the liquidity, EMA, trend, and volatility/volume conditions. This is a research shortlist, not a trade recommendation.
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
