# COMPLETE REPLACEMENT FOR: app/dashboard.py
# NSE Industry Momentum Monitor with 5-day Leadership Heatmap
#
# New features:
# - 5-session score history for every Basic Industry
# - Crossing-60 watchlist
# - Four rotation buckets
# - Selected-industry 30-session trend chart
# - No Pandas Styler
# - No duplicate-column merges
# - No treemap/scatter clutter

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
TOP_INDUSTRIES = 12
TOP_STOCKS = 20
HEATMAP_SESSIONS = 5
TREND_SESSIONS = 30

INK = "#0F172A"
MUTED = "#64748B"
GREEN = "#15803D"
RED = "#B91C1C"
BLUE = "#1D4ED8"
AMBER = "#B45309"

st.set_page_config(
    page_title="NSE Industry Momentum Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1380px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
    h1 { color: #0F172A; font-size: 2rem; font-weight: 750; letter-spacing: -.03em; }
    h2, h3 { color: #0F172A; font-weight: 700; }
    [data-testid="stMetricValue"] { color: #0F172A; font-size: 1.55rem; font-weight: 750; }
    [data-testid="stMetricLabel"] { color: #64748B; font-size: .72rem; font-weight: 650; text-transform: uppercase; letter-spacing: .05em; }
    [data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 10px; overflow: hidden; }
    .note { color: #64748B; font-size: .88rem; margin-top: -.35rem; margin-bottom: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def normalize_score(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    if len(valid) and valid.max() <= 1.5:
        return series * 100.0
    return series


def fmt_number(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_integer(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_return(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100.0:,.1f}%"
    except (TypeError, ValueError):
        return "—"


def chart_style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=25, t=44, b=10),
        font=dict(family="Inter, sans-serif", size=12, color=INK),
        title_font=dict(size=14, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def prepare_groups(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Neutral Transition"
    if "leadership_score" not in data.columns:
        data["leadership_score"] = data.get("strength_score", 0.0)
    if "actionability_score" not in data.columns:
        data["actionability_score"] = 0.0
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    data["leadership_score"] = normalize_score(data["leadership_score"]).fillna(0.0)
    data["actionability_score"] = normalize_score(data["actionability_score"]).fillna(0.0)
    return data


def prepare_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["symbol", "basic_industry", "industry", "sector"]:
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


def date_picker(all_dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = all_dates[-1]
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest
    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in all_dates:
        selected = latest
        st.session_state[state_key] = latest
    index = all_dates.index(selected)

    left, right = st.columns([5.8, 2.2])
    with left:
        st.subheader("Analysis Date")
    with right:
        previous, calendar, next_button = st.columns([0.4, 1.6, 0.4])
        with previous:
            if st.button("‹", key=f"{key}_previous", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = all_dates[index - 1]
                st.rerun()
        with calendar:
            chosen = st.date_input("Date", value=selected.date(), min_value=all_dates[0].date(), max_value=latest.date(), key=f"{key}_calendar", label_visibility="collapsed")
        with next_button:
            if st.button("›", key=f"{key}_next", disabled=index == len(all_dates) - 1, use_container_width=True):
                st.session_state[state_key] = all_dates[index + 1]
                st.rerun()

    valid = [date for date in all_dates if date <= pd.Timestamp(chosen)]
    resolved = valid[-1] if valid else all_dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"Data as of {resolved.strftime('%d %b %Y')}")
    return resolved


def show_table(data: pd.DataFrame, height: int, links: bool = False, progress_columns: list[str] | None = None) -> None:
    view = data.copy()
    view.columns = [str(column) for column in view.columns]
    view = view.loc[:, ~view.columns.duplicated(keep="first")]
    config = {}
    if links and "Chart" in view.columns:
        config["Chart"] = st.column_config.LinkColumn("Chart", display_text="Open ↗")
    for column in progress_columns or []:
        if column in view.columns:
            config[column] = st.column_config.ProgressColumn(column, min_value=0, max_value=100, format="%.0f")
    st.dataframe(view, use_container_width=True, hide_index=True, height=height, column_config=config)


def current_snapshot(history: pd.DataFrame, selected_date: pd.Timestamp, group_column: str) -> pd.DataFrame:
    data = history[history["date"] == selected_date].copy()
    data = data[numeric_column(data, "members") >= SMALL_GROUP_LIMIT].copy()
    data = data.drop_duplicates(group_column, keep="last")
    data["Leadership"] = normalize_score(data["leadership_score"]).fillna(0.0)
    data["Actionability"] = normalize_score(data["actionability_score"]).fillna(0.0)
    return data


def score_history(history: pd.DataFrame, group_column: str, selected_date: pd.Timestamp, sessions: int) -> pd.DataFrame:
    dates = sorted(pd.Timestamp(value) for value in pd.to_datetime(history["date"].dropna().unique()))
    dates = [date for date in dates if date <= selected_date]
    dates = dates[-sessions:]
    if not dates:
        return pd.DataFrame()

    rows = history[history["date"].isin(dates)].copy()
    rows = rows[numeric_column(rows, "members") >= SMALL_GROUP_LIMIT].copy()
    rows = rows.drop_duplicates(["date", group_column], keep="last")
    rows["Score"] = normalize_score(rows["leadership_score"]).fillna(0.0)
    pivot = rows.pivot(index=group_column, columns="date", values="Score")
    pivot = pivot.reindex(columns=dates)
    pivot.columns = [date.strftime("%d %b") for date in dates]
    pivot = pivot.reset_index()
    return pivot


def build_heatmap(history: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    heat = score_history(history, "basic_industry", selected_date, HEATMAP_SESSIONS)
    if heat.empty:
        return heat

    score_columns = list(heat.columns[1:])
    heat["Current Score"] = heat[score_columns[-1]]
    heat["5D Change"] = heat[score_columns[-1]] - heat[score_columns[0]]
    heat["Zone"] = heat["Current Score"].map(score_zone)
    heat = heat.sort_values(
        ["Current Score", "5D Change"],
        ascending=[False, False],
    )
    heat.insert(0, "Rank", range(1, len(heat) + 1))
    return heat


def score_zone(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "Unknown"

    if value >= 80:
        return "Very Strong"
    if value >= 70:
        return "Strong"
    if value >= 60:
        return "Constructive"
    if value >= 50:
        return "Emerging"
    if value >= 40:
        return "Weak"
    return "Very Weak"


def category(value: object, change: object) -> str:
    try:
        score = float(value)
        delta = float(change)
    except (TypeError, ValueError):
        return "Unknown"

    if score >= 60 and delta > 0:
        return "Strong & Improving"
    if 50 <= score < 60 and delta > 0:
        return "Emerging"
    if score >= 60 and delta < 0:
        return "Strong but Fading"
    if score < 50:
        return "Weak"
    return "Neutral"


def heatmap_color(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "background-color: #F8FAFC; color: #64748B;"

    if score >= 80:
        return "background-color: #14532D; color: white; font-weight: 700;"
    if score >= 70:
        return "background-color: #15803D; color: white; font-weight: 700;"
    if score >= 60:
        return "background-color: #86EFAC; color: #14532D; font-weight: 700;"
    if score >= 50:
        return "background-color: #FEF3C7; color: #78350F; font-weight: 700;"
    if score >= 40:
        return "background-color: #FED7AA; color: #7C2D12; font-weight: 700;"
    return "background-color: #FECACA; color: #7F1D1D; font-weight: 700;"


def heatmap_table(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("Not enough history is available to build the 5-session heatmap.")
        return

    view = data.copy()
    score_columns = [
        column
        for column in view.columns
        if column not in {
            "Rank",
            "basic_industry",
            "Current Score",
            "5D Change",
            "Zone",
            "Category",
        }
    ]

    view = view.rename(columns={"basic_industry": "Basic Industry"})
    view["Current Score"] = view["Current Score"].round(1)
    view["5D Change"] = view["5D Change"].round(1)

    display_columns = [
        "Rank",
        "Basic Industry",
        *score_columns,
        "Current Score",
        "5D Change",
        "Zone",
        "Category",
    ]
    display_columns = [column for column in display_columns if column in view.columns]
    view = view[display_columns]

    numeric_for_style = view.copy()
    style_columns = [column for column in score_columns if column in view.columns]
    styled = view.style
    for column in style_columns:
        styled = styled.map(heatmap_color, subset=[column])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=max(420, min(900, len(view) * 30 + 70)),
    )


def percentile_rank(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    values = numeric_column(frame, column, float("nan"))
    return values.rank(pct=True, ascending=ascending).fillna(0.5)


# =============================================================================
# INDUSTRY MONITOR
# =============================================================================

def industry_monitor_tab(basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("Industry Leadership Rotation")
    st.markdown(
        '<p class="note">See the path of each Basic Industry over the last five sessions, identify industries crossing 60, and inspect the selected industry over 30 sessions.</p>',
        unsafe_allow_html=True,
    )

    current = current_snapshot(basic_history, selected_date, "basic_industry")
    heat = build_heatmap(basic_history, selected_date)

    if current.empty or heat.empty:
        st.warning("Not enough Basic Industry history is available for this date.")
        return

    current = current.merge(
        heat[["basic_industry", "Current Score", "5D Change"]],
        on="basic_industry",
        how="left",
    )
    current["Category"] = current.apply(
        lambda row: category(row["Current Score"], row["5D Change"]),
        axis=1,
    )

    current_score = current["Current Score"]
    score_change = current["5D Change"]

    crossing_60 = current[(current_score >= 60) & (current_score - score_change < 60)].copy()
    strong_improving = current[(current_score >= 60) & (score_change > 0)].copy()
    emerging = current[(current_score >= 50) & (current_score < 60) & (score_change > 0)].copy()
    strong_fading = current[(current_score >= 60) & (score_change < 0)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crossing Above 60", fmt_number(len(crossing_60), 0))
    c2.metric("Strong & Improving", fmt_number(len(strong_improving), 0))
    c3.metric("Emerging 50–60", fmt_number(len(emerging), 0))
    c4.metric("Strong but Fading", fmt_number(len(strong_fading), 0))

    st.markdown("### Leadership Path: Last 5 Sessions")
    st.markdown(
        '<p class="note">Green shades indicate stronger Leadership Score. The final column shows the current score and the five-session change.</p>',
        unsafe_allow_html=True,
    )
    heatmap_table(heat)

    st.markdown("### Industries Crossing Above 60")
    if crossing_60.empty:
        st.info("No Basic Industry crossed above a Leadership Score of 60 in the available five-session window.")
    else:
        crossing = crossing_60.sort_values("5D Change", ascending=False).copy()
        crossing = crossing.rename(columns={
            "basic_industry": "Basic Industry",
            "Current Score": "Current Score",
            "5D Change": "5D Change",
            "Actionability": "Actionability %",
            "regime": "Regime",
        })
        columns = ["Basic Industry", "Current Score", "5D Change", "Actionability %", "Regime"]
        crossing = crossing[[column for column in columns if column in crossing.columns]]
        for column in ["Current Score", "5D Change", "Actionability %"]:
            if column in crossing.columns:
                crossing[column] = crossing[column].round(1)
        show_table(crossing, max(180, 38 * len(crossing) + 60), progress_columns=["Current Score", "Actionability %"])

    st.markdown("### Rotation Buckets")
    bucket_order = ["Strong & Improving", "Emerging", "Strong but Fading", "Weak", "Neutral"]
    bucket_counts = current["Category"].value_counts().reindex(bucket_order).fillna(0)
    figure = go.Figure(
        go.Bar(
            x=bucket_counts.index,
            y=bucket_counts.values,
            marker_color=[GREEN, BLUE, AMBER, RED, MUTED],
            text=bucket_counts.values.astype(int),
            textposition="outside",
        )
    )
    figure.update_layout(title="Industry rotation buckets", xaxis_title=None, yaxis_title="Number of industries")
    st.plotly_chart(chart_style(figure, 300), use_container_width=True)

    st.markdown("### Selected Industry Trend")
    choices = sorted(current["basic_industry"].dropna().unique().tolist())
    selected_industry = st.selectbox("Select Basic Industry", choices, key="selected_industry_trend")

    trend_dates = sorted(pd.Timestamp(value) for value in pd.to_datetime(basic_history["date"].dropna().unique()))
    trend_dates = [date for date in trend_dates if date <= selected_date][-TREND_SESSIONS:]
    trend = basic_history[
        (basic_history["date"].isin(trend_dates))
        & (basic_history["basic_industry"] == selected_industry)
    ].copy()

    if trend.empty:
        st.info("No historical values are available for the selected industry.")
        return

    trend = trend.drop_duplicates(["date", "basic_industry"], keep="last")
    trend["Leadership"] = normalize_score(trend["leadership_score"]).fillna(0.0)
    trend["Actionability"] = normalize_score(trend["actionability_score"]).fillna(0.0)
    trend = trend.sort_values("date")

    selected_row = current[current["basic_industry"] == selected_industry].iloc[0]
    current_value = float(selected_row["Current Score"])
    change_value = float(selected_row["5D Change"])
    actionability_value = float(selected_row["Actionability"])
    regime_value = selected_row.get("regime", "Unknown")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Leadership", f"{current_value:.1f}")
    m2.metric("5D Change", f"{change_value:+.1f}")
    m3.metric("Actionability", f"{actionability_value:.1f}%")
    m4.metric("Regime", str(regime_value))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=trend["date"],
            y=trend["Leadership"],
            mode="lines+markers",
            name="Leadership Score",
            line=dict(color=GREEN, width=3),
            marker=dict(size=7),
        )
    )
    figure.add_hrect(y0=0, y1=50, fillcolor="#FEE2E2", opacity=0.25, line_width=0)
    figure.add_hrect(y0=50, y1=60, fillcolor="#FEF3C7", opacity=0.25, line_width=0)
    figure.add_hrect(y0=60, y1=70, fillcolor="#DCFCE7", opacity=0.25, line_width=0)
    figure.add_hrect(y0=70, y1=100, fillcolor="#BBF7D0", opacity=0.25, line_width=0)
    figure.add_hline(y=50, line_dash="dot", line_color="#B45309")
    figure.add_hline(y=60, line_dash="dot", line_color="#15803D")
    figure.add_hline(y=70, line_dash="dot", line_color="#14532D")
    figure.update_layout(
        title=f"{selected_industry}: Leadership Score trend",
        xaxis_title=None,
        yaxis_title="Leadership Score",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(chart_style(figure, 380), use_container_width=True)


# =============================================================================
# TOP INDIVIDUAL SETUPS
# =============================================================================

def stock_setups_tab(
    basic_history: pd.DataFrame,
    stock_history: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> None:
    st.subheader("Top Individual Setups")
    st.markdown(
        '<p class="note">No industry hard filter. Industry Rank, Leadership and Regime are context only.</p>',
        unsafe_allow_html=True,
    )

    basic = basic_history[basic_history["date"] == selected_date].drop_duplicates("basic_industry", keep="last").copy()
    stocks = stock_history[stock_history["date"] == selected_date].copy()

    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return

    basic = basic.sort_values("leadership_score", ascending=False).reset_index(drop=True)
    basic["Industry Rank"] = range(1, len(basic) + 1)
    lookup = basic.set_index("basic_industry")[["Industry Rank", "leadership_score", "regime"]]

    established = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Established Qualified", fmt_integer(len(established)))
    c2.metric("IPO Qualified", fmt_integer(len(ipo)))
    c3.metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top 20 Established Setups")
    if established.empty:
        st.info("No established stocks pass the upstream setup gate on this date.")
    else:
        established["Priority Score"] = (
            0.30 * established.get("tight_3d_range", pd.Series(index=established.index)).rank(pct=True, ascending=False).fillna(0.5)
            + 0.25 * established.get("vol_ratio_50", pd.Series(index=established.index)).rank(pct=True, ascending=False).fillna(0.5)
            + 0.20 * established.get("gain_6m", pd.Series(index=established.index)).rank(pct=True).fillna(0.5)
            + 0.15 * established.get("up_down_ratio", pd.Series(index=established.index)).rank(pct=True).fillna(0.5)
            + 0.10 * established.get("stock_strength_score", pd.Series(index=established.index)).rank(pct=True).fillna(0.5)
        ) * 100.0
        established["Industry Rank"] = established["basic_industry"].map(lookup["Industry Rank"])
        established["Industry Leadership"] = normalize_score(established["basic_industry"].map(lookup["leadership_score"]))
        established["Industry Regime"] = established["basic_industry"].map(lookup["regime"])
        established = established.sort_values("Priority Score", ascending=False).head(TOP_STOCKS).reset_index(drop=True)
        established.insert(0, "Rank", range(1, len(established) + 1))
        established["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + established["symbol"].astype(str)
        view = established.rename(columns={"symbol":"Symbol", "basic_industry":"Basic Industry", "close":"Close", "tight_3d_range":"Tightness (3D)", "vol_ratio_50":"Vol vs 50D", "gain_6m":"6M Gain", "nearest_ema_tag":"EMA Zone", "momentum_badge":"Momentum"})
        columns = ["Rank","Symbol","Chart","Basic Industry","Industry Rank","Industry Leadership","Industry Regime","Close","Priority Score","Tightness (3D)","Vol vs 50D","6M Gain","EMA Zone","Momentum"]
        view = view[[column for column in columns if column in view.columns]]
        for column in ["Close", "Priority Score", "Vol vs 50D"]:
            if column in view.columns: view[column] = view[column].map(lambda value: fmt_number(value, 2))
        for column in ["Tightness (3D)", "6M Gain"]:
            if column in view.columns: view[column] = view[column].map(fmt_return)
        if "Industry Rank" in view.columns: view["Industry Rank"] = view["Industry Rank"].map(fmt_integer)
        if "Industry Leadership" in view.columns: view["Industry Leadership"] = view["Industry Leadership"].round(1)
        show_table(view, 560, links=True, progress_columns=["Industry Leadership"])

    st.markdown("### Top 20 IPO Setups")
    if ipo.empty:
        st.info("No IPO stocks pass the upstream IPO setup gate on this date.")
    else:
        ipo["Avg Turnover (Cr)"] = numeric_column(ipo, "ipo_turnover_avg") / 10_000_000.0
        if "ipo_setup_score" in ipo.columns:
            ipo["Setup Score"] = numeric_column(ipo, "ipo_setup_score")
        else:
            ipo["Setup Score"] = (
                0.25 * ipo.get("tight_3d_range", pd.Series(index=ipo.index)).rank(pct=True, ascending=False).fillna(0.5)
                + 0.20 * ipo.get("vol_ratio_50", pd.Series(index=ipo.index)).rank(pct=True, ascending=False).fillna(0.5)
                + 0.20 * ipo.get("vwap_premium", pd.Series(index=ipo.index)).rank(pct=True).fillna(0.5)
                + 0.20 * ipo.get("retracement_from_listing_high", pd.Series(index=ipo.index)).rank(pct=True, ascending=False).fillna(0.5)
                + 0.15 * ipo.get("hh_hl_count", pd.Series(index=ipo.index)).rank(pct=True).fillna(0.5)
            ) * 100.0
        ipo["Industry Rank"] = ipo["basic_industry"].map(lookup["Industry Rank"])
        ipo["Industry Leadership"] = normalize_score(ipo["basic_industry"].map(lookup["leadership_score"]))
        ipo["Industry Regime"] = ipo["basic_industry"].map(lookup["regime"])
        ipo = ipo.sort_values("Setup Score", ascending=False).head(TOP_STOCKS).reset_index(drop=True)
        ipo.insert(0, "Rank", range(1, len(ipo) + 1))
        ipo["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo["symbol"].astype(str)
        view = ipo.rename(columns={"symbol":"Symbol", "basic_industry":"Basic Industry", "close":"Close", "days_listed":"Days Listed", "ipo_phase":"Phase", "vwap_premium":"Above VWAP", "retracement_from_listing_high":"Off High"})
        columns = ["Rank","Symbol","Chart","Basic Industry","Industry Rank","Industry Leadership","Industry Regime","Close","Days Listed","Phase","Setup Score","Above VWAP","Off High","Avg Turnover (Cr)"]
        view = view[[column for column in columns if column in view.columns]]
        for column in ["Close", "Setup Score", "Avg Turnover (Cr)"]:
            if column in view.columns: view[column] = view[column].map(lambda value: fmt_number(value, 2))
        for column in ["Above VWAP", "Off High"]:
            if column in view.columns: view[column] = view[column].map(fmt_return)
        if "Days Listed" in view.columns: view["Days Listed"] = view["Days Listed"].map(fmt_integer)
        if "Industry Rank" in view.columns: view["Industry Rank"] = view["Industry Rank"].map(fmt_integer)
        if "Industry Leadership" in view.columns: view["Industry Leadership"] = view["Industry Leadership"].round(1)
        show_table(view, max(260, 40 * len(view) + 60), links=True, progress_columns=["Industry Leadership"])


# =============================================================================
# GROUP DETAIL TABS
# =============================================================================

def group_detail_tab(history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp, group_column: str, title: str) -> None:
    current = current_snapshot(history, selected_date, group_column)
    if current.empty:
        st.warning(f"No {title} data is available for this date.")
        return

    display_name = "Basic Industry" if group_column == "basic_industry" else "Industry"
    current = current.sort_values("Leadership", ascending=False).reset_index(drop=True)
    current.insert(0, "Rank", range(1, len(current) + 1))
    table = current.rename(columns={group_column: display_name, "Leadership":"Leadership Score", "Actionability":"Actionability %", "regime":"Regime", "members":"Stocks"})
    columns = ["Rank", display_name, "Leadership Score", "Actionability %", "Regime", "Stocks"]
    table = table[[column for column in columns if column in table.columns]]
    for column in ["Leadership Score", "Actionability %"]:
        if column in table.columns: table[column] = table[column].round(1)
    if "Stocks" in table.columns: table["Stocks"] = table["Stocks"].map(fmt_integer)
    show_table(table, max(360, min(760, 35 * len(table) + 60)), progress_columns=["Leadership Score", "Actionability %"])

    st.markdown(f"### {title} Constituents")
    selected_group = st.selectbox(title, table[display_name].tolist(), key=f"{group_column}_selector")
    stock_group_column = "basic_industry" if group_column == "basic_industry" else "industry"
    stocks = stock_history[(stock_history["date"] == selected_date) & (stock_history[stock_group_column] == selected_group)].copy()
    if stocks.empty:
        st.info("No constituent stock records are available.")
        return
    if "ret_20d" in stocks.columns: stocks = stocks.sort_values("ret_20d", ascending=False)
    stocks = stocks.head(30).reset_index(drop=True)
    stocks.insert(0, "Rank", range(1, len(stocks) + 1))
    stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
    view = stocks.rename(columns={"symbol":"Symbol", "close":"Close", "ret_20d":"20D Return", "ret_60d":"60D Return", "gain_6m":"6M Gain", "stock_strength_score":"Strength"})
    columns = ["Rank","Symbol","Chart","Close","20D Return","60D Return","6M Gain","Strength","established_buy_setup","ipo_buy_setup"]
    view = view[[column for column in columns if column in view.columns]]
    for column in ["20D Return", "60D Return", "6M Gain"]:
        if column in view.columns: view[column] = view[column].map(fmt_return)
    for column in ["Close", "Strength"]:
        if column in view.columns: view[column] = view[column].map(lambda value: fmt_number(value, 2))
    show_table(view, 420, links=True)


def methodology_tab() -> None:
    st.subheader("How to Read This Dashboard")
    st.markdown(
        """
### Leadership Heatmap

Each row is a Basic Industry. Each score column is one of the last five available trading sessions.

- Below 40: Very weak.
- 40–50: Weak.
- 50–60: Emerging.
- 60–70: Constructive.
- 70–80: Strong.
- Above 80: Very strong.

### Crossing Above 60

An industry appears in the Crossing Above 60 list when its current Leadership Score is at least 60 while its first score in the five-session window was below 60.

### Rotation Buckets

- Strong & Improving: current score at least 60 and positive five-session change.
- Emerging: current score between 50 and 60 and positive change.
- Strong but Fading: current score at least 60 and negative change.
- Weak: current score below 50.

### Top Setups

Established and IPO stocks are selected from their individual upstream setup flags. Industry rank and leadership are shown only as context.
        """
    )


def main() -> None:
    required_files = [BASIC_HISTORY_FILE, INDUSTRY_HISTORY_FILE, STOCK_HISTORY_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]

    if missing:
        st.error("Required dashboard data files are missing. Run the data workflow first.")
        st.code("\n".join(missing))
        st.stop()

    basic_history = prepare_groups(load_parquet(str(BASIC_HISTORY_FILE)), "basic_industry")
    industry_history = prepare_groups(load_parquet(str(INDUSTRY_HISTORY_FILE)), "industry")
    stock_history = prepare_stocks(load_parquet(str(STOCK_HISTORY_FILE)))

    all_dates = sorted(set(trading_dates(basic_history) + trading_dates(industry_history) + trading_dates(stock_history)))

    if not all_dates:
        st.error("No valid trading dates were found in the processed data.")
        st.stop()

    st.title("NSE Industry Momentum Monitor")
    sync = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption("Prepared market data: " + sync.replace("T", " ").replace("Z", " IST"))

    tabs = st.tabs(["Industry Monitor", "Top Setups", "Basic Industry", "Industry", "Methodology"])

    with tabs[0]:
        industry_monitor_tab(basic_history, date_picker(all_dates, "monitor"))
    with tabs[1]:
        stock_setups_tab(basic_history, stock_history, date_picker(all_dates, "setups"))
    with tabs[2]:
        group_detail_tab(basic_history, stock_history, date_picker(all_dates, "basic"), "basic_industry", "Basic Industry")
    with tabs[3]:
        group_detail_tab(industry_history, stock_history, date_picker(all_dates, "industry"), "industry", "Industry")
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
