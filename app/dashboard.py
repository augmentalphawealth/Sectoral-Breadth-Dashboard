# app/dashboard.py
# NSE Industry Momentum Monitor

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

INK = "#0F172A"
MUTED = "#64748B"
GREEN = "#15803D"
DARK_GREEN = "#166534"
LIGHT_GREEN = "#DCFCE7"
RED = "#B91C1C"
LIGHT_RED = "#FEE2E2"
AMBER = "#B45309"
LIGHT_AMBER = "#FEF3C7"
PURPLE = "#7C3AED"

st.set_page_config(
    page_title="NSE Industry Momentum Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1480px; padding-top: 1.15rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:0.75rem 0.95rem;}
        [data-testid="stMetricLabel"] {font-size:0.76rem; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;}
        [data-testid="stMetricValue"] {font-weight:700; color:#0F172A;}
        .improver-card {border:1px solid #E2E8F0; border-left:5px solid #15803D; border-radius:11px; padding:0.72rem 0.85rem; margin:0.38rem 0; background:#FFFFFF;}
        .improver-name {font-weight:700; color:#0F172A; font-size:0.95rem;}
        .improver-meta {color:#64748B; font-size:0.77rem; margin-top:0.2rem;}
        .improver-number {font-weight:800; font-size:1.02rem; text-align:right;}
        .status-pill {display:inline-block; padding:0.16rem 0.52rem; border-radius:999px; font-size:0.75rem; font-weight:700; white-space:nowrap;}
        @media (max-width: 800px) {
            .block-container {padding-left:0.7rem; padding-right:0.7rem;}
            .improver-name {font-size:0.86rem;}
            .improver-number {font-size:0.9rem;}
        }
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
    if not valid.empty and valid.abs().max() <= 1.5:
        return series * 100.0
    return series


def format_number(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_signed(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):+,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_integer(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def format_percent(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        number = float(value)
        if abs(number) <= 1.5:
            number *= 100.0
        return f"{number:,.1f}%"
    except (TypeError, ValueError):
        return "—"


def score_color(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return MUTED
    if score >= 70:
        return DARK_GREEN
    if score >= 60:
        return GREEN
    if score >= 50:
        return AMBER
    return RED


def change_color(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return MUTED
    if value > 0.05:
        return GREEN
    if value < -0.05:
        return RED
    return MUTED


def change_indicator(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "⚪ —"
    if number > 0.05:
        return f"🟢 {format_signed(number)}"
    if number < -0.05:
        return f"🔴 {format_signed(number)}"
    return f"⚪ {format_signed(number)}"


def quality_indicator(value: object, lower_is_better: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "⚪ —"
    if lower_is_better:
        if number <= 0.7:
            return f"🟢 {format_number(number)}"
        if number <= 1.0:
            return f"🟡 {format_number(number)}"
        return f"🔴 {format_number(number)}"
    if number >= 1.5:
        return f"🟢 {format_number(number)}"
    if number >= 1.0:
        return f"🟡 {format_number(number)}"
    return f"🔴 {format_number(number)}"


def leadership_status(score: object, change: object) -> tuple[str, str, str]:
    score_value = float(score) if pd.notna(score) else 0.0
    change_value = float(change) if pd.notna(change) else 0.0
    if score_value >= 70 and change_value > 0:
        return "Strong leader · Accelerating", DARK_GREEN, LIGHT_GREEN
    if score_value >= 70:
        return "Strong leadership", DARK_GREEN, LIGHT_GREEN
    if score_value >= 60 and change_value > 0:
        return "Building leadership", GREEN, LIGHT_GREEN
    if score_value >= 60:
        return "Positive transition", GREEN, LIGHT_GREEN
    if score_value >= 50 and change_value > 0:
        return "Improving · Watchlist", AMBER, LIGHT_AMBER
    if score_value >= 50:
        return "Neutral transition", AMBER, LIGHT_AMBER
    if change_value > 0:
        return "Improving · Not yet confirmed", AMBER, LIGHT_AMBER
    return "Weak leadership", RED, LIGHT_RED


def apply_chart_style(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=8, r=25, t=45, b=20),
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color=INK),
        title_font=dict(size=14, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    figure.update_yaxes(showgrid=False)
    return figure


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame


def prepare_group_data(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Neutral Transition"
    if "leadership_score" not in data.columns:
        data["leadership_score"] = data.get("strength_score", 0.0)
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    data["leadership_score"] = normalize_score(data["leadership_score"]).fillna(0.0)
    return data


def prepare_stock_data(frame: pd.DataFrame) -> pd.DataFrame:
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


def get_trading_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if "date" not in frame.columns:
        return []
    return sorted(pd.Timestamp(value) for value in frame["date"].dropna().unique())


def format_sync_time(raw_text: str) -> str:
    text = raw_text.strip()
    if not text or text.lower() == "not available":
        return "Not available"
    try:
        timestamp = pd.Timestamp(text.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Kolkata")
        else:
            timestamp = timestamp.tz_convert("Asia/Kolkata")
        return timestamp.strftime("%d %b %Y, %I:%M %p IST")
    except (TypeError, ValueError):
        return text.replace("T", " ").replace("Z", "")


def date_picker(all_dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = all_dates[-1]
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest
    selected = pd.Timestamp(st.session_state[state_key]).normalize()
    if selected not in all_dates:
        selected = latest
        st.session_state[state_key] = latest
    selected_index = all_dates.index(selected)

    previous, calendar, next_button, selected_label = st.columns([0.5, 2.0, 0.5, 3.4])
    with previous:
        if st.button("‹", key=f"{key}_previous", disabled=selected_index == 0, use_container_width=True):
            st.session_state[state_key] = all_dates[selected_index - 1]
            st.rerun()
    with calendar:
        requested = st.date_input(
            "Analysis date",
            value=selected.date(),
            min_value=all_dates[0].date(),
            max_value=latest.date(),
            key=f"{key}_calendar",
            label_visibility="collapsed",
            format="DD/MM/YYYY",
        )
    with next_button:
        if st.button("›", key=f"{key}_next", disabled=selected_index == len(all_dates) - 1, use_container_width=True):
            st.session_state[state_key] = all_dates[selected_index + 1]
            st.rerun()
    with selected_label:
        st.markdown(f"<div style='padding-top:0.34rem; color:{MUTED}; font-size:0.88rem;'>Analysis date: <b style='color:{INK};'>{selected.strftime('%d %b %Y')}</b></div>", unsafe_allow_html=True)

    requested_timestamp = pd.Timestamp(requested).normalize()
    if requested_timestamp in all_dates:
        resolved = requested_timestamp
    else:
        prior_dates = [date for date in all_dates if date <= requested_timestamp]
        resolved = prior_dates[-1] if prior_dates else all_dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    if requested_timestamp != resolved:
        st.caption(f"{requested_timestamp.strftime('%d %b %Y')} has no EOD data. Showing prior trading session: {resolved.strftime('%d %b %Y')}.")
    return resolved


def show_table(data: pd.DataFrame, height: int, chart_links: bool = False) -> None:
    view = data.copy()
    view.columns = [str(column) for column in view.columns]
    view = view.loc[:, ~view.columns.duplicated(keep="first")]
    config: dict[str, object] = {}
    if chart_links and "Chart" in view.columns:
        config["Chart"] = st.column_config.LinkColumn("Chart", display_text="Open ↗")
    st.dataframe(view, use_container_width=True, hide_index=True, height=height, column_config=config)


def current_snapshot(history: pd.DataFrame, selected_date: pd.Timestamp, group_column: str) -> pd.DataFrame:
    data = history[history["date"] == selected_date].copy()
    if "members" in data.columns:
        data = data[numeric_column(data, "members") >= SMALL_GROUP_LIMIT].copy()
    data = data.drop_duplicates(group_column, keep="last")
    data["Leadership Score"] = normalize_score(data["leadership_score"]).fillna(0.0)
    return data


def add_5_session_change(history: pd.DataFrame, current: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if current.empty:
        return current
    current_date = current["date"].iloc[0]
    earlier_dates = [date for date in get_trading_dates(history) if date < current_date]
    if len(earlier_dates) < 5:
        current["5D Leadership Change"] = 0.0
        return current
    prior_date = earlier_dates[-5]
    prior = history[history["date"] == prior_date].drop_duplicates(group_column, keep="last").copy()
    prior["Prior Leadership"] = normalize_score(prior["leadership_score"]).fillna(0.0)
    result = current.merge(prior[[group_column, "Prior Leadership"]], on=group_column, how="left")
    result["Prior Leadership"] = result["Prior Leadership"].fillna(result["Leadership Score"])
    result["5D Leadership Change"] = result["Leadership Score"] - result["Prior Leadership"]
    return result


def priority_score(frame: pd.DataFrame) -> pd.Series:
    return 0.65 * frame["Leadership Score"] + 0.35 * frame["5D Leadership Change"].clip(lower=0)


def render_improver_cards(current: pd.DataFrame) -> None:
    improving = current[current["5D Leadership Change"] > 0].copy()
    if improving.empty:
        st.info("No Basic Industry improved in Leadership Score over the last five available trading sessions.")
        return
    improving["Improver Priority"] = priority_score(improving)
    improving = improving.sort_values(["Improver Priority", "5D Leadership Change", "Leadership Score"], ascending=[False, False, False]).head(TOP_INDUSTRIES).reset_index(drop=True)
    st.markdown("### Leadership Improvers")
    st.caption("Ranking rewards current leadership quality and five-session improvement together.")
    for rank, row in improving.iterrows():
        status, status_color, status_bg = leadership_status(row["Leadership Score"], row["5D Leadership Change"])
        st.markdown(
            f"""
            <div class="improver-card" style="border-left-color:{status_color};">
                <div style="display:flex; justify-content:space-between; gap:12px; align-items:start;">
                    <div style="min-width:0;">
                        <div class="improver-name">{rank + 1}. {clean_text(row.get('basic_industry'))}</div>
                        <div class="improver-meta"><span class="status-pill" style="color:{status_color}; background:{status_bg};">{status}</span></div>
                    </div>
                    <div style="display:flex; gap:18px; flex-shrink:0;">
                        <div class="improver-number" style="color:{score_color(row['Leadership Score'])};">{format_number(row['Leadership Score'])}<div class="improver-meta">Current score</div></div>
                        <div class="improver-number" style="color:{change_color(row['5D Leadership Change'])};">{format_signed(row['5D Leadership Change'])}<div class="improver-meta">5-session change</div></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_selected_industry(basic_history: pd.DataFrame, selected_date: pd.Timestamp, current: pd.DataFrame) -> None:
    ranked = current.copy()
    ranked["Improver Priority"] = priority_score(ranked)
    ranked = ranked.sort_values(["Improver Priority", "5D Leadership Change", "Leadership Score"], ascending=[False, False, False])
    options = ranked["basic_industry"].tolist()
    if not options:
        st.info("No Basic Industry is available for the selected date.")
        return
    selected_group = st.selectbox("Select Basic Industry", options, key="selected_basic_industry", help="Starts with the highest improvement priority and descends.")
    row = current[current["basic_industry"] == selected_group].iloc[-1]
    score = float(row["Leadership Score"])
    change = float(row["5D Leadership Change"])
    regime = clean_text(row.get("regime", "Neutral Transition"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Current leadership", format_number(score))
    c2.metric("5-session change", format_signed(change))
    c3.metric("Current regime", regime)
    history = basic_history[(basic_history["basic_industry"] == selected_group) & (basic_history["date"] <= selected_date)].copy().sort_values("date").tail(30)
    if history.empty:
        st.info("No trend history is available for this industry.")
        return
    history["Leadership Score"] = normalize_score(history["leadership_score"]).fillna(0.0)
    figure = go.Figure(go.Scatter(x=history["date"], y=history["Leadership Score"], mode="lines+markers", line=dict(color=score_color(score), width=3), marker=dict(size=7), hovertemplate="<b>%{x|%d %b %Y}</b><br>Leadership score: %{y:.1f}<extra></extra>"))
    for threshold, color in [(70, DARK_GREEN), (60, GREEN), (50, AMBER)]:
        figure.add_hline(y=threshold, line_dash="dot", line_color=color, opacity=0.85)
    figure.update_layout(title=f"{selected_group}: Leadership Score trend", xaxis_title=None, yaxis_title="Leadership Score")
    figure.update_yaxes(range=[0, 100])
    st.plotly_chart(apply_chart_style(figure, 390), use_container_width=True)


def industry_monitor_tab(basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    current = current_snapshot(basic_history, selected_date, "basic_industry")
    if current.empty:
        st.warning("No Basic Industry data is available for the selected date.")
        return
    current = add_5_session_change(basic_history, current, "basic_industry")
    c1, c2, c3 = st.columns(3)
    c1.metric("Industries tracked", format_integer(len(current)))
    c2.metric("Leadership improving", format_integer((current["5D Leadership Change"] > 0).sum()))
    c3.metric("Leadership weakening", format_integer((current["5D Leadership Change"] < 0).sum()))
    render_improver_cards(current)
    st.markdown("### Industry leadership table")
    table = current[["basic_industry", "Leadership Score", "5D Leadership Change"]].copy()
    table["Status"] = table.apply(lambda row: leadership_status(row["Leadership Score"], row["5D Leadership Change"])[0], axis=1)
    table = table.sort_values(["Leadership Score", "5D Leadership Change"], ascending=[False, False]).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table = table.rename(columns={"basic_industry": "Basic Industry"})
    table["Leadership Score"] = table["Leadership Score"].map(lambda value: format_number(value, 1))
    table["5D Leadership Change"] = table["5D Leadership Change"].map(change_indicator)
    show_table(table, max(360, min(760, 35 * len(table) + 60)))
    st.markdown("### Selected industry trend")
    render_selected_industry(basic_history, selected_date, current)


def percentile_rank(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    return numeric_column(frame, column).rank(pct=True, ascending=ascending).fillna(0.0)


def prior_move_series(data: pd.DataFrame) -> pd.Series:
    for column in ["gain_6m", "ret_60d", "ret_20d", "ret_120d"]:
        if column in data.columns:
            return numeric_column(data, column)
    return pd.Series(0.0, index=data.index)


def volume_dryup_series(data: pd.DataFrame) -> pd.Series:
    for column in ["vol_ratio_50", "volume_ratio_50", "vol_ratio", "volume_ratio"]:
        if column in data.columns:
            return numeric_column(data, column)
    return pd.Series(0.0, index=data.index)


def tightness_series(data: pd.DataFrame) -> pd.Series:
    for column in ["tight_3d_range", "tightness_3d", "range_3d_pct"]:
        if column in data.columns:
            return numeric_column(data, column)
    return pd.Series(0.0, index=data.index)


def render_stock_table(stocks: pd.DataFrame, kind: str, limit: int) -> None:
    st.markdown(f"### Top {limit} {kind} Setups")
    if stocks.empty:
        st.info(f"No {kind.lower()} stocks pass the upstream setup gate on this date.")
        return
    data = stocks.copy()
    tightness = tightness_series(data)
    volume_ratio = volume_dryup_series(data)
    prior_move = prior_move_series(data)
    if kind == "Established":
        data["Priority Score"] = (
            0.35 * tightness.rank(pct=True, ascending=True)
            + 0.25 * volume_ratio.rank(pct=True, ascending=True)
            + 0.25 * prior_move.rank(pct=True, ascending=False)
            + 0.15 * numeric_column(data, "stock_strength_score").rank(pct=True, ascending=False)
        ) * 100.0
    else:
        data["Priority Score"] = numeric_column(data, "ipo_setup_score")
        if (data["Priority Score"] == 0).all():
            data["Priority Score"] = (
                0.35 * tightness.rank(pct=True, ascending=True)
                + 0.25 * volume_ratio.rank(pct=True, ascending=True)
                + 0.25 * prior_move.rank(pct=True, ascending=False)
                + 0.15 * numeric_column(data, "stock_strength_score").rank(pct=True, ascending=False)
            ) * 100.0
    data["Tightness (3D)"] = tightness
    data["Volume vs 50D"] = volume_ratio
    data["Prior Move"] = prior_move
    data = data.sort_values("Priority Score", ascending=False).head(limit).reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    view = data.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry"})
    keep = ["Rank", "Symbol", "Chart", "Basic Industry", "Priority Score", "Tightness (3D)", "Volume vs 50D", "Prior Move"]
    view = view[[column for column in keep if column in view.columns]]
    if "Priority Score" in view.columns:
        view["Priority Score"] = view["Priority Score"].map(lambda value: format_number(value, 1))
    if "Tightness (3D)" in view.columns:
        view["Tightness (3D)"] = view["Tightness (3D)"].map(lambda value: quality_indicator(value, lower_is_better=True))
    if "Volume vs 50D" in view.columns:
        view["Volume vs 50D"] = view["Volume vs 50D"].map(lambda value: quality_indicator(value, lower_is_better=True))
    if "Prior Move" in view.columns:
        view["Prior Move"] = view["Prior Move"].map(format_percent)
    show_table(view, max(280, 38 * len(view) + 60), chart_links=True)


def stock_setups_tab(stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    stocks = stock_history[stock_history["date"] == selected_date].copy()
    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return
    established = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Established qualified", format_integer(len(established)))
    c2.metric("IPO qualified", format_integer(len(ipo)))
    c3.metric("Scan date", selected_date.strftime("%d %b %Y"))
    st.caption("Tightness and Volume vs 50D: 🟢 lower values are constructive consolidation/dry-up; 🟡 is neutral; 🔴 is less constructive. Prior Move uses the best available upstream return field.")
    render_stock_table(established, "Established", TOP_STOCKS)
    render_stock_table(ipo, "IPO", TOP_STOCKS)


def group_detail_tab(history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp, group_column: str, title: str) -> None:
    current = current_snapshot(history, selected_date, group_column)
    if current.empty:
        st.warning(f"No {title} data is available for the selected date.")
        return
    current = add_5_session_change(history, current, group_column)
    current = current.sort_values(["Leadership Score", "5D Leadership Change"], ascending=[False, False]).reset_index(drop=True)
    current.insert(0, "Rank", range(1, len(current) + 1))
    display_name = "Basic Industry" if group_column == "basic_industry" else "Industry"
    table = current.rename(columns={group_column: display_name, "members": "Stocks"})
    table["Status"] = table.apply(lambda row: leadership_status(row["Leadership Score"], row["5D Leadership Change"])[0], axis=1)
    keep = ["Rank", display_name, "Leadership Score", "5D Leadership Change", "Status", "Stocks"]
    table = table[[column for column in keep if column in table.columns]]
    table["Leadership Score"] = table["Leadership Score"].map(lambda value: format_number(value, 1))
    table["5D Leadership Change"] = table["5D Leadership Change"].map(change_indicator)
    if "Stocks" in table.columns:
        table["Stocks"] = table["Stocks"].map(format_integer)
    show_table(table, max(360, min(760, 35 * len(table) + 60)))

    st.markdown(f"### {title} constituents")
    options = current.sort_values("5D Leadership Change", ascending=False)[group_column].tolist()
    selected_group = st.selectbox(title, options, key=f"{group_column}_selector", help="Starts with maximum five-session improvement and descends.")
    stock_group_column = "basic_industry" if group_column == "basic_industry" else "industry"
    stocks = stock_history[(stock_history["date"] == selected_date) & (stock_history[stock_group_column] == selected_group)].copy()
    if stocks.empty:
        st.info("No constituent stock records are available for this selected trading date.")
        return
    if "ret_20d" in stocks.columns:
        stocks = stocks.sort_values("ret_20d", ascending=False)
    stocks = stocks.head(30).reset_index(drop=True)
    stocks.insert(0, "Rank", range(1, len(stocks) + 1))
    stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
    view = stocks.rename(columns={"symbol": "Symbol", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain", "stock_strength_score": "Strength"})
    keep = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "established_buy_setup", "ipo_buy_setup"]
    view = view[[column for column in keep if column in view.columns]]
    for column in ["20D Return", "60D Return", "6M Gain"]:
        if column in view.columns:
            view[column] = view[column].map(format_percent)
    for column in ["Close", "Strength"]:
        if column in view.columns:
            view[column] = view[column].map(lambda value: format_number(value, 1))
    show_table(view, 440, chart_links=True)


def methodology_tab() -> None:
    st.subheader("Detailed methodology and dashboard guide")
    st.markdown(
        """
        ## Purpose
        This monitor is a market-research workspace for identifying Basic Industries that are already showing relative leadership, improving in leadership, and producing individual stock setups. It is not a buy/sell recommendation system.

        ## Dates and available EOD data
        - The Analysis Date selector uses only the EOD dates present in the dashboard data files.
        - If a calendar date is a weekend, exchange holiday, or has not been downloaded, the dashboard automatically resolves it to the most recent available prior trading session.
        - The label beside the date selector tells you the exact EOD session currently being shown.
        - Last Data Refresh is the timestamp recorded by the data workflow. It is displayed in Indian Standard Time when a valid timestamp is available.

        ## Industry Leadership Score
        - Leadership Score is a relative-strength-style score scaled to 0–100.
        - Higher values represent stronger current leadership versus the rest of the tracked universe.
        - Scores at or above 70 are shown as strong leadership.
        - Scores from 60 to 69.9 are constructive/positive transition.
        - Scores from 50 to 59.9 are neutral-transition/watchlist territory.
        - Scores below 50 are weak leadership unless there is a sustained improvement in later sessions.

        ## Five-session Leadership Change
        - 5D Leadership Change equals the current Leadership Score minus the Leadership Score from five available trading sessions earlier.
        - A positive number means the industry improved relative to the market over that period.
        - A negative number means leadership deteriorated.
        - The Industry Leadership Table uses 🟢 for positive change, 🔴 for negative change, and ⚪ for near-zero change.
        - Change must always be interpreted with the current score. A rise from 40 to 55 is constructive but is not equivalent to a rise from 70 to 80.

        ## Leadership Improvers
        - The compact Leadership Improvers cards combine current score and recent improvement.
        - Their internal ranking uses 65% current Leadership Score and 35% positive five-session improvement.
        - The purpose is to prioritize industries that are both strong now and getting stronger, rather than over-emphasizing weak industries that merely bounced.
        - Strong Leader · Accelerating normally means a score of at least 70 with positive five-session change.
        - Improving · Watchlist means momentum improved but the current score still needs confirmation.

        ## Selected Industry Trend
        - The dropdown starts from the strongest improvement-priority industry and descends.
        - The line chart shows the most recent 30 available trading sessions through the selected analysis date.
        - Dotted levels at 50, 60 and 70 provide context for weak, transitional and strong leadership zones.
        - Use this view to judge whether an improvement is persistent, a recovery from weakness, or merely a short-term bounce.

        ## Top Setup Stocks
        - These tables contain stocks that passed the upstream Established or IPO setup flags.
        - The dashboard does not use industry leadership, industry rank, or regime as a hard filter for stocks. A good individual setup can occur in a mixed group, and a strong group can contain extended stocks.
        - Priority Score ranks qualifying stocks within the displayed category. It is a research-ordering score, not a probability of success.

        ## Setup-table fields
        - Tightness (3D): uses the best available upstream three-day tightness/range field. Lower values usually indicate a narrower consolidation and are shown as more constructive.
        - Volume vs 50D: uses the best available volume-ratio field. Values below 1.0 indicate volume below the 50-day normal level. Lower values are often constructive during a base because they can indicate volume dry-up; this should be confirmed visually before a breakout.
        - Prior Move: uses the best available upstream move field in this order: 6M Gain, 60D Return, 20D Return, then 120D Return. It shows whether the stock had an earlier advance before consolidation.
        - 🟢 on Tightness or Volume vs 50D means lower/more constructive under the dashboard thresholds. 🟡 is intermediate. 🔴 is less constructive. These are screening cues, not standalone trading signals.
        - Open ↗ opens the matching NSE symbol in TradingView for chart confirmation.

        ## Suggested workflow
        1. Begin with Leadership Improvers and identify industries with a high current score plus positive five-session improvement.
        2. Open the Selected Industry Trend and check whether the improvement is sustained.
        3. Review Top Setup Stocks for tightness, prior move and volume dry-up.
        4. Open the chart and manually check base structure, price/volume behaviour, key moving averages, resistance and liquidity.
        5. Define entry, invalidation level, stop-loss, position size and market-risk conditions independently.

        ## Limitations
        - Values depend on the accuracy and completeness of the upstream EOD pipeline.
        - A missing session, small industry membership, corporate action, symbol mapping issue, or stale cache can affect results.
        - Past price/volume structure does not guarantee a future move. Always use risk management and independent judgement.
        """
    )


def main() -> None:
    required_files = [BASIC_HISTORY_FILE, INDUSTRY_HISTORY_FILE, STOCK_HISTORY_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        st.error("Required dashboard data files are missing. Run the data workflow first.")
        st.code("\n".join(missing))
        st.stop()

    basic_history = prepare_group_data(load_parquet(str(BASIC_HISTORY_FILE)), "basic_industry")
    industry_history = prepare_group_data(load_parquet(str(INDUSTRY_HISTORY_FILE)), "industry")
    stock_history = prepare_stock_data(load_parquet(str(STOCK_HISTORY_FILE)))
    all_dates = sorted(set(get_trading_dates(basic_history) + get_trading_dates(industry_history) + get_trading_dates(stock_history)))
    if not all_dates:
        st.error("No valid trading dates were found in the processed data.")
        st.stop()

    st.title("NSE Industry Momentum Monitor")
    raw_sync = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Last data refresh: {format_sync_time(raw_sync)}")

    tabs = st.tabs(["Industry Monitor", "Top Setups", "Basic Industry", "Industry", "Methodology"])
    with tabs[0]:
        industry_monitor_tab(basic_history, date_picker(all_dates, "monitor"))
    with tabs[1]:
        stock_setups_tab(stock_history, date_picker(all_dates, "setups"))
    with tabs[2]:
        group_detail_tab(basic_history, stock_history, date_picker(all_dates, "basic"), "basic_industry", "Basic Industry")
    with tabs[3]:
        group_detail_tab(industry_history, stock_history, date_picker(all_dates, "industry"), "industry", "Industry")
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
