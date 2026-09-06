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
        .block-container {max-width: 1480px; padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:0.8rem 1rem;}
        [data-testid="stMetricLabel"] {font-size:0.78rem; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;}
        [data-testid="stMetricValue"] {font-weight:700; color:#0F172A;}
        .improver-card {border:1px solid #E2E8F0; border-left:5px solid #15803D; border-radius:11px; padding:0.75rem 0.9rem; margin:0.4rem 0; background:#FFFFFF;}
        .improver-name {font-weight:700; color:#0F172A; font-size:0.96rem;}
        .improver-meta {color:#64748B; font-size:0.78rem; margin-top:0.24rem;}
        .improver-number {font-weight:800; font-size:1.04rem; text-align:right;}
        .status-pill {display:inline-block; padding:0.16rem 0.55rem; border-radius:999px; font-size:0.76rem; font-weight:700; white-space:nowrap;}
        @media (max-width: 800px) {
            .block-container {padding-left:0.7rem; padding-right:0.7rem;}
            .improver-name {font-size:0.87rem;}
            .improver-number {font-size:0.92rem;}
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
        change = float(value)
    except (TypeError, ValueError):
        return MUTED
    if change > 0.05:
        return GREEN
    if change < -0.05:
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

    heading, previous, calendar, next_button = st.columns([3.8, 0.45, 2.0, 0.45])
    with heading:
        st.markdown("#### Analysis date")
        st.caption("Choose an available trading session")
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

    valid_dates = [date for date in all_dates if date <= pd.Timestamp(requested)]
    resolved = valid_dates[-1] if valid_dates else all_dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"Showing: {resolved.strftime('%d %b %Y')}")
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
    improving = improving.sort_values(
        ["Improver Priority", "5D Leadership Change", "Leadership Score"],
        ascending=[False, False, False],
    ).head(TOP_INDUSTRIES).reset_index(drop=True)
    st.markdown("### Leadership Improvers")
    st.caption("Ranks current leadership quality and five-session improvement together. A strong leader improving further ranks above a weak rebound.")
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
    ranked = ranked.sort_values(
        ["Improver Priority", "5D Leadership Change", "Leadership Score"],
        ascending=[False, False, False],
    )
    options = ranked["basic_industry"].tolist()
    if not options:
        st.info("No Basic Industry is available for the selected date.")
        return
    selected_group = st.selectbox(
        "Select Basic Industry",
        options,
        key="selected_basic_industry",
        help="Industries begin with the highest improvement priority and descend.",
    )
    row = current[current["basic_industry"] == selected_group].iloc[-1]
    score = float(row["Leadership Score"])
    change = float(row["5D Leadership Change"])
    regime = clean_text(row.get("regime", "Neutral Transition"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Current leadership", format_number(score))
    c2.metric("5-session change", format_signed(change))
    c3.metric("Current regime", regime)

    history = basic_history[(basic_history["basic_industry"] == selected_group) & (basic_history["date"] <= selected_date)].copy()
    history = history.sort_values("date").tail(30)
    if history.empty:
        st.info("No trend history is available for this industry.")
        return
    history["Leadership Score"] = normalize_score(history["leadership_score"]).fillna(0.0)
    figure = go.Figure(
        go.Scatter(
            x=history["date"],
            y=history["Leadership Score"],
            mode="lines+markers",
            line=dict(color=score_color(score), width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Leadership score: %{y:.1f}<extra></extra>",
        )
    )
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
    improving = int((current["5D Leadership Change"] > 0).sum())
    weakening = int((current["5D Leadership Change"] < 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Industries tracked", format_integer(len(current)))
    c2.metric("Leadership improving", format_integer(improving))
    c3.metric("Leadership weakening", format_integer(weakening))

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
    st.caption("The dropdown begins with the highest improvement priority and descends.")
    render_selected_industry(basic_history, selected_date, current)


def percentile_rank(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    return numeric_column(frame, column).rank(pct=True, ascending=ascending).fillna(0.0)


def stock_setups_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    basic = current_snapshot(basic_history, selected_date, "basic_industry")
    stocks = stock_history[stock_history["date"] == selected_date].copy()
    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return
    basic = add_5_session_change(basic_history, basic, "basic_industry")
    basic = basic.sort_values("Leadership Score", ascending=False).drop_duplicates("basic_industry").reset_index(drop=True)
    basic["Industry Rank"] = range(1, len(basic) + 1)
    lookup = basic.set_index("basic_industry")[["Industry Rank", "Leadership Score", "regime"]]
    established = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Established qualified", format_integer(len(established)))
    c2.metric("IPO qualified", format_integer(len(ipo)))
    c3.metric("Scan date", selected_date.strftime("%d %b %Y"))
    render_stock_table(established, lookup, "Established", TOP_STOCKS)
    render_stock_table(ipo, lookup, "IPO", TOP_STOCKS)


def render_stock_table(stocks: pd.DataFrame, lookup: pd.DataFrame, kind: str, limit: int) -> None:
    st.markdown(f"### Top {limit} {kind} Setups")
    if stocks.empty:
        st.info(f"No {kind.lower()} stocks pass the upstream setup gate on this date.")
        return
    data = stocks.copy()
    if kind == "Established":
        data["Priority Score"] = (
            0.30 * percentile_rank(data, "tight_3d_range", ascending=True)
            + 0.25 * percentile_rank(data, "vol_ratio_50", ascending=True)
            + 0.20 * percentile_rank(data, "gain_6m", ascending=False)
            + 0.15 * percentile_rank(data, "up_down_ratio", ascending=False)
            + 0.10 * percentile_rank(data, "stock_strength_score", ascending=False)
        ) * 100.0
    else:
        data["Priority Score"] = numeric_column(data, "ipo_setup_score")
        if (data["Priority Score"] == 0).all():
            data["Priority Score"] = (
                0.25 * percentile_rank(data, "tight_3d_range", ascending=True)
                + 0.20 * percentile_rank(data, "vol_ratio_50", ascending=True)
                + 0.20 * percentile_rank(data, "vwap_premium", ascending=False)
                + 0.20 * percentile_rank(data, "retracement_from_listing_high", ascending=True)
                + 0.15 * percentile_rank(data, "hh_hl_count", ascending=False)
            ) * 100.0
    data["Industry Rank"] = data["basic_industry"].map(lookup["Industry Rank"])
    data["Industry Leadership"] = data["basic_industry"].map(lookup["Leadership Score"])
    data["Industry Regime"] = data["basic_industry"].map(lookup["regime"])
    data = data.sort_values("Priority Score", ascending=False).head(limit).reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    view = data.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close"})
    keep = ["Rank", "Symbol", "Chart", "Basic Industry", "Industry Rank", "Industry Leadership", "Industry Regime", "Close", "Priority Score"]
    view = view[[column for column in keep if column in view.columns]]
    for column in ["Close", "Priority Score", "Industry Leadership"]:
        if column in view.columns:
            view[column] = view[column].map(lambda value: format_number(value, 1))
    if "Industry Rank" in view.columns:
        view["Industry Rank"] = view["Industry Rank"].map(format_integer)
    show_table(view, max(260, 37 * len(view) + 60), chart_links=True)


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
    selected_group = st.selectbox(title, options, key=f"{group_column}_selector", help="Starts with maximum 5-session improvement and descends.")
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
    st.subheader("How to read the monitor")
    st.markdown(
        """
        ### Leadership Score
        - **Leadership Score** is the current relative-strength score for a group, shown on a 0–100 scale.
        - **5D Leadership Change** equals the current score minus the score five available trading sessions earlier.
        - The Leadership Improvers section ranks both current leadership and positive improvement so strong industries improving further receive priority over weak rebounds.

        ### Colour guide
        - Green: positive leadership momentum.
        - Red: negative leadership momentum.
        - Amber: transition/watchlist condition.
        - Dark green: strong leadership, normally 70 or above.

        This dashboard is a research tool, not a trade recommendation.
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
        stock_setups_tab(basic_history, stock_history, date_picker(all_dates, "setups"))
    with tabs[2]:
        group_detail_tab(basic_history, stock_history, date_picker(all_dates, "basic"), "basic_industry", "Basic Industry")
    with tabs[3]:
        group_detail_tab(industry_history, stock_history, date_picker(all_dates, "industry"), "industry", "Industry")
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
