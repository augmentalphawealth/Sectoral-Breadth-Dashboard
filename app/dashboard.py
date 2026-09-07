# app/dashboard.py
# Fast snapshot-only Streamlit dashboard.
# GitHub Actions prepares every calculation and every date snapshot. This app
# only reads the selected small snapshot and presents it.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATES_FILE = PROCESSED / "dashboard_dates.parquet"
SNAPSHOT_ROOT = PROCESSED / "dashboard_snapshots"
SYNC_FILE = PROCESSED / "last_sync.txt"

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

st.set_page_config(
    page_title="NSE Industry Momentum Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width:1480px; padding-top:1rem; padding-bottom:2rem;}
    [data-testid="stMetric"] {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:.72rem .9rem;}
    [data-testid="stMetricLabel"] {font-size:.75rem; color:#64748B; text-transform:uppercase; letter-spacing:.04em;}
    [data-testid="stMetricValue"] {font-weight:700; color:#0F172A;}
    .improver-card {border:1px solid #E2E8F0; border-left:5px solid #15803D; border-radius:11px; padding:.68rem .82rem; margin:.34rem 0; background:#FFFFFF;}
    .improver-name {font-weight:700; color:#0F172A; font-size:.94rem;}
    .improver-meta {color:#64748B; font-size:.76rem; margin-top:.2rem;}
    .improver-number {font-weight:800; font-size:1.0rem; text-align:right;}
    .status-pill {display:inline-block; padding:.16rem .50rem; border-radius:999px; font-size:.74rem; font-weight:700; white-space:nowrap;}
    div.stButton > button[kind="tertiary"] {padding:0; min-height:0; border:0; color:#1D4ED8; font-size:.74rem; justify-content:flex-start;}
    div.stButton > button[kind="tertiary"]:hover {color:#1E40AF; text-decoration:underline;}
    @media (max-width:800px) {.block-container {padding-left:.7rem; padding-right:.7rem;}.improver-name {font-size:.85rem;}.improver-number {font-size:.89rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def number(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_number(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_signed(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):+,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_integer(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        raw = float(value)
        percent = raw * 100.0 if abs(raw) <= 1.5 else raw
        return f"{percent:,.1f}%"
    except (TypeError, ValueError):
        return "—"


def score_color(value: object) -> str:
    value = number(value)
    if value >= 70:
        return DARK_GREEN
    if value >= 60:
        return GREEN
    if value >= 50:
        return AMBER
    return RED


def change_color(value: object) -> str:
    value = number(value)
    if value > 0.05:
        return GREEN
    if value < -0.05:
        return RED
    return MUTED


def change_indicator(value: object) -> str:
    value = number(value)
    if value > 0.05:
        return f"🟢 {format_signed(value)}"
    if value < -0.05:
        return f"🔴 {format_signed(value)}"
    return f"⚪ {format_signed(value)}"


def leadership_status(score: object, change: object) -> tuple[str, str, str]:
    score_value = number(score)
    change_value = number(change)
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
def load_dates(path: str, modified: float) -> list[pd.Timestamp]:
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        raise ValueError("dashboard_dates.parquet is missing the date column")
    dates = sorted(pd.Timestamp(value).normalize() for value in pd.to_datetime(frame["date"], errors="coerce").dropna().unique())
    if not dates:
        raise ValueError("dashboard_dates.parquet has no valid dates")
    return dates


@st.cache_data(show_spinner=False)
def load_snapshot(path: str, modified: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame


def snapshot_path(selected_date: pd.Timestamp, filename: str) -> Path:
    return SNAPSHOT_ROOT / selected_date.strftime("%Y-%m-%d") / filename


def load_selected_snapshot(selected_date: pd.Timestamp, filename: str) -> pd.DataFrame:
    path = snapshot_path(selected_date, filename)
    if not path.exists():
        raise FileNotFoundError(f"Prepared snapshot is unavailable: {path.relative_to(ROOT)}")
    return load_snapshot(str(path), path.stat().st_mtime)


def load_trend(group_kind: str, selected_date: pd.Timestamp, group_name: str) -> pd.DataFrame:
    filename = f"{group_kind}_history.parquet"
    path = PROCESSED / f"dashboard_{filename}"
    if not path.exists():
        return pd.DataFrame()
    history = load_snapshot(str(path), path.stat().st_mtime)
    group_column = group_kind
    if group_column not in history.columns or "date" not in history.columns:
        return pd.DataFrame()
    result = history[(history[group_column].map(clean_text) == group_name) & (history["date"] <= selected_date)].copy()
    return result.sort_values("date").tail(30)


def format_sync_time() -> str:
    if not SYNC_FILE.exists():
        return "Not available"
    text = SYNC_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return "Not available"
    try:
        timestamp = pd.Timestamp(text.replace("Z", "+00:00"))
        timestamp = timestamp.tz_localize("Asia/Kolkata") if timestamp.tzinfo is None else timestamp.tz_convert("Asia/Kolkata")
        return timestamp.strftime("%d %b %Y, %I:%M %p IST")
    except (TypeError, ValueError):
        return text.replace("T", " ").replace("Z", "")


def resolve_date(requested: object, dates: list[pd.Timestamp]) -> pd.Timestamp:
    requested_date = pd.Timestamp(requested).normalize()
    if requested_date in dates:
        return requested_date
    earlier = [date for date in dates if date <= requested_date]
    return earlier[-1] if earlier else dates[0]


def global_date_picker(dates: list[pd.Timestamp]) -> pd.Timestamp:
    state_key = "global_analysis_date"
    widget_key = "global_analysis_date_calendar"
    if state_key not in st.session_state:
        st.session_state[state_key] = dates[-1]
    selected = resolve_date(st.session_state[state_key], dates)
    st.session_state[state_key] = selected

    previous, calendar, next_button, label, spacer = st.columns([0.28, 1.15, 0.28, 1.45, 3.84])
    index = dates.index(selected)
    with previous:
        if st.button("‹", key="global_previous_date", disabled=index == 0, use_container_width=True):
            st.session_state[state_key] = dates[index - 1]
            st.rerun()
    with calendar:
        requested = st.date_input(
            "Analysis date",
            value=selected.date(),
            min_value=dates[0].date(),
            max_value=dates[-1].date(),
            key=f"{widget_key}_{selected.strftime('%Y%m%d')}",
            label_visibility="collapsed",
            format="DD/MM/YYYY",
        )
    with next_button:
        if st.button("›", key="global_next_date", disabled=index == len(dates) - 1, use_container_width=True):
            st.session_state[state_key] = dates[index + 1]
            st.rerun()

    resolved = resolve_date(requested, dates)
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    with label:
        st.markdown(f"<div style='padding-top:.35rem; color:{MUTED}; font-size:.82rem;'>Analysis date:<br><b style='color:{INK};'>{resolved.strftime('%d %b %Y')}</b></div>", unsafe_allow_html=True)
    if pd.Timestamp(requested).normalize() != resolved:
        st.caption(f"{pd.Timestamp(requested).strftime('%d %b %Y')} has no prepared EOD snapshot. Showing {resolved.strftime('%d %b %Y')}.")
    return resolved


def show_table(data: pd.DataFrame, height: int, chart_links: bool = False) -> None:
    view = data.copy()
    view.columns = [str(column) for column in view.columns]
    view = view.loc[:, ~view.columns.duplicated(keep="first")]
    config: dict[str, object] = {}
    if chart_links and "Chart" in view.columns:
        config["Chart"] = st.column_config.LinkColumn("Chart", display_text="Open ↗")
    st.dataframe(view, use_container_width=True, hide_index=True, height=height, column_config=config)


def get_score_column(frame: pd.DataFrame) -> str:
    if "leadership_score" in frame.columns:
        return "leadership_score"
    if "strength_score" in frame.columns:
        return "strength_score"
    return ""


def get_change_column(frame: pd.DataFrame) -> str:
    return "leadership_change_5d" if "leadership_change_5d" in frame.columns else ""


def select_group_everywhere(group_kind: str, group_name: str) -> None:
    st.session_state[f"selected_{group_kind}_constituents"] = group_name
    st.session_state[f"selected_{group_kind}_trend"] = group_name
    st.session_state["selection_notice"] = f"{group_kind.replace('_', ' ').title()}: {group_name}"
    st.rerun()


def render_improver_cards(frame: pd.DataFrame, group_column: str, title: str) -> None:
    score_column = get_score_column(frame)
    change_column = get_change_column(frame)
    if not score_column or not change_column or group_column not in frame.columns:
        st.info(f"Prepared {title} snapshot does not contain the required leadership fields.")
        return
    data = frame.copy()
    data["_score"] = pd.to_numeric(data[score_column], errors="coerce").fillna(0.0)
    data["_change"] = pd.to_numeric(data[change_column], errors="coerce").fillna(0.0)
    if "improver_priority" in data.columns:
        data["_priority"] = pd.to_numeric(data["improver_priority"], errors="coerce").fillna(0.0)
    else:
        data["_priority"] = 0.65 * data["_score"] + 0.35 * data["_change"].clip(lower=0)
    data = data[data["_change"] > 0].sort_values(["_priority", "_change", "_score"], ascending=[False, False, False]).head(TOP_INDUSTRIES).reset_index(drop=True)
    if data.empty:
        st.info(f"No {title} group improved over the last five available trading sessions.")
        return
    st.markdown(f"### {title} Leadership Improvers")
    for rank, row in data.iterrows():
        name = clean_text(row[group_column])
        status, status_color, status_bg = leadership_status(row["_score"], row["_change"])
        st.markdown(
            f"<div class='improver-card' style='border-left-color:{status_color};'><div style='display:flex;justify-content:space-between;gap:12px;align-items:start;'><div style='min-width:0;'><div class='improver-name'>{rank + 1}. {name}</div><div class='improver-meta'><span class='status-pill' style='color:{status_color};background:{status_bg};'>{status}</span></div></div><div style='display:flex;gap:18px;flex-shrink:0;'><div class='improver-number' style='color:{score_color(row["_score"])};'>{format_number(row["_score"])}<div class='improver-meta'>Current score</div></div><div class='improver-number' style='color:{change_color(row["_change"])};'>{format_signed(row["_change"])}<div class='improver-meta'>5-session change</div></div></div></div></div>",
            unsafe_allow_html=True,
        )
        if st.button("↗ constituents + chart", key=f"open_{group_column}_{rank}_{name}", type="tertiary"):
            select_group_everywhere(group_column, name)


def render_leadership_table(frame: pd.DataFrame, group_column: str, title: str) -> None:
    score_column = get_score_column(frame)
    change_column = get_change_column(frame)
    if not score_column or group_column not in frame.columns:
        st.info(f"Prepared {title} snapshot does not contain leadership data.")
        return
    data = frame.copy()
    data["_score"] = pd.to_numeric(data[score_column], errors="coerce").fillna(0.0)
    data["_change"] = pd.to_numeric(data[change_column], errors="coerce").fillna(0.0) if change_column else 0.0
    data = data.sort_values(["_score", "_change"], ascending=[False, False]).reset_index(drop=True)
    table = pd.DataFrame({
        "Rank": range(1, len(data) + 1),
        title: data[group_column].map(clean_text),
        "Leadership Score": data["_score"].map(format_number),
        "5D Leadership Change": data["_change"].map(change_indicator),
        "Status": [leadership_status(score, change)[0] for score, change in zip(data["_score"], data["_change"])],
    })
    st.markdown(f"### {title} leadership table")
    show_table(table, max(320, min(720, 34 * len(table) + 60)))


def render_constituents(stock: pd.DataFrame, groups: pd.DataFrame, group_column: str, title: str) -> None:
    if group_column not in stock.columns or group_column not in groups.columns:
        st.info(f"No prepared {title} constituent data is available.")
        return
    change_column = get_change_column(groups)
    ordered = groups.copy()
    if change_column:
        ordered["_change"] = pd.to_numeric(ordered[change_column], errors="coerce").fillna(0.0)
        ordered = ordered.sort_values("_change", ascending=False)
    options = [clean_text(value) for value in ordered[group_column].dropna().unique()]
    if not options:
        return
    state_key = f"selected_{group_column}_constituents"
    selected = st.session_state.get(state_key)
    if selected not in options:
        selected = options[0]
    selected = st.selectbox(f"Select {title} for constituents", options, index=options.index(selected), key=f"{state_key}_widget")
    st.session_state[state_key] = selected
    st.session_state[f"selected_{group_column}_trend"] = selected

    data = stock[stock[group_column].map(clean_text) == selected].copy()
    if data.empty:
        st.info(f"No EOD constituent stock records are available for {selected} on this date.")
        return
    if "ret_20d" in data.columns:
        data = data.sort_values("ret_20d", ascending=False)
    data = data.head(30).reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    rename = {"symbol":"Symbol", "close":"Close", "ret_20d":"20D Return", "ret_60d":"60D Return", "gain_6m":"6M Gain", "stock_strength_score":"Strength"}
    view = data.rename(columns=rename)
    keep = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "established_buy_setup", "ipo_buy_setup"]
    view = view[[column for column in keep if column in view.columns]]
    for column in ["20D Return", "60D Return", "6M Gain"]:
        if column in view.columns:
            view[column] = view[column].map(format_percent)
    for column in ["Close", "Strength"]:
        if column in view.columns:
            view[column] = view[column].map(format_number)
    st.markdown(f"### {title} constituents")
    show_table(view, 420, chart_links=True)


def render_trend(groups: pd.DataFrame, selected_date: pd.Timestamp, group_kind: str, group_column: str, title: str) -> None:
    if group_column not in groups.columns:
        return
    change_column = get_change_column(groups)
    score_column = get_score_column(groups)
    ranked = groups.copy()
    if score_column:
        ranked["_score"] = pd.to_numeric(ranked[score_column], errors="coerce").fillna(0.0)
    else:
        ranked["_score"] = 0.0
    if change_column:
        ranked["_change"] = pd.to_numeric(ranked[change_column], errors="coerce").fillna(0.0)
    else:
        ranked["_change"] = 0.0
    ranked["_priority"] = pd.to_numeric(ranked.get("improver_priority", 0.65 * ranked["_score"] + 0.35 * ranked["_change"].clip(lower=0)), errors="coerce").fillna(0.0)
    options = [clean_text(value) for value in ranked.sort_values(["_priority", "_change"], ascending=[False, False])[group_column].dropna().unique()]
    if not options:
        return
    state_key = f"selected_{group_column}_trend"
    selected = st.session_state.get(state_key)
    if selected not in options:
        selected = options[0]
    selected = st.selectbox(f"Select {title} trend", options, index=options.index(selected), key=f"{state_key}_widget")
    st.session_state[state_key] = selected

    selected_row = ranked[ranked[group_column].map(clean_text) == selected]
    score = number(selected_row["_score"].iloc[-1]) if not selected_row.empty else 0.0
    change = number(selected_row["_change"].iloc[-1]) if not selected_row.empty else 0.0
    regime = clean_text(selected_row["regime"].iloc[-1]) if "regime" in selected_row.columns and not selected_row.empty else "Neutral Transition"
    c1, c2, c3 = st.columns(3)
    c1.metric("Current leadership", format_number(score))
    c2.metric("5-session change", format_signed(change))
    c3.metric("Current regime", regime)

    history = load_trend(group_kind, selected_date, selected)
    if history.empty or "leadership_score" not in history.columns:
        st.info(f"No prepared trend history is available for {selected}.")
        return
    history["_score"] = pd.to_numeric(history["leadership_score"], errors="coerce").fillna(0.0)
    figure = go.Figure(go.Scatter(
        x=history["date"], y=history["_score"], mode="lines+markers",
        line=dict(color=score_color(score), width=3), marker=dict(size=7),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Leadership score: %{y:.1f}<extra></extra>",
    ))
    for threshold, color in [(70, DARK_GREEN), (60, GREEN), (50, AMBER)]:
        figure.add_hline(y=threshold, line_dash="dot", line_color=color, opacity=.85)
    figure.update_layout(title=f"{selected}: Leadership Score trend", xaxis_title=None, yaxis_title="Leadership Score")
    figure.update_yaxes(range=[0, 100])
    st.plotly_chart(apply_chart_style(figure, 390), use_container_width=True)


def render_group_tab(groups: pd.DataFrame, stock: pd.DataFrame, selected_date: pd.Timestamp, group_column: str, title: str) -> None:
    render_improver_cards(groups, group_column, title)
    render_leadership_table(groups, group_column, title)
    render_constituents(stock, groups, group_column, title)
    st.markdown(f"### Selected {title.lower()} trend")
    render_trend(groups, selected_date, group_column, group_column, title)


def stock_metric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    for column in names:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(0.0, index=frame.index)


def render_setup_table(data: pd.DataFrame, title: str) -> None:
    st.markdown(f"### {title}")
    if data.empty:
        st.info(f"No stocks pass the prepared {title.lower()} screen on this date.")
        return
    frame = data.copy().head(TOP_STOCKS).reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + frame["symbol"].astype(str)
    frame["Tightness (3D)"] = stock_metric(frame, ["tight_3d_range", "tightness_3d", "range_3d_pct"])
    frame["Volume vs 50D"] = stock_metric(frame, ["vol_ratio_50", "volume_ratio_50", "vol_ratio", "volume_ratio"])
    frame["Prior Move"] = stock_metric(frame, ["gain_6m", "ret_60d", "ret_20d", "ret_120d"])
    rename = {"symbol":"Symbol", "basic_industry":"Basic Industry", "buy_priority_score":"Priority Score", "ipo_setup_score":"Priority Score"}
    view = frame.rename(columns=rename)
    if "Priority Score" not in view.columns:
        view["Priority Score"] = "—"
    keep = ["Rank", "Symbol", "Chart", "Basic Industry", "Priority Score", "Tightness (3D)", "Volume vs 50D", "Prior Move"]
    view = view[[column for column in keep if column in view.columns]]
    if "Priority Score" in view.columns:
        view["Priority Score"] = view["Priority Score"].map(format_number)
    for column in ["Tightness (3D)", "Volume vs 50D", "Prior Move"]:
        if column in view.columns:
            view[column] = view[column].map(format_percent)
    show_table(view, max(250, 38 * len(view) + 60), chart_links=True)


def top_setups_tab(selected_date: pd.Timestamp) -> None:
    try:
        established = load_selected_snapshot(selected_date, "top_buy_candidates.parquet")
        ipo = load_selected_snapshot(selected_date, "ipo_watchlist.parquet")
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Established qualified", format_integer(len(established)))
    c2.metric("IPO qualified", format_integer(len(ipo)))
    c3.metric("Scan date", selected_date.strftime("%d %b %Y"))
    st.caption("Tightness, Volume vs 50D and Prior Move are precomputed GitHub-side values, displayed as percentages with one decimal.")
    render_setup_table(established, "Top Established Setups")
    render_setup_table(ipo, "Top IPO Setups")


def methodology_tab() -> None:
    st.subheader("Methodology and data architecture")
    st.markdown(
        """
        ## Fast dashboard design
        GitHub Actions performs feature calculation, ranking, five-session change calculation, snapshot construction, classification updates and validation. Streamlit only reads the selected ready-made snapshot and displays it.

        ## Global date
        One compact Analysis Date control applies to every tab. The date list comes from `dashboard_dates.parquet`, which contains only dates for which GitHub Actions created prepared snapshots. If a calendar date is unavailable, the dashboard uses the most recent prior prepared EOD date.

        ## Classification hierarchy
        Sector, Industry and Basic Industry are sourced from the classified master. A hierarchy is complete only when all three are present. No dashboard logic guesses a missing classification. Records without verified mapping remain explicitly `Unclassified` and are tracked by the GitHub classification audit.

        ## Leadership
        Leadership Score is a 0–100 relative-strength-style measure computed in GitHub Actions. Five-session change is the current score minus the score five available trading sessions earlier. Improver Priority is 65% current leadership plus 35% positive five-session change.

        ## Navigation
        The `↗ constituents + chart` control selects the same group for both the constituent list and trend chart. The rerun is fast because only a small prepared date snapshot is read.

        ## Stock setups
        Established and IPO setup lists are produced by GitHub Actions. Tightness (3D), Volume vs 50D and Prior Move are precomputed. They are shown as percentages with one decimal; they require chart context and therefore do not use simplistic red/green flags.

        ## Limits
        This dashboard is a research tool, not a trade recommendation. Data availability depends on EOD source coverage, exchange holidays, corporate actions and classification-source publication timing.
        """
    )


def main() -> None:
    if not DATES_FILE.exists() or not SNAPSHOT_ROOT.exists():
        st.error("Prepared snapshot data is not available yet. Run the EOD GitHub Actions workflow once after deploying the new pipeline.")
        st.stop()

    try:
        dates = load_dates(str(DATES_FILE), DATES_FILE.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not load prepared dashboard dates: {exc}")
        st.stop()

    st.title("NSE Industry Momentum Monitor")
    st.caption(f"Last data refresh: {format_sync_time()}")
    selected_date = global_date_picker(dates)

    try:
        basic = load_selected_snapshot(selected_date, "basic_industry_snapshot.parquet")
        industry = load_selected_snapshot(selected_date, "industry_snapshot.parquet")
        stock = load_selected_snapshot(selected_date, "stock_snapshot.parquet")
    except Exception as exc:
        st.error(f"Prepared data for {selected_date.strftime('%d %b %Y')} could not be loaded: {exc}")
        st.stop()

    # Sector is stored in the stock snapshot. Sector aggregate output is also
    # available in dashboard_sector_history.parquet for trends. Until a
    # sector_snapshot is written, this simple current-date aggregation is only
    # display grouping, not a scoring calculation.
    sector = pd.DataFrame()
    sector_snapshot = snapshot_path(selected_date, "sector_snapshot.parquet")
    if sector_snapshot.exists():
        sector = load_snapshot(str(sector_snapshot), sector_snapshot.stat().st_mtime)

    tabs = st.tabs(["Industry Monitor", "Sector", "Industry", "Top Setups", "Methodology"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        change_column = get_change_column(basic)
        changes = pd.to_numeric(basic[change_column], errors="coerce").fillna(0.0) if change_column else pd.Series(dtype=float)
        c1.metric("Industries tracked", format_integer(len(basic)))
        c2.metric("Leadership improving", format_integer((changes > 0).sum()))
        c3.metric("Leadership weakening", format_integer((changes < 0).sum()))
        render_group_tab(basic, stock, selected_date, "basic_industry", "Basic Industry")
    with tabs[1]:
        if sector.empty:
            st.info("Prepared Sector snapshot is not available for this date yet. Run the upgraded EOD workflow after the snapshot builder has been updated to publish sector snapshots.")
        else:
            render_group_tab(sector, stock, selected_date, "sector", "Sector")
    with tabs[2]:
        render_group_tab(industry, stock, selected_date, "industry", "Industry")
    with tabs[3]:
        top_setups_tab(selected_date)
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
