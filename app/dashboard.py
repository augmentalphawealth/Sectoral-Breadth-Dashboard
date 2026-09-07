# app/dashboard.py
# Fast snapshot-only Streamlit dashboard.
# GitHub Actions prepares every calculation and every date snapshot. This app
# only reads the selected prepared snapshot and displays it.

from __future__ import annotations

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
MAX_CONSTITUENTS = 30

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
    .status-pill {display:inline-block; padding:.16rem .50rem; border-radius:999px; font-size:.74rem; font-weight:700; white-space:nowrap;}
    div.stButton > button[kind="secondary"] {text-align:left; justify-content:flex-start; white-space:normal; min-height:2.25rem;}
    @media (max-width:800px) {.block-container {padding-left:.7rem; padding-right:.7rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def to_number(value: object, default: float = 0.0) -> float:
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
    value = to_number(value)
    if value >= 70:
        return DARK_GREEN
    if value >= 60:
        return GREEN
    if value >= 50:
        return AMBER
    return RED


def leadership_status(score: object, change: object) -> tuple[str, str, str]:
    score_value = to_number(score)
    change_value = to_number(change)
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


def load_selected_snapshot(selected_date: pd.Timestamp, filename: str, required: bool = True) -> pd.DataFrame:
    path = snapshot_path(selected_date, filename)
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(f"Prepared snapshot is unavailable: {path.relative_to(ROOT)}")
        return pd.DataFrame()
    return load_snapshot(str(path), path.stat().st_mtime)


def get_score_column(frame: pd.DataFrame) -> str:
    for name in ["leadership_score", "strength_score", "strength"]:
        if name in frame.columns:
            return name
    return ""


def get_change_column(frame: pd.DataFrame) -> str:
    for name in ["leadership_change_5d", "leadership_score_change_5d", "strength_change_5d", "score_change_5d"]:
        if name in frame.columns:
            return name
    return ""


def get_count_column(frame: pd.DataFrame) -> str:
    for name in ["member_count", "constituent_count", "stock_count", "n_stocks"]:
        if name in frame.columns:
            return name
    return ""


def get_status_column(frame: pd.DataFrame) -> str:
    for name in ["status", "market_status", "leadership_status", "regime"]:
        if name in frame.columns:
            return name
    return ""


def load_trend(group_column: str, selected_date: pd.Timestamp, group_name: str) -> pd.DataFrame:
    path = PROCESSED / f"dashboard_{group_column}_history.parquet"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    history = load_snapshot(str(path), path.stat().st_mtime)
    if group_column not in history.columns or "date" not in history.columns:
        return pd.DataFrame()
    return history[(history[group_column].map(clean_text) == group_name) & (history["date"] <= selected_date)].copy().sort_values("date").tail(60)


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
    if state_key not in st.session_state:
        st.session_state[state_key] = dates[-1]
    selected = resolve_date(st.session_state[state_key], dates)
    st.session_state[state_key] = selected

    previous, calendar, next_button, label, _ = st.columns([0.28, 1.15, 0.28, 1.45, 3.84])
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
            key=f"global_analysis_date_calendar_{selected.strftime('%Y%m%d')}",
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


def show_table(data: pd.DataFrame, chart_links: bool = False) -> None:
    """Render only real rows. Never pass height=None to Streamlit."""
    view = data.copy()
    view.columns = [str(column) for column in view.columns]
    view = view.loc[:, ~view.columns.duplicated(keep="first")]
    if view.empty:
        st.info("No rows are available for this selection.")
        return
    config: dict[str, object] = {}
    if chart_links and "Chart" in view.columns:
        config["Chart"] = st.column_config.LinkColumn("Chart", display_text="Open ↗")
    st.dataframe(view, use_container_width=True, hide_index=True, column_config=config)


def group_metrics(groups: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if groups.empty or group_column not in groups.columns:
        return pd.DataFrame(columns=["Rank", "Group", "Leadership Score", "5D Leadership Change", "No. of Stocks", "Status", "_score", "_change"])
    score_column = get_score_column(groups)
    change_column = get_change_column(groups)
    count_column = get_count_column(groups)
    status_column = get_status_column(groups)
    data = groups.copy()
    data["_name"] = data[group_column].map(clean_text)
    data["_score"] = pd.to_numeric(data[score_column], errors="coerce").fillna(0.0) if score_column else 0.0
    data["_change"] = pd.to_numeric(data[change_column], errors="coerce").fillna(0.0) if change_column else 0.0
    data["_count"] = pd.to_numeric(data[count_column], errors="coerce") if count_column else pd.NA
    data = data.drop_duplicates("_name", keep="first").sort_values(["_score", "_change"], ascending=[False, False]).reset_index(drop=True)
    status = data[status_column].map(clean_text) if status_column else [leadership_status(score, change)[0] for score, change in zip(data["_score"], data["_change"])]
    return pd.DataFrame({
        "Rank": range(1, len(data) + 1),
        "Group": data["_name"],
        "Leadership Score": data["_score"].map(format_number),
        "5D Leadership Change": data["_change"].map(format_signed),
        "No. of Stocks": data["_count"].map(format_integer),
        "Status": status,
        "_score": data["_score"],
        "_change": data["_change"],
    })


def select_group(group_column: str, group_name: str) -> None:
    st.session_state[f"selected_{group_column}"] = group_name
    st.session_state["selection_notice"] = f"{group_column.replace('_', ' ').title()}: {group_name}"
    st.rerun()


def render_improver_cards(groups: pd.DataFrame, group_column: str, title: str) -> None:
    metrics = group_metrics(groups, group_column)
    if metrics.empty:
        return
    candidates = metrics[metrics["_change"] > 0].copy()
    candidates["_priority"] = 0.65 * candidates["_score"] + 0.35 * candidates["_change"].clip(lower=0)
    candidates = candidates.sort_values(["_priority", "_change", "_score"], ascending=[False, False, False]).head(TOP_INDUSTRIES)
    if candidates.empty:
        st.info(f"No {title} groups improved over the last five available trading sessions.")
        return
    st.markdown(f"### {title} leadership improvers")
    for _, row in candidates.iterrows():
        name = str(row["Group"])
        status, status_color, status_bg = leadership_status(row["_score"], row["_change"])
        name_column, values_column = st.columns([4, 1])
        with name_column:
            if st.button(name, key=f"improver_{group_column}_{name}", type="secondary", use_container_width=True):
                select_group(group_column, name)
            st.markdown(f"<span class='status-pill' style='color:{status_color};background:{status_bg};'>{status}</span>", unsafe_allow_html=True)
        with values_column:
            st.markdown(f"<div style='font-weight:800;color:{score_color(row['_score'])};'>{format_number(row['_score'])}</div><div style='font-size:.76rem;color:{MUTED};'>Score</div><div style='font-weight:800;color:{GREEN if row['_change'] > 0 else RED};margin-top:.25rem;'>{format_signed(row['_change'])}</div><div style='font-size:.76rem;color:{MUTED};'>5D change</div>", unsafe_allow_html=True)


def render_group_selector(metrics: pd.DataFrame, group_column: str, title: str) -> str | None:
    if metrics.empty:
        st.info(f"Prepared {title} snapshot does not contain leadership data.")
        return None
    st.markdown(f"### {title} leadership")
    st.caption("Leadership score, five-day change, constituent count and status. Click a group name below to update its constituents and chart in the same tab.")
    show_table(metrics[["Rank", "Leadership Score", "5D Leadership Change", "No. of Stocks", "Status"]])
    st.markdown("##### Open constituents + chart")
    for start in range(0, len(metrics), 3):
        row = metrics.iloc[start:start + 3]
        columns = st.columns(3)
        for column, (_, item) in zip(columns, row.iterrows()):
            with column:
                label = str(item["Group"])
                if st.button(label, key=f"group_{group_column}_{label}", type="secondary", use_container_width=True):
                    select_group(group_column, label)
    options = metrics["Group"].tolist()
    state_key = f"selected_{group_column}"
    selected = st.session_state.get(state_key)
    if selected not in options:
        selected = options[0]
        st.session_state[state_key] = selected
    return selected


def constituent_table(stock: pd.DataFrame, group_column: str, group_name: str) -> pd.DataFrame:
    if stock.empty or group_column not in stock.columns or "symbol" not in stock.columns:
        return pd.DataFrame()
    data = stock.copy()
    data["_group"] = data[group_column].map(clean_text)
    data["symbol"] = data["symbol"].fillna("").astype(str).str.strip()
    data = data[(data["_group"] == group_name) & (data["symbol"] != "")].copy()
    if data.empty:
        return pd.DataFrame()
    sort_column = next((name for name in ["buy_priority_score", "stock_strength_score", "strength_score", "strength", "ret_20d"] if name in data.columns), None)
    if sort_column:
        data["_sort"] = pd.to_numeric(data[sort_column], errors="coerce").fillna(0.0)
        data = data.sort_values("_sort", ascending=False)
    data = data.head(MAX_CONSTITUENTS).reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    rename = {"symbol": "Symbol", "close": "Close", "ret_20d": "20D Return", "ret_60d": "60D Return", "gain_6m": "6M Gain", "stock_strength_score": "Strength", "established_buy_setup": "Established setup", "ipo_buy_setup": "IPO setup"}
    view = data.rename(columns=rename)
    keep = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "Established setup", "IPO setup"]
    view = view[[column for column in keep if column in view.columns]]
    for column in ["20D Return", "60D Return", "6M Gain"]:
        if column in view.columns:
            view[column] = view[column].map(format_percent)
    for column in ["Close", "Strength"]:
        if column in view.columns:
            view[column] = view[column].map(format_number)
    return view


def render_trend(group_column: str, selected_date: pd.Timestamp, group_name: str) -> None:
    history = load_trend(group_column, selected_date, group_name)
    if history.empty:
        st.info(f"No prepared trend history is available for {group_name}.")
        return
    score_column = get_score_column(history)
    if not score_column:
        st.info(f"Prepared trend history for {group_name} does not contain a leadership score.")
        return
    history["_score"] = pd.to_numeric(history[score_column], errors="coerce").fillna(0.0)
    change_column = get_change_column(history)
    latest_score = history["_score"].iloc[-1]
    latest_change = pd.to_numeric(history[change_column], errors="coerce").fillna(0.0).iloc[-1] if change_column else 0.0
    c1, c2 = st.columns(2)
    c1.metric("Current leadership", format_number(latest_score))
    c2.metric("5-session change", format_signed(latest_change))
    figure = go.Figure(go.Scatter(
        x=history["date"], y=history["_score"], mode="lines+markers",
        line=dict(color=score_color(latest_score), width=3), marker=dict(size=6),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Leadership score: %{y:.1f}<extra></extra>",
    ))
    for threshold, color in [(70, DARK_GREEN), (60, GREEN), (50, AMBER)]:
        figure.add_hline(y=threshold, line_dash="dot", line_color=color, opacity=.8)
    figure.update_layout(title=f"{group_name}: leadership score trend", xaxis_title=None, yaxis_title="Leadership score")
    figure.update_yaxes(range=[0, 100])
    st.plotly_chart(apply_chart_style(figure, 365), use_container_width=True)


def render_group_tab(groups: pd.DataFrame, stock: pd.DataFrame, selected_date: pd.Timestamp, group_column: str, title: str) -> None:
    render_improver_cards(groups, group_column, title)
    st.divider()
    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        selected = render_group_selector(group_metrics(groups, group_column), group_column, title)
    with right:
        st.markdown(f"### {title} constituents and chart")
        if selected is None:
            return
        st.markdown(f"**Selected {title}:** {selected}")
        view = constituent_table(stock, group_column, selected)
        if view.empty:
            st.info(f"No usable stock constituent records are available for {selected} on {selected_date:%d %b %Y}.")
        else:
            st.caption(f"{len(view):,} constituent stock(s) shown for {selected} on {selected_date:%d %b %Y}.")
            show_table(view, chart_links=True)
        st.markdown("##### Leadership trend")
        render_trend(group_column, selected_date, selected)


def stock_metric(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    for column in names:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(0.0, index=frame.index)


def render_setup_table(data: pd.DataFrame, title: str) -> None:
    st.markdown(f"### {title}")
    if data.empty or "symbol" not in data.columns:
        st.info(f"No stocks pass the prepared {title.lower()} screen on this date.")
        return
    frame = data.copy()
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.strip()
    frame = frame[frame["symbol"] != ""].head(TOP_STOCKS).reset_index(drop=True)
    if frame.empty:
        st.info(f"No stocks pass the prepared {title.lower()} screen on this date.")
        return
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + frame["symbol"].astype(str)
    frame["Tightness (3D)"] = stock_metric(frame, ["tight_3d_range", "tightness_3d", "range_3d_pct"])
    frame["Volume vs 50D"] = stock_metric(frame, ["vol_ratio_50", "volume_ratio_50", "vol_ratio", "volume_ratio"])
    frame["Prior Move"] = stock_metric(frame, ["gain_6m", "ret_60d", "ret_20d", "ret_120d"])
    view = frame.rename(columns={"symbol": "Symbol", "basic_industry": "Basic Industry", "buy_priority_score": "Priority Score", "ipo_setup_score": "Priority Score"})
    if "Priority Score" not in view.columns:
        view["Priority Score"] = "—"
    keep = ["Rank", "Symbol", "Chart", "Basic Industry", "Priority Score", "Tightness (3D)", "Volume vs 50D", "Prior Move"]
    view = view[[column for column in keep if column in view.columns]]
    if "Priority Score" in view.columns:
        view["Priority Score"] = view["Priority Score"].map(format_number)
    for column in ["Tightness (3D)", "Volume vs 50D", "Prior Move"]:
        if column in view.columns:
            view[column] = view[column].map(format_percent)
    show_table(view, chart_links=True)


def top_setups_tab(selected_date: pd.Timestamp) -> None:
    established = load_selected_snapshot(selected_date, "top_buy_candidates.parquet", required=False)
    ipo = load_selected_snapshot(selected_date, "ipo_watchlist.parquet", required=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Established qualified", format_integer(len(established)))
    c2.metric("IPO qualified", format_integer(len(ipo)))
    c3.metric("Scan date", selected_date.strftime("%d %b %Y"))
    st.caption("All setup metrics are prepared by the EOD pipeline before the dashboard loads.")
    render_setup_table(established, "Top Established Setups")
    render_setup_table(ipo, "Top IPO Setups")


def methodology_tab() -> None:
    st.subheader("Methodology and data architecture")
    st.markdown(
        """
## Prepared EOD architecture
GitHub Actions prepares NSE price data, joins the verified classification master, calculates stock features, aggregates Basic Industry / Industry / Sector features, calculates leadership and five-session changes, then writes the date-specific snapshots. Streamlit reads these prepared files only.

## Stock calculations
For closing price \(P_t\), the prepared return measures are based on trading-session price changes:

- **20D Return:** \((P_t / P_{t-20}) - 1\), shown as a percentage.
- **60D Return:** \((P_t / P_{t-60}) - 1\), shown as a percentage.
- **6M Gain:** the precomputed medium-term price-performance field, using approximately six months of trading sessions.
- **Strength:** the EOD pipeline’s stock-level technical leadership score. Higher values indicate stronger prepared relative-strength / momentum conditions.
- **Established setup** and **IPO setup:** precomputed binary rules identifying stocks that satisfy the respective upstream setup filters on the selected EOD date.
- **Priority Score:** when present in the prepared setup list, this upstream ranking uses prepared inputs such as range tightness, volume ratio / volume expansion, prior move, up-down-volume behaviour, and strength. The dashboard does not recalculate or change the score.

## Basic Industry, Industry and Sector leadership
A classified stock contributes to its Basic Industry, Industry, and Sector. The EOD group-feature pipeline aggregates prepared constituent inputs to produce the group fields.

- **Leadership Score:** a prepared 0–100 composite reading of group-level constituent momentum, strength, breadth, and setup-related conditions. Groups rank from highest to lowest current score.
- **5D Leadership Change:** current leadership score minus the score five available trading sessions earlier. A positive value indicates improving leadership over that interval.
- **No. of Stocks:** the prepared classified constituent count used for the group-date calculation.
- **Status:** a supplied pipeline status when available; otherwise it is derived from score and five-day change. Scores of 70+ are strong, 60–69.9 are positive / building, 50–59.9 are transitional / watchlist, and below 50 are weak unless their five-day change is improving.

## Navigation
Each group name is a real Streamlit control. Clicking it immediately selects that group and updates the constituent stock table and leadership chart in the same tab. The stock list is filtered only from that selected date’s `stock_snapshot.parquet`, so constituents from different dates are never mixed.

## Limits
This is a technical breadth and research dashboard, not investment advice. Rankings may change as prices, volumes, listings, corporate actions, and classification coverage change. Review liquidity, disclosures, results, valuation, charts, and risk before acting.
        """
    )


def main() -> None:
    if not DATES_FILE.exists() or not SNAPSHOT_ROOT.exists():
        st.error("Prepared snapshot data is not available yet. Run the EOD GitHub Actions workflow after deploying the data pipeline.")
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
    sector = load_selected_snapshot(selected_date, "sector_snapshot.parquet", required=False)

    if st.session_state.get("selection_notice"):
        st.success(f"Showing constituents and chart for {st.session_state.pop('selection_notice')}.")

    tabs = st.tabs(["Basic Industry", "Top Setups", "Industry", "Sector", "Methodology"])
    with tabs[0]:
        changes = pd.to_numeric(basic[get_change_column(basic)], errors="coerce").fillna(0.0) if get_change_column(basic) else pd.Series(dtype=float)
        c1, c2, c3 = st.columns(3)
        c1.metric("Basic industries tracked", format_integer(len(basic)))
        c2.metric("Leadership improving", format_integer((changes > 0).sum()))
        c3.metric("Leadership weakening", format_integer((changes < 0).sum()))
        render_group_tab(basic, stock, selected_date, "basic_industry", "Basic Industry")
    with tabs[1]:
        top_setups_tab(selected_date)
    with tabs[2]:
        render_group_tab(industry, stock, selected_date, "industry", "Industry")
    with tabs[3]:
        if sector.empty:
            st.warning(
                f"Sector data has not been published for {selected_date:%d %b %Y}. "
                "Run the updated EOD workflow after sector snapshot generation is deployed, then refresh the app."
            )
        else:
            render_group_tab(sector, stock, selected_date, "sector", "Sector")
    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
