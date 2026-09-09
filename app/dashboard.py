# app/dashboard.py
# Fast snapshot-only Streamlit dashboard with intraday support and deep linking.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
    .header-row {font-weight:600; font-size:0.85rem; color:#64748B; border-bottom:1px solid #E2E8F0; padding-bottom:0.5rem; margin-bottom:0.5rem;}
    .table-row {font-size:0.9rem; padding:0.4rem 0; border-bottom:1px solid #F1F5F9;}
    @media (max-width:800px) {.block-container {padding-left:.7rem; padding-right:.7rem;}.improver-name {font-size:.85rem;}.improver-number {font-size:.89rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def handle_scroll():
    if "scroll_target" in st.session_state:
        target = st.session_state["scroll_target"]
        js = f"""
        <script>
            var el = window.parent.document.getElementById('{target}');
            if (el) {{ el.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}
        </script>
        """
        components.html(js, height=0)
        del st.session_state["scroll_target"]


def clean_text(value: object) -> str:
    if value is None or pd.isna(value): return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def number(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value): return default
        return float(value)
    except (TypeError, ValueError): return default


def format_number(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value): return "—"
    try: return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError): return "—"


def format_signed(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value): return "—"
    try: return f"{float(value):+,.{decimals}f}"
    except (TypeError, ValueError): return "—"


def format_integer(value: object) -> str:
    if value is None or pd.isna(value): return "—"
    try: return f"{int(round(float(value))):,}"
    except (TypeError, ValueError): return "—"


def format_percent(value: object) -> str:
    if value is None or pd.isna(value): return "—"
    try:
        raw = float(value)
        percent = raw * 100.0 if abs(raw) <= 1.5 else raw
        return f"{percent:,.1f}%"
    except (TypeError, ValueError): return "—"


def score_color(value: object) -> str:
    value = number(value)
    if value >= 70: return DARK_GREEN
    if value >= 60: return GREEN
    if value >= 50: return AMBER
    return RED


def change_color(value: object) -> str:
    value = number(value)
    if value > 0.05: return GREEN
    if value < -0.05: return RED
    return MUTED


def change_indicator(value: object) -> str:
    value = number(value)
    if value > 0.05: return f"🟢 {format_signed(value)}"
    if value < -0.05: return f"🔴 {format_signed(value)}"
    return f"⚪ {format_signed(value)}"


def leadership_status(score: object, change: object) -> tuple[str, str, str]:
    score_value = number(score)
    change_value = number(change)
    if score_value >= 70 and change_value > 0: return "Strong leader · Accelerating", DARK_GREEN, LIGHT_GREEN
    if score_value >= 70: return "Strong leadership", DARK_GREEN, LIGHT_GREEN
    if score_value >= 60 and change_value > 0: return "Building leadership", GREEN, LIGHT_GREEN
    if score_value >= 60: return "Positive transition", GREEN, LIGHT_GREEN
    if score_value >= 50 and change_value > 0: return "Improving · Watchlist", AMBER, LIGHT_AMBER
    if score_value >= 50: return "Neutral transition", AMBER, LIGHT_AMBER
    if change_value > 0: return "Improving · Not confirmed", AMBER, LIGHT_AMBER
    return "Weak leadership", RED, LIGHT_RED


def apply_chart_style(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(
        height=height, margin=dict(l=8, r=25, t=45, b=20),
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=INK),
        title_font=dict(size=14, color=INK), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    figure.update_yaxes(showgrid=False)
    return figure


@st.cache_data(show_spinner=False)
def load_dates(path: str, modified: float) -> list[pd.Timestamp]:
    frame = pd.read_parquet(path)
    return sorted(pd.Timestamp(value).normalize() for value in pd.to_datetime(frame["date"], errors="coerce").dropna().unique())


@st.cache_data(show_spinner=False)
def load_snapshot(path: str, modified: float) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns: frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame


def snapshot_path(selected_date: pd.Timestamp, filename: str) -> Path:
    return SNAPSHOT_ROOT / selected_date.strftime("%Y-%m-%d") / filename


def load_selected_snapshot(selected_date: pd.Timestamp, filename: str) -> pd.DataFrame:
    path = snapshot_path(selected_date, filename)
    if not path.exists(): raise FileNotFoundError(f"Prepared snapshot is unavailable: {path.relative_to(ROOT)}")
    return load_snapshot(str(path), path.stat().st_mtime)


def load_trend(group_kind: str, selected_date: pd.Timestamp, group_name: str) -> pd.DataFrame:
    filename = f"{group_kind}_history.parquet"
    path = PROCESSED / f"dashboard_{filename}"
    if not path.exists(): return pd.DataFrame()
    history = load_snapshot(str(path), path.stat().st_mtime)
    if group_kind not in history.columns or "date" not in history.columns: return pd.DataFrame()
    result = history[(history[group_kind].map(clean_text) == group_name) & (history["date"] <= selected_date)].copy()
    return result.sort_values("date").tail(30)


def resolve_date(requested: object, dates: list[pd.Timestamp]) -> pd.Timestamp:
    requested_date = pd.Timestamp(requested).normalize()
    if requested_date in dates: return requested_date
    earlier = [date for date in dates if date <= requested_date]
    return earlier[-1] if earlier else dates[0]


def global_date_picker(dates: list[pd.Timestamp]) -> pd.Timestamp:
    state_key = "global_analysis_date"
    if state_key not in st.session_state: st.session_state[state_key] = dates[-1]
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
            "Analysis date", value=selected.date(), min_value=dates[0].date(), max_value=dates[-1].date(),
            label_visibility="collapsed", format="DD/MM/YYYY"
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
    return resolved


def show_table(data: pd.DataFrame, height: int, chart_links: bool = False) -> None:
    view = data.copy()
    view.columns = [str(column) for column in view.columns]
    # Safety feature: Prevent PyArrow duplicate column crashes natively
    view = view.loc[:, ~view.columns.duplicated(keep="first")]
    config = {"Chart": st.column_config.LinkColumn("Chart", display_text="Open ↗")} if chart_links and "Chart" in view.columns else {}
    st.dataframe(view, use_container_width=True, hide_index=True, height=height, column_config=config)


def select_group_everywhere(group_kind: str, group_name: str) -> None:
    st.session_state[f"selected_{group_kind}_constituents"] = group_name
    st.session_state[f"selected_{group_kind}_trend"] = group_name
    st.session_state["scroll_target"] = f"constituents_anchor_{group_kind}"
    st.rerun()


def render_improver_cards(frame: pd.DataFrame, group_column: str, title: str) -> None:
    score_column = "leadership_score" if "leadership_score" in frame.columns else "strength_score"
    if score_column not in frame.columns: return
    
    data = frame.copy()
    data["members"] = pd.to_numeric(data.get("members", 0), errors="coerce").fillna(0).astype(int)
    # Cards only focus on groups with >= 5 members to prevent noise from micro-industries
    data = data[data["members"] >= 5]
    
    data["_score"] = pd.to_numeric(data[score_column], errors="coerce").fillna(0.0)
    data["_change"] = pd.to_numeric(data.get("leadership_change_5d", 0), errors="coerce").fillna(0.0)
    data["_priority"] = pd.to_numeric(data.get("improver_priority", 0.65 * data["_score"] + 0.35 * data["_change"].clip(lower=0)), errors="coerce").fillna(0.0)
    data = data[data["_change"] > 0].sort_values(["_priority", "_change"], ascending=[False, False]).head(TOP_INDUSTRIES).reset_index(drop=True)
    
    if data.empty: return
    st.markdown(f"### {title} Leadership Improvers")
    for rank, row in data.iterrows():
        name = clean_text(row[group_column])
        members = format_integer(row["members"])
        status, status_color, status_bg = leadership_status(row["_score"], row["_change"])
        st.markdown(
            f"<div class='improver-card' style='border-left-color:{status_color};'><div style='display:flex;justify-content:space-between;gap:12px;align-items:start;'><div style='min-width:0;'><div class='improver-name'>{rank + 1}. {name}</div><div class='improver-meta'><span class='status-pill' style='color:{status_color};background:{status_bg};'>{status}</span> | {members} Members</div></div><div style='display:flex;gap:18px;flex-shrink:0;'><div class='improver-number' style='color:{score_color(row["_score"])};'>{format_number(row["_score"])}<div class='improver-meta'>Current score</div></div><div class='improver-number' style='color:{change_color(row["_change"])};'>{format_signed(row["_change"])}<div class='improver-meta'>5-session change</div></div></div></div></div>",
            unsafe_allow_html=True,
        )
        if st.button("↗ Jump to Constituents + Chart", key=f"open_imp_{group_column}_{rank}_{name}", type="tertiary"):
            select_group_everywhere(group_column, name)


def render_leadership_table(frame: pd.DataFrame, group_column: str, title: str) -> None:
    score_column = "leadership_score" if "leadership_score" in frame.columns else "strength_score"
    if score_column not in frame.columns: return
    
    data = frame.copy()
    data["_score"] = pd.to_numeric(data[score_column], errors="coerce").fillna(0.0)
    data["_change"] = pd.to_numeric(data.get("leadership_change_5d", 0), errors="coerce").fillna(0.0)
    data["members"] = pd.to_numeric(data.get("members", 0), errors="coerce").fillna(0).astype(int)
    
    # Split the dataset: >= 5 members vs < 5 members
    main_data = data[data["members"] >= 5].sort_values(["_score", "_change"], ascending=[False, False]).reset_index(drop=True)
    small_data = data[data["members"] < 5].sort_values(["_score", "_change"], ascending=[False, False]).reset_index(drop=True)

    def build_table(df: pd.DataFrame, table_title: str, is_main: bool):
        st.markdown(f"### {table_title}")
        if df.empty:
            st.info(f"No records found for {table_title}.")
            return
            
        with st.container(height=max(120, min(550, 42 * len(df) + 60))):
            hcols = st.columns([0.6, 2.5, 1, 1.2, 1.2, 2.0, 1.5])
            hcols[0].markdown("<div class='header-row'>Rank</div>", unsafe_allow_html=True)
            hcols[1].markdown(f"<div class='header-row'>{title}</div>", unsafe_allow_html=True)
            hcols[2].markdown("<div class='header-row'>Members</div>", unsafe_allow_html=True)
            hcols[3].markdown("<div class='header-row'>Score</div>", unsafe_allow_html=True)
            hcols[4].markdown("<div class='header-row'>5D Chg</div>", unsafe_allow_html=True)
            hcols[5].markdown("<div class='header-row'>Status</div>", unsafe_allow_html=True)
            hcols[6].markdown("<div class='header-row'>Action</div>", unsafe_allow_html=True)

            for idx, row in df.iterrows():
                cols = st.columns([0.6, 2.5, 1, 1.2, 1.2, 2.0, 1.5])
                cols[0].markdown(f"<div class='table-row'>{idx + 1}</div>", unsafe_allow_html=True)
                cols[1].markdown(f"<div class='table-row'><b>{clean_text(row[group_column])}</b></div>", unsafe_allow_html=True)
                cols[2].markdown(f"<div class='table-row'>{format_integer(row['members'])}</div>", unsafe_allow_html=True)
                cols[3].markdown(f"<div class='table-row' style='color:{score_color(row['_score'])}'><b>{format_number(row['_score'])}</b></div>", unsafe_allow_html=True)
                cols[4].markdown(f"<div class='table-row'>{change_indicator(row['_change'])}</div>", unsafe_allow_html=True)
                cols[5].markdown(f"<div class='table-row' style='font-size:0.8rem; color:{MUTED};'>{leadership_status(row['_score'], row['_change'])[0]}</div>", unsafe_allow_html=True)
                key_prefix = "main" if is_main else "small"
                if cols[6].button("↗ Constituents", key=f"tbl_btn_{group_column}_{key_prefix}_{idx}", type="tertiary"):
                    select_group_everywhere(group_column, row[group_column])

    # Render Main Table
    build_table(main_data, f"{title} Leadership", is_main=True)
    
    # Render Small Table (only if data exists)
    if not small_data.empty:
        build_table(small_data, f"Small {title} (< 5 Stocks)", is_main=False)


def render_constituents(stock: pd.DataFrame, groups: pd.DataFrame, group_column: str, title: str) -> None:
    st.markdown(f"<div id='constituents_anchor_{group_column}' style='padding-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"### {title} Constituents")
    if group_column not in stock.columns or group_column not in groups.columns: return

    # Dropdown includes ALL industries (both large and small)
    ordered = groups.copy()
    ordered["_change"] = pd.to_numeric(ordered.get("leadership_change_5d", 0), errors="coerce").fillna(0.0)
    options = [clean_text(value) for value in ordered.sort_values("_change", ascending=False)[group_column].dropna().unique()]
    
    state_key = f"selected_{group_column}_constituents"
    selected = st.session_state.get(state_key, options[0] if options else "Unclassified")
    selected = st.selectbox(f"Select {title}", options, index=options.index(selected) if selected in options else 0, key=f"{state_key}_widget")
    st.session_state[state_key] = selected
    st.session_state[f"selected_{group_column}_trend"] = selected

    data = stock[stock[group_column].map(clean_text) == selected].copy()
    if data.empty: return

    data = data.sort_values("ret_20d", ascending=False).head(30).reset_index(drop=True)
    data.insert(0, "Rank", range(1, len(data) + 1))
    data["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + data["symbol"].astype(str)
    
    view = data.rename(columns={"symbol":"Symbol", "close":"Close", "ret_20d":"20D Return", "ret_60d":"60D Return", "gain_6m":"6M Gain", "stock_strength_score":"Strength"})
    keep = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength", "established_buy_setup"]
    view = view[[c for c in keep if c in view.columns]]
    for c in ["20D Return", "60D Return", "6M Gain"]:
        if c in view.columns: view[c] = view[c].map(format_percent)
    for c in ["Close", "Strength"]:
        if c in view.columns: view[c] = view[c].map(format_number)
    show_table(view, 420, chart_links=True)


def render_trend(groups: pd.DataFrame, selected_date: pd.Timestamp, group_kind: str, group_column: str, title: str) -> None:
    st.markdown(f"### Selected {title.lower()} trend")
    if group_column not in groups.columns: return
    
    ranked = groups.copy()
    ranked["_score"] = pd.to_numeric(ranked.get("leadership_score", 0), errors="coerce").fillna(0.0)
    ranked["_change"] = pd.to_numeric(ranked.get("leadership_change_5d", 0), errors="coerce").fillna(0.0)
    ranked["_priority"] = pd.to_numeric(ranked.get("improver_priority", 0.65 * ranked["_score"] + 0.35 * ranked["_change"].clip(lower=0)), errors="coerce").fillna(0.0)
    
    options = [clean_text(v) for v in ranked.sort_values(["_priority", "_change"], ascending=[False, False])[group_column].dropna().unique()]
    state_key = f"selected_{group_column}_trend"
    selected = st.session_state.get(state_key, options[0] if options else "Unclassified")
    selected = st.selectbox(f"View trend for {title}", options, index=options.index(selected) if selected in options else 0, key=f"{state_key}_widget")
    
    selected_row = ranked[ranked[group_column].map(clean_text) == selected]
    score = number(selected_row["_score"].iloc[-1]) if not selected_row.empty else 0.0
    change = number(selected_row["_change"].iloc[-1]) if not selected_row.empty else 0.0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current leadership", format_number(score))
    c2.metric("5-session change", format_signed(change))
    c3.metric("Current regime", clean_text(selected_row["regime"].iloc[-1]) if "regime" in selected_row.columns and not selected_row.empty else "Neutral")

    history = load_trend(group_kind, selected_date, selected)
    if history.empty: return
    history["_score"] = pd.to_numeric(history.get("leadership_score", 0), errors="coerce").fillna(0.0)
    
    fig = go.Figure(go.Scatter(x=history["date"], y=history["_score"], mode="lines+markers", line=dict(color=score_color(score), width=3), marker=dict(size=7)))
    for y, c in [(70, DARK_GREEN), (60, GREEN), (50, AMBER)]: fig.add_hline(y=y, line_dash="dot", line_color=c, opacity=.85)
    st.plotly_chart(apply_chart_style(fig.update_layout(title=f"{selected} Trend", yaxis_range=[0, 100]), 390), use_container_width=True)


def render_group_tab(groups: pd.DataFrame, stock: pd.DataFrame, selected_date: pd.Timestamp, group_column: str, title: str) -> None:
    render_improver_cards(groups, group_column, title)
    render_leadership_table(groups, group_column, title)
    render_constituents(stock, groups, group_column, title)
    render_trend(groups, selected_date, group_column, group_column, title)


def top_setups_tab(selected_date: pd.Timestamp, basic: pd.DataFrame) -> None:
    try:
        established = load_selected_snapshot(selected_date, "top_buy_candidates.parquet")
        ipo = load_selected_snapshot(selected_date, "ipo_watchlist.parquet")
    except FileNotFoundError as e:
        st.error(str(e))
        return

    if not basic.empty and "basic_industry" in basic.columns and "members" in basic.columns:
        member_map = basic.set_index("basic_industry")["members"].to_dict()
        established["Ind. Members"] = established.get("basic_industry", pd.Series()).map(lambda x: member_map.get(x, 0))
        ipo["Ind. Members"] = ipo.get("basic_industry", pd.Series()).map(lambda x: member_map.get(x, 0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Established qualified", format_integer(len(established)))
    c2.metric("IPO qualified", format_integer(len(ipo)))
    c3.metric("Scan date", selected_date.strftime("%d %b %Y"))

    def render_setup(data: pd.DataFrame, title: str):
        st.markdown(f"### {title}")
        if data.empty: return
        df = data.copy().head(TOP_STOCKS).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        df["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + df["symbol"].astype(str)
        df["Tightness (3D)"] = df.get("tight_3d_range", 0)
        df["Volume vs 50D"] = df.get("vol_ratio_50", 0)
        df["Prior Move"] = df.get("gain_6m", 0)
        
        # Safely extract the priority score to prevent PyArrow duplication crashes
        if "IPO" in title:
            df["Priority Score"] = df.get("ipo_setup_score", "—")
        else:
            df["Priority Score"] = df.get("buy_priority_score", "—")
            
        view = df.rename(columns={"symbol":"Symbol", "basic_industry":"Basic Industry"})
        keep = ["Rank", "Symbol", "Chart", "Basic Industry", "Ind. Members", "Priority Score", "Tightness (3D)", "Volume vs 50D", "Prior Move"]
        view = view[[c for c in keep if c in view.columns]]
        
        for c in ["Tightness (3D)", "Volume vs 50D", "Prior Move"]:
            if c in view.columns: view[c] = view[c].map(format_percent)
        show_table(view, max(250, 38 * len(view) + 60), chart_links=True)

    render_setup(established, "Top Established Setups")
    render_setup(ipo, "Top IPO Setups")


def render_intraday_tab():
    intraday_file = PROCESSED / "intraday_sector_movers.parquet"
    stocks_file = PROCESSED / "intraday_top_stocks.parquet"
    
    st.markdown("### Intraday Sector Movers")
    if intraday_file.exists():
        df = pd.read_parquet(intraday_file)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Intraday sector data is not available. The intraday workflow runs every 30 minutes during market hours.")
        
    st.markdown("### Intraday Top Stocks")
    if stocks_file.exists():
        stdf = pd.read_parquet(stocks_file)
        st.dataframe(stdf, use_container_width=True, hide_index=True)
    else:
        st.info("Intraday stocks data is not available.")


def main() -> None:
    handle_scroll()
    if not DATES_FILE.exists() or not SNAPSHOT_ROOT.exists():
        st.error("Prepared snapshot data is not available yet. Run the EOD GitHub Actions workflow.")
        st.stop()

    dates = load_dates(str(DATES_FILE), DATES_FILE.stat().st_mtime)
    st.title("NSE Industry Momentum Monitor")
    selected_date = global_date_picker(dates)

    basic = load_selected_snapshot(selected_date, "basic_industry_snapshot.parquet")
    industry = load_selected_snapshot(selected_date, "industry_snapshot.parquet")
    stock = load_selected_snapshot(selected_date, "stock_snapshot.parquet")
    sector = load_snapshot(str(snapshot_path(selected_date, "sector_snapshot.parquet")), 0) if snapshot_path(selected_date, "sector_snapshot.parquet").exists() else pd.DataFrame()

    tabs = st.tabs(["Industry Monitor", "Sector", "Industry", "Top Setups", "Intraday (Live)", "Methodology"])
    with tabs[0]: render_group_tab(basic, stock, selected_date, "basic_industry", "Basic Industry")
    with tabs[1]: render_group_tab(sector, stock, selected_date, "sector", "Sector") if not sector.empty else st.info("Sector snapshot missing.")
    with tabs[2]: render_group_tab(industry, stock, selected_date, "industry", "Industry")
    with tabs[3]: top_setups_tab(selected_date, basic)
    with tabs[4]: render_intraday_tab()
    with tabs[5]: st.write("GitHub Actions builds everything. Streamlit only reads and displays.")


if __name__ == "__main__":
    main()
