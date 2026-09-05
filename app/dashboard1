# COMPLETE REPLACEMENT: app/dashboard.py
# Industry-momentum-first redesign.
# Focus: what is moving up, what is moving down, and why.
# No treemap, no scatter, no raw Excel-style mega-table, no fake zero setup chart.

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
TOP_N = 12
SMALL_GROUP_LIMIT = 5

INK = "#0F172A"
MUTED = "#64748B"
GREEN = "#15803D"
RED = "#B91C1C"
BLUE = "#1D4ED8"
AMBER = "#B45309"
PALETTE = {
    "Fresh Leader (HUNT)": GREEN,
    "Extended Leader (WAIT)": AMBER,
    "Speculative Coil (AVOID)": "#7C3AED",
    "Dead (AVOID)": RED,
    "Neutral Transition": MUTED,
}

st.set_page_config(page_title="NSE Industry Momentum Monitor", page_icon="â—ˆ", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {max-width:1380px;padding-top:1.2rem;padding-bottom:2.5rem}
h1 {font-size:2rem;font-weight:750;letter-spacing:-.035em;color:#0F172A}
h2,h3 {font-weight:700;color:#0F172A;letter-spacing:-.02em}
[data-testid="stMetricValue"] {font-size:1.55rem;font-weight:750;color:#0F172A}
[data-testid="stMetricLabel"] {font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.05em;font-weight:650}
[data-testid="stDataFrame"] {border:1px solid #E2E8F0;border-radius:10px;overflow:hidden}
.section-note {color:#64748B;font-size:.86rem;margin-top:-.35rem;margin-bottom:.8rem}
</style>
""", unsafe_allow_html=True)


def clean(value: object) -> str:
    if value is None or pd.isna(value): return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns: return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def score100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    return values * 100 if len(valid) and valid.max() <= 1.5 else values


def fmt(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value): return "â€”"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError): return "â€”"


def fmt_int(value: object) -> str:
    try:
        if value is None or pd.isna(value): return "â€”"
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError): return "â€”"


def fmt_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value): return "â€”"
        return f"{float(value) * 100:,.1f}%"
    except (TypeError, ValueError): return "â€”"


def chart_style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8,r=20,t=42,b=8),
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14, color=INK),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(showgrid=False),
    )
    return fig


@st.cache_data(show_spinner=False)
def load(path: str) -> pd.DataFrame:
    data = pd.read_parquet(path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    return data


def prep_groups(data: pd.DataFrame, key: str) -> pd.DataFrame:
    data = data.copy()
    if key not in data.columns: data[key] = "Unclassified"
    if "regime" not in data.columns: data["regime"] = "Neutral Transition"
    if "leadership_score" not in data.columns: data["leadership_score"] = 0.0
    if "actionability_score" not in data.columns: data["actionability_score"] = 0.0
    data[key] = data[key].map(clean)
    data["regime"] = data["regime"].map(clean)
    data["leadership_score"] = score100(data["leadership_score"]).fillna(0)
    data["actionability_score"] = score100(data["actionability_score"]).fillna(0)
    return data


def prep_stocks(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for col in ["symbol", "basic_industry", "industry", "sector"]:
        if col not in data.columns: data[col] = "Unclassified"
        data[col] = data[col].map(clean)
    for col in ["established_buy_setup", "ipo_buy_setup"]:
        if col not in data.columns: data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
    return data


def dates(data: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.Timestamp(x) for x in pd.to_datetime(data["date"].dropna().unique()) ) if "date" in data.columns else []


def picker(all_dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = all_dates[-1]
    state = f"{key}_date"
    if state not in st.session_state: st.session_state[state] = latest
    selected = pd.Timestamp(st.session_state[state])
    if selected not in all_dates: selected = latest; st.session_state[state] = latest
    idx = all_dates.index(selected)
    left, right = st.columns([5.8,2.2])
    with left: st.subheader("Analysis Date")
    with right:
        p, cal, n = st.columns([.4,1.6,.4])
        with p:
            if st.button("â€¹", key=f"{key}_p", disabled=idx == 0, use_container_width=True): st.session_state[state] = all_dates[idx-1]; st.rerun()
        with cal:
            chosen = st.date_input("Date", value=selected.date(), min_value=all_dates[0].date(), max_value=latest.date(), key=f"{key}_cal", label_visibility="collapsed")
        with n:
            if st.button("â€º", key=f"{key}_n", disabled=idx == len(all_dates)-1, use_container_width=True): st.session_state[state] = all_dates[idx+1]; st.rerun()
    valid = [d for d in all_dates if d <= pd.Timestamp(chosen)]
    result = valid[-1] if valid else all_dates[0]
    if result != selected: st.session_state[state] = result; st.rerun()
    st.caption(f"Data as of {result.strftime('%d %b %Y')}")
    return result


def show_table(data: pd.DataFrame, height: int = 400, progress: list[str] | None = None) -> None:
    data = data.copy()
    data.columns = [str(x) for x in data.columns]
    data = data.loc[:, ~data.columns.duplicated(keep="first")]
    config = {}
    for col in progress or []:
        if col in data.columns: config[col] = st.column_config.ProgressColumn(col, min_value=0, max_value=100, format="%.0f")
    st.dataframe(data, use_container_width=True, hide_index=True, height=height, column_config=config)


def industry_snapshot(history: pd.DataFrame, selected_date: pd.Timestamp, key: str) -> pd.DataFrame:
    current = history[history["date"] == selected_date].copy()
    current = current[num(current, "members") >= SMALL_GROUP_LIMIT].copy()
    current = current.drop_duplicates(key, keep="last")
    current["Leadership"] = score100(current["leadership_score"]).fillna(0)
    current["Actionability"] = score100(current["actionability_score"]).fillna(0)
    return current


def add_movement(history: pd.DataFrame, current: pd.DataFrame, key: str, lookback: int) -> pd.DataFrame:
    unique_dates = sorted(pd.to_datetime(history["date"].dropna().unique()))
    prior_dates = [d for d in unique_dates if d < current["date"].iloc[0]] if not current.empty else []
    prior_date = prior_dates[-lookback] if len(prior_dates) >= lookback else (prior_dates[0] if prior_dates else None)
    if prior_date is None:
        current["Prior Leadership"] = current["Leadership"]
        current["Leadership Change"] = 0.0
        current["Prior Actionability"] = current["Actionability"]
        current["Actionability Change"] = 0.0
        return current
    prior = history[history["date"] == prior_date].copy()
    prior = prior.drop_duplicates(key, keep="last")
    prior["Prior Leadership"] = score100(prior["leadership_score"]).fillna(0)
    prior["Prior Actionability"] = score100(prior["actionability_score"]).fillna(0)
    prior = prior[[key, "Prior Leadership", "Prior Actionability"]]
    current = current.merge(prior, on=key, how="left")
    current["Prior Leadership"] = current["Prior Leadership"].fillna(current["Leadership"])
    current["Prior Actionability"] = current["Prior Actionability"].fillna(current["Actionability"])
    current["Leadership Change"] = current["Leadership"] - current["Prior Leadership"]
    current["Actionability Change"] = current["Actionability"] - current["Prior Actionability"]
    return current


def movement_chart(data: pd.DataFrame, value_col: str, title: str, color: str, label_col: str) -> None:
    frame = data.sort_values(value_col).copy()
    fig = go.Figure(go.Bar(
        x=frame[value_col], y=frame[label_col], orientation="h",
        marker_color=[GREEN if x >= 0 else RED for x in frame[value_col]],
        text=frame[value_col].round(1), textposition="outside",
        hovertemplate="%{y}<br>Change: %{x:.1f} points<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Score change (points)", yaxis_title=None, shapes=[dict(type="line", x0=0,x1=0,y0=-.5,y1=len(frame)-.5,line=dict(color="#94A3B8",width=1))])
    st.plotly_chart(chart_style(fig, max(300, 56 + 28 * len(frame))), use_container_width=True)


def industry_monitor_tab(basic_history: pd.DataFrame, industry_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("Industry Momentum Monitor")
    st.markdown('<p class="section-note">The main question: which industries are improving, which are deteriorating, and where is setup density appearing?</p>', unsafe_allow_html=True)

    current = industry_snapshot(basic_history, selected_date, "basic_industry")
    if current.empty:
        st.warning("No Basic Industry data is available for this date.")
        return
    current["date"] = selected_date
    current = add_movement(basic_history, current, "basic_industry", 5)

    rising = current.nlargest(TOP_N, "Leadership Change").copy()
    falling = current.nsmallest(TOP_N, "Leadership Change").copy()
    setup_rising = current.nlargest(TOP_N, "Actionability").copy()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Industries Tracked", fmt_int(len(current)))
    c2.metric("Improving Leadership", fmt_int((current["Leadership Change"] > 0).sum()))
    c3.metric("Deteriorating Leadership", fmt_int((current["Leadership Change"] < 0).sum()))
    c4.metric("Industries With Setups", fmt_int((current["Actionability"] > 0).sum()))

    st.markdown("### Leadership Moving Up")
    st.markdown('<p class="section-note">Five-session change in Leadership Score. Positive values mean the industry is gaining relative strength.</p>', unsafe_allow_html=True)
    movement_chart(rising, "Leadership Change", "Industries with the strongest Leadership improvement", GREEN, "basic_industry")

    st.markdown("### Leadership Moving Down")
    st.markdown('<p class="section-note">The deterioration list is deliberately separate so weakening groups are visible instead of hidden inside an average.</p>', unsafe_allow_html=True)
    movement_chart(falling, "Leadership Change", "Industries with the largest Leadership deterioration", RED, "basic_industry")

    st.markdown("### Current Setup Density")
    st.markdown('<p class="section-note">This view uses the actual Actionability field. Industries with zero are removed from the chart; they remain visible in the table below.</p>', unsafe_allow_html=True)
    nonzero = current[current["Actionability"] > 0].nlargest(TOP_N, "Actionability").sort_values("Actionability")
    if nonzero.empty:
        st.info("The current Basic Industry file contains zero Actionability values for this date. The setup chart is hidden rather than showing a misleading wall of zeros.")
    else:
        fig = go.Figure(go.Bar(x=nonzero["Actionability"], y=nonzero["basic_industry"], orientation="h", marker_color=BLUE, text=nonzero["Actionability"].round(1), textposition="outside", hovertemplate="%{y}<br>Actionability: %{x:.1f}%<extra></extra>"))
        fig.update_layout(title="Industries with the highest current setup density", xaxis=dict(range=[0, max(nonzero["Actionability"].max()*1.2, 5)], title="Actionability (%)"), yaxis_title=None)
        st.plotly_chart(chart_style(fig, max(300, 56 + 28*len(nonzero))), use_container_width=True)

    st.markdown("### Industry Decision Board")
    board = current.sort_values("Leadership Change", ascending=False).copy()
    board.insert(0, "Rank", range(1, len(board)+1))
    board["Direction"] = board["Leadership Change"].map(lambda x: "â†‘ Improving" if x > .25 else ("â†“ Weakening" if x < -.25 else "â†’ Flat"))
    board = board.rename(columns={"basic_industry":"Basic Industry","Leadership":"Leadership Score","Actionability":"Actionability %","members":"Stocks","regime":"Regime","Leadership Change":"5D Leadership Î”","Actionability Change":"5D Actionability Î”"})
    cols=["Rank","Basic Industry","Direction","Leadership Score","5D Leadership Î”","Actionability %","5D Actionability Î”","Regime","Stocks"]
    board=board[[c for c in cols if c in board.columns]]
    for c in ["Leadership Score","5D Leadership Î”","Actionability %","5D Actionability Î”"]:
        if c in board.columns: board[c]=board[c].round(1)
    if "Stocks" in board.columns: board["Stocks"]=board["Stocks"].map(fmt_int)
    show_table(board, max(360, min(760, 35*len(board)+60)), progress=["Leadership Score","Actionability %"])


def stock_setups_tab(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("Top Individual Setups")
    st.markdown('<p class="section-note">Individual ranking only. Industry direction is displayed beside each stock; it does not block selection.</p>', unsafe_allow_html=True)
    basic = basic_history[basic_history["date"] == selected_date].drop_duplicates("basic_industry").copy()
    stocks = stock_history[stock_history["date"] == selected_date].copy()
    if stocks.empty: st.warning("No stock data available."); return
    lookup = basic.set_index("basic_industry")[["leadership_score","regime"]].rename(columns={"leadership_score":"Industry Score","regime":"Industry Regime"})
    buy = stocks[stocks["established_buy_setup"] == 1].copy()
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy()
    for frame, score_name in [(buy,"Priority Score"),(ipo,"Setup Score")]:
        if frame.empty: continue
        if score_name == "Priority Score":
            frame[score_name]=(0.30*frame.get("tight_3d_range",pd.Series(index=frame.index)).rank(pct=True,ascending=False).fillna(.5)+0.25*frame.get("vol_ratio_50",pd.Series(index=frame.index)).rank(pct=True,ascending=False).fillna(.5)+0.20*frame.get("gain_6m",pd.Series(index=frame.index)).rank(pct=True).fillna(.5)+0.15*frame.get("up_down_ratio",pd.Series(index=frame.index)).rank(pct=True).fillna(.5)+0.10*frame.get("stock_strength_score",pd.Series(index=frame.index)).rank(pct=True).fillna(.5))*100
        elif "ipo_setup_score" in frame.columns: frame[score_name]=pd.to_numeric(frame["ipo_setup_score"],errors="coerce").fillna(0)
        else: frame[score_name]=frame.get("tight_3d_range",pd.Series(index=frame.index)).rank(pct=True,ascending=False).fillna(.5)*100
        frame["Industry Score"]=frame["basic_industry"].map(lookup["Industry Score"])
        frame["Industry Regime"]=frame["basic_industry"].map(lookup["Industry Regime"])
    c1,c2,c3=st.columns(3); c1.metric("Established Qualified",fmt_int(len(buy))); c2.metric("IPO Qualified",fmt_int(len(ipo))); c3.metric("Date",selected_date.strftime("%d %b %Y"))
    for frame,title,score in [(buy,"Top 20 Established", "Priority Score"),(ipo,"Top 20 IPO", "Setup Score")]:
        st.markdown(f"### {title}")
        if frame.empty: st.info(f"No {title.lower()} pass the upstream setup gate."); continue
        view=frame.sort_values(score,ascending=False).head(20).copy().reset_index(drop=True); view.insert(0,"Rank",range(1,len(view)+1))
        view["Chart"]="https://in.tradingview.com/chart/?symbol=NSE:"+view["symbol"].astype(str)
        view=view.rename(columns={"symbol":"Symbol","basic_industry":"Basic Industry","close":"Close","Industry Score":"Industry Leadership","Industry Regime":"Industry Regime"})
        cols=["Rank","Symbol","Chart","Basic Industry","Industry Leadership","Industry Regime","Close",score,"gain_6m","nearest_ema_tag","momentum_badge"]
        view=view[[c for c in cols if c in view.columns]]
        for c in [score,"Industry Leadership","Close"]:
            if c in view.columns: view[c]=view[c].round(1)
        show_table(view, max(260,40*len(view)+50), progress=["Industry Leadership"])


def detail_tab(history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp, key: str, label: str) -> None:
    data = industry_snapshot(history, selected_date, key)
    if data.empty: st.warning(f"No {label} data available."); return
    data = add_movement(history, data.assign(date=selected_date), key, 5)
    data = data.sort_values("Leadership", ascending=False).reset_index(drop=True)
    data.insert(0,"Rank",range(1,len(data)+1))
    name_col = "Basic Industry" if key == "basic_industry" else "Industry"
    data = data.rename(columns={key:name_col,"Leadership":"Leadership Score","Actionability":"Actionability %","members":"Stocks","regime":"Regime","Leadership Change":"5D Leadership Î”"})
    cols=["Rank",name_col,"Leadership Score","5D Leadership Î”","Actionability %","Regime","Stocks"]
    data=data[[c for c in cols if c in data.columns]]
    for c in ["Leadership Score","5D Leadership Î”","Actionability %"]:
        if c in data.columns: data[c]=data[c].round(1)
    show_table(data, max(360,min(760,35*len(data)+60)), progress=["Leadership Score","Actionability %"])
    chosen=st.selectbox(name_col,data[name_col].tolist(),key=f"{key}_select")
    stock_col="basic_industry" if key=="basic_industry" else "industry"
    stocks=stock_history[(stock_history["date"]==selected_date)&(stock_history[stock_col]==chosen)].copy()
    if stocks.empty: st.info("No constituent records found."); return
    if "ret_20d" in stocks.columns: stocks=stocks.sort_values("ret_20d",ascending=False)
    stocks=stocks.head(30).reset_index(drop=True); stocks.insert(0,"Rank",range(1,len(stocks)+1)); stocks["Chart"]="https://in.tradingview.com/chart/?symbol=NSE:"+stocks["symbol"].astype(str)
    stocks=stocks.rename(columns={"symbol":"Symbol","close":"Close","ret_20d":"20D Return","ret_60d":"60D Return","stock_strength_score":"Strength"})
    cols=["Rank","Symbol","Chart","Close","20D Return","60D Return","Strength","established_buy_setup","ipo_buy_setup"]
    stocks=stocks[[c for c in cols if c in stocks.columns]]
    show_table(stocks,420,links=True)


def methodology() -> None:
    st.subheader("How to read this monitor")
    st.markdown("""
- **Leadership Score** is the current relative strength of an industry based on price velocity, breadth/EMA alignment, and institutional-volume behavior.
- **5D Leadership Î”** is the current score minus the score from the prior available session five sessions earlier. Positive means improving; negative means weakening.
- **Actionability %** is the percentage of tracked constituents currently passing the upstream stock-setup rules. It is not the same thing as Leadership Score.
- **Top Individual Setups** ranks stocks across all industries. Industry direction is context, not a filter.
- The monitor intentionally hides a setup-density chart when the processed file contains only zeros, because plotting 100 zero bars is not informative.

Use the workflow in this order: identify industries moving up, inspect whether Actionability is non-zero, then inspect individual stocks and finally validate the chart manually.
""")


def main() -> None:
    required=[BASIC_HISTORY_FILE,INDUSTRY_HISTORY_FILE,STOCK_HISTORY_FILE]
    missing=[str(x.relative_to(ROOT)) for x in required if not x.exists()]
    if missing: st.error("Required dashboard files are missing."); st.code("\n".join(missing)); st.stop()
    basic=prep_groups(load(str(BASIC_HISTORY_FILE)),"basic_industry")
    industry=prep_groups(load(str(INDUSTRY_HISTORY_FILE)),"industry")
    stocks=prep_stocks(load(str(STOCK_HISTORY_FILE)))
    all_dates=sorted(set(dates(basic)+dates(industry)+dates(stocks)))
    if not all_dates: st.error("No valid trading dates found."); st.stop()
    st.title("NSE Industry Momentum Monitor")
    sync=SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Prepared market data: {sync.replace('T',' ').replace('Z',' IST')}")
    tabs=st.tabs(["Industry Monitor","Top Setups","Basic Industry","Industry","Methodology"])
    with tabs[0]: industry_monitor_tab(basic,industry,date_picker(all_dates,"monitor"))
    with tabs[1]: stock_setups_tab(basic,stocks,date_picker(all_dates,"setups"))
    with tabs[2]: detail_tab(basic,stocks,date_picker(all_dates,"basic"),"basic_industry","Basic Industry")
    with tabs[3]: detail_tab(industry,stocks,date_picker(all_dates,"industry"),"industry","Industry")
    with tabs[4]: methodology()


if __name__ == "__main__": main()
