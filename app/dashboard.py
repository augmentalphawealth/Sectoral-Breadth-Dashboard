from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = PROCESSED / "dashboard_basic_industry_history.parquet"
INDUSTRY_HISTORY_FILE = PROCESSED / "dashboard_industry_history.parquet"
STOCK_HISTORY_FILE = PROCESSED / "dashboard_stock_history.parquet"
SYNC_FILE = PROCESSED / "last_sync.txt"
SMALL_GROUP_LIMIT = 5

st.set_page_config(
    page_title="NSE Sectoral Leadership Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1600px; padding-top: 1rem; padding-bottom: 2rem; background-color: #F8FAFC; }
    h1, h2, h3 { letter-spacing: -0.03em; color: #0F172A; }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; }
    [data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 6px; background-color: #FFFFFF; }
    div[data-baseweb="tab-list"] { gap: 24px; }
    div[data-baseweb="tab"] { padding: 8px 16px; font-weight: 600; }
    div[data-testid="stDateInput"] label { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value): return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def as_number(value: object) -> float | None:
    if value is None or pd.isna(value): return None
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def fmt_int(value: object) -> str:
    number = as_number(value)
    return "—" if number is None else f"{int(round(number)):,}"


def fmt_num(value: object, decimals: int = 2) -> str:
    number = as_number(value)
    return "—" if number is None else f"{number:,.{decimals}f}"


def fmt_pct(value: object, source_is_fraction: bool) -> str:
    number = as_number(value)
    if number is None: return "—"
    if source_is_fraction: number *= 100.0
    return f"{number:,.2f}%"


def cell_styler(val, metric_type):
    num = as_number(val)
    if num is None: return ""
    
    if metric_type == "range":
        if num <= 4.0: return "background-color: #10B981; color: #ffffff; font-weight: bold;"
        if num <= 6.0: return "background-color: #F59E0B; color: #ffffff; font-weight: bold;"
    elif metric_type == "vol":
        if num <= 0.5: return "background-color: #10B981; color: #ffffff; font-weight: bold;"
    elif metric_type == "score":
        if num >= 80: return "background-color: #1e3a8a; color: #ffffff; font-weight: bold;"
        if num >= 60: return "background-color: #3B82F6; color: #ffffff; font-weight: bold;"
    
    return "color: #475569;"


def style_execution_table(df: pd.DataFrame):
    return df.style.map(lambda x: cell_styler(x, "range"), subset=["3-Day Range %"]) \
                   .map(lambda x: cell_styler(x, "vol"), subset=["Vol vs 50D"]) \
                   .map(lambda x: cell_styler(x, "score"), subset=["Priority Score"])


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns: frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def ensure_group_columns(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns: data[group_column] = "Unclassified"
    if "regime" not in data.columns: data["regime"] = "Unclassified"
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    if "leadership_score" not in data.columns and "strength_score" in data.columns:
        data["leadership_score"] = data["strength_score"]
    if "actionability_score" not in data.columns: data["actionability_score"] = 0.0
    return data


def ensure_stock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["symbol", "industry", "basic_industry", "sector"]:
        if column not in data.columns: data[column] = "Unclassified"
        data[column] = data[column].map(clean_text)
    return data


def trading_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(frame["date"].dropna().unique())
    return sorted(pd.Timestamp(date) for date in dates)


def render_macro_hero(basic_latest: pd.DataFrame):
    st.markdown("### Market Regime Pulse")
    regime_counts = basic_latest["regime"].value_counts(normalize=True) * 100
    
    color_map = {
        "Fresh Leader (HUNT)": "#10B981", 
        "Extended Leader (WAIT)": "#F59E0B", 
        "Neutral Transition": "#94A3B8", 
        "Speculative Coil (AVOID)": "#64748B", 
        "Dead (AVOID)": "#EF4444"
    }
    
    fig = go.Figure()
    for regime, pct in regime_counts.items():
        fig.add_trace(go.Bar(
            y=["Market Regime"], x=[pct], name=regime, orientation='h',
            marker=dict(color=color_map.get(regime, "#CBD5E1")),
            text=f"{pct:.0f}%", textposition="inside", insidetextanchor="middle"
        ))
    
    fig.update_layout(
        barmode='stack', height=100, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_top_sectors_chart(basic_latest: pd.DataFrame):
    eligible = basic_latest[(basic_latest["members"] >= SMALL_GROUP_LIMIT) & (basic_latest["basic_industry"] != "Unclassified")]
    top15 = eligible.sort_values("leadership_score", ascending=True).tail(15)
    
    fig = px.bar(
        top15, x="leadership_score", y="basic_industry", orientation='h',
        title="Top 15 Sectors by Leadership Score",
        labels={"leadership_score": "Score", "basic_industry": ""},
        color="leadership_score", color_continuous_scale="Blues"
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def top_buy_setups_view(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    stock_latest = stock_history[stock_history["date"] == selected_date].copy()

    eligible_df = basic_latest[(basic_latest["members"] >= SMALL_GROUP_LIMIT) & (basic_latest["leadership_score"] >= 70.0)]
    top_leaders = eligible_df.sort_values("leadership_score", ascending=False)["basic_industry"].tolist()
    if not top_leaders:
        top_leaders = basic_latest[basic_latest["members"] >= SMALL_GROUP_LIMIT].sort_values("leadership_score", ascending=False).head(5)["basic_industry"].tolist()

    buy_candidates = stock_latest[(stock_latest["basic_industry"].isin(top_leaders)) & (stock_latest["established_buy_setup"] == 1)].copy()
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("### Actionable Setups Matrix")
        if not buy_candidates.empty:
            chart_df = buy_candidates.copy()
            chart_df["Bubble Size"] = 1.0 / (chart_df["vol_ratio_50"] + 0.1)
            chart_df["6M Gain %"] = chart_df["gain_6m"] * 100
            chart_df["3D Range %"] = chart_df["tight_3d_range"] * 100
            
            fig = px.scatter(
                chart_df, x="3D Range %", y="6M Gain %", size="Bubble Size", color="basic_industry", hover_name="symbol",
                hover_data={"3D Range %": ":.2f", "6M Gain %": ":.1f", "vol_ratio_50": ":.2f", "Bubble Size": False},
                labels={"3D Range %": "3-Day Squeeze % (Tighter →)", "6M Gain %": "6M Momentum Gain %"},
            )
            fig.update_layout(xaxis=dict(autorange="reversed"), height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candidates match strict liquidity and setup gates today.")
            
    with col2:
        render_top_sectors_chart(basic_latest)

    st.markdown("### ⚡ Execution Table: Curated Setups")
    if buy_candidates.empty: return

    sort_cols, sort_dirs = [], []
    for c in ["buy_priority_score", "tight_3d_range", "vol_ratio_50"]:
        if c in buy_candidates.columns:
            sort_cols.append(c)
            sort_dirs.append(False if c == "buy_priority_score" else True)
            
    if sort_cols: buy_candidates = buy_candidates.sort_values(sort_cols, ascending=sort_dirs)
    
    buy_candidates = buy_candidates.head(20).reset_index(drop=True)
    buy_candidates.insert(0, "Rank", range(1, len(buy_candidates) + 1))
    buy_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy_candidates["symbol"].astype(str)

    # Format explicitly for display
    display_df = buy_candidates[["Rank", "symbol", "basic_industry", "buy_priority_score", "tight_3d_range", "vol_ratio_50", "gain_6m", "Chart"]].copy()
    display_df.rename(columns={
        "symbol": "Symbol", "basic_industry": "Basic Industry", "buy_priority_score": "Priority Score",
        "tight_3d_range": "3-Day Range %", "vol_ratio_50": "Vol vs 50D", "gain_6m": "6M Gain %"
    }, inplace=True)

    display_df["3-Day Range %"] = display_df["3-Day Range %"] * 100
    display_df["6M Gain %"] = display_df["6M Gain %"] * 100

    st.dataframe(
        style_execution_table(display_df),
        use_container_width=True, hide_index=True,
        column_config={
            "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
            "Priority Score": st.column_config.NumberColumn(format="%.1f"),
            "3-Day Range %": st.column_config.NumberColumn(format="%.2f%%"),
            "Vol vs 50D": st.column_config.NumberColumn(format="%.2fx"),
            "6M Gain %": st.column_config.NumberColumn(format="+%.1f%%"),
        }
    )


def ipo_watchlist_view(stock_latest: pd.DataFrame, top_leaders: list):
    st.markdown("### 🚀 IPO Liquidity Watchlist")
    ipos = stock_latest[(stock_latest["basic_industry"].isin(top_leaders)) & (stock_latest["ipo_buy_setup"] == 1)].copy()
    
    if ipos.empty:
        st.info("No newly listed IPOs (≥ 5 Cr Turnover) in leading sectors show tight action today.")
        return
        
    ipos["Avg Turnover (Cr)"] = ipos["ipo_turnover_avg"] / 10000000.0
    ipos["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipos["symbol"].astype(str)
    ipos = ipos.sort_values("daily_range", ascending=True).reset_index(drop=True)
    ipos.insert(0, "Rank", range(1, len(ipos) + 1))

    display_df = ipos[["Rank", "symbol", "basic_industry", "daily_range", "Avg Turnover (Cr)", "Chart"]].copy()
    display_df.rename(columns={"symbol": "Symbol", "basic_industry": "Industry", "daily_range": "1-Day Range %"}, inplace=True)
    display_df["1-Day Range %"] = display_df["1-Day Range %"] * 100

    st.dataframe(
        display_df.style.map(lambda x: cell_styler(x, "range"), subset=["1-Day Range %"]),
        use_container_width=True, hide_index=True,
        column_config={
            "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
            "1-Day Range %": st.column_config.NumberColumn(format="%.2f%%"),
            "Avg Turnover (Cr)": st.column_config.NumberColumn(format="₹%.1f Cr"),
        }
    )


def main() -> None:
    required = [BASIC_HISTORY_FILE, INDUSTRY_HISTORY_FILE, STOCK_HISTORY_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Required dashboard files are missing. Run the data workflow first.")
        st.stop()

    basic_history = ensure_group_columns(load_parquet(str(BASIC_HISTORY_FILE)), "basic_industry")
    industry_history = ensure_group_columns(load_parquet(str(INDUSTRY_HISTORY_FILE)), "industry")
    stock_history = ensure_stock_columns(load_parquet(str(STOCK_HISTORY_FILE)))

    dates = trading_dates(stock_history)
    selected_date = pd.Timestamp(dates[-1])

    header_left, header_right = st.columns([7, 3])
    with header_left:
        st.title("NSE Sectoral Leadership Terminal")
        sync_text = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
        st.caption(f"Data as of {sync_text.replace('T', ' ').replace('Z', ' IST')}")
    with header_right:
        selected_date = st.date_input("Historical Scan Date", value=selected_date.date(), min_value=dates[0].date(), max_value=dates[-1].date())
        selected_date = pd.Timestamp(selected_date)
        if selected_date not in dates:
            valid_dates = [d for d in dates if d <= selected_date]
            selected_date = valid_dates[-1] if valid_dates else dates[0]

    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    stock_latest = stock_history[stock_history["date"] == selected_date].copy()

    render_macro_hero(basic_latest)

    top_leaders = basic_latest[(basic_latest["members"] >= SMALL_GROUP_LIMIT) & (basic_latest["leadership_score"] >= 70.0)]["basic_industry"].tolist()

    tabs = st.tabs(["🎯 Top Buy Setups", "🚀 IPO Watchlist", "📊 Sector Map & Methodology"])

    with tabs[0]:
        top_buy_setups_view(basic_history, stock_history, selected_date)
    with tabs[1]:
        ipo_watchlist_view(stock_latest, top_leaders)
    with tabs[2]:
        st.markdown("### 2-Axis System Methodology")
        st.markdown(
            """
            **Axis 1: Leadership Score (0–100, Determines Hunting Ground)**
            *   **35% Price Velocity:** Cross-sectional percentile rank of 20D and 60D equal-weighted returns.
            *   **35% Structural Alignment:** Percentage of constituents stacked cleanly (20 EMA > 50 EMA > 200 EMA).
            *   **30% Institutional Volume:** 50-Day Cumulative Up/Down Volume Ratio.
            
            **Axis 2: Micro Setup Density (The Buy Setups Tab)**
            Extracts Mainboard `EQ` stocks from leading industries that pass:
            1.  **Liquidity:** 20D Avg Turnover $\ge$ 5 Crore.
            2.  **Uptrend:** Price > 50 EMA > 200 EMA.
            3.  **Strike Zone:** Resting within -1% to +5% of 10/20/50 EMA.
            4.  **Prior Advance:** 6-Month Gain $\ge 30\%$.
            5.  **Volatility Squeeze:** 3-Day Range $\le 1.2 \times \text{ATR}_{14}$.
            6.  **Volume Dry-Up:** Today's volume $\le 0.5\times$ 50D Average.
            
            *IPOs (<150 days) bypass moving averages and are tracked via tight daily ranges and absolute 5 Cr Turnover.*
            """
        )

if __name__ == "__main__":
    main()
