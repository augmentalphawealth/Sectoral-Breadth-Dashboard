from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

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
    /* Institutional Slate Theme */
    .stApp { background-color: #F8FAFC; }
    .block-container { max-width: 1600px; padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: -0.03em; color: #0F172A; font-family: 'Inter', sans-serif; }
    
    /* Clean Metric Cards */
    [data-testid="stMetric"] { 
        background: #FFFFFF; 
        border: 1px solid #E2E8F0; 
        padding: 1rem; 
        border-radius: 8px; 
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Streamlit Dataframe Overrides */
    [data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 8px; overflow: hidden; }
    
    /* Time Travel Slider Fixes */
    div[data-testid="stDateInput"] label { display: none; }
    div[data-testid="stButton"] > button { min-height: 34px; border-radius: 6px; }
    
    /* Tab Styling */
    div[data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #E2E8F0; }
    div[data-baseweb="tab"] { padding: 12px 16px; font-weight: 600; color: #64748B; }
    div[data-baseweb="tab"][aria-selected="true"] { color: #0F172A; border-bottom-color: #3B82F6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# DATA LOADING & SAFETY HANDLERS
# =====================================================================

def clean_text(value: object) -> str:
    if value is None or pd.isna(value): return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"

@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns: 
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame

def ensure_group_columns(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns: data[group_column] = "Unclassified"
    if "regime" not in data.columns: data["regime"] = "Unclassified"
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    if "leadership_score" not in data.columns and "strength_score" in data.columns:
        data["leadership_score"] = data["strength_score"]
    if "actionability_score" not in data.columns: 
        data["actionability_score"] = 0.0
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

def section_header(title: str, dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = pd.Timestamp(dates[-1])
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state: st.session_state[state_key] = latest
    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in dates:
        selected = latest
        st.session_state[state_key] = latest
    index = dates.index(selected)

    heading, controls = st.columns([5.5, 2.5])
    with heading:
        st.subheader(title)
    with controls:
        previous, calendar, next_button = st.columns([0.4, 1.55, 0.4])
        with previous:
            if st.button("‹", key=f"{key}_previous", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = dates[index - 1]
                st.rerun()
        with calendar:
            chosen = st.date_input("Date", value=selected.date(), min_value=pd.Timestamp(dates[0]).date(), max_value=latest.date(), key=f"{key}_calendar")
        with next_button:
            if st.button("›", key=f"{key}_next", disabled=index == len(dates) - 1, use_container_width=True):
                st.session_state[state_key] = dates[index + 1]
                st.rerun()

    requested = pd.Timestamp(chosen)
    valid_dates = [date for date in dates if date <= requested]
    resolved = valid_dates[-1] if valid_dates else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"Historical Scan As Of: **{resolved.strftime('%d %b %Y')}**")
    return resolved


# =====================================================================
# UI COMPONENTS & CHARTS
# =====================================================================

def render_setup_matrix(buy_candidates: pd.DataFrame):
    """Interactive Scatter Plot for visually hunting setups."""
    if buy_candidates.empty: return
    
    chart_df = buy_candidates.copy()
    # Safely generate columns for charting
    chart_df["3D Range %"] = chart_df.get("tight_3d_range", pd.Series([0]*len(chart_df))) * 100
    chart_df["6M Gain %"] = chart_df.get("gain_6m", pd.Series([0]*len(chart_df))) * 100
    
    # Inverse volume ratio for bubble size (drier volume = bigger bubble)
    vol_ratio = chart_df.get("vol_ratio_50", pd.Series([1]*len(chart_df)))
    chart_df["Dryness Bubble"] = 1.0 / (vol_ratio.clip(lower=0.1) + 0.1)
    
    fig = px.scatter(
        chart_df, x="3D Range %", y="6M Gain %", size="Dryness Bubble", 
        color="basic_industry", hover_name="symbol",
        hover_data={"3D Range %": ":.2f", "6M Gain %": ":.1f", "Dryness Bubble": False},
        labels={"3D Range %": "3-Day Squeeze % (Tighter →)", "6M Gain %": "6M Gain % (Stronger ↑)"},
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    
    fig.update_layout(
        xaxis=dict(autorange="reversed", showgrid=True, gridcolor="#E2E8F0"), 
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        height=400, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_sector_bar_chart(basic_latest: pd.DataFrame):
    """Clean Top 15 Sector Bar chart."""
    eligible = basic_latest[(basic_latest.get("members", 0) >= SMALL_GROUP_LIMIT) & (basic_latest["basic_industry"] != "Unclassified")]
    if eligible.empty: return
    
    top15 = eligible.sort_values("leadership_score", ascending=True).tail(15)
    
    fig = px.bar(
        top15, x="leadership_score", y="basic_industry", orientation='h',
        labels={"leadership_score": "Leadership Score", "basic_industry": ""},
        color="actionability_score", color_continuous_scale="Blues"
    )
    fig.update_layout(
        height=450, margin=dict(l=0, r=0, t=10, b=0), 
        coloraxis_colorbar=dict(title="Setup<br>Density %", thickness=15),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# =====================================================================
# TAB VIEWS
# =====================================================================

def top_buy_setups_view(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    stock_latest = stock_history[stock_history["date"] == selected_date].copy()

    eligible_df = basic_latest[(basic_latest.get("members", 0) >= SMALL_GROUP_LIMIT) & (basic_latest.get("leadership_score", 0) >= 70.0)]
    top_leaders = eligible_df.sort_values("leadership_score", ascending=False)["basic_industry"].tolist()
    
    if not top_leaders:
        top_leaders = basic_latest[basic_latest.get("members", 0) >= SMALL_GROUP_LIMIT].sort_values("leadership_score", ascending=False).head(5)["basic_industry"].tolist()

    buy_candidates = stock_latest[(stock_latest["basic_industry"].isin(top_leaders)) & (stock_latest.get("established_buy_setup", 0) == 1)].copy()

    # --- TOP METRICS ---
    m_col = st.columns(4)
    m_col[0].metric("Actionable Buy Setups", fmt_int(len(buy_candidates)))
    m_col[1].metric("Leading Basic Industries", fmt_int(len(top_leaders)))
    m_col[2].metric("Market Breadth (Top Tier)", f"{len(top_leaders)} / {len(basic_latest[basic_latest.get('members', 0) >= SMALL_GROUP_LIMIT])}")
    m_col[3].metric("Data Sync", selected_date.strftime("%d %b %Y"))

    st.markdown("---")
    
    # --- INTERACTIVE CHART ---
    st.markdown("### The Setup Matrix")
    st.caption("Visually hunt the best setups. Top-Right corner = Tighter Squeeze & Higher Historical Momentum.")
    if buy_candidates.empty:
        st.info("No established stocks currently meet all 5 criteria in leading sectors today.")
    else:
        render_setup_matrix(buy_candidates)

    # --- EXECUTION TABLE ---
    st.markdown("### Curated Execution Table")
    if not buy_candidates.empty:
        # Sort Logic
        if "buy_priority_score" not in buy_candidates.columns: buy_candidates["buy_priority_score"] = 0.0
        sort_cols = [c for c in ["buy_priority_score", "tight_3d_range", "vol_ratio_50"] if c in buy_candidates.columns]
        sort_dirs = [False if c == "buy_priority_score" else True for c in sort_cols]
        if sort_cols: buy_candidates = buy_candidates.sort_values(sort_cols, ascending=sort_dirs)
        
        buy_candidates = buy_candidates.head(20).reset_index(drop=True)
        buy_candidates.insert(0, "Rank", range(1, len(buy_candidates) + 1))
        buy_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy_candidates["symbol"].astype(str)

        display_buy = buy_candidates.rename(columns={
            "symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
            "buy_priority_score": "Priority Score", "tight_3d_range": "3-Day Range",
            "vol_ratio_50": "Vol vs 50D", "gain_6m": "6M Gain"
        })

        keep_cols = ["Rank", "Symbol", "Chart", "Basic Industry", "Close", "Priority Score", "3-Day Range", "Vol vs 50D", "6M Gain"]
        display_buy = display_buy[[col for col in keep_cols if col in display_buy.columns]]

        # Using Streamlit Column Config for beautiful, interactive tables
        st.dataframe(
            display_buy,
            use_container_width=True, hide_index=True, height=450,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Symbol": st.column_config.TextColumn(weight="bold"),
                "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗", width="small"),
                "Priority Score": st.column_config.ProgressColumn("Score (0-100)", format="%.1f", min_value=0, max_value=100),
                "3-Day Range": st.column_config.NumberColumn("3-Day Range", format="%.3f"),
                "Vol vs 50D": st.column_config.NumberColumn("Volume (vs 50D)", format="%.2fx"),
                "6M Gain": st.column_config.NumberColumn("6M Gain", format="%.2f"),
            }
        )


def ipo_watchlist_view(stock_history: pd.DataFrame, basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.markdown("### 🚀 High-Liquidity IPO Watchlist")
    st.caption("Newly listed stocks (< 150 days) in leading sectors. Daily range ≤ 5% and average turnover > 5 Crore.")
    
    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    top_leaders = basic_latest[basic_latest.get("leadership_score", 0) >= 70.0]["basic_industry"].tolist()
    
    ipo_candidates = stock_history[
        (stock_history["date"] == selected_date) & 
        (stock_history["basic_industry"].isin(top_leaders)) & 
        (stock_history.get("ipo_buy_setup", 0) == 1)
    ].copy()

    if ipo_candidates.empty:
        st.info("No newly listed IPO stocks in leading industries currently meet criteria.")
        return

    ipo_candidates["Avg Turnover (Cr)"] = ipo_candidates.get("ipo_turnover_avg", pd.Series(dtype=float)) / 10000000.0
    ipo_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo_candidates["symbol"].astype(str)
    
    if "daily_range" in ipo_candidates.columns:
        ipo_candidates = ipo_candidates.sort_values("daily_range", ascending=True)
        
    ipo_candidates = ipo_candidates.reset_index(drop=True)
    ipo_candidates.insert(0, "Rank", range(1, len(ipo_candidates) + 1))

    display_ipo = ipo_candidates.rename(columns={
        "symbol": "Symbol", "basic_industry": "Basic Industry", "close": "Close",
        "daily_range": "1-Day Range", "ret_20d": "20D Return"
    })
    
    keep_ipo = ["Rank", "Symbol", "Chart", "Basic Industry", "Close", "1-Day Range", "Avg Turnover (Cr)", "20D Return"]
    display_ipo = display_ipo[[col for col in keep_ipo if col in display_ipo.columns]]
    
    st.dataframe(
        display_ipo,
        use_container_width=True, hide_index=True, height=400,
        column_config={
            "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
            "1-Day Range": st.column_config.NumberColumn(format="%.3f"),
            "Avg Turnover (Cr)": st.column_config.NumberColumn("Turnover (₹ Cr)", format="₹%.1f Cr"),
            "20D Return": st.column_config.NumberColumn(format="%.3f"),
        }
    )


def sector_leadership_view(basic_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.markdown("### Basic Industry Leadership")
    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("#### Top 15 Sectors")
        render_sector_bar_chart(basic_latest)
        
    with col2:
        st.markdown("#### Sector Data")
        if "members" in basic_latest.columns:
            basic_latest = basic_latest[basic_latest["members"] >= SMALL_GROUP_LIMIT]
            
        basic_latest = basic_latest.sort_values("leadership_score", ascending=False).reset_index(drop=True)
        basic_latest.insert(0, "Rank", range(1, len(basic_latest) + 1))
        
        display_basic = basic_latest.rename(columns={
            "basic_industry": "Basic Industry", "leadership_score": "Leadership", 
            "actionability_score": "Setup %", "regime": "State", "members": "Stocks",
            "eq_ret_20d": "20D Ret"
        })
        
        keep = ["Rank", "Basic Industry", "State", "Leadership", "Setup %", "Stocks", "20D Ret"]
        display_basic = display_basic[[c for c in keep if c in display_basic.columns]]
        
        st.dataframe(
            display_basic, use_container_width=True, hide_index=True, height=450,
            column_config={
                "State": st.column_config.TextColumn(width="medium"),
                "Leadership": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "Setup %": st.column_config.NumberColumn(format="%.1f%%"),
                "20D Ret": st.column_config.NumberColumn(format="%.3f"),
            }
        )

def methodology_view() -> None:
    st.subheader("Methodology (2-Axis System)")
    st.markdown(
        """
        **Axis 1: Leadership Score (Macro Institutional Trend, 0–100)**
        *   **35% Price Velocity:** Median 20-day and 60-day equal-weighted returns percentile-ranked cross-sectionally.
        *   **35% Structural Alignment:** Percentage of constituents in full EMA alignment (20 EMA > 50 EMA > 200 EMA).
        *   **30% Institutional Volume:** 50-Day Cumulative Up/Down Volume Ratio.

        **Axis 2: Actionability Score (Micro Setup Density %)**
        Displays the exact raw percentage of stocks in that industry passing the **5-Rule Setup Gauntlet**:
        1.  **Trend:** Price > 50 EMA > 200 EMA.
        2.  **Prior Advance:** 6-Month Advance $\ge 30\%$.
        3.  **Strike Zone:** Price resting within -1% to +5% of 10 EMA, 20 EMA, or 50 EMA.
        4.  **Coil:** 3-Day Squeeze $\le 1.2 \times \text{ATR}_{14}$.
        5.  **Dry-Up:** Today's volume $\le 0.5\times$ the 50-day average.

        **Top Buy Setups Filter**
        Stocks are only displayed on the Top Buy Setups list if their parent industry has a Leadership Score $\ge 70$ AND the individual stock passes all 5 rules.
        """
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

    st.title("NSE Sectoral Leadership Terminal")
    sync_text = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Backend Synced: {sync_text.replace('T', ' ').replace('Z', ' IST')}")

    tabs = st.tabs(["🎯 Top Buy Setups", "🚀 IPO Watchlist", "📊 Sector Leadership", "🏢 Broad Industry", "📖 Methodology"])

    with tabs[0]:
        buy_setup_date = section_header("Setup Discovery Engine", trading_dates(stock_history), "buys")
        top_buy_setups_view(basic_history, stock_history, buy_setup_date)
    with tabs[1]:
        ipo_date = section_header("IPO Discovery Engine", trading_dates(stock_history), "ipos")
        ipo_watchlist_view(stock_history, basic_history, ipo_date)
    with tabs[2]:
        sector_date = section_header("Basic Industry Macro", trading_dates(basic_history), "basic")
        sector_leadership_view(basic_history, sector_date)
    with tabs[3]:
        ind_date = section_header("Broad Industry Macro", trading_dates(industry_history), "industry")
        st.info("Broad Industry table leverages the same logic as Basic Industries. Refer to the Sector Leadership tab for granular rotation.")
    with tabs[4]:
        methodology_view()

if __name__ == "__main__":
    main()
