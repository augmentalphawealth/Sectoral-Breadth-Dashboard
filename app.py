import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import os

st.set_page_config(page_title="Industry Breadth Engine", layout="wide", initial_sidebar_state="collapsed")

# --- INITIALIZE SESSION STATE FOR SYNC TRACKING ---
if 'sync_in_progress' not in st.session_state:
    st.session_state.sync_in_progress = False
    st.session_state.sync_start_time = 0
    st.session_state.pre_sync_time = ""

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
    header { visibility: hidden; }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff; border-radius: 12px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); border: 1px solid #e2e8f0; margin-bottom: 12px;
    }
    .card-title { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 12px; font-weight: 600; color: #64748b; margin-top: 2px; }
    
    .exec-banner {
        background-color: #0f172a; color: #f8fafc; padding: 14px 18px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #38bdf8;
    }
    .exec-heading { font-size: 13px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 8px; }
    .exec-bullet { font-size: 12.5px; color: #e2e8f0; margin-bottom: 4px; line-height: 1.4; }
    
    .stButton>button { border-radius: 6px; font-weight: 600; font-size: 12px; padding: 0.35rem 0.6rem; }
    </style>
""", unsafe_allow_html=True)

REPO_OWNER = "augmentalphawealth"
REPO_NAME = "Industry-Breadth-Dashboard"
BRANCH = "main"

def trigger_github_action(workflow_name, button_label):
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        st.error("GitHub Token missing in Streamlit Secrets!")
        return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{workflow_name}/dispatches"
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}
    with st.status(f"🚀 Triggering {button_label}...", expanded=False) as status:
        res = requests.post(url, headers=headers, json={"ref": BRANCH})
        if res.status_code == 204:
            status.update(label="✅ Workflow started successfully.", state="complete")
            time.sleep(1)
            return True
        else:
            status.update(label=f"❌ Failed to trigger ({res.status_code}). Check Secrets.", state="error")
            return False

# 1. Data Loader
@st.cache_data(ttl=30, show_spinner=False)
def load_engine_data():
    matrix_df = pd.read_csv("industry_breadth_matrix.csv") if os.path.exists("industry_breadth_matrix.csv") else pd.DataFrame()
    stocks_df = pd.read_parquet("latest_stocks_snapshot.parquet") if os.path.exists("latest_stocks_snapshot.parquet") else pd.DataFrame()
    sync_time = "Unknown"
    if os.path.exists("last_sync.txt"):
        with open("last_sync.txt", "r") as f:
            sync_time = f.read().strip()
    return matrix_df, stocks_df, sync_time

matrix_df, stocks_df, last_sync = load_engine_data()

# Header Row
head_c1, head_c2 = st.columns([3.2, 1.8])
with head_c1:
    st.markdown("<h2 style='margin:0; font-weight:800; color:#0f172a;'>🏭 INDUSTRY & SUB-SECTOR BREADTH</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0; color:#64748b; font-size:13px; font-weight:600;'>Real-Time Institutional Capital Migration & Industry Momentum Rank</p>", unsafe_allow_html=True)
with head_c2:
    st.markdown(f"<div style='text-align:right; font-size:12px; font-weight:700; color:#475569; margin-top:2px; margin-bottom:4px;'>Last Sync: <span style='color:#0284c7;'>{last_sync}</span></div>", unsafe_allow_html=True)
    if st.button("⚡ Live Intraday Sync", use_container_width=True):
        if trigger_github_action("intraday_update.yml", "Intraday Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync
            st.rerun()

# --- SYNC WARNING BANNER & AUTO-REFRESH LOGIC ---
if st.session_state.sync_in_progress:
    elapsed_time = time.time() - st.session_state.sync_start_time
    if elapsed_time < 300:
        st.warning(f"⏳ **Live Intraday Sync in Progress:** The robot is streaming live quotes from Angel One. The dashboard will **auto-refresh** once complete. (Elapsed: {int(elapsed_time)}s)", icon="🤖")
        time.sleep(10)
        _, _, curr_sync_time = load_engine_data()
        if curr_sync_time != st.session_state.pre_sync_time and curr_sync_time != "Unknown":
            st.session_state.sync_in_progress = False
            st.cache_data.clear()
            st.rerun()
        else:
            st.rerun()
    else:
        st.session_state.sync_in_progress = False
        st.error("Sync timed out. Please click Live Intraday Sync again.")

st.markdown("<hr style='margin: 8px 0px 14px 0px;'>", unsafe_allow_html=True)

if matrix_df.empty:
    st.warning("⏳ Data is currently compiling. Please run the workflow in GitHub Actions to generate the database.")
    st.stop()

# 2. Algorithmic Executive Summary
vol_breakout_inds = matrix_df[matrix_df['Volume_Shocks'] >= 2].sort_values('Volume_Shocks', ascending=False)['Industry'].head(3).tolist()
vol_str = ", ".join(vol_breakout_inds) if vol_breakout_inds else "No concentrated volume shock clusters"

early_bottom_inds = matrix_df[(matrix_df['Pct_Above_20'] >= 50) & (matrix_df['Pct_Above_200'] < 40)]['Industry'].head(3).tolist()
bottom_str = ", ".join(early_bottom_inds) if early_bottom_inds else "None (Broad trends aligned with 200 EMA)"

exhaustion_inds = matrix_df[(matrix_df['Pct_Above_50'] >= 80) & (matrix_df['Pct_Above_20'] < 55)]['Industry'].head(3).tolist()
exhaust_str = ", ".join(exhaustion_inds) if exhaustion_inds else "None (Leaders maintaining fast momentum)"

top5_focus = matrix_df.head(5)['Industry'].tolist()
top5_str = " • ".join(top5_focus)

lagging_inds = matrix_df[matrix_df['Pct_Above_50'] < 30]['Industry'].tail(3).tolist()
lagging_str = ", ".join(lagging_inds) if lagging_inds else "No severely damaged sectors"

st.markdown(f"""
<div class='exec-banner'>
    <div class='exec-heading'>⚡ Live Algorithmic Briefing</div>
    <div class='exec-bullet'>🚀 <b>Volume Shock Clusters:</b> Institutional turnover surging in <b>{vol_str}</b>.</div>
    <div class='exec-bullet'>🌱 <b>Early Reversals / Bottoms:</b> Reclaiming 20 EMA from base: <b>{bottom_str}</b>.</div>
    <div class='exec-bullet'>⚠️ <b>Exhaustion / Breakdown Risk:</b> Extended leaders losing short-term structure: <b>{exhaust_str}</b>.</div>
    <div class='exec-bullet'>🎯 <b>Top 5 Actionable Focus Industries:</b> <b>{top5_str}</b>.</div>
    <div class='exec-bullet'>🛡️ <b>Capital Outflow / Lagging:</b> Damaged moving average breadth (Avoid): <b>{lagging_str}</b>.</div>
</div>
""", unsafe_allow_html=True)

# 3. Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(border=True):
        st.markdown("<div class='card-title'>Tracked Industries</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(matrix_df)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-sub'>{len(stocks_df)} Active Equities</div>", unsafe_allow_html=True)
with m2:
    with st.container(border=True):
        leading_count = len(matrix_df[matrix_df['Thrust_Score'] >= 70])
        st.markdown("<div class='card-title'>Leading Industries (>70 Score)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='color:#16a34a;'>{leading_count}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-sub'>Aggressive Breadth & Momentum</div>", unsafe_allow_html=True)
with m3:
    with st.container(border=True):
        chop_count = len(matrix_df[(matrix_df['Thrust_Score'] >= 40) & (matrix_df['Thrust_Score'] < 70)])
        st.markdown("<div class='card-title'>Neutral / Selective</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='color:#f59e0b;'>{chop_count}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-sub'>Stock-Specific Opportunities</div>", unsafe_allow_html=True)
with m4:
    with st.container(border=True):
        lag_count = len(matrix_df[matrix_df['Thrust_Score'] < 40])
        st.markdown("<div class='card-title'>Lagging / Risk-Off</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='color:#dc2626;'>{lag_count}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-sub'>Avoid Capital Allocation</div>", unsafe_allow_html=True)

# 4. Live Intraday Movers Row
if 'Avg_Daily_Gain' in matrix_df.columns:
    st.markdown("### ⚡ Live Market Movers (Today's % Change)")
    c_gain, c_loss = st.columns(2)
    
    top_gainers = matrix_df.sort_values('Avg_Daily_Gain', ascending=False).head(5)
    top_losers = matrix_df.sort_values('Avg_Daily_Gain', ascending=True).head(5)
    
    with c_gain:
        with st.container(border=True):
            st.markdown("<div class='card-title' style='color:#16a34a;'>🟢 Top 5 Gaining Industries Today</div>", unsafe_allow_html=True)
            fig_g = go.Figure(go.Bar(
                x=top_gainers['Avg_Daily_Gain'],
                y=top_gainers['Industry'],
                orientation='h',
                marker_color='#22c55e',
                text=[f"+{v:.2f}%" for v in top_gainers['Avg_Daily_Gain']],
                textposition='auto'
            ))
            fig_g.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            fig_g.update_xaxes(visible=False)
            st.plotly_chart(fig_g, use_container_width=True, config={'displayModeBar': False})
            
    with c_loss:
        with st.container(border=True):
            st.markdown("<div class='card-title' style='color:#dc2626;'>🔴 Top 5 Lagging Industries Today</div>", unsafe_allow_html=True)
            fig_l = go.Figure(go.Bar(
                x=top_losers['Avg_Daily_Gain'],
                y=top_losers['Industry'],
                orientation='h',
                marker_color='#ef4444',
                text=[f"{v:.2f}%" for v in top_losers['Avg_Daily_Gain']],
                textposition='auto'
            ))
            fig_l.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            fig_l.update_xaxes(visible=False)
            st.plotly_chart(fig_l, use_container_width=True, config={'displayModeBar': False})

# 5. Main Breadth Matrix
st.markdown("### 📊 Sub-Sector Breadth Matrix")

display_cols = ['Industry', 'Total_Stocks', 'Thrust_Score', 'Pct_Above_20', 'Pct_Above_50', 'Pct_Above_200', 'Pct_Near_52W', 'Volume_Shocks']
rename_map = {
    'Total_Stocks': 'Universe',
    'Thrust_Score': 'Thrust Score',
    'Pct_Above_20': '% > 20 EMA',
    'Pct_Above_50': '% > 50 EMA',
    'Pct_Above_200': '% > 200 EMA',
    'Pct_Near_52W': '% Near 52W High',
    'Volume_Shocks': 'Vol Shocks'
}

if 'Avg_Daily_Gain' in matrix_df.columns:
    display_cols.insert(3, 'Avg_Daily_Gain')
    rename_map['Avg_Daily_Gain'] = 'Today Avg Gain %'

display_df = matrix_df[display_cols].rename(columns=rename_map)

st.dataframe(
    display_df.style.background_gradient(
        subset=['Thrust Score', '% > 20 EMA', '% > 50 EMA', '% > 200 EMA', '% Near 52W High'],
        cmap='RdYlGn', vmin=10, vmax=90
    ),
    use_container_width=True,
    height=400
)

# 6. Instant Stock-Level Drill-Down
st.markdown("---")
st.markdown("### 🔍 Sub-Sector Constituent Drill-Down")

sel_c1, _ = st.columns([2.5, 2])
with sel_c1:
    selected_ind = st.selectbox("Select an Industry to inspect underlying constituents:", options=matrix_df['Industry'].tolist())

if not stocks_df.empty and selected_ind:
    sub_df = stocks_df[stocks_df['Industry'] == selected_ind].copy()
    
    sub_df['EMA Status'] = np.where(
        sub_df['Above_20_EMA'] & sub_df['Above_50_EMA'] & sub_df['Above_200_EMA'], "🟢 Above All (20/50/200)",
        np.where(sub_df['Above_200_EMA'], "🟡 Above 200 EMA", "🔴 Below 200 EMA")
    )
    sub_df['Vol / 20D Avg'] = (sub_df['Volume'] / sub_df['Vol_20D_Avg'].replace(0, np.nan)).round(2).fillna(1.0).astype(str) + "x"
    sub_df['Dist 52W High'] = sub_df['Dist_52W_High'].round(1).astype(str) + "%"
    sub_df['Daily Gain %'] = sub_df['Daily_Pct'].round(2)
    
    out_table = sub_df[['Symbol', 'Close', 'Daily Gain %', 'Vol / 20D Avg', 'Dist 52W High', 'EMA Status']].sort_values('Daily Gain %', ascending=False)
    
    st.write(f"**Found {len(out_table)} constituent equities in `{selected_ind}`:**")
    st.dataframe(
        out_table.style.format({'Close': '₹{:.2f}'}),
        use_container_width=True,
        height=320
    )
