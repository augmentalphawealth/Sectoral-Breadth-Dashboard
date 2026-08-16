import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import os

st.set_page_config(page_title="Industry Engine", layout="wide", initial_sidebar_state="collapsed")

if 'sync_in_progress' not in st.session_state:
    st.session_state.sync_in_progress = False
    st.session_state.sync_start_time = 0
    st.session_state.pre_sync_time = ""

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
    header { visibility: hidden; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border-radius: 12px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); border: 1px solid #e2e8f0; margin-bottom: 12px; }
    .card-title { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 12px; font-weight: 600; color: #64748b; margin-top: 2px; }
    .exec-banner { background-color: #0f172a; color: #f8fafc; padding: 14px 18px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #38bdf8; }
    .exec-heading { font-size: 13px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 8px; }
    .exec-bullet { font-size: 12.5px; color: #e2e8f0; margin-bottom: 4px; line-height: 1.4; }
    .stButton>button { border-radius: 6px; font-weight: 600; font-size: 12px; padding: 0.35rem 0.6rem; }
    </style>
""", unsafe_allow_html=True)

# Replace with your actual GitHub username
REPO_OWNER = "YOUR_GITHUB_USERNAME_HERE"
REPO_NAME = "Industry-Breadth-Dashboard"

def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token: return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/intraday_update.yml/dispatches"
    res = requests.post(url, headers={"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}, json={"ref": "main"})
    return res.status_code == 204

@st.cache_data(ttl=30, show_spinner=False)
def load_data():
    matrix = pd.read_csv("industry_breadth_matrix.csv") if os.path.exists("industry_breadth_matrix.csv") else pd.DataFrame()
    stocks = pd.read_parquet("latest_stocks_snapshot.parquet") if os.path.exists("latest_stocks_snapshot.parquet") else pd.DataFrame()
    sync_time = open("last_sync.txt", "r").read().strip() if os.path.exists("last_sync.txt") else "Unknown"
    return matrix, stocks, sync_time

matrix_df, stocks_df, last_sync = load_data()

head_c1, head_c2 = st.columns([3.2, 1.8])
with head_c1:
    st.markdown("<h2 style='margin:0; font-weight:800; color:#0f172a;'>🏭 INDUSTRY BREADTH ENGINE</h2>", unsafe_allow_html=True)
with head_c2:
    st.markdown(f"<div style='text-align:right; font-size:12px; font-weight:700; color:#475569;'>Last Sync: <span style='color:#0284c7;'>{last_sync}</span></div>", unsafe_allow_html=True)
    if st.button("⚡ Trigger Live Sync", use_container_width=True):
        if trigger_github_action():
            st.session_state.sync_in_progress, st.session_state.sync_start_time, st.session_state.pre_sync_time = True, time.time(), last_sync
            st.rerun()

if st.session_state.sync_in_progress:
    if time.time() - st.session_state.sync_start_time < 180:
        st.warning("⏳ Live sync running... Dashboard will auto-refresh.")
        time.sleep(10)
        _, _, curr_time = load_data()
        if curr_time != st.session_state.pre_sync_time:
            st.session_state.sync_in_progress = False
            st.cache_data.clear()
        st.rerun()
    else: st.session_state.sync_in_progress = False

if matrix_df.empty:
    st.warning("Data compiling. Run the GitHub Action first.")
    st.stop()

v_break = matrix_df[matrix_df['Vol_Shock_Count'] >= 2]['Industry'].head(3).tolist()
bottoms = matrix_df[(matrix_df['Pct_Above_20'] >= 50) & (matrix_df['Pct_Above_200'] < 40)]['Industry'].head(3).tolist()
exhaust = matrix_df[(matrix_df['Pct_Above_50'] >= 80) & (matrix_df['Pct_Above_20'] < 50)]['Industry'].head(3).tolist()

st.markdown(f"""
<div class='exec-banner'>
    <div class='exec-heading'>⚡ Algorithmic Executive Briefing</div>
    <div class='exec-bullet'>🚀 <b>Volume Breakouts:</b> {", ".join(v_break) if v_break else "None detected"}</div>
    <div class='exec-bullet'>🌱 <b>Early Reversals:</b> {", ".join(bottoms) if bottoms else "None (Aligned with long trends)"}</div>
    <div class='exec-bullet'>⚠️ <b>Exhaustion Risk:</b> {", ".join(exhaust) if exhaust else "None (Leaders holding fast momentum)"}</div>
    <div class='exec-bullet'>🎯 <b>Top 3 Focus:</b> {" • ".join(matrix_df['Industry'].head(3).tolist())}</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Tracked</div><div class='metric-value'>{len(matrix_df)}</div>", unsafe_allow_html=True)
with m2:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Leading</div><div class='metric-value' style='color:#16a34a;'>{len(matrix_df[matrix_df['Thrust_Score'] >= 70])}</div>", unsafe_allow_html=True)
with m3:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Neutral</div><div class='metric-value' style='color:#f59e0b;'>{len(matrix_df[(matrix_df['Thrust_Score'] >= 40) & (matrix_df['Thrust_Score'] < 70)])}</div>", unsafe_allow_html=True)
with m4:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Risk-Off</div><div class='metric-value' style='color:#dc2626;'>{len(matrix_df[matrix_df['Thrust_Score'] < 40])}</div>", unsafe_allow_html=True)

st.markdown("### 📊 Industry Matrix")
disp_cols = ['Industry', 'Total_Stocks', 'Thrust_Score'] + (['Avg_Daily_Gain'] if 'Avg_Daily_Gain' in matrix_df else []) + ['Pct_Above_20', 'Pct_Above_50', 'Pct_Above_200', 'Pct_Near_52W', 'Vol_Shock_Count']
st.dataframe(matrix_df[disp_cols].style.background_gradient(subset=['Thrust_Score', 'Pct_Above_20', 'Pct_Above_50', 'Pct_Above_200'], cmap='RdYlGn', vmin=10, vmax=90), use_container_width=True, height=400)

st.markdown("### 🔍 Stock Drill-Down")
selected_ind = st.selectbox("Select Industry:", options=matrix_df['Industry'].tolist())
if selected_ind and not stocks_df.empty:
    sub_df = stocks_df[stocks_df['Industry'] == selected_ind].copy()
    sub_df['EMA Status'] = np.where(sub_df['Above_20'] & sub_df['Above_50'] & sub_df['Above_200'], "🟢 Strong", np.where(sub_df['Above_200'], "🟡 >200", "🔴 Weak"))
    sub_df['Vol Surge'] = (sub_df['Volume'] / sub_df['Vol_20D_Avg'].replace(0, np.nan)).round(1).astype(str) + "x"
    st.dataframe(sub_df[['Symbol', 'Close', 'Daily_Pct', 'Vol Surge', 'Dist_52W_High', 'EMA Status']].sort_values('Daily_Pct', ascending=False), use_container_width=True)
