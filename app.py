import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests

st.set_page_config(page_title="Situational Awareness", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .block-container { padding-top: 1.5rem; max-width: 98%; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 15px; }
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; font-weight: 600; font-size: 13px; }
    .card-title { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .metric-val { font-size: 24px; font-weight: 800; color: #f8fafc; }
    .stDataFrame { font-size: 12px; }
    /* Force dark mode table rendering */
    [data-testid="stDataFrame"] div[role="grid"] { background-color: #1e293b !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    hist = pd.read_parquet("historical_breadth_matrix.parquet") if os.path.exists("historical_breadth_matrix.parquet") else pd.DataFrame()
    sync = open("last_sync.txt", "r").read().strip() if os.path.exists("last_sync.txt") else "Unknown"
    return hist, sync

hist_df, last_sync = load_data()

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown("<h2 style='text-align:center; margin-bottom: 0px;'>SITUATIONAL AWARENESS: SECTORAL BREADTH</h2>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; font-size:12px; color:#38bdf8; font-weight:600;'>Last Sync: {last_sync}</div>", unsafe_allow_html=True)

if hist_df.empty:
    st.warning("Data compiling. Awaiting initial GitHub Action run.")
    st.stop()

# --- SCHEMA VALIDATION SHIELD ---
# Prevents raw Python crashes if Streamlit boots before the backend finishes updating the Parquet file.
required_columns = ['Pct_50E', 'MCO', 'Thrust_3D', 'TRIN', 'Pct_Froth', 'Pct_200E', 'Pct_20E']
missing_cols = [col for col in required_columns if col not in hist_df.columns]

if missing_cols:
    st.warning(f"⚠️ **Database Upgrade in Progress.** The backend engine is currently building the new institutional metrics. Please wait for the GitHub Action to finish writing the new data. (Missing columns: {', '.join(missing_cols)})")
    st.stop()

# --- UNIVERSAL TIME MACHINE ---
hist_df['Date'] = pd.to_datetime(hist_df['Date'])
dates_avail = sorted(hist_df['Date'].dt.date.unique(), reverse=True)

st.markdown("<hr style='border-color: #334155; margin: 15px 0px;'>", unsafe_allow_html=True)
sc1, sc2, sc3 = st.columns([1, 2, 1])
with sc2:
    selected_date = st.selectbox("📅 SELECT MARKET STATE DATE", options=dates_avail)

df = hist_df[hist_df['Date'].dt.date == selected_date].copy()

# --- STATE MACHINE LOGIC (Using EMA) ---
# Phase 1: Leading (Strong EMA, Positive MCO, High Thrust)
leading = df[(df['Pct_50E'] > 60) & (df['MCO'] > 0) & (df['Thrust_3D'] > 1.2)]['Industry'].tolist()
# Phase 2: Bottoming (Capitulation TRIN > 1.5, Price < 200E but 20E > 50E crossing)
bottoming = df[(df['TRIN'] > 1.5) & (df['Pct_200E'] < 40) & (df['Pct_20E'] > df['Pct_50E'])]['Industry'].tolist()
# Phase 3: Exhaustion (Speculative froth > 15%, Complacency TRIN < 0.6)
exhausted = df[(df['Pct_Froth'] > 15) & (df['TRIN'] < 0.6)]['Industry'].tolist()

m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Total Sectors</div><div class='metric-val'>{len(df)}</div>", unsafe_allow_html=True)
with m2:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Leading Momentum</div><div class='metric-val' style='color:#22c55e;'>{len(leading)}</div>", unsafe_allow_html=True)
with m3:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Capitulation / Bottoming</div><div class='metric-val' style='color:#38bdf8;'>{len(bottoming)}</div>", unsafe_allow_html=True)
with m4:
    with st.container(border=True): st.markdown(f"<div class='card-title'>Exhaustion Risk</div><div class='metric-val' style='color:#ef4444;'>{len(exhausted)}</div>", unsafe_allow_html=True)

st.markdown(f"""
<div style='background-color:#1e293b; padding:15px; border-radius:8px; border:1px solid #334155; margin-bottom: 20px;'>
    <div style='font-size:13px; font-weight:800; color:#38bdf8; margin-bottom:8px;'>SITUATIONAL TAGS ({selected_date})</div>
    <div style='font-size:13px; color:#cbd5e1; margin-bottom:4px;'>🟢 <b>Leading:</b> {", ".join(leading[:5]) if leading else "None"}</div>
    <div style='font-size:13px; color:#cbd5e1; margin-bottom:4px;'>🔵 <b>Bottoming:</b> {", ".join(bottoming[:5]) if bottoming else "None"}</div>
    <div style='font-size:13px; color:#cbd5e1;'>🔴 <b>Exhaustion/Froth:</b> {", ".join(exhausted[:5]) if exhausted else "None"}</div>
</div>
""", unsafe_allow_html=True)

# --- THE INSTITUTIONAL MATRIX ---
st.markdown("### 🧬 Institutional Breadth Matrix")

df['TRIN'] = df['TRIN'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Too Small")
df['MCO'] = df['MCO'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "Too Small")

rename_map = {
    'Universe': 'Stocks',
    'Avg_Return': 'Avg %',
    'Pct_20E': '% > 20 EMA',
    'Pct_50E': '% > 50 EMA',
    'Pct_200E': '% > 200 EMA',
    'Pct_NH': '% New Highs',
    'Pct_NL': '% New Lows',
    'Vol_Breadth': 'Vol Brdth',
    'Thrust_3D': '3D Thrust',
    'Pct_Froth': '% Froth (1M>25%)'
}
display_df = df.rename(columns=rename_map).sort_values('% > 20 EMA', ascending=False)
cols = ['Industry', 'Stocks', 'Avg %', '% > 20 EMA', '% > 50 EMA', '% > 200 EMA', '% New Highs', '% New Lows', 'Vol Brdth', '3D Thrust', '% Froth (1M>25%)', 'TRIN', 'MCO']

format_dict = {'Avg %': '{:.2f}%', '% > 20 EMA': '{:.1f}%', '% > 50 EMA': '{:.1f}%', '% > 200 EMA': '{:.1f}%', '% New Highs': '{:.1f}%', '% New Lows': '{:.1f}%', 'Vol Brdth': '{:.2f}x', '3D Thrust': '{:.2f}x', '% Froth (1M>25%)': '{:.1f}%'}

st.dataframe(
    display_df[cols].style
        .background_gradient(subset=['% > 20 EMA', '% > 50 EMA', '% > 200 EMA', '% New Highs'], cmap='RdYlGn', vmin=10, vmax=90)
        .background_gradient(subset=['% New Lows', '% Froth (1M>25%)'], cmap='Reds', vmin=0, vmax=20)
        .format(format_dict),
    use_container_width=True, 
    height=600
)
