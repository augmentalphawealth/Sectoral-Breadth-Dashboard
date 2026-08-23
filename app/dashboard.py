from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'

st.set_page_config(page_title='Basic Industry Situational Awareness', layout='wide')


def read_parquet(path: str) -> pd.DataFrame:
    file_path = DATA / path
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(file_path)


def read_last_sync() -> str:
    path = DATA / 'last_sync.txt'
    return path.read_text(encoding='utf-8').strip() if path.exists() else 'Not available'

basic = read_parquet('dashboard_basic_industry_latest.parquet')
industry = read_parquet('dashboard_industry_latest.parquet')
watch = read_parquet('dashboard_stock_watchlist_latest.parquet')

st.title('Basic Industry Situational Awareness')
st.caption(f'Last sync: {read_last_sync()}')

if basic.empty:
    st.warning('Dashboard tables not found. Run the EOD pipeline first.')
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Basic industries tracked', int(basic['basic_industry'].nunique()))
c2.metric('Strong groups', int((basic['regime'] == 'Strong').sum()))
c3.metric('Emerging groups', int((basic['regime'] == 'Emerging').sum()))
c4.metric('VCP ready stocks', int(watch['vcp_ready'].sum()) if 'vcp_ready' in watch.columns else 0)

left, right = st.columns([1.4, 1])
with left:
    st.subheader('Basic industry leaderboard')
    cols = ['basic_industry', 'strength_score', 'regime', 'eq_ret_20d', 'eq_ret_60d', 'pct_above_50', 'trend_template_pct', 'acc_minus_dist', 'breakout_count', 'vcp_ready_count']
    show = basic[cols].copy()
    st.dataframe(show, use_container_width=True, hide_index=True)

with right:
    st.subheader('Strength map')
    heat = basic[['basic_industry', 'strength_score', 'regime']].copy()
    heat['tile'] = 'Basic Industry'
    fig = px.treemap(heat, path=['tile', 'basic_industry'], values='strength_score', color='strength_score', color_continuous_scale='RdYlGn')
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=480)
    st.plotly_chart(fig, use_container_width=True)

st.subheader('Industry leaderboard')
icols = ['industry', 'strength_score', 'regime', 'eq_ret_20d', 'eq_ret_60d', 'pct_above_50', 'trend_template_pct', 'acc_minus_dist', 'breakout_count', 'vcp_ready_count']
st.dataframe(industry[icols], use_container_width=True, hide_index=True)

st.subheader('Actionable stock watchlist')
wcols = ['symbol', 'industry', 'basic_industry', 'close', 'ret_20d', 'trend_template_pass', 'vcp_ready', 'breakout_55', 'dist_52w_high', 'quality_rank']
existing = [c for c in wcols if c in watch.columns]
st.dataframe(watch[existing], use_container_width=True, hide_index=True)
