# Requirements: streamlit>=1.60.0, plotly>=6.0.0
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
SMALL_GROUP_LIMIT = 5
TOP_N_SETUPS = 20
MAX_LABELED_BUBBLES = 15
MAX_LABELED_TILES = 25
INK = "#1F2937"
PALETTE = {"Fresh Leader (HUNT)":"#2E7D63","Extended Leader (WAIT)":"#D98E3B","Speculative Coil (AVOID)":"#8B5FBF","Dead (AVOID)":"#B0483C","Neutral Transition":"#9AA5B1"}
CHART_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK, size=13)

def styled_fig(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10,r=10,t=48,b=10), font=CHART_FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(size=15,color=INK), legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0))
    return fig

st.set_page_config(page_title="NSE Sectoral Breadth & Buy Setups", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{padding-top:1.6rem;padding-bottom:2.5rem;max-width:1440px}[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:700}[data-testid="stMetricLabel"]{font-size:.8rem;color:#64748b;text-transform:uppercase;letter-spacing:.03em}h1{font-weight:700;letter-spacing:-.02em}h3{font-weight:600;margin-top:1.6rem;color:#1F2937}[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;border:1px solid #e5e7eb}</style>""", unsafe_allow_html=True)

def clean_text(value):
    if value is None or pd.isna(value): return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"

def as_number(value):
    if value is None or pd.isna(value): return None
    try: return float(str(value).replace(",","").replace("%","").strip())
    except (TypeError,ValueError): return None

def fmt_int(value):
    number=as_number(value); return "—" if number is None else f"{int(round(number)):,}"

def fmt_num(value, decimals=2):
    number=as_number(value); return "—" if number is None else f"{number:,.{decimals}f}"

def fmt_pct(value, source_is_fraction=True):
    number=as_number(value)
    if number is None: return "—"
    if source_is_fraction: number*=100
    return f"{number:,.2f}%"

def heat_color(value):
    score=as_number(value)
    if score is None: return "background-color:#f8fafc;color:#64748b"
    if score>=80: return "background-color:#14532d;color:#fff;font-weight:700"
    if score>=70: return "background-color:#166534;color:#fff;font-weight:700"
    if score>=60: return "background-color:#22c55e;color:#052e16;font-weight:700"
    if score>=50: return "background-color:#86efac;color:#14532d;font-weight:700"
    if score>=40: return "background-color:#fef3c7;color:#78350f;font-weight:700"
    if score>=30: return "background-color:#fed7aa;color:#7c2d12;font-weight:700"
    return "background-color:#fecaca;color:#7f1d1d;font-weight:700"

def actionability_color(value):
    pct=as_number(value)
    if pct is None: return ""
    if pct>=20: return "background-color:#1e3a8a;color:#fff;font-weight:700"
    if pct>0: return "background-color:#bfdbfe;color:#1e3a8a;font-weight:700"
    return "color:#94a3b8"

def style_with_heatmap(raw, display):
    styles=pd.DataFrame("", index=display.index, columns=display.columns)
    for col in ["Leadership Score","Industry Leadership Score","Strength","Buy Setup Score","Buy Priority Score"]:
        if col in raw.columns and col in display.columns: styles.loc[:,col]=raw[col].map(heat_color)
    if "Actionability (Setup %)" in raw.columns and "Actionability (Setup %)" in display.columns: styles.loc[:,"Actionability (Setup %)"]=raw["Actionability (Setup %)"].map(actionability_color)
    return display.style.apply(lambda _: styles, axis=None)

@st.cache_data(show_spinner=False)
def load_parquet(path):
    frame=pd.read_parquet(path)
    if "date" in frame.columns: frame["date"]=pd.to_datetime(frame["date"]).dt.normalize()
    return frame

def ensure_group_columns(frame, group_column):
    data=frame.copy()
    if group_column not in data.columns: data[group_column]="Unclassified"
    if "regime" not in data.columns: data["regime"]="Unclassified"
    data[group_column]=data[group_column].map(clean_text); data["regime"]=data["regime"].map(clean_text)
    if "leadership_score" not in data.columns and "strength_score" in data.columns: data["leadership_score"]=data["strength_score"]
    if "actionability_score" not in data.columns: data["actionability_score"]=0.0
    return data

def ensure_stock_columns(frame):
    data=frame.copy()
    for col in ["symbol","industry","basic_industry","sector"]:
        if col not in data.columns: data[col]="Unclassified"
        data[col]=data[col].map(clean_text)
    return data

def trading_dates(frame): return sorted(pd.Timestamp(x) for x in pd.to_datetime(frame["date"].dropna().unique()))

def global_date_navigator(dates,key):
    latest=pd.Timestamp(dates[-1]); state_key=f"{key}_selected_date"
    if state_key not in st.session_state: st.session_state[state_key]=latest
    selected=pd.Timestamp(st.session_state[state_key])
    if selected not in dates: selected=latest; st.session_state[state_key]=latest
    index=dates.index(selected)
    heading,controls=st.columns([5.7,2.3])
    with heading: st.subheader("Historical Date")
    with controls:
        previous,calendar,next_button=st.columns([.4,1.55,.4])
        with previous:
            if st.button("‹",key=f"{key}_previous",disabled=index==0,use_container_width=True): st.session_state[state_key]=dates[index-1]; st.rerun()
        with calendar:
            chosen=st.date_input("Historical date",value=selected.date(),min_value=pd.Timestamp(dates[0]).date(),max_value=latest.date(),key=f"{key}_calendar",label_visibility="collapsed")
        with next_button:
            if st.button("›",key=f"{key}_next",disabled=index==len(dates)-1,use_container_width=True): st.session_state[state_key]=dates[index+1]; st.rerun()
    valid=[d for d in dates if d<=pd.Timestamp(chosen)]; resolved=valid[-1] if valid else dates[0]
    if resolved!=selected: st.session_state[state_key]=resolved; st.rerun()
    st.caption(f"As of {resolved.strftime('%d %b %Y')}")
    return resolved

def make_group_table(frame, group_column):
    wanted=[group_column,"leadership_score","actionability_score","regime","members","nh_nl_net","eq_ret_1d","eq_ret_5d","eq_ret_20d","eq_ret_60d","pct_above_50","pct_above_200","acc_minus_dist","breakout_count","vcp_ready_count"]
    data=frame[[c for c in wanted if c in frame.columns]].copy()
    data=data.rename(columns={group_column:"Basic Industry" if group_column=="basic_industry" else "Industry","leadership_score":"Leadership Score","actionability_score":"Actionability (Setup %)","regime":"Trading State","members":"Constituent Stocks","nh_nl_net":"Net New Highs (%)","eq_ret_1d":"1D Return","eq_ret_5d":"5D Return","eq_ret_20d":"20D Return","eq_ret_60d":"60D Return","pct_above_50":"Stocks Above 50 EMA","pct_above_200":"Stocks Above 200 EMA","acc_minus_dist":"Accumulation − Distribution","breakout_count":"Breakouts","vcp_ready_count":"VCP Ready"})
    if "Leadership Score" in data.columns: data=data.sort_values("Leadership Score",ascending=False)
    data.insert(0,"Rank",range(1,len(data)+1)); return data.reset_index(drop=True)

def format_group_table(frame):
    data=frame.copy()
    for col in ["1D Return","5D Return","20D Return","60D Return","Stocks Above 50 EMA","Stocks Above 200 EMA"]:
        if col in data.columns: data[col]=data[col].apply(lambda x:fmt_pct(x,True))
    for col in ["Rank","Constituent Stocks","Accumulation − Distribution","Breakouts","VCP Ready"]:
        if col in data.columns: data[col]=data[col].apply(fmt_int)
    for col in ["Leadership Score","Actionability (Setup %)","Net New Highs (%)"]:
        if col in data.columns: data[col]=data[col].apply(fmt_num)
    return data

def format_stock_table(frame):
    data=frame.copy()
    for col in ["1D Return","5D Return","20D Return","60D Return","Distance from 52W High","6M Gain","Candle Range","Price Tightness (3D)"]:
        if col in data.columns: data[col]=data[col].apply(lambda x:fmt_pct(x,True))
    for col in ["Rank","Industry Rank","Heavy Volume Days (6M)"]:
        if col in data.columns: data[col]=data[col].apply(fmt_int)
    for col in ["Strength","Close","Buy Setup Score","Buy Priority Score","Setup Score","Vol Contraction (vs 50D)","50D Up/Down Vol","14D ATR","Avg Turnover (Cr)","Industry Leadership Score"]:
        if col in data.columns: data[col]=data[col].apply(fmt_num)
    return data

def overview_tab(basic_history,industry_history,stock_history,selected_date):
    basic=basic_history[basic_history["date"]==selected_date].copy(); kpi_dates=basic_history["date"].sort_values().unique()[-20:]; basic_kpi=basic_history[basic_history["date"].isin(kpi_dates)]
    hunt=basic_kpi[basic_kpi["regime"]=="Fresh Leader (HUNT)"].groupby("date")["basic_industry"].nunique(); wait=basic_kpi[basic_kpi["regime"]=="Extended Leader (WAIT)"].groupby("date")["basic_industry"].nunique(); total=basic_kpi.groupby("date")["basic_industry"].nunique(); avg=basic_kpi.groupby("date")["actionability_score"].mean()
    k1,k2,k3,k4,k5=st.columns(5); k1.metric("Latest Data",selected_date.strftime("%d %b %Y"),chart_data=hunt,chart_type="area"); k2.metric("Basic Industries",fmt_int(basic["basic_industry"].nunique()),chart_data=total,chart_type="line"); k3.metric("Fresh Leaders (HUNT)",f"{(basic['regime']=='Fresh Leader (HUNT)').sum()}",chart_data=hunt,chart_type="bar"); k4.metric("Extended Leaders (WAIT)",f"{(basic['regime']=='Extended Leader (WAIT)').sum()}",chart_data=wait,chart_type="bar"); k5.metric("Avg Actionability",fmt_num(basic["actionability_score"].mean(),1),chart_data=avg,chart_type="area")
    st.markdown("### Sector Leadership Treemap"); treemap_df=basic[basic["members"]>=SMALL_GROUP_LIMIT].copy().sort_values("members",ascending=False); idx=treemap_df.head(MAX_LABELED_TILES).index; treemap_df["tile_label"]=treemap_df["basic_industry"].where(treemap_df.index.isin(idx),""); fig=go.Figure(go.Treemap(labels=treemap_df["basic_industry"],parents=[""]*len(treemap_df),values=treemap_df["members"],text=treemap_df["tile_label"],textinfo="text",marker=dict(colors=treemap_df["leadership_score"],colorscale="Greens",showscale=True),hovertemplate="%{label}<br>Members: %{value}<br>Leadership: %{marker.color:.1f}<extra></extra>")); fig.update_layout(title="Sector Size colored by Leadership Score — top 25 tiles labeled"); st.plotly_chart(styled_fig(fig,420),use_container_width=True)
    st.markdown("### Regime Composition (Last 60 Sessions)"); dates=basic_history["date"].sort_values().unique()[-60:]; rc=basic_history[basic_history["date"].isin(dates)].groupby(["date","regime"]).size().unstack(fill_value=0).reset_index(); fig=go.Figure();
    for regime in PALETTE:
        if regime in rc.columns: fig.add_trace(go.Scatter(x=rc["date"],y=rc[regime],stackgroup="one",name=regime,fillcolor=PALETTE[regime],line=dict(width=0)))
    fig.update_layout(title="Regime Mix Over Time",yaxis_title="Count"); st.plotly_chart(styled_fig(fig,320),use_container_width=True)
    st.markdown("### Market Breadth Gauges"); g1,g2=st.columns(2)
    for col,title in [("pct_above_50","% Stocks Above 50 EMA"),("pct_above_200","% Stocks Above 200 EMA")]:
        value=basic[col].mean() if col in basic.columns else 0; fig=go.Figure(go.Indicator(mode="gauge+number",value=value,title=dict(text=title,font=dict(size=14)),gauge=dict(axis=dict(range=[0,100]),steps=[dict(range=[0,30],color="#fecaca"),dict(range=[30,70],color="#fef3c7"),dict(range=[70,100],color="#86efac")]))); (g1 if col=="pct_above_50" else g2).plotly_chart(styled_fig(fig,270),use_container_width=True)
    st.markdown("### Leadership vs Actionability Scatter"); sdf=basic[basic["members"]>=SMALL_GROUP_LIMIT].copy(); idx=sdf.nlargest(MAX_LABELED_BUBBLES,"leadership_score").index; sdf["label"]=sdf["basic_industry"].where(sdf.index.isin(idx),""); fig=go.Figure(go.Scatter(x=sdf["leadership_score"],y=sdf["actionability_score"],mode="markers+text",text=sdf["label"],textposition="top center",marker=dict(size=sdf["members"]*2,color=sdf["regime"].map(PALETTE).fillna("#9AA5B1"),line=dict(width=1,color="white")),hovertext=sdf["basic_industry"],hovertemplate="%{hovertext}<br>Leadership: %{x:.1f}<br>Actionability: %{y:.1f}%<extra></extra>")); fig.update_layout(title="Top 15 leaders labeled; hover for all industries",xaxis_title="Leadership Score",yaxis_title="Actionability (Setup %)",xaxis=dict(range=[0,100]),yaxis=dict(range=[0,max(sdf["actionability_score"].max()*1.2,30)])); st.plotly_chart(styled_fig(fig,400),use_container_width=True)
    st.markdown("### Current Basic Industry Leadership"); raw=make_group_table(basic[basic["members"]>=SMALL_GROUP_LIMIT],"basic_industry"); st.dataframe(style_with_heatmap(raw,format_group_table(raw)),use_container_width=True,hide_index=True,height=480)

def safe_rank_score(frame, column, ascending=False):
    if column not in frame.columns: return pd.Series(0.5,index=frame.index,dtype=float)
    values=pd.to_numeric(frame[column],errors="coerce"); ranks=values.rank(pct=True,ascending=ascending); return ranks.fillna(0.5)

def top_buy_tab(basic_history,stock_history,selected_date):
    st.subheader("🎯 Top Individual Buy Setups"); st.caption("No industry gate. Individual flags are used for selection; industry rank and leadership are context only.")
    basic=basic_history[basic_history["date"]==selected_date].copy(); stocks=stock_history[stock_history["date"]==selected_date].copy()
    context=basic[["basic_industry","leadership_score","regime"]].copy().sort_values("leadership_score",ascending=False).reset_index(drop=True); context.insert(0,"Industry Rank",range(1,len(context)+1)); context=context.rename(columns={"basic_industry":"Basic Industry","leadership_score":"Industry Leadership Score","regime":"Industry Regime"})
    buy=stocks[stocks.get("established_buy_setup",pd.Series(0,index=stocks.index)).fillna(0).astype(int)==1].copy(); ipo=stocks[stocks.get("ipo_buy_setup",pd.Series(0,index=stocks.index)).fillna(0).astype(int)==1].copy()
    a,b,c,d=st.columns(4); a.metric("Total Qualified Setups",fmt_int(len(buy))); b.metric("Total IPO Setups",fmt_int(len(ipo))); c.metric("Shortlist Size",f"Top {TOP_N_SETUPS}"); d.metric("Scan Date",selected_date.strftime("%d %b %Y"))
    st.markdown("### Top 20 Established Buy Setups")
    if buy.empty: st.info("No stocks currently pass the individual hard gates on this date.")
    else:
        buy["Buy Priority Score"]=(.30*safe_rank_score(buy,"tight_3d_range",True)+.25*safe_rank_score(buy,"vol_ratio_50",True)+.20*safe_rank_score(buy,"gain_6m")+.15*safe_rank_score(buy,"up_down_ratio")+.10*safe_rank_score(buy,"stock_strength_score"))*100
        buy=buy.merge(context[["Basic Industry","Industry Rank","Industry Leadership Score"]],left_on="basic_industry",right_on="Basic Industry",how="left").sort_values("Buy Priority Score",ascending=False).head(TOP_N_SETUPS).reset_index(drop=True); buy.insert(0,"Rank",range(1,len(buy)+1)); buy["Chart"]="https://in.tradingview.com/chart/?symbol=NSE:"+buy["symbol"].astype(str)
        display=buy.rename(columns={"symbol":"Symbol","basic_industry":"Basic Industry","close":"Close","tight_3d_range":"Price Tightness (3D)","vol_ratio_50":"Vol Contraction (vs 50D)","gain_6m":"6M Gain","nearest_ema_tag":"EMA Proximity","momentum_badge":"Momentum"}); keep=["Rank","Symbol","Chart","Basic Industry","Industry Rank","Industry Leadership Score","Close","Buy Priority Score","Price Tightness (3D)","Vol Contraction (vs 50D)","6M Gain","EMA Proximity","Momentum"]; display=display[[x for x in keep if x in display.columns]]; raw_display=display.copy(); formatted=format_stock_table(display.copy()); st.dataframe(style_with_heatmap(raw_display,formatted),use_container_width=True,hide_index=True,height=560,column_config={"Chart":st.column_config.LinkColumn("TradingView",display_text="Open ↗")})
    st.markdown("### Top 20 IPO Setups"); st.caption("Newly listed stocks (<150 days), no industry filter, ranked by Setup Score.")
    if ipo.empty: st.info("No IPO stocks currently pass the individual IPO gates on this date.")
    else:
        if "ipo_turnover_avg" in ipo.columns: ipo["Avg Turnover (Cr)"]=pd.to_numeric(ipo["ipo_turnover_avg"],errors="coerce")/10000000
        else: ipo["Avg Turnover (Cr)"]=None
        ipo=ipo.merge(context[["Basic Industry","Industry Rank","Industry Leadership Score"]],left_on="basic_industry",right_on="Basic Industry",how="left")
        if "ipo_setup_score" not in ipo.columns:
            ipo["ipo_setup_score"]=(.25*safe_rank_score(ipo,"tight_3d_range",True)+.20*safe_rank_score(ipo,"vol_ratio_50",True)+.20*safe_rank_score(ipo,"vwap_premium")+.20*safe_rank_score(ipo,"retracement_from_listing_high",True)+.15*safe_rank_score(ipo,"hh_hl_count"))*100
        ipo=ipo.sort_values("ipo_setup_score",ascending=False).head(TOP_N_SETUPS).reset_index(drop=True); ipo.insert(0,"Rank",range(1,len(ipo)+1)); ipo["Chart"]="https://in.tradingview.com/chart/?symbol=NSE:"+ipo["symbol"].astype(str)
        display=ipo.rename(columns={"symbol":"Symbol","basic_industry":"Basic Industry","close":"Close","days_listed":"Days Listed","ipo_phase":"Phase","ipo_setup_score":"Setup Score","vwap_premium":"Above VWAP","retracement_from_listing_high":"Off Post-List High"}); keep=["Rank","Symbol","Chart","Basic Industry","Industry Rank","Industry Leadership Score","Close","Days Listed","Phase","Setup Score","Above VWAP","Off Post-List High","Avg Turnover (Cr)"]; display=display[[x for x in keep if x in display.columns]]; raw_display=display.copy(); formatted=format_stock_table(display.copy()); st.dataframe(style_with_heatmap(raw_display,formatted),use_container_width=True,hide_index=True,height=max(240,40*len(formatted)+60),column_config={"Chart":st.column_config.LinkColumn("TradingView",display_text="Open ↗")})
    st.markdown("### Full Basic Industry Leaderboard — context only"); st.dataframe(context,use_container_width=True,hide_index=True,height=360)

def basic_industry_tab(basic_history,stock_history):
    date=global_date_navigator(trading_dates(basic_history),"basic"); selected=basic_history[basic_history["date"]==date].copy(); filters=st.columns([1.45,.85,.85])
    with filters[0]: regimes=st.multiselect("Trading State filter",list(PALETTE),default=list(PALETTE),key="basic_regimes")
    with filters[1]: minimum=st.number_input("Minimum stocks",min_value=1,value=SMALL_GROUP_LIMIT,step=1,key="basic_minimum")
    with filters[2]: ranking=st.selectbox("Ranking",["Highest Leadership","Lowest Leadership"],key="basic_sort")
    if regimes: selected=selected[selected["regime"].isin(regimes)]
    selected=selected[selected["members"]>=minimum]; table=make_group_table(selected,"basic_industry")
    if ranking=="Lowest Leadership" and "Leadership Score" in table: table=table.sort_values("Leadership Score").reset_index(drop=True); table["Rank"]=range(1,len(table)+1)
    if table.empty: st.warning("No Basic Industries match the selected filters."); return
    st.dataframe(style_with_heatmap(table,format_group_table(table)),use_container_width=True,hide_index=True,height=410)
    group=st.selectbox("Basic Industry",table["Basic Industry"].tolist(),key="basic_group_selector"); data=stock_history[(stock_history["date"]==date)&(stock_history["basic_industry"]==group)].copy()
    if not data.empty: data["Chart"]="https://in.tradingview.com/chart/?symbol=NSE:"+data["symbol"].astype(str); data=data.sort_values("ret_20d",ascending=False) if "ret_20d" in data else data; data=data.reset_index(drop=True); data.insert(0,"Rank",range(1,len(data)+1)); display=data.rename(columns={"symbol":"Symbol","close":"Close","ret_20d":"20D Return","ret_60d":"60D Return","gain_6m":"6M Gain","stock_strength_score":"Strength"}); keep=["Rank","Symbol","Chart","Close","20D Return","60D Return","6M Gain","Strength"]; display=display[[x for x in keep if x in display.columns]]; st.dataframe(style_with_heatmap(display,format_stock_table(display)),use_container_width=True,hide_index=True,height=320,column_config={"Chart":st.column_config.LinkColumn("TradingView",display_text="Open ↗")})

def industry_tab(industry_history,stock_history):
    date=global_date_navigator(trading_dates(industry_history),"industry"); selected=industry_history[industry_history["date"]==date].copy(); selected=selected[selected["members"]>=SMALL_GROUP_LIMIT]; table=make_group_table(selected,"industry")
    if table.empty: st.warning("No Industry data is available for this date."); return
    st.dataframe(style_with_heatmap(table,format_group_table(table)),use_container_width=True,hide_index=True,height=470); group=st.selectbox("Industry",table["Industry"].tolist(),key="industry_group_selector"); data=stock_history[(stock_history["date"]==date)&(stock_history["industry"]==group)].copy()
    if not data.empty: data["Chart"]="https://in.tradingview.com/chart/?symbol=NSE:"+data["symbol"].astype(str); data=data.sort_values("ret_20d",ascending=False) if "ret_20d" in data else data; data=data.reset_index(drop=True); data.insert(0,"Rank",range(1,len(data)+1)); display=data.rename(columns={"symbol":"Symbol","basic_industry":"Basic Industry","close":"Close","ret_20d":"20D Return","ret_60d":"60D Return","gain_6m":"6M Gain","stock_strength_score":"Strength","nearest_ema_tag":"EMA Proximity","momentum_badge":"Momentum"}); keep=["Rank","Symbol","Chart","Basic Industry","Close","20D Return","60D Return","6M Gain","Strength","EMA Proximity","Momentum"]; display=display[[x for x in keep if x in display.columns]]; st.dataframe(style_with_heatmap(display,format_stock_table(display)),use_container_width=True,hide_index=True,height=320,column_config={"Chart":st.column_config.LinkColumn("TradingView",display_text="Open ↗")})

def methodology_tab():
    st.subheader("Methodology (2-Axis System)"); st.markdown("""**Top Buy Setups:** Individual stock gates only; no industry Leadership filter. Industry Rank and Leadership Score are context columns. Established setups are ranked by tightness 30%, volume contraction 25%, 6-month gain 20%, up/down ratio 15%, and stock strength 10%. IPO setups are ranked by tightness 25%, dry-up 20%, VWAP premium 20%, retracement 20%, and HH-HL structure 15%. Liquidity threshold is 5 Crore average turnover.\n\n**Leadership:** 35% price velocity, 35% structural alignment, and 30% institutional volume, smoothed with a 3-day EWM.\n\n**Actionability:** Liquidity, price/trend gates, power, coil, and dry-up measurements.\n\nThe dashboard is a research shortlist, not a trade recommendation. Verify price, liquidity, and risk before acting.""")

def main():
    required=[BASIC_HISTORY_FILE,INDUSTRY_HISTORY_FILE,STOCK_HISTORY_FILE]; missing=[str(x.relative_to(ROOT)) for x in required if not x.exists()]
    if missing: st.error("Required dashboard files are missing. Run the data workflow first."); st.code("\n".join(missing)); st.stop()
    basic=ensure_group_columns(load_parquet(str(BASIC_HISTORY_FILE)),"basic_industry"); industry=ensure_group_columns(load_parquet(str(INDUSTRY_HISTORY_FILE)),"industry"); stocks=ensure_stock_columns(load_parquet(str(STOCK_HISTORY_FILE)))
    st.title("NSE Sectoral Breadth & 2-Axis Setup Engine"); sync=SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"; st.caption(f"Data as of {sync.replace('T',' ').replace('Z',' IST')}")
    dates=sorted(set(trading_dates(basic)+trading_dates(industry)+trading_dates(stocks))); tabs=st.tabs(["🎯 Top Buy Setups","Overview","Basic Industry","Industry","Methodology"])
    with tabs[0]: top_buy_tab(basic,stocks,global_date_navigator(dates,"buys"))
    with tabs[1]: overview_tab(basic,industry,stocks,global_date_navigator(dates,"overview"))
    with tabs[2]: basic_industry_tab(basic,stocks)
    with tabs[3]: industry_tab(industry,stocks)
    with tabs[4]: methodology_tab()

if __name__=="__main__": main()
