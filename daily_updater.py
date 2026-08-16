import pandas as pd
import numpy as np
import requests
import datetime
import pyotp
import os
import csv
import time
from io import StringIO
from SmartApi import SmartConnect
import logzero
import logging

logzero.logger.setLevel(logging.FATAL)

api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

if not all([api_key, client_code, login_pin, totp_secret]): exit(1)

try:
    smartApi = SmartConnect(api_key=api_key)
    session = smartApi.generateSession(client_code, login_pin, pyotp.TOTP(totp_secret).now())
    if not session.get('status'): exit(1)
except Exception: exit(1)

scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(scrip_url, timeout=30).json()
df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))].copy()
symbol_to_token = dict(zip(nse_stocks['symbol'], nse_stocks['token']))
all_symbols = list(symbol_to_token.keys())

# --- THE BULLETPROOF TAXONOMY SYNC ---
nse_map = {}
if os.path.exists("sector_map.csv"):
    try:
        with open("sector_map.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym, ind = (row.get("Symbol") or "").strip().upper(), (row.get("Industry") or row.get("Sector") or "").strip()
                if sym and ind: nse_map[f"{sym}-EQ"] = ind
    except: pass

try:
    indices = ["ind_nifty500list.csv", "ind_niftymicrocap250_list.csv", "ind_niftysmallcap250list.csv"]
    for filename in indices:
        resp = requests.get(f"https://niftyindices.com/IndexConstituent/{filename}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code == 200:
            for r in csv.DictReader(StringIO(resp.content.decode("utf-8-sig", errors="replace"))):
                sym = f"{str(r.get('Symbol')).strip().upper()}-EQ"
                ind = str(r.get("Industry") or "").strip()
                if sym and ind and sym not in nse_map: nse_map[sym] = ind.title()
except: pass

sym_to_ind = {}
for sym in all_symbols:
    if sym in nse_map: sym_to_ind[sym] = nse_map[sym]
    else:
        clean_sym, refined_ind = sym.replace("-EQ", "").upper(), "Emerging Equities"
        for kw, tag in [("BANK", "Private Sector Bank"), ("FIN", "NBFC"), ("TECH", "IT Services"), ("AUTO", "Auto Components"), ("PHARMA", "Pharmaceuticals"), ("CHEM", "Specialty Chemicals")]:
            if kw in clean_sym: refined_ind = tag; break
        sym_to_ind[sym] = refined_ind

pd.DataFrame(list(sym_to_ind.items()), columns=["Symbol", "Industry"]).to_parquet("master_stock_industry.parquet", index=False)

# --- BATCH EOD PRICE DATA FETCH ---
ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
today_dt = pd.to_datetime(datetime.datetime.now(ist_tz).strftime("%Y-%m-%d"))

all_tokens = list(symbol_to_token.values())
chunks = [all_tokens[i:i + 35] for i in range(0, len(all_tokens), 35)]
fetched_data = []

for chunk in chunks:
    for attempt in range(4):
        try:
            res = smartApi.getMarketData("FULL", {"NSE": chunk})
            if res and res.get('status') and res.get('data'):
                for item in res['data'].get('fetched', []):
                    sym = item.get('tradingSymbol', '')
                    if sym: fetched_data.append({"Date": today_dt, "Symbol": sym, "Industry": sym_to_ind.get(sym, "Emerging Equities"), "Open": float(item.get('open', 0)), "High": float(item.get('high', 0)), "Low": float(item.get('low', 0)), "Close": float(item.get('ltp', 0)), "Volume": float(item.get('tradeVolume', 0) or 0)})
                break
        except: time.sleep(1.5)
    time.sleep(0.12)

if not fetched_data: exit(1)
df_today = pd.DataFrame(fetched_data)

# --- DATA MERGE & 6-YEAR EXPANSION ---
hist_file = "industry_historical_cache.parquet"
if os.path.exists(hist_file):
    df_hist = pd.read_parquet(hist_file)
    df_hist['Industry'] = df_hist['Symbol'].map(sym_to_ind).fillna("Emerging Equities")
    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.tz_localize(None).dt.normalize()
    df_combined = pd.concat([df_hist[df_hist['Date'] != today_dt], df_today], ignore_index=True)
else:
    df_combined = df_today.copy()

df_combined['Date'] = pd.to_datetime(df_combined['Date']).dt.tz_localize(None).dt.normalize()
# Expanded memory up to 6.8 years (2500 days)
df_combined = df_combined[df_combined['Date'] >= (today_dt - pd.Timedelta(days=2500))].sort_values(['Symbol', 'Date']).reset_index(drop=True)

df_combined['EMA_20'] = df_combined.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=10).mean())
df_combined['EMA_50'] = df_combined.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=20).mean())
df_combined['EMA_200'] = df_combined.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=50).mean())
df_combined['Daily_Pct'] = df_combined.groupby('Symbol')['Close'].pct_change() * 100
df_combined['Daily_Pct'] = df_combined['Daily_Pct'].replace([np.inf, -np.inf], 0).fillna(0)
df_combined['Vol_20D_Avg'] = df_combined.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
df_combined['Vol_Shock'] = (df_combined['Volume'] > (df_combined['Vol_20D_Avg'] * 1.5)) & (df_combined['Daily_Pct'] > 0)
df_combined['Rolling_52W_High'] = df_combined.groupby('Symbol')['High'].transform(lambda x: x.rolling(252, min_periods=30).max())
df_combined['Dist_52W_High'] = ((df_combined['Close'] - df_combined['Rolling_52W_High']) / df_combined['Rolling_52W_High']) * 100
df_combined['Near_52W_High'] = df_combined['Dist_52W_High'] >= -3.0
df_combined['Above_20'] = df_combined['Close'] > df_combined['EMA_20']
df_combined['Above_50'] = df_combined['Close'] > df_combined['EMA_50']
df_combined['Above_200'] = df_combined['Close'] > df_combined['EMA_200']

df_combined.to_parquet(hist_file, index=False)

# --- THE TIME MACHINE: BUILD 6-YEAR HISTORICAL MATRIX ---
print("Building 6-Year Historical Timeline Matrix...")
hist_matrix = df_combined.groupby(['Date', 'Industry']).agg(Total_Stocks=('Symbol', 'count'), Avg_Daily_Gain=('Daily_Pct', 'mean'), Above_20=('Above_20', 'sum'), Above_50=('Above_50', 'sum'), Above_200=('Above_200', 'sum'), Vol_Shock_Count=('Vol_Shock', 'sum'), Near_52W_Count=('Near_52W_High', 'sum')).reset_index()
hist_matrix = hist_matrix[hist_matrix['Total_Stocks'] >= 3].copy()

for col, agg_col in [('Pct_Above_20', 'Above_20'), ('Pct_Above_50', 'Above_50'), ('Pct_Above_200', 'Above_200'), ('Pct_Near_52W', 'Near_52W_Count')]:
    hist_matrix[col] = (hist_matrix[agg_col] / hist_matrix['Total_Stocks'] * 100).round(1)

hist_matrix['Thrust_Score'] = ((hist_matrix['Pct_Above_20'] * 0.30) + (hist_matrix['Pct_Above_50'] * 0.25) + (hist_matrix['Pct_Above_200'] * 0.20) + (hist_matrix['Pct_Near_52W'] * 0.15) + ((hist_matrix['Vol_Shock_Count'] / hist_matrix['Total_Stocks'] * 100).clip(upper=100) * 0.10)).round(0).astype(int)
hist_matrix['Avg_Daily_Gain'] = hist_matrix['Avg_Daily_Gain'].round(2)
hist_matrix.to_parquet("historical_breadth_matrix.parquet", index=False)

# --- LIVE EOD SNAPSHOT FOR FAST DASHBOARD ---
latest_df = df_combined[df_combined['Date'] == today_dt].copy()
latest_df[['Symbol', 'Industry', 'Close', 'Daily_Pct', 'Volume', 'Vol_20D_Avg', 'Dist_52W_High', 'Above_20', 'Above_50', 'Above_200', 'Vol_Shock']].to_parquet("latest_stocks_snapshot.parquet", index=False)

matrix = hist_matrix[hist_matrix['Date'] == today_dt].sort_values('Thrust_Score', ascending=False).reset_index(drop=True)
matrix.to_csv("industry_breadth_matrix.csv", index=False)

with open("last_sync.txt", "w") as f:
    f.write(datetime.datetime.now(ist_tz).strftime('%d %b %Y, %I:%M %p IST (EOD Sync)'))
