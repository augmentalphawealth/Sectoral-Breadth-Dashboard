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
import yfinance as yf
import logzero
import logging

logzero.logger.setLevel(logging.FATAL)

# --- ANGEL ONE AUTHENTICATION ---
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

# --- THE MASS BULK CRAWLER (UNLEASHED) ---
industry_cache_file = "master_stock_industry.parquet"
if os.path.exists(industry_cache_file):
    df_ind = pd.read_parquet(industry_cache_file)
    sym_to_ind = dict(zip(df_ind['Symbol'], df_ind['Industry']))
else:
    sym_to_ind = {}

# 1. Instant Bulk-Load from NiftyIndices
try:
    headers = {"User-Agent": "Mozilla/5.0"}
    indices = ["ind_nifty500list.csv", "ind_niftymicrocap250_list.csv", "ind_niftysmallcap250list.csv", "ind_niftytotalmarket_list.csv"]
    for filename in indices:
        resp = requests.get(f"https://niftyindices.com/IndexConstituent/{filename}", headers=headers, timeout=10)
        if resp.status_code == 200:
            for r in csv.DictReader(StringIO(resp.content.decode("utf-8-sig", errors="replace"))):
                sym = f"{str(r.get('Symbol')).strip().upper()}-EQ"
                ind = str(r.get("Industry") or "").strip().title()
                if sym in all_symbols and ind and ind.upper() not in ["", "NAN", "NONE"]: 
                    sym_to_ind[sym] = ind
except: pass

# 2. Find who is still missing
for sym in all_symbols:
    if sym not in sym_to_ind:
        sym_to_ind[sym] = "Emerging Equities"

missing_symbols = [sym for sym, ind in sym_to_ind.items() if ind == "Emerging Equities"]

# 3. MASS YAHOO CRAWLER (NO LIMITS)
if missing_symbols:
    print(f"🚀 MASS CRAWLER ACTIVATED: Fetching {len(missing_symbols)} missing stocks via Yahoo Finance. This will take ~15 minutes...")
    
    # Custom session to prevent Yahoo blocks
    yf_session = requests.Session()
    yf_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    for i, sym in enumerate(missing_symbols):
        clean_sym = sym.replace('-EQ', '')
        try:
            ticker = yf.Ticker(f"{clean_sym}.NS", session=yf_session)
            info = ticker.info
            raw_ind = info.get('industry') or info.get('sector') or "Emerging Equities"
            sym_to_ind[sym] = raw_ind.title()
        except Exception:
            pass 
            
        # This will print progress into your GitHub Action logs!
        if (i + 1) % 100 == 0 or (i + 1) == len(missing_symbols):
            print(f"Mapped {i + 1} / {len(missing_symbols)} microcaps...")
            
        time.sleep(0.3) # 0.3s delay to prevent Yahoo from crashing the robot

# 4. Institutional Granularity Filter (Splits Banks & NBFCs)
psu_banks = {"SBIN", "PNB", "BOB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "CENTRALBK", "IOB", "UCOBANK", "MAHABANK", "PSB"}
for sym, ind in sym_to_ind.items():
    clean_sym = sym.replace("-EQ", "").upper()
    ind_upper = str(ind).upper()
    if ind_upper in ["FINANCIAL SERVICES", "BANKS", "FINANCE", "REGIONAL BANKS", "CREDIT SERVICES"]:
        is_bank = "BANK" in clean_sym or "BANC" in clean_sym or clean_sym in psu_banks or clean_sym in {"HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK"}
        is_insurance = "INSUR" in clean_sym or "LIFE" in clean_sym or "ASSURANCE" in clean_sym or clean_sym in {"LIC", "GICRE", "LICI"}
        if is_bank: sym_to_ind[sym] = "PSU Bank" if clean_sym in psu_banks else "Private Bank"
        elif is_insurance: sym_to_ind[sym] = "Insurance"
        else: sym_to_ind[sym] = "NBFC"

# Permanently save the perfected mapping
pd.DataFrame(list(sym_to_ind.items()), columns=["Symbol", "Industry"]).to_parquet(industry_cache_file, index=False)

# --- BATCH EOD PRICE DATA FETCH ---
ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
today_dt = pd.to_datetime(datetime.datetime.now(ist_tz).strftime("%Y-%m-%d"))

all_tokens = list(symbol_to_token.values())
chunks = [all_tokens[i:i + 35] for i in range(0, len(all_tokens), 35)]
fetched_data = []

print("Fetching EOD Quotes...")
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

# --- DATA MERGE & AUTO-CURE FOR HISTORICAL DATA ---
hist_file = "industry_historical_cache.parquet"
if os.path.exists(hist_file):
    df_hist = pd.read_parquet(hist_file)
    
    # ⚡ THE CURE: Instantly rewrites the 6-year history with the new correct names
    df_hist['Industry'] = df_hist['Symbol'].map(sym_to_ind).fillna("Emerging Equities")
    
    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.tz_localize(None).dt.normalize()
    df_combined = pd.concat([df_hist[df_hist['Date'] != today_dt], df_today], ignore_index=True)
else:
    df_combined = df_today.copy()

df_combined['Date'] = pd.to_datetime(df_combined['Date']).dt.tz_localize(None).dt.normalize()
df_combined = df_combined[df_combined['Date'] >= (today_dt - pd.Timedelta(days=2500))].sort_values(['Symbol', 'Date']).reset_index(drop=True)

print("Crunching EMAs and Volumetric Thrusts...")
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
hist_matrix = df_combined.groupby(['Date', 'Industry']).agg(Total_Stocks=('Symbol', 'count'), Avg_Daily_Gain=('Daily_Pct', 'mean'), Above_20=('Above_20', 'sum'), Above_50=('Above_50', 'sum'), Above_200=('Above_200', 'sum'), Vol_Shock_Count=('Vol_Shock', 'sum'), Near_52W_Count=('Near_52W_High', 'sum')).reset_index()
hist_matrix = hist_matrix[hist_matrix['Total_Stocks'] >= 3].copy()

for col, agg_col in [('Pct_Above_20', 'Above_20'), ('Pct_Above_50', 'Above_50'), ('Pct_Above_200', 'Above_200'), ('Pct_Near_52W', 'Near_52W_Count')]:
    hist_matrix[col] = (hist_matrix[agg_col] / hist_matrix['Total_Stocks'] * 100).round(1)

hist_matrix['Thrust_Score'] = ((hist_matrix['Pct_Above_20'] * 0.30) + (hist_matrix['Pct_Above_50'] * 0.25) + (hist_matrix['Pct_Above_200'] * 0.20) + (hist_matrix['Pct_Near_52W'] * 0.15) + ((hist_matrix['Vol_Shock_Count'] / hist_matrix['Total_Stocks'] * 100).clip(upper=100) * 0.10)).round(0).astype(int)
hist_matrix['Avg_Daily_Gain'] = hist_matrix['Avg_Daily_Gain'].round(2)
hist_matrix.to_parquet("historical_breadth_matrix.parquet", index=False)

# --- LIVE EOD SNAPSHOT FOR DASHBOARD ---
latest_df = df_combined[df_combined['Date'] == today_dt].copy()
latest_df[['Symbol', 'Industry', 'Close', 'Daily_Pct', 'Volume', 'Vol_20D_Avg', 'Dist_52W_High', 'Above_20', 'Above_50', 'Above_200', 'Vol_Shock']].to_parquet("latest_stocks_snapshot.parquet", index=False)

matrix = hist_matrix[hist_matrix['Date'] == today_dt].sort_values('Thrust_Score', ascending=False).reset_index(drop=True)
matrix.to_csv("industry_breadth_matrix.csv", index=False)

with open("last_sync.txt", "w") as f:
    f.write(datetime.datetime.now(ist_tz).strftime('%d %b %Y, %I:%M %p IST (EOD Sync)'))
