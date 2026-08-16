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

# --- AUTHENTICATION ---
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

scrip_master = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json", timeout=30).json()
df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))].copy()
symbol_to_token = dict(zip(nse_stocks['symbol'], nse_stocks['token']))
all_symbols = list(symbol_to_token.keys())

# --- TAXONOMY CRAWLER (ISIN / NIFTY / YAHOO) ---
industry_cache_file = "master_stock_industry.parquet"
sym_to_ind = dict(zip(pd.read_parquet(industry_cache_file)['Symbol'], pd.read_parquet(industry_cache_file)['Industry'])) if os.path.exists(industry_cache_file) else {}

try:
    for filename in ["ind_nifty500list.csv", "ind_niftymicrocap250_list.csv", "ind_niftysmallcap250list.csv", "ind_niftytotalmarket_list.csv"]:
        resp = requests.get(f"https://niftyindices.com/IndexConstituent/{filename}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code == 200:
            for r in csv.DictReader(StringIO(resp.content.decode("utf-8-sig", errors="replace"))):
                sym, ind = f"{str(r.get('Symbol')).strip().upper()}-EQ", str(r.get("Industry") or "").strip().title()
                if sym in all_symbols and ind and ind.upper() not in ["", "NAN", "NONE"]: sym_to_ind[sym] = ind
except: pass

for sym in all_symbols:
    if sym not in sym_to_ind: sym_to_ind[sym] = "Emerging Equities"

missing_symbols = [sym for sym, ind in sym_to_ind.items() if ind == "Emerging Equities"]
if missing_symbols:
    yf_session = requests.Session()
    yf_session.headers.update({"User-Agent": "Mozilla/5.0"})
    for sym in missing_symbols[:50]: # Drip crawl 50 per day to avoid bans
        try:
            info = yf.Ticker(f"{sym.replace('-EQ', '')}.NS", session=yf_session).info
            sym_to_ind[sym] = (info.get('industry') or info.get('sector') or "Emerging Equities").title()
        except: pass
        time.sleep(0.3)

psu_banks = {"SBIN", "PNB", "BOB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "CENTRALBK", "IOB", "UCOBANK", "MAHABANK", "PSB"}
for sym, ind in sym_to_ind.items():
    clean_sym = sym.replace("-EQ", "").upper()
    if str(ind).upper() in ["FINANCIAL SERVICES", "BANKS", "FINANCE", "REGIONAL BANKS", "CREDIT SERVICES"]:
        if "BANK" in clean_sym or "BANC" in clean_sym or clean_sym in psu_banks or clean_sym in {"HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK"}: sym_to_ind[sym] = "PSU Bank" if clean_sym in psu_banks else "Private Bank"
        elif "INSUR" in clean_sym or "LIFE" in clean_sym or clean_sym in {"LIC", "GICRE", "LICI"}: sym_to_ind[sym] = "Insurance"
        else: sym_to_ind[sym] = "NBFC"

pd.DataFrame(list(sym_to_ind.items()), columns=["Symbol", "Industry"]).to_parquet(industry_cache_file, index=False)

# --- DAILY DATA FETCH ---
ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
today_dt = pd.to_datetime(datetime.datetime.now(ist_tz).strftime("%Y-%m-%d"))
chunks = [list(symbol_to_token.values())[i:i + 35] for i in range(0, len(symbol_to_token), 35)]
fetched_data = []

for chunk in chunks:
    for attempt in range(4):
        try:
            res = smartApi.getMarketData("FULL", {"NSE": chunk})
            if res and res.get('data'):
                for item in res['data'].get('fetched', []):
                    sym = item.get('tradingSymbol', '')
                    if sym: fetched_data.append({"Date": today_dt, "Symbol": sym, "Industry": sym_to_ind.get(sym, "Emerging Equities"), "Open": float(item.get('open', 0)), "High": float(item.get('high', 0)), "Low": float(item.get('low', 0)), "Close": float(item.get('ltp', 0)), "Volume": float(item.get('tradeVolume', 0) or 0)})
                break
        except: time.sleep(1.5)
    time.sleep(0.1)

if not fetched_data: exit(1)

# --- MERGE & VECTORIZED STOCK MATH ---
hist_file = "industry_historical_cache.parquet"
if os.path.exists(hist_file):
    df_hist = pd.read_parquet(hist_file)
    df_hist['Industry'] = df_hist['Symbol'].map(sym_to_ind).fillna("Emerging Equities")
    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.tz_localize(None).dt.normalize()
    df_combined = pd.concat([df_hist[df_hist['Date'] != today_dt], pd.DataFrame(fetched_data)], ignore_index=True)
else:
    df_combined = pd.DataFrame(fetched_data)

df_combined['Date'] = pd.to_datetime(df_combined['Date']).dt.tz_localize(None).dt.normalize()
df_combined = df_combined.sort_values(['Symbol', 'Date']).reset_index(drop=True)

print("Calculating 8 Institutional Parameters...")

# 1. Simple Exponential Moving Averages (EMA)
df_combined['EMA_20'] = df_combined.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=10).mean())
df_combined['EMA_50'] = df_combined.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=20).mean())
df_combined['EMA_200'] = df_combined.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=50).mean())

df_combined['Above_20E'] = (df_combined['Close'] > df_combined['EMA_20']).astype(int)
df_combined['Above_50E'] = (df_combined['Close'] > df_combined['EMA_50']).astype(int)
df_combined['Above_200E'] = (df_combined['Close'] > df_combined['EMA_200']).astype(int)

# 2. Advance/Decline & Volume Splitting
df_combined['Prev_Close'] = df_combined.groupby('Symbol')['Close'].shift(1)
df_combined['Is_Advance'] = (df_combined['Close'] > df_combined['Prev_Close']).astype(int)
df_combined['Is_Decline'] = (df_combined['Close'] < df_combined['Prev_Close']).astype(int)
df_combined['Up_Vol'] = df_combined['Volume'] * df_combined['Is_Advance']
df_combined['Down_Vol'] = df_combined['Volume'] * df_combined['Is_Decline']

# 3. NH / NL (2% Threshold)
df_combined['52W_High'] = df_combined.groupby('Symbol')['High'].transform(lambda x: x.rolling(252, min_periods=30).max())
df_combined['52W_Low'] = df_combined.groupby('Symbol')['Low'].transform(lambda x: x.rolling(252, min_periods=30).min())
df_combined['Is_NH'] = (df_combined['Close'] >= (df_combined['52W_High'] * 0.98)).astype(int)
df_combined['Is_NL'] = (df_combined['Close'] <= (df_combined['52W_Low'] * 1.02)).astype(int)

# 4. 1-Month 25% Movers
df_combined['Close_20D_Ago'] = df_combined.groupby('Symbol')['Close'].shift(20)
df_combined['Is_25P_Mover'] = ((df_combined['Close'] / df_combined['Close_20D_Ago'] - 1) >= 0.25).astype(int)

# 5. Volatility / Thrust safety
df_combined['Daily_Pct'] = df_combined['Close'].pct_change() * 100
df_combined['Daily_Pct'] = df_combined['Daily_Pct'].replace([np.inf, -np.inf], 0).fillna(0)

# Save raw cache
cutoff = today_dt - pd.Timedelta(days=1500)
df_combined = df_combined[df_combined['Date'] >= cutoff].copy()
df_combined.to_parquet(hist_file, index=False)

# --- INDUSTRY AGGREGATION & COMPLEX OSCILLATORS ---
print("Aggregating Industry Breadth Matrix...")
agg_df = df_combined.groupby(['Date', 'Industry']).agg(
    Universe=('Symbol', 'count'),
    Avg_Return=('Daily_Pct', 'mean'),
    Above_20E=('Above_20E', 'sum'),
    Above_50E=('Above_50E', 'sum'),
    Above_200E=('Above_200E', 'sum'),
    Advances=('Is_Advance', 'sum'),
    Declines=('Is_Decline', 'sum'),
    UpVol=('Up_Vol', 'sum'),
    DownVol=('Down_Vol', 'sum'),
    NH=('Is_NH', 'sum'),
    NL=('Is_NL', 'sum'),
    Movers_25P=('Is_25P_Mover', 'sum')
).reset_index()

agg_df = agg_df.sort_values(['Industry', 'Date']).reset_index(drop=True)

# Math: Percentages
for col, agg_col in [('Pct_20E', 'Above_20E'), ('Pct_50E', 'Above_50E'), ('Pct_200E', 'Above_200E'), ('Pct_NH', 'NH'), ('Pct_NL', 'NL'), ('Pct_Froth', 'Movers_25P')]:
    agg_df[col] = (agg_df[agg_col] / agg_df['Universe'] * 100).round(1)

# Math: Volume Breadth Ratio
agg_df['Vol_Breadth'] = (agg_df['UpVol'] / agg_df['DownVol'].replace(0, np.nan)).fillna(1.0).round(2)

# Math: 3-Day Breakout Thrust
agg_df['Adv_3D_Sum'] = agg_df.groupby('Industry')['Advances'].transform(lambda x: x.rolling(3).sum())
agg_df['Dec_3D_Sum'] = agg_df.groupby('Industry')['Declines'].transform(lambda x: x.rolling(3).sum())
agg_df['Thrust_3D'] = (agg_df['Adv_3D_Sum'] / agg_df['Dec_3D_Sum'].replace(0, np.nan)).fillna(1.0).round(2)

# Math: TRIN & MCO (Strictly bounded by Universe >= 10)
agg_df['Net_Adv'] = agg_df['Advances'] - agg_df['Declines']
agg_df['MCO'] = agg_df.groupby('Industry')['Net_Adv'].transform(lambda x: x.ewm(span=19, adjust=False).mean() - x.ewm(span=39, adjust=False).mean())

ad_ratio = agg_df['Advances'] / agg_df['Declines'].replace(0, 0.001)
vol_ratio = agg_df['UpVol'] / agg_df['DownVol'].replace(0, 0.001)
agg_df['TRIN'] = (ad_ratio / vol_ratio.replace(0, 0.001)).round(2)

mask_small = agg_df['Universe'] < 10
agg_df.loc[mask_small, 'MCO'] = np.nan
agg_df.loc[mask_small, 'TRIN'] = np.nan
agg_df['MCO'] = agg_df['MCO'].round(1)

cols_to_keep = ['Date', 'Industry', 'Universe', 'Avg_Return', 'Pct_20E', 'Pct_50E', 'Pct_200E', 'Pct_NH', 'Pct_NL', 'Vol_Breadth', 'Thrust_3D', 'Pct_Froth', 'TRIN', 'MCO']
final_matrix = agg_df[cols_to_keep].copy()

final_matrix.to_parquet("historical_breadth_matrix.parquet", index=False)

live_matrix = final_matrix[final_matrix['Date'] == today_dt].copy()
live_matrix.to_csv("industry_breadth_matrix.csv", index=False)

with open("last_sync.txt", "w") as f:
    f.write(datetime.datetime.now(ist_tz).strftime('%d %b %Y, %I:%M %p IST (EOD Sync)'))
    
