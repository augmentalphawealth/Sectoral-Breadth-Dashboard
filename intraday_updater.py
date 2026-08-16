import pandas as pd
import numpy as np
import requests
import datetime
import pyotp
import os
import time
from SmartApi import SmartConnect
import logzero
import logging

logzero.logger.setLevel(logging.FATAL)

# ---------------------------------------------------------
# 1. AUTHENTICATE WITH ANGEL ONE
# ---------------------------------------------------------
api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

if not all([api_key, client_code, login_pin, totp_secret]):
    print("❌ Missing Angel One credentials.")
    exit(1)

print("Authenticating for Live Intraday Industry Snapshot...")
try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if not session.get('status'):
        print("❌ Login Failed.")
        exit(1)
except Exception as e:
    print("❌ Authentication Error:", e)
    exit(1)

# ---------------------------------------------------------
# 2. LOAD HISTORICAL CACHE & INDUSTRY MAPPING
# ---------------------------------------------------------
hist_file = "industry_historical_cache.parquet"
industry_cache_file = "master_stock_industry.parquet"

if not os.path.exists(hist_file):
    print("❌ Historical cache not found. Please run the EOD update first to build the base.")
    exit(1)

df_hist = pd.read_parquet(hist_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date'])

if os.path.exists(industry_cache_file):
    df_ind_master = pd.read_parquet(industry_cache_file)
    sym_to_industry = dict(zip(df_ind_master['Symbol'], df_ind_master['Industry']))
else:
    sym_to_industry = {}

# ---------------------------------------------------------
# 3. BATCH FETCH LIVE QUOTES
# ---------------------------------------------------------
print("Fetching Scrip Master Tokens...")
scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(scrip_url, timeout=30).json()

df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))]
symbol_to_token = dict(zip(nse_stocks['symbol'], nse_stocks['token']))

all_tokens = list(symbol_to_token.values())
chunk_size = 40
chunks = [all_tokens[i:i + chunk_size] for i in range(0, len(all_tokens), chunk_size)]

ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
now_ist = datetime.datetime.now(ist_offset)
today_dt = pd.to_datetime(now_ist.strftime("%Y-%m-%d"))
time_str = now_ist.strftime("%I:%M %p")

new_rows = []
print(f"Streaming live quotes for {len(all_tokens)} scrips...")
for chunk in chunks:
    for attempt in range(3):
        try:
            params = {"mode": "FULL", "exchangeTokens": {"NSE": chunk}}
            res = smartApi.getMarketData(params["mode"], params["exchangeTokens"])
            if res and isinstance(res, dict):
                if res.get('errorcode') == 'AB1021':
                    time.sleep(2)
                    continue
                if res.get('status') and res.get('data'):
                    for item in res['data'].get('fetched', []):
                        sym = item.get('tradingSymbol', '')
                        new_rows.append({
                            "Date": today_dt,
                            "Symbol": sym,
                            "Industry": sym_to_industry.get(sym, "Emerging Equities"),
                            "Open": float(item.get('open', 0)),
                            "High": float(item.get('high', 0)),
                            "Low": float(item.get('low', 0)),
                            "Close": float(item.get('ltp', 0)),
                            "Volume": float(item.get('tradeVolume', 0) or item.get('totBuyQuan', 0))
                        })
                    break
        except Exception:
            time.sleep(1)
    time.sleep(0.10)

if not new_rows:
    print("❌ No live data fetched.")
    exit(1)

df_live = pd.DataFrame(new_rows)

# ---------------------------------------------------------
# 4. OVERLAY LIVE INTRADAY ON CACHE & CALCULATE MATH
# ---------------------------------------------------------
df_hist = df_hist[df_hist['Date'] != today_dt]
df_combined = pd.concat([df_hist, df_live], ignore_index=True)
df_combined = df_combined.sort_values(['Symbol', 'Date']).reset_index(drop=True)

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

df_combined['Above_20_EMA'] = df_combined['Close'] > df_combined['EMA_20']
df_combined['Above_50_EMA'] = df_combined['Close'] > df_combined['EMA_50']
df_combined['Above_200_EMA'] = df_combined['Close'] > df_combined['EMA_200']
df_combined['Up_4_Pct'] = df_combined['Daily_Pct'] >= 4.0

# ---------------------------------------------------------
# 5. AGGREGATE LIVE INDUSTRY MATRIX
# ---------------------------------------------------------
latest_df = df_combined[df_combined['Date'] == today_dt].copy()

latest_df[['Symbol', 'Industry', 'Close', 'Daily_Pct', 'Volume', 'Vol_20D_Avg', 'Dist_52W_High', 'Above_20_EMA', 'Above_50_EMA', 'Above_200_EMA', 'Vol_Shock']].to_parquet("latest_stocks_snapshot.parquet", index=False)

matrix = latest_df.groupby('Industry').agg(
    Total_Stocks=('Symbol', 'count'),
    Avg_Daily_Gain=('Daily_Pct', 'mean'),
    Above_20=('Above_20_EMA', 'sum'),
    Above_50=('Above_50_EMA', 'sum'),
    Above_200=('Above_200_EMA', 'sum'),
    Up_4_Count=('Up_4_Pct', 'sum'),
    Vol_Shock_Count=('Vol_Shock', 'sum'),
    Near_52W_Count=('Near_52W_High', 'sum')
).reset_index()

matrix = matrix[matrix['Total_Stocks'] >= 3].copy()

matrix['Pct_Above_20'] = (matrix['Above_20'] / matrix['Total_Stocks'] * 100).round(1)
matrix['Pct_Above_50'] = (matrix['Above_50'] / matrix['Total_Stocks'] * 100).round(1)
matrix['Pct_Above_200'] = (matrix['Above_200'] / matrix['Total_Stocks'] * 100).round(1)
matrix['Pct_Up_4'] = (matrix['Up_4_Count'] / matrix['Total_Stocks'] * 100).round(1)
matrix['Pct_Near_52W'] = (matrix['Near_52W_Count'] / matrix['Total_Stocks'] * 100).round(1)
matrix['Volume_Shocks'] = matrix['Vol_Shock_Count']
matrix['Avg_Daily_Gain'] = matrix['Avg_Daily_Gain'].round(2)

matrix['Thrust_Score'] = (
    (matrix['Pct_Above_20'] * 0.30) +
    (matrix['Pct_Above_50'] * 0.25) +
    (matrix['Pct_Above_200'] * 0.20) +
    (matrix['Pct_Near_52W'] * 0.15) +
    ((matrix['Volume_Shocks'] / matrix['Total_Stocks'] * 100).clip(upper=100) * 0.10)
).round(0).astype(int)

matrix = matrix.sort_values('Thrust_Score', ascending=False).reset_index(drop=True)
matrix.to_csv("industry_breadth_matrix.csv", index=False)

with open("last_sync.txt", "w") as f:
    f.write(f"Today, {time_str} IST (⚡ LIVE INTRADAY)")

print(f"✅ Live Intraday Matrix updated successfully at {time_str} IST.")
