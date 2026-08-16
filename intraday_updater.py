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

hist_file = "industry_historical_cache.parquet"
if not os.path.exists(hist_file):
    print("No cache found. Run EOD first.")
    exit(1)
df_hist = pd.read_parquet(hist_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date'])

ind_master = pd.read_parquet("master_stock_industry.parquet") if os.path.exists("master_stock_industry.parquet") else pd.DataFrame()
sym_to_ind = dict(zip(ind_master['Symbol'], ind_master['Industry'])) if not ind_master.empty else {}

scrip_master = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json").json()
df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))]
tokens = list(dict(zip(nse_stocks['symbol'], nse_stocks['token'])).values())
chunks = [tokens[i:i + 35] for i in range(0, len(tokens), 35)]

now_ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, 30)))
today_dt = pd.to_datetime(now_ist.strftime("%Y-%m-%d"))
fetched = []

for chunk in chunks:
    for attempt in range(3):
        try:
            res = smartApi.getMarketData("FULL", {"NSE": chunk})
            if res and isinstance(res, dict) and res.get('status') and res.get('data'):
                for item in res['data'].get('fetched', []):
                    sym = item.get('tradingSymbol', '')
                    fetched.append({
                        "Date": today_dt, "Symbol": sym, "Industry": sym_to_ind.get(sym, "Emerging Equities"),
                        "Open": float(item.get('open', 0)), "High": float(item.get('high', 0)),
                        "Low": float(item.get('low', 0)), "Close": float(item.get('ltp', 0)),
                        "Volume": float(item.get('tradeVolume', 0) or 0)
                    })
                break
        except Exception: time.sleep(1)
    time.sleep(0.1)

if not fetched: exit(1)
df_live = pd.DataFrame(fetched)

df_hist = df_hist[df_hist['Date'] != today_dt]
df_combined = pd.concat([df_hist, df_live], ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)

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

latest_df = df_combined[df_combined['Date'] == today_dt].copy()
latest_df[['Symbol', 'Industry', 'Close', 'Daily_Pct', 'Volume', 'Vol_20D_Avg', 'Dist_52W_High', 'Above_20', 'Above_50', 'Above_200', 'Vol_Shock']].to_parquet("latest_stocks_snapshot.parquet", index=False)

matrix = latest_df.groupby('Industry').agg(
    Total_Stocks=('Symbol', 'count'), Avg_Daily_Gain=('Daily_Pct', 'mean'), Above_20=('Above_20', 'sum'), Above_50=('Above_50', 'sum'),
    Above_200=('Above_200', 'sum'), Vol_Shock_Count=('Vol_Shock', 'sum'), Near_52W_Count=('Near_52W_High', 'sum')
).reset_index()

matrix = matrix[matrix['Total_Stocks'] >= 3].copy()
for col, agg_col in [('Pct_Above_20', 'Above_20'), ('Pct_Above_50', 'Above_50'), ('Pct_Above_200', 'Above_200'), ('Pct_Near_52W', 'Near_52W_Count')]:
    matrix[col] = (matrix[agg_col] / matrix['Total_Stocks'] * 100).round(1)

matrix['Thrust_Score'] = ((matrix['Pct_Above_20'] * 0.30) + (matrix['Pct_Above_50'] * 0.25) + (matrix['Pct_Above_200'] * 0.20) + (matrix['Pct_Near_52W'] * 0.15) + ((matrix['Vol_Shock_Count'] / matrix['Total_Stocks'] * 100).clip(upper=100) * 0.10)).round(0).astype(int)
matrix['Avg_Daily_Gain'] = matrix['Avg_Daily_Gain'].round(2)
matrix = matrix.sort_values('Thrust_Score', ascending=False).reset_index(drop=True)
matrix.to_csv("industry_breadth_matrix.csv", index=False)

with open("last_sync.txt", "w") as f:
    f.write(now_ist.strftime('%d %b %Y, %I:%M %p IST (⚡ LIVE)'))
print("✅ Intraday Run Complete.")
