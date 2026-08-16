import pandas as pd
import numpy as np
import requests
import datetime
import pyotp
import os
import time
from io import StringIO
from SmartApi import SmartConnect
import logzero
import logging

logzero.logger.setLevel(logging.FATAL)

# --- THE INSTITUTIONAL INDUSTRY REFINEMENT ENGINE ---
def refine_industry(raw_industry, symbol):
    ind = str(raw_industry).strip().upper()
    sym = str(symbol).replace("-EQ", "").strip().upper()
    
    # 1. Surgical Split for the Massive Financial Sector
    if ind in ["FINANCIAL SERVICES", "BANKS", "FINANCE"]:
        psu_banks = {"SBIN", "PNB", "BOB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "CENTRALBK", "IOB", "UCOBANK", "MAHABANK", "PSB"}
        is_bank = "BANK" in sym or "BANC" in sym or sym in psu_banks or sym in {"HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK"}
        is_insurance = "INSUR" in sym or "LIFE" in sym or "ASSURANCE" in sym or sym in {"LIC", "GICRE", "LICI", "NEWINDIA"}
        
        if is_bank: return "PSU Bank" if sym in psu_banks else "Private Bank"
        elif is_insurance: return "Insurance"
        else: return "NBFC"
        
    # 2. Algorithmic Fallback for blank/missing NSE data
    if ind in ["", "NAN", "NONE", "EMERGING EQUITIES"]:
        for kw, tag in [("BANK", "Private Bank"), ("FIN", "NBFC"), ("TECH", "Software & IT"), ("AUTO", "Automobiles & Parts"), ("PHARMA", "Pharmaceuticals"), ("CHEM", "Specialty Chemicals"), ("POWER", "Power Generation"), ("METAL", "Metals & Mining")]:
            if kw in sym: return tag
        return "Emerging Equities"
        
    # 3. Clean Formatting for everything else
    return str(raw_industry).title()

# --- ANGEL ONE AUTHENTICATION ---
api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

if not all([api_key, client_code, login_pin, totp_secret]):
    print("❌ Missing Angel One credentials.")
    exit(1)

try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if not session.get('status'):
        print("❌ Login Failed.")
        exit(1)
except Exception as e:
    print("❌ Auth Error:", e)
    exit(1)

print("Fetching Angel One Scrip Master...")
scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(scrip_url, timeout=30).json()
df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))].copy()
symbol_to_token = dict(zip(nse_stocks['symbol'], nse_stocks['token']))
all_symbols = list(symbol_to_token.keys())

# --- INCREMENTAL INDUSTRY TAXONOMY SYNC ---
industry_cache_file = "master_stock_industry.parquet"
if os.path.exists(industry_cache_file):
    df_ind = pd.read_parquet(industry_cache_file)
    cached_symbols = set(df_ind['Symbol'])
else:
    df_ind = pd.DataFrame(columns=['Symbol', 'Industry'])
    cached_symbols = set()

new_listings = set(all_symbols) - cached_symbols
if new_listings:
    print(f"🆕 Syncing {len(new_listings)} new listings (IPOs) against NSE Registry...")
    nse_map = {}
    try:
        nse_res = requests.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if nse_res.status_code == 200:
            nse_csv = pd.read_csv(StringIO(nse_res.text))
            for _, r in nse_csv.iterrows():
                ind = str(r.get('BASIC_INDUSTRY') or r.get('INDUSTRY') or '').strip()
                if ind: nse_map[f"{str(r.get('SYMBOL')).strip()}-EQ"] = ind
    except Exception:
        pass

    new_rows = []
    for sym in new_listings:
        raw_ind = nse_map.get(sym, "")
        # Apply the smart refinement engine on new IPOs immediately
        refined_ind = refine_industry(raw_ind, sym)
        new_rows.append({"Symbol": sym, "Industry": refined_ind})
    
    df_ind = pd.concat([df_ind, pd.DataFrame(new_rows)], ignore_index=True)
    df_ind.to_parquet(industry_cache_file, index=False)

sym_to_ind = dict(zip(df_ind['Symbol'], df_ind['Industry']))

# --- BATCH EOD PRICE DATA FETCH ---
# Ensure timestamp is set correctly in IST timezone, removing intraday time 
ist_timezone = datetime.timezone(datetime.timedelta(hours=5, 30))
today_dt = pd.to_datetime(datetime.datetime.now(ist_timezone).strftime("%Y-%m-%d"))

all_tokens = list(symbol_to_token.values())
chunks = [all_tokens[i:i + 35] for i in range(0, len(all_tokens), 35)]
fetched_data = []

print("Fetching EOD Quotes...")
for chunk in chunks:
    for attempt in range(4):
        try:
            res = smartApi.getMarketData("FULL", {"NSE": chunk})
            if res and isinstance(res, dict) and res.get('status') and res.get('data'):
                for item in res['data'].get('fetched', []):
                    sym = item.get('tradingSymbol', '')
                    if not sym: continue
                    fetched_data.append({
                        "Date": today_dt,
                        "Symbol": sym,
                        "Industry": sym_to_ind.get(sym, refine_industry("", sym)), # Strict fallback ensuring no blanks
                        "Open": float(item.get('open', 0)),
                        "High": float(item.get('high', 0)),
                        "Low": float(item.get('low', 0)),
                        "Close": float(item.get('ltp', 0)),
                        "Volume": float(item.get('tradeVolume', 0) or 0)
                    })
                break
        except Exception:
            time.sleep(1.5)
    time.sleep(0.12)

if not fetched_data:
    print("❌ Failed to fetch current market snapshot.")
    exit(1)
df_today = pd.DataFrame(fetched_data)

# --- DATA MERGE & DUPLICATION SHIELD ---
hist_file = "industry_historical_cache.parquet"
if not os.path.exists(hist_file) and os.path.exists("nse_6yr_historical.parquet"):
    print("Migrating legacy 6-year history to new cache format...")
    df_hist = pd.read_parquet("nse_6yr_historical.parquet")
    df_hist['Industry'] = df_hist['Symbol'].map(sym_to_ind).fillna("Emerging Equities")
elif os.path.exists(hist_file):
    df_hist = pd.read_parquet(hist_file)
else:
    df_hist = pd.DataFrame()

if not df_hist.empty:
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])
    # Strict anti-duplication shield: drops prior pulls from today
    df_hist = df_hist[df_hist['Date'] != today_dt]
    df_combined = pd.concat([df_hist, df_today], ignore_index=True)
else:
    df_combined = df_today.copy()

# Keep 300 days memory footprint to maintain high server speed
cutoff_date = today_dt - pd.Timedelta(days=300)
df_combined = df_combined[df_combined['Date'] >= cutoff_date]
df_combined = df_combined.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# --- MATHEMATICAL ENGINE ---
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

# Save updated cache safely
df_combined.to_parquet(hist_file, index=False)

# --- INDUSTRY BREADTH AGGREGATION MATRIX ---
latest_df = df_combined[df_combined['Date'] == today_dt].copy()

# Lightweight file specifically generated for ultra-fast Streamlit drill-down clicks
latest_df[['Symbol', 'Industry', 'Close', 'Daily_Pct', 'Volume', 'Vol_20D_Avg', 'Dist_52W_High', 'Above_20', 'Above_50', 'Above_200', 'Vol_Shock']].to_parquet("latest_stocks_snapshot.parquet", index=False)

matrix = latest_df.groupby('Industry').agg(
    Total_Stocks=('Symbol', 'count'),
    Avg_Daily_Gain=('Daily_Pct', 'mean'),
    Above_20=('Above_20', 'sum'),
    Above_50=('Above_50', 'sum'),
    Above_200=('Above_200', 'sum'),
    Vol_Shock_Count=('Vol_Shock', 'sum'),
    Near_52W_Count=('Near_52W_High', 'sum')
).reset_index()

# Drop meaningless noise (e.g. 1 random stock miscategorized)
matrix = matrix[matrix['Total_Stocks'] >= 3].copy()

for col, agg_col in [('Pct_Above_20', 'Above_20'), ('Pct_Above_50', 'Above_50'), ('Pct_Above_200', 'Above_200'), ('Pct_Near_52W', 'Near_52W_Count')]:
    matrix[col] = (matrix[agg_col] / matrix['Total_Stocks'] * 100).round(1)

# Core Institutional Thrust Algorithm (Max 100 Score)
matrix['Thrust_Score'] = (
    (matrix['Pct_Above_20'] * 0.30) + 
    (matrix['Pct_Above_50'] * 0.25) + 
    (matrix['Pct_Above_200'] * 0.20) + 
    (matrix['Pct_Near_52W'] * 0.15) + 
    ((matrix['Vol_Shock_Count'] / matrix['Total_Stocks'] * 100).clip(upper=100) * 0.10)
).round(0).astype(int)

matrix = matrix.sort_values('Thrust_Score', ascending=False).reset_index(drop=True)
matrix.to_csv("industry_breadth_matrix.csv", index=False)

with open("last_sync.txt", "w") as f:
    f.write(datetime.datetime.now(ist_timezone).strftime('%d %b %Y, %I:%M %p IST (EOD Sync)'))

print("✅ EOD Run Complete. Matrix generated successfully.")
