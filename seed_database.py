import pandas as pd
import requests
import time
import datetime
import os
import sys
from io import StringIO
from SmartApi import SmartConnect
import pyotp
import logzero
import logging

logzero.logger.setLevel(logging.FATAL)

print("=========================================================")
print("  ONE-TIME CLOUD DATABASE SEEDER (1-YEAR HISTORY)")
print("=========================================================")

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
    sys.exit(1)

try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if not session.get('status'):
        print("❌ Login Failed.")
        sys.exit(1)
    print("✅ Angel One Authentication Successful.")
except Exception as e:
    print("❌ Auth Error:", e)
    sys.exit(1)

# --- FETCH MASTER LIST & NSE REGISTRY ---
print("Fetching Scrip Master & NSE Registry...")
scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(scrip_url, timeout=30).json()
df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))]
tokens_to_fetch = nse_stocks[['symbol', 'token']].to_dict('records')

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

# --- FETCH 1 YEAR OF DATA ---
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)
from_date_str = start_date.strftime("%Y-%m-%d 09:15")
to_date_str = end_date.strftime("%Y-%m-%d 15:30")

all_ohlc_data = []
total_stocks = len(tokens_to_fetch)

print(f"Downloading historical data for {total_stocks} stocks. This will take ~15 minutes...")

for i, stock in enumerate(tokens_to_fetch):
    for attempt in range(3):
        try:
            params = {
                "exchange": "NSE",
                "symboltoken": stock['token'],
                "interval": "ONE_DAY",
                "fromdate": from_date_str, 
                "todate": to_date_str
            }
            hist_data = smartApi.getCandleData(params)
            
            if hist_data and hist_data.get('status') == False:
                if hist_data.get('errorcode') == 'AB1021':
                    time.sleep(2)
                    continue 
            
            if hist_data and hist_data.get('data'):
                df_temp = pd.DataFrame(hist_data['data'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df_temp['Symbol'] = stock['symbol']
                
                # Apply Intelligent Industry Mapping
                raw_ind = nse_map.get(stock['symbol'], "")
                df_temp['Industry'] = refine_industry(raw_ind, stock['symbol'])
                
                all_ohlc_data.append(df_temp)
            break 
            
        except Exception:
            time.sleep(1) 
            
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1} / {total_stocks} stocks fetched...")
        
    time.sleep(0.4) # Strict rate limiting protection

# --- SAVE & COMPILE ---
if all_ohlc_data:
    final_df = pd.concat(all_ohlc_data, ignore_index=True)
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp']).dt.normalize()
    final_df = final_df.rename(columns={'Timestamp': 'Date'})
    
    # Extract unique stocks and their refined industries
    master_ind = final_df[['Symbol', 'Industry']].drop_duplicates(subset=['Symbol'], keep='last')
    master_ind.to_parquet("master_stock_industry.parquet", index=False)
    
    # Save the huge price history file
    output_file = "industry_historical_cache.parquet"
    final_df.to_parquet(output_file, index=False)
    
    print(f"✅ SUCCESS! 1-Year Database generated and saved to {output_file}.")
else:
    print("❌ Failed to fetch data.")
    sys.exit(1)
