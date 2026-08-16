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

# 1. Fetch Master List
print("Fetching Scrip Master...")
scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(scrip_url, timeout=30).json()
df_scrip = pd.DataFrame(scrip_master)
nse_stocks = df_scrip[(df_scrip['exch_seg'] == 'NSE') & (df_scrip['symbol'].str.endswith('-EQ'))]
tokens_to_fetch = nse_stocks[['symbol', 'token']].to_dict('records')

# 2. Map Industries
print("Mapping Industries...")
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

# 3. Fetch 1 Year of Data (Needed for 200 EMA & 52W High)
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

from_date_str = start_date.strftime("%Y-%m-%d 09:15")
to_date_str = end_date.strftime("%Y-%m-%d 15:30")

all_ohlc_data = []
total_stocks = len(tokens_to_fetch)

print(f"Downloading historical data for {total_stocks} stocks. This will take ~15 minutes to avoid rate limits...")

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
                
                # Apply Industry mapping or fallback
                clean_name = stock['symbol'].replace("-EQ", "")
                ind = nse_map.get(stock['symbol'])
                if not ind:
                    for kw, tag in [("BANK", "Banking"), ("FIN", "Financial Services"), ("TECH", "Software & IT"), ("AUTO", "Automobiles"), ("PHARMA", "Pharmaceuticals"), ("CHEM", "Chemicals"), ("POWER", "Power"), ("METAL", "Metals")]:
                        if kw in clean_name:
                            ind = tag
                            break
                df_temp['Industry'] = ind or "Emerging Equities"
                
                all_ohlc_data.append(df_temp)
            break 
            
        except Exception:
            time.sleep(1) 
            
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1} / {total_stocks} stocks fetched...")
        
    time.sleep(0.4) # Strict pause to respect Angel One rate limits

# 4. Save and Compile
if all_ohlc_data:
    final_df = pd.concat(all_ohlc_data, ignore_index=True)
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp']).dt.normalize()
    final_df = final_df.rename(columns={'Timestamp': 'Date'})
    
    # Save the master industry mapping file too
    master_ind = final_df[['Symbol', 'Industry']].drop_duplicates()
    master_ind.to_parquet("master_stock_industry.parquet", index=False)
    
    # Save historical cache
    output_file = "industry_historical_cache.parquet"
    final_df.to_parquet(output_file, index=False)
    
    print(f"✅ SUCCESS! Database generated and saved to {output_file}.")
else:
    print("❌ Failed to fetch data.")
    sys.exit(1)
