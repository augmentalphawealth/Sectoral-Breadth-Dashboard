import pandas as pd
import requests
import datetime
import pyotp
import os
import time
import sys
from SmartApi import SmartConnect

print("Logging in to Angel One for LIVE INTRADAY Sync...")
api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if session['status'] == False:
        print("❌ Login Failed! Verify your GitHub Secrets credentials.")
        sys.exit(1)
except Exception as e:
    print("❌ Error logging in:", e)
    sys.exit(1)

# Fetch Scrip Master for NSE EQ
print("Fetching Scrip Master...")
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(url).json()
nse_tokens = [item['token'] for item in scrip_master if item['exch_seg'] == 'NSE' and item['symbol'].endswith('-EQ')]

# Batching to bypass firewall/rate limits safely
chunk_size = 25
chunks = [nse_tokens[i:i + chunk_size] for i in range(0, len(nse_tokens), chunk_size)]

advances = 0
declines = 0
unchanged = 0

print(f"Fetching Live Prices for {len(nse_tokens)} stocks in batches...")

for i, chunk in enumerate(chunks):
    for attempt in range(5):
        try:
            params = {"mode": "FULL", "exchangeTokens": {"NSE": chunk}}
            res = smartApi.getMarketData(params["mode"], params["exchangeTokens"])
            
            if res and isinstance(res, dict):
                if res.get('errorcode') == 'AB1021':
                    time.sleep(2) 
                    continue
                
                if res.get('status') and res.get('data'):
                    fetched_data = res['data'].get('fetched', [])
                    for item in fetched_data:
                        ltp = float(item.get('ltp', 0))
                        prev_close = float(item.get('close', 0))
                        
                        # Categorize market breadth
                        if ltp > prev_close:
                            advances += 1
                        elif ltp < prev_close:
                            declines += 1
                        else:
                            unchanged += 1
                    break 
        except Exception:
            time.sleep(2)
            
    time.sleep(0.4) # Rate limit protection between batches
    
    if (i + 1) % 20 == 0:
        print(f"Progress: {(i + 1) * chunk_size} stocks processed...")

total_universe = advances + declines + unchanged

# Generate strictly formatted CSV output for the Dashboard to read
ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
now = datetime.datetime.now(ist_offset)

live_data = pd.DataFrame([{
    "Date": now.strftime('%Y-%m-%d'),
    "Time": now.strftime('%H:%M'),
    "Advances": advances,
    "Declines": declines,
    "Total_Universe": total_universe
}])

live_data.to_csv("live_intraday_breadth.csv", index=False)
print(f"✅ Live Sync Complete! Advances: {advances} | Declines: {declines} | Unchanged: {unchanged} | Total: {total_universe}")
