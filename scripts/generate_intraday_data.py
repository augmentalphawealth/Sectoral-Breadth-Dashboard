"""
Generate Intraday Sector Movers Data

This script runs every 30 minutes during market hours (9:15 AM - 3:30 PM IST)
to generate:
- data/processed/intraday_sector_movers.parquet
- data/processed/intraday_top_stocks.parquet

These files power the "Intraday Sector Movers" panel in the dashboard.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# NSE sectoral indices (Yahoo Finance symbols)
SECTORAL_INDICES = {
    "NIFTY BANK": "^NSXBK",
    "NIFTY IT": "^NSXIT",
    "NIFTY PHARMA": "^NSXPHAR",
    "NIFTY AUTO": "^NSXAUTO",
    "NIFTY FIN SERVICE": "^NSXFIN",
    "NIFTY METAL": "^NSXMETAL",
    "NIFTY REALTY": "^NSXREALTY",
    "NIFTY MEDIA": "^NSXMEDIA",
    "NIFTY PRIVATE BANK": "^NSXBANK",
    "NIFTY COMMODITIES": "^NSXCOMM",
    "NIFTY CONSUMPTION": "^NSXCONSUM",
    "NIFTY ENERGY": "^NSXENERGY",
    "NIFTY INFRASTRUCTURE": "^NSXINFRA",
    "NIFTY FMCG": "^NSXFMCG",
    "NIFTY HEALTHCARE": "^NSXHEALTH",
}

# Mapping from Yahoo sector names to dashboard basic_industry names
YAHOO_TO_DASHBOARD = {
    "NIFTY BANK": "Banks",
    "NIFTY IT": "IT Services",
    "NIFTY PHARMA": "Pharmaceuticals",
    "NIFTY AUTO": "Automobiles",
    "NIFTY FIN SERVICE": "Financial Services",
    "NIFTY METAL": "Metals",
    "NIFTY REALTY": "Real Estate",
    "NIFTY MEDIA": "Media & Entertainment",
    "NIFTY PRIVATE BANK": "Private Banks",
    "NIFTY COMMODITIES": "Commodities",
    "NIFTY CONSUMPTION": "Consumer Goods",
    "NIFTY ENERGY": "Energy",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "NIFTY FMCG": "FMCG",
    "NIFTY HEALTHCARE": "Healthcare",
}


def get_intraday_sector_data() -> pd.DataFrame:
    """
    Fetch intraday performance for all NSE sectoral indices.
    Returns DataFrame with sector performance metrics.
    """
    print("Fetching intraday sector data...")
    
    sector_data = []
    
    for sector_name, yahoo_symbol in SECTORAL_INDICES.items():
        try:
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period="5d", interval="15m")
            
            if hist.empty:
                print(f"  ⚠️  No data for {sector_name} ({yahoo_symbol})")
                continue
            
            # Get latest close and previous day's close
            latest_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest_close
            
            # Calculate intraday return
            intraday_return = (latest_close - prev_close) / prev_close if prev_close > 0 else 0
            
            # Get today's high and low
            today_high = hist["High"].max()
            today_low = hist["Low"].min()
            
            sector_data.append({
                "basic_industry": YAHOO_TO_DASHBOARD.get(sector_name, sector_name),
                "intraday_return": intraday_return,
                "latest_close": latest_close,
                "day_high": today_high,
                "day_low": today_low,
                "volume": hist["Volume"].iloc[-1] if "Volume" in hist.columns else 0,
            })
            
            print(f"  ✅ {sector_name}: {intraday_return*100:+.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error fetching {sector_name}: {e}")
            continue
    
    if not sector_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(sector_data)
    
    # Add rankings and scores
    df = df.sort_values("intraday_return", ascending=False).reset_index(drop=True)
    df["sector_rank"] = df.index + 1
    
    # Calculate intraday strength score (0-100 scale)
    max_return = df["intraday_return"].max()
    min_return = df["intraday_return"].min()
    range_return = max_return - min_return if max_return != min_return else 1
    
    df["intraday_strength_score"] = ((df["intraday_return"] - min_return) / range_return * 100).round(2)
    
    return df


def get_top_stocks_by_sector() -> pd.DataFrame:
    """
    Get top gaining stocks in each sector for intraday view.
    This is a simplified version - in production you'd fetch from your actual stock data.
    """
    print("Fetching top stocks by sector...")
    
    # For now, return empty DataFrame with expected columns
    # In production, this would query your stock database
    columns = [
        "basic_industry", "Symbol", "Daily_Pct", "Volume_Surge", 
        "Is_Breakout", "Sector_Rank"
    ]
    
    return pd.DataFrame(columns=columns)


def is_market_hours() -> bool:
    """Check if current time is within NSE market hours (9:15 AM - 3:30 PM IST)"""
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now = datetime.datetime.now(IST)
    
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    
    # Check if weekday
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    return market_open <= now.time() <= market_close


def main() -> None:
    print("=" * 60)
    print("Intraday Sector Data Generator")
    print("=" * 60)
    
    # Check market hours
    if not is_market_hours():
        print("⚠️  Outside market hours. Skipping intraday data generation.")
        print("   Market hours: 9:15 AM - 3:30 PM IST, Monday-Friday")
        return
    
    print(f"\n🕐 Market is OPEN. Generating intraday data...")
    print(f"   Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directory exists
    PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Generate sector data
    sector_df = get_intraday_sector_data()
    
    if sector_df.empty:
        print("\n❌ No sector data retrieved. Exiting.")
        return
    
    # Generate top stocks data
    stocks_df = get_top_stocks_by_sector()
    
    # Save to parquet
    sector_file = PROCESSED / "intraday_sector_movers.parquet"
    stocks_file = PROCESSED / "intraday_top_stocks.parquet"
    
    sector_df.to_parquet(sector_file, index=False)
    stocks_df.to_parquet(stocks_file, index=False)
    
    print(f"\n✅ Successfully generated:")
    print(f"   📁 {sector_file}")
    print(f"   📁 {stocks_file}")
    print(f"\n   Sectors: {len(sector_df)}")
    print(f"   Top stocks: {len(stocks_df)}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
