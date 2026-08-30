from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INTRADAY_STOCK_FILE = Path("../Situational-Awareness/dashboard_data.parquet").resolve()
MASTER_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"
OUTPUT_FILE = PROCESSED / "intraday_sector_movers.parquet"

TODAY = pd.Timestamp.today().normalize()


def main() -> None:
    if not INTRADAY_STOCK_FILE.exists():
        raise FileNotFoundError(
            f"Missing intraday stock data: {INTRADAY_STOCK_FILE}
"
            "Run Situational-Awareness intraday workflow first."
        )

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Missing master classification: {MASTER_FILE}
"
            "Run EOD workflow first."
        )

    print("========== INTRADAY SECTOR PANEL BUILD START ==========")

    # Read intraday stock data
    intraday = pd.read_parquet(INTRADAY_STOCK_FILE)
    print(f"Intraday stocks: {len(intraday)}")

    # Read master classification
    master = pd.read_parquet(MASTER_FILE)
    print(f"Master classifications: {len(master)}")

    # Filter to today's intraday data
    intraday["Date"] = pd.to_datetime(intraday["Date"]).dt.normalize()
    intraday_today = intraday[intraday["Date"] == TODAY].copy()

    if intraday_today.empty:
        raise ValueError(f"No intraday data found for {TODAY.date()}")

    # Merge with sector classification
    intraday_today = intraday_today.merge(
        master[["symbol", "basic_industry", "industry", "sector"]].drop_duplicates(subset=["symbol"]),
        left_on="Symbol",
        right_on="symbol",
        how="left",
    )

    # Fill unclassified
    for col in ["basic_industry", "industry", "sector"]:
        intraday_today[col] = intraday_today[col].fillna("Unclassified")

    # Calculate sector-level metrics
    sector_metrics = intraday_today.groupby("basic_industry").agg(
        members=("Symbol", "nunique"),
        avg_intraday_return=("Daily_Pct", "mean"),
        median_intraday_return=("Daily_Pct", "median"),
        pct_gainers=("Gainer", "mean"),  # % of stocks that are gainers
        pct_above_20_ema=("Above_20_EMA", "mean"),
        pct_above_50_ema=("Above_50_EMA", "mean"),
        pct_above_200_ema=("Above_200_EMA", "mean"),
        volume_surge_count=("Volume_Surge", "sum"),
        volume_surge_pct=("Volume_Surge", "mean"),
        breakout_count=("Is_Breakout", "sum"),
        breakout_pct=("Is_Breakout", "mean"),
        total_up_volume=("Up_Volume", "sum"),
        total_down_volume=("Down_Volume", "sum"),
        new_high_count=("New_52W_High", "sum"),
        new_low_count=("New_52W_Low", "sum"),
    ).reset_index()

    # Calculate net metrics
    sector_metrics["net_volume"] = sector_metrics["total_up_volume"] - sector_metrics["total_down_volume"]
    sector_metrics["net_breakouts"] = sector_metrics["breakout_count"]
    sector_metrics["net_new_highs"] = sector_metrics["new_high_count"] - sector_metrics["new_low_count"]

    # Calculate intraday strength score
    # Weight: 40% return, 30% volume surge, 20% breakout, 10% breadth
    sector_metrics["intraday_strength_score"] = (
        (sector_metrics["avg_intraday_return"].rank(pct=True) * 40)
        + (sector_metrics["volume_surge_pct"].rank(pct=True) * 30)
        + (sector_metrics["breakout_pct"].rank(pct=True) * 20)
        + (sector_metrics["pct_gainers"].rank(pct=True) * 10)
    )

    # Rank sectors
    sector_metrics = sector_metrics.sort_values(
        "intraday_strength_score",
        ascending=False,
    ).reset_index(drop=True)
    sector_metrics["sector_rank"] = range(1, len(sector_metrics) + 1)

    # Add stock lists for top 10 sectors
    top_10_sectors = sector_metrics.head(10)["basic_industry"].tolist()
    top_stocks = intraday_today[intraday_today["basic_industry"].isin(top_10_sectors)].copy()
    top_stocks = top_stocks.sort_values(
        ["basic_industry", "Daily_Pct"],
        ascending=[True, False],
    )

    # Save outputs
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sector_metrics.to_parquet(OUTPUT_FILE, index=False)

    # Also save top stocks for dashboard
    top_stocks_file = PROCESSED / "intraday_top_stocks.parquet"
    top_stocks.to_parquet(top_stocks_file, index=False)

    print("========== INTRADAY SECTOR PANEL BUILD COMPLETE ==========")
    print(f"Sectors: {len(sector_metrics)}")
    print(f"Top stocks: {len(top_stocks)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Top stocks: {top_stocks_file}")

    # Print top 5 sectors
    print("
Top 5 Sectors by Intraday Strength:")
    for _, row in sector_metrics.head(5).iterrows():
        print(f"  {row['sector_rank']}. {row['basic_industry']}: "
              f"{row['avg_intraday_return']:.2f}% return, "
              f"{row['volume_surge_pct']*100:.1f}% volume surge, "
              f"{row['breakout_count']} breakouts")


if __name__ == "__main__":
    main()
