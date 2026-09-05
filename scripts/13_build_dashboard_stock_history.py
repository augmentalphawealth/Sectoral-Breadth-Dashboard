# scripts/13_build_dashboard_stock_history.py
# Extended to retain setup_precision_score, nearest_ema_tag, momentum_badge in history

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED / "stock_daily_features.parquet"
OUTPUT_FILE = PROCESSED / "dashboard_stock_history.parquet"

RECENT_TRADING_DAYS = 200  # was 400 -- that put this file at 92.5MB, ~7MB under GitHub's
                           # 100MB hard limit, with no room for new columns. If a git push
                           # of this file ever gets rejected for size, the live dashboard
                           # silently stops updating with no visible error -- same failure
                           # mode as the missing prices.parquet, just one file over.
HIGH_STRENGTH_THRESHOLD = 70.0

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    required_columns = [
        "date", "symbol", "industry", "basic_industry", "sector", "close",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d", "above_20", "above_50",
        "above_200", "dist_52w_high", "trend_template_pass", "acc_day",
        "dist_day", "breakout_55", "vcp_ready", "stock_strength_score",
        "high_strength_flag", "established_buy_setup", "ipo_buy_setup",
        "buy_setup_score", "gain_6m", "daily_range", "vol_ratio_50",
        "vol_2x_count_6m", "up_down_ratio", "atr_14", "range_3d", 
        "tight_3d_range", "ipo_turnover_avg", "actionable_setup_pass",
        "setup_precision_score", "nearest_ema_tag", "momentum_badge",
        "ipo_setup_score", "vwap_premium", "retracement_from_listing_high",
        "days_listed", "ipo_phase", "hh_hl_streak_5d",
    ]

    keep_cols = [c for c in required_columns if c in df.columns]
    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip()
    df["industry"] = df["industry"].fillna("Unclassified").astype(str).str.strip()
    df["basic_industry"] = df["basic_industry"].fillna("Unclassified").astype(str).str.strip()
    df["sector"] = df["sector"].fillna("Unclassified").astype(str).str.strip()

    unique_dates = sorted(df["date"].dropna().unique())
    if not unique_dates:
        raise ValueError("No valid dates found in stock features file")

    kept_dates = unique_dates[-RECENT_TRADING_DAYS:]
    df = df[df["date"].isin(kept_dates)].copy()
    df = df.drop_duplicates(subset=["date", "symbol"], keep="last")

    group_keys = ["date", "basic_industry"]

    if "stock_strength_score" in df.columns:
        df["stock_rank_in_basic_industry"] = (
            df.groupby(group_keys)["stock_strength_score"].rank(method="dense", ascending=False).astype("Int64")
        )
        df["stock_strength_percentile_in_basic_industry"] = (
            df.groupby(group_keys)["stock_strength_score"].rank(pct=True, ascending=True) * 100
        )

    if "high_strength_flag" in df.columns:
        df["high_strength_flag"] = df["high_strength_flag"].astype(int)
        df["basic_industry_members"] = df.groupby(group_keys)["symbol"].transform("nunique")
        df["high_strength_count"] = df.groupby(group_keys)["high_strength_flag"].transform("sum")
        df["pct_high_strength"] = df["high_strength_count"] / df["basic_industry_members"].clip(lower=1) * 100

    if "stock_rank_in_basic_industry" in df.columns:
        df = df.sort_values(["date", "basic_industry", "stock_rank_in_basic_industry", "symbol"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print("========== STOCK HISTORY BUILD COMPLETE (v2) ==========")


if __name__ == "__main__":
    main()
