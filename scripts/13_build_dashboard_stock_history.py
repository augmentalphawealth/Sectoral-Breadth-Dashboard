# scripts/13_build_dashboard_stock_history.py
# Produces a compact stock history source for GitHub snapshot generation.
# Streamlit must not load this file. It is a GitHub-Actions-only intermediate.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INPUT_FILE = PROCESSED / "stock_daily_features.parquet"
OUTPUT_FILE = PROCESSED / "dashboard_stock_history.parquet"
METADATA_FILE = PROCESSED / "dashboard_stock_history_metadata.json"

# Retain the existing rolling window. Historical dashboard viewing becomes fast
# because Streamlit will use daily snapshots, not this combined history file.
RECENT_TRADING_DAYS = 200

DISPLAY_COLUMNS = [
    "date",
    "symbol",
    "sector",
    "industry",
    "basic_industry",
    "close",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "above_20",
    "above_50",
    "above_200",
    "dist_52w_high",
    "trend_template_pass",
    "acc_day",
    "dist_day",
    "breakout_55",
    "vcp_ready",
    "stock_strength_score",
    "high_strength_flag",
    "established_buy_setup",
    "ipo_buy_setup",
    "buy_setup_score",
    "gain_6m",
    "daily_range",
    "vol_ratio_50",
    "vol_2x_count_6m",
    "up_down_ratio",
    "atr_14",
    "range_3d",
    "tight_3d_range",
    "ipo_turnover_avg",
    "actionable_setup_pass",
    "setup_precision_score",
    "nearest_ema_tag",
    "momentum_badge",
    "ipo_setup_score",
    "vwap_premium",
    "retracement_from_listing_high",
    "days_listed",
    "ipo_phase",
    "hh_hl_streak_5d",
]


def clean_group(series: pd.Series) -> pd.Series:
    return series.fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")


def main() -> None:
    print("========== STOCK HISTORY BUILD START ==========")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {INPUT_FILE}")

    source = pd.read_parquet(INPUT_FILE)
    required = ["date", "symbol", "sector", "industry", "basic_industry"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Stock feature file missing required columns: {missing}")

    available_columns = [column for column in DISPLAY_COLUMNS if column in source.columns]
    data = source[available_columns].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].fillna("").astype(str).str.strip()
    for column in ["sector", "industry", "basic_industry"]:
        data[column] = clean_group(data[column])

    data = data.dropna(subset=["date"]).copy()
    data = data[data["symbol"] != ""].copy()
    data = data.drop_duplicates(subset=["date", "symbol"], keep="last")

    dates = sorted(pd.Timestamp(date) for date in data["date"].dropna().unique())
    if not dates:
        raise ValueError("No valid dates found in stock features file")
    kept_dates = dates[-RECENT_TRADING_DAYS:]
    data = data[data["date"].isin(kept_dates)].copy()

    numeric_columns = [
        "stock_strength_score", "buy_setup_score", "setup_precision_score",
        "ipo_setup_score", "ret_20d", "ret_60d", "gain_6m", "vol_ratio_50",
        "tight_3d_range", "daily_range", "up_down_ratio",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    group_keys = ["date", "basic_industry"]
    if "stock_strength_score" in data.columns:
        data["stock_rank_in_basic_industry"] = (
            data.groupby(group_keys)["stock_strength_score"]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )
        data["stock_strength_percentile_in_basic_industry"] = (
            data.groupby(group_keys)["stock_strength_score"].rank(pct=True, ascending=True) * 100.0
        ).round(1)

    if "high_strength_flag" in data.columns:
        data["high_strength_flag"] = pd.to_numeric(data["high_strength_flag"], errors="coerce").fillna(0).astype(int)
        data["basic_industry_members"] = data.groupby(group_keys)["symbol"].transform("nunique")
        data["high_strength_count"] = data.groupby(group_keys)["high_strength_flag"].transform("sum")
        data["pct_high_strength"] = (
            data["high_strength_count"] / data["basic_industry_members"].clip(lower=1) * 100.0
        ).round(1)

    sort_columns = ["date", "sector", "industry", "basic_industry"]
    if "stock_rank_in_basic_industry" in data.columns:
        sort_columns.append("stock_rank_in_basic_industry")
    sort_columns.append("symbol")
    data = data.sort_values(sort_columns).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_FILE, index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rows": int(len(data)),
        "symbols": int(data["symbol"].nunique()),
        "dates": int(len(kept_dates)),
        "start_date": kept_dates[0].strftime("%Y-%m-%d"),
        "latest_date": kept_dates[-1].strftime("%Y-%m-%d"),
        "streamlit_usage": "Do not load in Streamlit; use per-date dashboard snapshots instead.",
    }
    METADATA_FILE.write_text(pd.Series(metadata).to_json(indent=2), encoding="utf-8")

    print("========== STOCK HISTORY BUILD COMPLETE ==========")
    print(f"Rows: {metadata['rows']:,}")
    print(f"Symbols: {metadata['symbols']:,}")
    print(f"Dates: {metadata['dates']:,} ({metadata['start_date']} to {metadata['latest_date']})")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
