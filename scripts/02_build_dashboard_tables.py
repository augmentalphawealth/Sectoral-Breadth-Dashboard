# scripts/02_build_dashboard_tables.py
# Builds latest dashboard tables and compact all-date display snapshots.

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from utils import p, read_parquet_safe, write_parquet

SMALL_GROUP_LIMIT = 5
MAX_PER_INDUSTRY = 4
TOP_BUY_COUNT = 20
IPO_COUNT = 15

SNAPSHOT_ROOT_NAME = "dashboard_snapshots"


def latest(df: pd.DataFrame) -> pd.DataFrame:
    max_date = pd.to_datetime(df["date"]).max()
    return df[pd.to_datetime(df["date"]) == max_date].copy()


def pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(pct=True, ascending=higher_is_better, na_option="keep") * 100.0


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def prepare_stock_snapshot(stock: pd.DataFrame) -> pd.DataFrame:
    data = stock.copy()
    required = [
        "date", "symbol", "industry", "basic_industry", "sector",
        "ret_20d", "ret_60d", "above_50", "above_200", "breakout_55",
        "vcp_ready", "stock_strength_score", "high_strength_flag",
        "established_buy_setup", "ipo_buy_setup", "buy_setup_score",
        "gain_6m", "daily_range", "vol_ratio_50", "vol_2x_count_6m",
        "actionable_setup_pass", "up_down_ratio", "atr_14", "range_3d",
        "tight_3d_range", "ipo_turnover_avg", "close", "nearest_ema_tag",
        "momentum_badge", "trend_template_pass", "dist_52w_high",
    ]
    require_columns(data, required, "Stock feature file")
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    for column in ["industry", "basic_industry", "sector"]:
        data[column] = data[column].fillna("Unclassified").astype(str).str.strip()
    data["symbol"] = data["symbol"].fillna("").astype(str).str.strip()
    data["stock_strength_score"] = pd.to_numeric(data["stock_strength_score"], errors="coerce")
    data["high_strength_flag"] = pd.to_numeric(data["high_strength_flag"], errors="coerce").fillna(0).astype(int)
    group_keys = ["date", "basic_industry"]
    data["basic_industry_members"] = data.groupby(group_keys)["symbol"].transform("nunique")
    data["high_strength_count_snapshot"] = data.groupby(group_keys)["high_strength_flag"].transform("sum")
    data["pct_high_strength_snapshot"] = data["high_strength_count_snapshot"] / data["basic_industry_members"].clip(lower=1) * 100.0
    return data


def add_stock_priority(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["buy_priority_score"] = (
        0.30 * pct_rank(frame["tight_3d_range"], higher_is_better=False)
        + 0.25 * pct_rank(frame["vol_ratio_50"], higher_is_better=False)
        + 0.20 * pct_rank(frame["gain_6m"], higher_is_better=True)
        + 0.15 * pct_rank(frame["up_down_ratio"], higher_is_better=True)
        + 0.10 * pct_rank(frame["stock_strength_score"], higher_is_better=True)
    )
    return frame


def build_snapshot_for_date(
    date: pd.Timestamp,
    basic_history: pd.DataFrame,
    industry_history: pd.DataFrame,
    stock_history: pd.DataFrame,
    output_root: Path,
) -> dict:
    date_key = date.strftime("%Y-%m-%d")
    output_dir = output_root / date_key
    output_dir.mkdir(parents=True, exist_ok=True)

    basic = basic_history[basic_history["date"] == date].copy()
    industry = industry_history[industry_history["date"] == date].copy()
    stocks = stock_history[stock_history["date"] == date].copy()

    basic = basic.sort_values(["leadership_score", "actionability_score"], ascending=[False, False])
    industry = industry.sort_values(["leadership_score", "actionability_score"], ascending=[False, False])

    if not basic.empty:
        basic["rank"] = range(1, len(basic) + 1)
        basic["positive_5d_change"] = basic.get("leadership_change_5d", 0.0).clip(lower=0)
        basic["improver_priority"] = 0.65 * basic["leadership_score"] + 0.35 * basic["positive_5d_change"]

    if not industry.empty:
        industry["rank"] = range(1, len(industry) + 1)

    stocks = add_stock_priority(stocks) if not stocks.empty else stocks
    if not stocks.empty:
        stocks["buy_priority_score"] = stocks["buy_priority_score"].fillna(0.0)
        stocks["stock_rank_in_basic_industry"] = stocks.groupby("basic_industry")["buy_priority_score"].rank(method="first", ascending=False)

    basic.to_parquet(output_dir / "basic_industry_snapshot.parquet", index=False)
    industry.to_parquet(output_dir / "industry_snapshot.parquet", index=False)
    stocks.to_parquet(output_dir / "stock_snapshot.parquet", index=False)

    buy = stocks[stocks["established_buy_setup"] == 1].copy() if not stocks.empty else stocks.copy()
    buy = buy.sort_values("buy_priority_score", ascending=False)
    buy["industry_rank"] = buy.groupby("basic_industry").cumcount()
    buy = buy[buy["industry_rank"] < MAX_PER_INDUSTRY].head(TOP_BUY_COUNT)
    ipo = stocks[stocks["ipo_buy_setup"] == 1].copy() if not stocks.empty else stocks.copy()
    if not ipo.empty:
        ipo = ipo.sort_values("daily_range", ascending=True).head(IPO_COUNT)
    buy.to_parquet(output_dir / "top_buy_candidates.parquet", index=False)
    ipo.to_parquet(output_dir / "ipo_watchlist.parquet", index=False)

    metadata = {
        "date": date_key,
        "basic_industry_rows": int(len(basic)),
        "industry_rows": int(len(industry)),
        "stock_rows": int(len(stocks)),
        "top_buy_rows": int(len(buy)),
        "ipo_rows": int(len(ipo)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    processed = p("data", "processed")
    basic = read_parquet_safe(processed / "basic_industry_daily_features.parquet")
    industry = read_parquet_safe(processed / "industry_daily_features.parquet")
    stock = prepare_stock_snapshot(read_parquet_safe(processed / "stock_daily_features.parquet"))

    for frame in [basic, industry]:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    snapshot_root = processed / SNAPSHOT_ROOT_NAME
    snapshot_root.mkdir(parents=True, exist_ok=True)

    dates = sorted(set(basic["date"].dropna()) & set(industry["date"].dropna()) & set(stock["date"].dropna()))
    if not dates:
        raise ValueError("No common EOD dates exist across Basic Industry, Industry and Stock features")

    # Remove only date folders that are no longer represented. Existing valid
    # snapshots remain reusable, which is helpful after a ten-day gap.
    valid_keys = {pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates}
    for child in snapshot_root.iterdir():
        if child.is_dir() and child.name not in valid_keys:
            shutil.rmtree(child)

    metadata = [build_snapshot_for_date(pd.Timestamp(date), basic, industry, stock, snapshot_root) for date in dates]
    pd.DataFrame(metadata).sort_values("date").to_parquet(processed / "dashboard_dates.parquet", index=False)

    latest_date = pd.Timestamp(dates[-1])
    latest_basic = basic[basic["date"] == latest_date].copy()
    latest_industry = industry[industry["date"] == latest_date].copy()
    latest_stock = stock[stock["date"] == latest_date].copy()

    strength_summary = latest_stock[["date", "basic_industry", "basic_industry_members", "high_strength_count_snapshot", "pct_high_strength_snapshot"]].drop_duplicates(["date", "basic_industry"])
    basic_latest = latest_basic.drop(columns=["high_strength_count", "pct_high_strength"], errors="ignore").merge(strength_summary, on=["date", "basic_industry"], how="left")
    basic_latest["basic_industry_members"] = basic_latest["basic_industry_members"].fillna(basic_latest["members"])
    basic_latest["high_strength_count"] = basic_latest["high_strength_count_snapshot"].fillna(0).astype(int)
    basic_latest["pct_high_strength"] = basic_latest["pct_high_strength_snapshot"].fillna(0.0)
    basic_latest = basic_latest.drop(columns=["high_strength_count_snapshot", "pct_high_strength_snapshot"], errors="ignore")
    basic_latest["small_industry"] = (basic_latest["members"] < SMALL_GROUP_LIMIT).astype(int)

    latest_stock = add_stock_priority(latest_stock)
    latest_stock = latest_stock.sort_values("buy_priority_score", ascending=False)
    latest_stock["_industry_rank"] = latest_stock.groupby("basic_industry").cumcount()
    top_buy = latest_stock[(latest_stock["established_buy_setup"] == 1) & (latest_stock["_industry_rank"] < MAX_PER_INDUSTRY)].head(TOP_BUY_COUNT).drop(columns="_industry_rank")
    ipo_watchlist = latest_stock[latest_stock["ipo_buy_setup"] == 1].sort_values("daily_range", ascending=True).head(IPO_COUNT)

    write_parquet(basic_latest, processed / "dashboard_basic_industry_latest.parquet")
    write_parquet(latest_industry.sort_values(["leadership_score", "actionability_score"], ascending=[False, False]), processed / "dashboard_industry_latest.parquet")
    write_parquet(top_buy, processed / "dashboard_top_buy_candidates.parquet")
    write_parquet(ipo_watchlist, processed / "dashboard_ipo_watchlist.parquet")

    summary = {"latest_date": latest_date.strftime("%Y-%m-%d"), "dates": len(dates), "snapshot_root": str(snapshot_root), "latest_stock_rows": int(len(latest_stock))}
    (processed / "dashboard_snapshot_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"dashboard snapshots ready: {len(dates)} dates through {latest_date.date()}")


if __name__ == "__main__":
    main()
