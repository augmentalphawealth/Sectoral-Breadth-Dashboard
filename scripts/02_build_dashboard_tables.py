# scripts/02_build_dashboard_tables.py
# Builds compact date snapshots for Basic Industry, Industry, Sector and stocks.

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from utils import p, read_parquet_safe, write_parquet

MAX_PER_INDUSTRY = 4
TOP_BUY_COUNT = 20
IPO_COUNT = 15
SNAPSHOT_ROOT_NAME = "dashboard_snapshots"


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def clean_group(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame[column] = frame[column].fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")
    return frame


def prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    data = frame.copy()
    require_columns(data, ["date"], label)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"]).copy()
    return data


def prepare_stock(stock: pd.DataFrame) -> pd.DataFrame:
    data = prepare(stock, "Stock feature file")
    required = ["symbol", "sector", "industry", "basic_industry", "established_buy_setup", "ipo_buy_setup"]
    require_columns(data, required, "Stock feature file")
    data["symbol"] = data["symbol"].fillna("").astype(str).str.strip()
    data = data[data["symbol"] != ""].drop_duplicates(["date", "symbol"], keep="last")
    for column in ["sector", "industry", "basic_industry"]:
        data = clean_group(data, column)
    for column in ["established_buy_setup", "ipo_buy_setup"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    return data


def add_priority(stock: pd.DataFrame) -> pd.DataFrame:
    data = stock.copy()
    def numeric(column: str) -> pd.Series:
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce")
        return pd.Series(0.0, index=data.index)
    def pct_rank(values: pd.Series, ascending: bool) -> pd.Series:
        return values.rank(pct=True, ascending=ascending, na_option="keep").fillna(0.0) * 100.0
    data["buy_priority_score"] = (
        0.30 * pct_rank(numeric("tight_3d_range"), False)
        + 0.25 * pct_rank(numeric("vol_ratio_50"), False)
        + 0.20 * pct_rank(numeric("gain_6m"), True)
        + 0.15 * pct_rank(numeric("up_down_ratio"), True)
        + 0.10 * pct_rank(numeric("stock_strength_score"), True)
    ).round(1)
    return data


def build_snapshot(date: pd.Timestamp, basic: pd.DataFrame, industry: pd.DataFrame, sector: pd.DataFrame, stock: pd.DataFrame, root: Path) -> dict:
    key = date.strftime("%Y-%m-%d")
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)

    basic_day = basic[basic["date"] == date].copy()
    industry_day = industry[industry["date"] == date].copy()
    sector_day = sector[sector["date"] == date].copy()
    stock_day = stock[stock["date"] == date].copy()

    for frame in [basic_day, industry_day, sector_day]:
        if "leadership_score" in frame.columns:
            frame.sort_values(["leadership_score", "actionability_score"], ascending=[False, False], inplace=True)

    stock_day = add_priority(stock_day)
    stock_day.sort_values("buy_priority_score", ascending=False, inplace=True)

    basic_day.to_parquet(folder / "basic_industry_snapshot.parquet", index=False)
    industry_day.to_parquet(folder / "industry_snapshot.parquet", index=False)
    sector_day.to_parquet(folder / "sector_snapshot.parquet", index=False)
    stock_day.to_parquet(folder / "stock_snapshot.parquet", index=False)

    buy = stock_day[stock_day["established_buy_setup"] == 1].copy()
    buy["_industry_rank"] = buy.groupby("basic_industry").cumcount()
    buy = buy[buy["_industry_rank"] < MAX_PER_INDUSTRY].drop(columns="_industry_rank").head(TOP_BUY_COUNT)
    ipo = stock_day[stock_day["ipo_buy_setup"] == 1].copy().head(IPO_COUNT)
    buy.to_parquet(folder / "top_buy_candidates.parquet", index=False)
    ipo.to_parquet(folder / "ipo_watchlist.parquet", index=False)

    metadata = {
        "date": key,
        "basic_industry_rows": int(len(basic_day)),
        "industry_rows": int(len(industry_day)),
        "sector_rows": int(len(sector_day)),
        "stock_rows": int(len(stock_day)),
        "top_buy_rows": int(len(buy)),
        "ipo_rows": int(len(ipo)),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    processed = p("data", "processed")
    basic = prepare(read_parquet_safe(processed / "basic_industry_daily_features.parquet"), "Basic Industry feature file")
    industry = prepare(read_parquet_safe(processed / "industry_daily_features.parquet"), "Industry feature file")
    sector = prepare(read_parquet_safe(processed / "sector_daily_features.parquet"), "Sector feature file")
    stock = prepare_stock(read_parquet_safe(processed / "stock_daily_features.parquet"))

    for frame, column in [(basic, "basic_industry"), (industry, "industry"), (sector, "sector")]:
        require_columns(frame, [column, "leadership_score", "members"], f"{column} feature file")
        clean_group(frame, column)

    common_dates = sorted(set(basic["date"]) & set(industry["date"]) & set(sector["date"]) & set(stock["date"]))
    if not common_dates:
        raise ValueError("No common dates across Basic Industry, Industry, Sector and Stock features")

    snapshot_root = processed / SNAPSHOT_ROOT_NAME
    snapshot_root.mkdir(parents=True, exist_ok=True)
    valid_keys = {pd.Timestamp(date).strftime("%Y-%m-%d") for date in common_dates}
    for child in snapshot_root.iterdir():
        if child.is_dir() and child.name not in valid_keys:
            shutil.rmtree(child)

    metadata = [build_snapshot(pd.Timestamp(date), basic, industry, sector, stock, snapshot_root) for date in common_dates]
    dates = pd.DataFrame(metadata).sort_values("date").reset_index(drop=True)
    dates.to_parquet(processed / "dashboard_dates.parquet", index=False)

    latest = pd.Timestamp(common_dates[-1])
    latest_basic = basic[basic["date"] == latest].copy()
    latest_industry = industry[industry["date"] == latest].copy()
    latest_sector = sector[sector["date"] == latest].copy()
    latest_stock = add_priority(stock[stock["date"] == latest].copy())
    latest_stock.sort_values("buy_priority_score", ascending=False, inplace=True)

    write_parquet(latest_basic, processed / "dashboard_basic_industry_latest.parquet")
    write_parquet(latest_industry, processed / "dashboard_industry_latest.parquet")
    write_parquet(latest_sector, processed / "dashboard_sector_latest.parquet")
    write_parquet(latest_stock.head(TOP_BUY_COUNT), processed / "dashboard_top_buy_candidates.parquet")
    write_parquet(latest_stock[latest_stock["ipo_buy_setup"] == 1].head(IPO_COUNT), processed / "dashboard_ipo_watchlist.parquet")

    summary = {"latest_date": latest.strftime("%Y-%m-%d"), "dates": len(common_dates), "sector_snapshots": len(common_dates)}
    (processed / "dashboard_snapshot_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"dashboard snapshots ready: {len(common_dates):,} dates through {latest.date()}")


if __name__ == "__main__":
    main()
