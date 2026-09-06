# scripts/03_run_eod_pipeline.py
# GitHub Actions EOD orchestration. All calculations and display-file creation
# happen here; Streamlit later reads only finished date snapshots.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def run(script_name: str) -> None:
    script_path = ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script missing: {script_path}")
    print(f"\n========== RUNNING {script_name} ==========")
    subprocess.run([sys.executable, str(script_path)], check=True)


def validate_outputs() -> None:
    required_files = [
        PROCESSED / "stock_daily_features.parquet",
        PROCESSED / "basic_industry_daily_features.parquet",
        PROCESSED / "industry_daily_features.parquet",
        PROCESSED / "sector_daily_features.parquet",
        PROCESSED / "dashboard_basic_industry_history.parquet",
        PROCESSED / "dashboard_industry_history.parquet",
        PROCESSED / "dashboard_sector_history.parquet",
        PROCESSED / "dashboard_stock_history.parquet",
        PROCESSED / "dashboard_dates.parquet",
        PROCESSED / "dashboard_snapshots",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("EOD pipeline did not produce required outputs:\n" + "\n".join(missing))

    dates = pd.read_parquet(PROCESSED / "dashboard_dates.parquet")
    if dates.empty or "date" not in dates.columns:
        raise ValueError("dashboard_dates.parquet is empty or missing the date column")
    dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
    dates = dates.dropna(subset=["date"])
    if dates.empty:
        raise ValueError("dashboard_dates.parquet contains no valid dates")

    latest = dates["date"].max().strftime("%Y-%m-%d")
    latest_snapshot = PROCESSED / "dashboard_snapshots" / latest
    expected_snapshot_files = [
        latest_snapshot / "basic_industry_snapshot.parquet",
        latest_snapshot / "industry_snapshot.parquet",
        latest_snapshot / "stock_snapshot.parquet",
        latest_snapshot / "top_buy_candidates.parquet",
        latest_snapshot / "ipo_watchlist.parquet",
        latest_snapshot / "metadata.json",
    ]
    absent = [str(path.relative_to(ROOT)) for path in expected_snapshot_files if not path.exists()]
    if absent:
        raise FileNotFoundError("Latest date snapshot is incomplete:\n" + "\n".join(absent))

    basic = pd.read_parquet(latest_snapshot / "basic_industry_snapshot.parquet")
    stock = pd.read_parquet(latest_snapshot / "stock_snapshot.parquet")
    if basic.empty:
        raise ValueError(f"Latest Basic Industry snapshot is empty: {latest}")
    if stock.empty:
        raise ValueError(f"Latest stock snapshot is empty: {latest}")

    print("========== OUTPUT VALIDATION PASSED ==========")
    print(f"Available dashboard dates: {len(dates):,}")
    print(f"Latest snapshot: {latest}")
    print(f"Latest Basic Industry rows: {len(basic):,}")
    print(f"Latest stock rows: {len(stock):,}")


def main() -> None:
    print("========== EOD PIPELINE START ==========")
    # Classification/universe refresh is intentionally handled by dedicated
    # workflows before this EOD workflow. This pipeline uses the verified,
    # classified master they have published.
    run("01_build_group_features.py")
    run("12_build_dashboard_history.py")
    run("13_build_dashboard_stock_history.py")
    run("02_build_dashboard_tables.py")
    validate_outputs()

    sync_path = PROCESSED / "last_sync.txt"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sync_path.write_text(timestamp, encoding="utf-8")

    print("========== EOD PIPELINE COMPLETE ==========")
    print(f"Last sync written: {sync_path}")
    print(f"Timestamp UTC: {timestamp}")


if __name__ == "__main__":
    main()
