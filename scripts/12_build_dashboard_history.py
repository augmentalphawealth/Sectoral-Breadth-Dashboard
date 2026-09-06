# scripts/12_build_dashboard_history.py
# Builds compact long-term trend files for Basic Industry, Industry and Sector.
# These files are created in GitHub Actions. Streamlit reads precomputed data
# and performs no leadership-score or five-session-change calculation.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "basic_industry": {
        "input": PROCESSED / "basic_industry_daily_features.parquet",
        "output": PROCESSED / "dashboard_basic_industry_history.parquet",
        "group_column": "basic_industry",
    },
    "industry": {
        "input": PROCESSED / "industry_daily_features.parquet",
        "output": PROCESSED / "dashboard_industry_history.parquet",
        "group_column": "industry",
    },
    "sector": {
        "input": PROCESSED / "sector_daily_features.parquet",
        "output": PROCESSED / "dashboard_sector_history.parquet",
        "group_column": "sector",
    },
}

DISPLAY_COLUMNS = [
    "date",
    "members",
    "strength_score",
    "leadership_score",
    "leadership_change_5d",
    "improver_priority",
    "actionability_score",
    "regime",
    "eq_ret_1d",
    "eq_ret_5d",
    "eq_ret_10d",
    "eq_ret_20d",
    "eq_ret_60d",
    "pct_above_20",
    "pct_above_50",
    "pct_above_200",
    "trend_template_pct",
    "new_high_55_pct",
    "new_high_252_pct",
    "acc_minus_dist",
    "breakout_count",
    "breakout_pct",
    "vcp_ready_count",
    "vcp_ready_pct",
    "high_strength_count",
    "pct_high_strength",
    "buy_volume_shock_count",
    "sell_volume_shock_count",
    "buy_volume_shock_pct",
    "sell_volume_shock_pct",
    "median_volume_shock",
    "small_industry",
    "median_dist_52w_high",
    "nh_nl_net",
]


def clean_group(series: pd.Series) -> pd.Series:
    return series.fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")


def build_history_table(input_file: Path, output_file: Path, group_column: str) -> dict:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing feature file: {input_file}")

    source = pd.read_parquet(input_file)
    required = ["date", group_column, "leadership_score"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{input_file.name} missing required columns: {missing}")

    selected_columns = list(dict.fromkeys([group_column] + [column for column in DISPLAY_COLUMNS if column in source.columns]))
    history = source[selected_columns].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history[group_column] = clean_group(history[group_column])
    history = history.dropna(subset=["date", group_column]).drop_duplicates(["date", group_column], keep="last")
    history = history.sort_values([group_column, "date"]).reset_index(drop=True)

    # Safety fallback: 01_build_group_features.py normally creates these.
    # This preserves the exact 5-available-trading-session meaning if an older
    # feature file is used to rebuild history.
    if "leadership_change_5d" not in history.columns:
        history["leadership_change_5d"] = history.groupby(group_column)["leadership_score"].diff(5).fillna(0.0).round(1)
    else:
        history["leadership_change_5d"] = pd.to_numeric(history["leadership_change_5d"], errors="coerce").fillna(0.0).round(1)

    if "improver_priority" not in history.columns:
        history["improver_priority"] = (
            0.65 * pd.to_numeric(history["leadership_score"], errors="coerce").fillna(0.0)
            + 0.35 * history["leadership_change_5d"].clip(lower=0)
        ).round(1)
    else:
        history["improver_priority"] = pd.to_numeric(history["improver_priority"], errors="coerce").fillna(0.0).round(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output_file, index=False)

    return {
        "rows": int(len(history)),
        "groups": int(history[group_column].nunique()),
        "start_date": str(history["date"].min().date()),
        "latest_date": str(history["date"].max().date()),
        "file": output_file.name,
        "columns": history.columns.tolist(),
    }


def main() -> None:
    print("========== DASHBOARD HISTORY BUILD START ==========")
    metadata: dict[str, object] = {}

    for name, config in FILES.items():
        result = build_history_table(config["input"], config["output"], config["group_column"])
        metadata[name] = result
        print(f"{name}: {result['rows']:,} rows, {result['groups']:,} groups, {result['start_date']} to {result['latest_date']}")

    metadata["generated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metadata_file = PROCESSED / "dashboard_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("========== DASHBOARD HISTORY BUILD COMPLETE ==========")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
