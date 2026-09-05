from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


FILES = {
    "industry": {
        "input": PROCESSED / "industry_daily_features.parquet",
        "output": PROCESSED / "dashboard_industry_history.parquet",
        "group_column": "industry",
    },
    "basic_industry": {
        "input": PROCESSED / "basic_industry_daily_features.parquet",
        "output": PROCESSED / "dashboard_basic_industry_history.parquet",
        "group_column": "basic_industry",
    },
}


COMMON_COLUMNS = [
    "date",
    "members",
    "strength_score",
    "leadership_score",
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


def build_history_table(
    input_file: Path,
    output_file: Path,
    group_column: str,
) -> dict:
    df = pd.read_parquet(input_file)

    required_columns = ["date", group_column]
    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"{input_file.name} is missing required columns: "
            f"{missing_required}"
        )

    selected_columns = [
        group_column,
        *[
            column
            for column in COMMON_COLUMNS
            if column in df.columns
        ],
    ]
    selected_columns = list(dict.fromkeys(selected_columns))

    history = df[selected_columns].copy()
    history["date"] = pd.to_datetime(history["date"])
    history[group_column] = history[group_column].fillna("Unclassified")

    history = history.dropna(subset=["date", group_column])
    history = history.drop_duplicates(
        subset=["date", group_column],
        keep="last",
    )
    history = history.sort_values(
        [group_column, "date"],
    ).reset_index(drop=True)

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

    metadata = {}

    for name, config in FILES.items():
        result = build_history_table(
            input_file=config["input"],
            output_file=config["output"],
            group_column=config["group_column"],
        )

        metadata[name] = result

        print(
            f"{name}: "
            f"{result['rows']} rows, "
            f"{result['groups']} groups, "
            f"{result['start_date']} to {result['latest_date']}"
        )

    metadata["generated_at_utc"] = pd.Timestamp.utcnow().isoformat()

    metadata_file = PROCESSED / "dashboard_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("========== DASHBOARD HISTORY BUILD COMPLETE ==========")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
