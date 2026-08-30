from __future__ import annotations

import pandas as pd

from utils import p, read_parquet_safe, write_parquet


HIGH_STRENGTH_THRESHOLD = 70.0
SMALL_GROUP_LIMIT = 5


def latest(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date"].max()
    return df[df["date"] == max_date].copy()


def build_stock_strength_snapshot(stock: pd.DataFrame) -> pd.DataFrame:
    data = latest(stock).copy()

    required = [
        "date",
        "symbol",
        "basic_industry",
        "ret_20d",
        "ret_60d",
        "above_50",
        "above_200",
        "breakout_55",
        "vcp_ready",
        "stock_strength_score",  # ✅ Now required from 01 (pre-computed)
        "high_strength_flag",    # ✅ Now required from 01 (pre-computed)
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "Stock feature file is missing required columns: "
            f"{missing}"
        )

    # ✅ NO RECOMPUTATION - Use pre-computed scores from 01_build_group_features.py
    data["stock_strength_score"] = pd.to_numeric(
        data["stock_strength_score"],
        errors="coerce",
    )

    group_keys = ["date", "basic_industry"]

    # ✅ Use pre-computed high_strength_flag from 01
    data["high_strength_flag"] = pd.to_numeric(
        data["high_strength_flag"],
        errors="coerce",
    ).fillna(0).astype(int)

    data["basic_industry_members"] = (
        data.groupby(group_keys)["symbol"].transform("nunique")
    )
    data["high_strength_count_snapshot"] = (
        data.groupby(group_keys)["high_strength_flag"].transform("sum")
    )
    data["pct_high_strength_snapshot"] = (
        data["high_strength_count_snapshot"]
        / data["basic_industry_members"].clip(lower=1)
        * 100
    )

    return data


def main() -> None:
    processed = p("data", "processed")

    basic = read_parquet_safe(
        processed / "basic_industry_daily_features.parquet"
    )
    industry = read_parquet_safe(
        processed / "industry_daily_features.parquet"
    )
    stock = read_parquet_safe(
        processed / "stock_daily_features.parquet"
    )

    stock_snapshot = build_stock_strength_snapshot(stock)

    strength_summary = (
        stock_snapshot[
            [
                "date",
                "basic_industry",
                "basic_industry_members",
                "high_strength_count_snapshot",
                "pct_high_strength_snapshot",
            ]
        ]
        .drop_duplicates(
            subset=["date", "basic_industry"],
            keep="last",
        )
        .rename(
            columns={
                "high_strength_count_snapshot": "high_strength_count_snapshot",
                "pct_high_strength_snapshot": "pct_high_strength_snapshot",
            }
        )
        .copy()
    )

    basic_latest = latest(basic).copy()

    for column in ["high_strength_count", "pct_high_strength"]:
        if column in basic_latest.columns:
            basic_latest = basic_latest.drop(columns=[column])

    basic_latest = basic_latest.merge(
        strength_summary,
        on=["date", "basic_industry"],
        how="left",
    )

    basic_latest["basic_industry_members"] = (
        basic_latest["basic_industry_members"]
        .fillna(basic_latest["members"])
    )
    basic_latest["high_strength_count"] = (
        basic_latest["high_strength_count_snapshot"]
        .fillna(0)
        .astype(int)
    )
    basic_latest["pct_high_strength"] = (
        basic_latest["pct_high_strength_snapshot"]
        .fillna(0.0)
    )
    basic_latest = basic_latest.drop(
        columns=[
            "high_strength_count_snapshot",
            "pct_high_strength_snapshot",
        ],
        errors="ignore",
    )

    basic_latest["small_industry"] = (
        basic_latest["members"] < SMALL_GROUP_LIMIT
    ).astype(int)

    basic_latest = basic_latest.sort_values(
        ["strength_score", "pct_high_strength", "eq_ret_20d"],
        ascending=[False, False, False],
    )

    industry_latest = latest(industry).sort_values(
        ["strength_score", "eq_ret_20d"],
        ascending=[False, False],
    )

    watch = stock_snapshot[
        (stock_snapshot["trend_template_pass"] == 1)
        | (stock_snapshot["vcp_ready"] == 1)
        | (stock_snapshot["breakout_55"] == 1)
        | (stock_snapshot["high_strength_flag"] == 1)
    ].copy()

    watch["quality_rank"] = (
        watch["trend_template_pass"] * 30
        + watch["vcp_ready"] * 25
        + watch["breakout_55"] * 20
        + watch["above_50"] * 10
        + watch["above_200"] * 10
        + (watch["dist_52w_high"] > -0.10).astype(int) * 5
    )

    watch = watch.sort_values(
        [
            "high_strength_flag",
            "quality_rank",
            "stock_strength_score",
            "ret_20d",
        ],
        ascending=[False, False, False, False],
    )

    write_parquet(
        basic_latest,
        processed / "dashboard_basic_industry_latest.parquet",
    )
    write_parquet(
        industry_latest,
        processed / "dashboard_industry_latest.parquet",
    )
    write_parquet(
        watch,
        processed / "dashboard_stock_watchlist_latest.parquet",
    )

    print("dashboard tables ready")
    print(
        "Basic Industries with high-strength participation data: "
        f"{basic_latest['pct_high_strength'].notna().sum()}"
    )


if __name__ == "__main__":
    main()
