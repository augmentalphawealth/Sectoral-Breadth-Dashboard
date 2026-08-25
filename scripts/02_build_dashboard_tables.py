from __future__ import annotations

import pandas as pd

from utils import p, read_parquet_safe, write_parquet


HIGH_STRENGTH_THRESHOLD = 70.0
SMALL_GROUP_LIMIT = 5


def latest(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date"].max()
    return df[df["date"] == max_date].copy()


def build_stock_strength_snapshot(
    stock: pd.DataFrame,
) -> pd.DataFrame:
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
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Stock feature file is missing required columns: "
            f"{missing}"
        )

    data["stock_strength_score"] = (
        data["ret_20d"].rank(pct=True) * 35
        + data["ret_60d"].rank(pct=True) * 35
        + data["above_50"] * 15
        + data["above_200"] * 10
        + data["breakout_55"] * 3
        + data["vcp_ready"] * 2
    )

    group_keys = ["date", "basic_industry"]

    data["high_strength_flag"] = (
        data["stock_strength_score"] >= HIGH_STRENGTH_THRESHOLD
    ).astype(int)

    data["basic_industry_members"] = (
        data.groupby(group_keys)["symbol"]
        .transform("nunique")
    )

    data["high_strength_count"] = (
        data.groupby(group_keys)["high_strength_flag"]
        .transform("sum")
    )

    data["pct_high_strength"] = (
        data["high_strength_count"]
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
                "high_strength_count",
                "pct_high_strength",
            ]
        ]
        .drop_duplicates(
            subset=["date", "basic_industry"],
            keep="last",
        )
        .copy()
    )

    basic_latest = latest(basic).merge(
        strength_summary,
        on=["date", "basic_industry"],
        how="left",
    )

    basic_latest["basic_industry_members"] = (
        basic_latest["basic_industry_members"]
        .fillna(basic_latest["members"])
    )

    basic_latest["high_strength_count"] = (
        basic_latest["high_strength_count"]
        .fillna(0)
        .astype(int)
    )

    basic_latest["pct_high_strength"] = (
        basic_latest["pct_high_strength"]
        .fillna(0.0)
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
        + (watch["dist_52w_high"] > -0.10).astype(int) *
