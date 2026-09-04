# scripts/02_build_dashboard_tables.py
# Single consolidated buy_priority_score (30% tightness, 25% vol, 20% gain, 15% up/down, 10% strength)

from __future__ import annotations

import pandas as pd
from utils import p, read_parquet_safe, write_parquet

HIGH_STRENGTH_THRESHOLD = 70.0
SMALL_GROUP_LIMIT = 5
MAX_PER_INDUSTRY = 4 


def latest(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date"].max()
    return df[df["date"] == max_date].copy()


def build_stock_strength_snapshot(stock: pd.DataFrame) -> pd.DataFrame:
    data = latest(stock).copy()

    required = [
        "date", "symbol", "basic_industry", "ret_20d", "ret_60d",
        "above_50", "above_200", "breakout_55", "vcp_ready",
        "stock_strength_score", "high_strength_flag", "established_buy_setup",
        "ipo_buy_setup", "buy_setup_score", "gain_6m", "daily_range",
        "vol_ratio_50", "vol_2x_count_6m", "actionable_setup_pass",
        "up_down_ratio", "atr_14", "range_3d", "tight_3d_range",
        "ipo_turnover_avg", "close", "nearest_ema_tag", "momentum_badge",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Stock feature file missing columns: {missing}")

    data["stock_strength_score"] = pd.to_numeric(data["stock_strength_score"], errors="coerce")
    group_keys = ["date", "basic_industry"]
    data["high_strength_flag"] = pd.to_numeric(data["high_strength_flag"], errors="coerce").fillna(0).astype(int)
    data["basic_industry_members"] = data.groupby(group_keys)["symbol"].transform("nunique")
    data["high_strength_count_snapshot"] = data.groupby(group_keys)["high_strength_flag"].transform("sum")
    data["pct_high_strength_snapshot"] = (
        data["high_strength_count_snapshot"] / data["basic_industry_members"].clip(lower=1) * 100
    )
    return data


def main() -> None:
    processed = p("data", "processed")

    basic = read_parquet_safe(processed / "basic_industry_daily_features.parquet")
    industry = read_parquet_safe(processed / "industry_daily_features.parquet")
    stock = read_parquet_safe(processed / "stock_daily_features.parquet")

    stock_snapshot = build_stock_strength_snapshot(stock)

    strength_summary = (
        stock_snapshot[
            ["date", "basic_industry", "basic_industry_members",
             "high_strength_count_snapshot", "pct_high_strength_snapshot"]
        ].drop_duplicates(subset=["date", "basic_industry"], keep="last").copy()
    )

    basic_latest = latest(basic).copy()
    for column in ["high_strength_count", "pct_high_strength"]:
        if column in basic_latest.columns:
            basic_latest = basic_latest.drop(columns=[column])

    basic_latest = basic_latest.merge(strength_summary, on=["date", "basic_industry"], how="left")
    basic_latest["basic_industry_members"] = basic_latest["basic_industry_members"].fillna(basic_latest["members"])
    basic_latest["high_strength_count"] = basic_latest["high_strength_count_snapshot"].fillna(0).astype(int)
    basic_latest["pct_high_strength"] = basic_latest["pct_high_strength_snapshot"].fillna(0.0)
    basic_latest = basic_latest.drop(columns=["high_strength_count_snapshot", "pct_high_strength_snapshot"], errors="ignore")
    basic_latest["small_industry"] = (basic_latest["members"] < SMALL_GROUP_LIMIT).astype(int)
    basic_latest = basic_latest.sort_values(
        ["leadership_score", "actionability_score", "eq_ret_20d"],
        ascending=[False, False, False],
    )

    industry_latest = latest(industry).sort_values(
        ["leadership_score", "actionability_score", "eq_ret_20d"],
        ascending=[False, False, False],
    )

    eligible_top_industries = (
        basic_latest[
            (basic_latest["members"] >= SMALL_GROUP_LIMIT)
            & (basic_latest["leadership_score"] >= 70.0)
            & (basic_latest["basic_industry"] != "Unclassified")
        ].head(15)["basic_industry"].dropna().unique().tolist()
    )
    if not eligible_top_industries:
        eligible_top_industries = (
            basic_latest[
                (basic_latest["members"] >= SMALL_GROUP_LIMIT)
                & (basic_latest["basic_industry"] != "Unclassified")
            ].head(5)["basic_industry"].dropna().unique().tolist()
        )

    top_buy_candidates = stock_snapshot[
        (stock_snapshot["basic_industry"].isin(eligible_top_industries))
        & (stock_snapshot["established_buy_setup"] == 1)
    ].copy()

    def _pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
        return s.rank(pct=True, ascending=higher_is_better, na_option="keep") * 100

    top_buy_candidates["buy_priority_score"] = (
        0.30 * _pct_rank(top_buy_candidates["tight_3d_range"], higher_is_better=False)
        + 0.25 * _pct_rank(top_buy_candidates["vol_ratio_50"], higher_is_better=False)
        + 0.20 * _pct_rank(top_buy_candidates["gain_6m"], higher_is_better=True)
        + 0.15 * _pct_rank(top_buy_candidates["up_down_ratio"], higher_is_better=True)
        + 0.10 * _pct_rank(top_buy_candidates["stock_strength_score"], higher_is_better=True)
    )

    top_buy_candidates = top_buy_candidates.sort_values("buy_priority_score", ascending=False)
    top_buy_candidates["_industry_rank"] = top_buy_candidates.groupby("basic_industry").cumcount()
    top_buy_candidates = (
        top_buy_candidates[top_buy_candidates["_industry_rank"] < MAX_PER_INDUSTRY]
        .sort_values("buy_priority_score", ascending=False)
        .head(20)
        .drop(columns="_industry_rank")
    )

    ipo_watchlist = stock_snapshot[
        (stock_snapshot["basic_industry"].isin(eligible_top_industries))
        & (stock_snapshot["ipo_buy_setup"] == 1)
    ].copy()
    ipo_watchlist = ipo_watchlist.sort_values("daily_range", ascending=True).head(15)

    legacy_required = ["trend_template_pass", "dist_52w_high"]
    missing_legacy = [c for c in legacy_required if c not in stock_snapshot.columns]
    if missing_legacy:
        raise ValueError(f"Legacy watchlist missing columns: {missing_legacy}")

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
        ["high_strength_flag", "quality_rank", "stock_strength_score", "ret_20d"],
        ascending=[False, False, False, False],
    )

    write_parquet(basic_latest, processed / "dashboard_basic_industry_latest.parquet")
    write_parquet(industry_latest, processed / "dashboard_industry_latest.parquet")
    write_parquet(top_buy_candidates, processed / "dashboard_top_buy_candidates.parquet")
    write_parquet(ipo_watchlist, processed / "dashboard_ipo_watchlist.parquet")
    write_parquet(watch, processed / "dashboard_stock_watchlist_latest.parquet")

    print("dashboard tables ready")


if __name__ == "__main__":
    main()
