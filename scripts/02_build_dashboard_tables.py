from __future__ import annotations
import pandas as pd
from utils import p, read_parquet_safe, write_parquet


def latest(df: pd.DataFrame) -> pd.DataFrame:
    mx = df['date'].max()
    return df[df['date'] == mx].copy()


def main():
    processed = p('data', 'processed')
    basic = read_parquet_safe(processed / 'basic_industry_daily_features.parquet')
    industry = read_parquet_safe(processed / 'industry_daily_features.parquet')
    stock = read_parquet_safe(processed / 'stock_daily_features.parquet')

    basic_latest = latest(basic).sort_values(['strength_score', 'eq_ret_20d'], ascending=[False, False])
    industry_latest = latest(industry).sort_values(['strength_score', 'eq_ret_20d'], ascending=[False, False])

    stock_latest = latest(stock)
    watch = stock_latest[
        (stock_latest['trend_template_pass'] == 1) |
        (stock_latest['vcp_ready'] == 1) |
        (stock_latest['breakout_55'] == 1)
    ].copy()
    watch['quality_rank'] = (
        watch['trend_template_pass'] * 30 +
        watch['vcp_ready'] * 25 +
        watch['breakout_55'] * 20 +
        watch['above_50'] * 10 +
        watch['above_200'] * 10 +
        (watch['dist_52w_high'] > -0.10).astype(int) * 5
    )
    watch = watch.sort_values(['quality_rank', 'ret_20d'], ascending=[False, False])

    write_parquet(basic_latest, processed / 'dashboard_basic_industry_latest.parquet')
    write_parquet(industry_latest, processed / 'dashboard_industry_latest.parquet')
    write_parquet(watch, processed / 'dashboard_stock_watchlist_latest.parquet')
    print('dashboard tables ready')


if __name__ == '__main__':
    main()
