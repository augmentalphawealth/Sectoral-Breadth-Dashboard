from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = PROCESSED / "dashboard_basic_industry_history.parquet"
INDUSTRY_HISTORY_FILE = PROCESSED / "dashboard_industry_history.parquet"
STOCK_HISTORY_FILE = PROCESSED / "dashboard_stock_history.parquet"
METADATA_FILE = PROCESSED / "dashboard_metadata.json"
SYNC_FILE = PROCESSED / "last_sync.txt"

st.set_page_config(
    page_title="NSE Sectoral Breadth",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def fmt_2dp(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.2f}"


def fmt_2dp_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:,.2f}%"


def fmt_1dp_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:,.1f}%"


def regime_label(regime: object) -> str:
    return clean_text(regime)


def strength_label(score: float) -> str:
    if pd.isna(score):
        return "Neutral"
    if score >= 70:
        return "Leader"
    if score >= 55:
        return "Strong"
    if score >= 40:
        return "Neutral"
    if score >= 25:
        return "Weak"
    return "Lagging"


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def ensure_group_columns(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = df.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Unclassified"
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    return data


def ensure_stock_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for column in ["symbol", "industry", "basic_industry", "sector"]:
        if column not in data.columns:
            data[column] = "Unclassified"
        data[column] = data[column].map(clean_text)
    return data


def get_trading_dates(history: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(history["date"].dropna().unique())
    return sorted([pd.Timestamp(date) for date in dates])


def date_navigation(
    available_dates: list[pd.Timestamp],
    widget_key: str,
) -> pd.Timestamp:
    if not available_dates:
        raise ValueError("No dates are available")
    latest_date = available_dates[-1]
    state_key = f"{widget_key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest_date
    selected_date = pd.Timestamp(st.session_state[state_key])
    if selected_date not in available_dates:
        selected_date = latest_date
        st.session_state[state_key] = selected_date
    current_index = available_dates.index(selected_date)
    col_previous, col_date, col_next, col_latest = st.columns(
        [0.6, 2.0, 0.6, 0.9]
    )
    if col_previous.button(
        "←",
        disabled=current_index == 0,
        key=f"{widget_key}_previous",
        use_container_width=True,
    ):
        st.session_state[state_key] = available_dates[current_index - 1]
        st.rerun()
    date_choice = col_date.selectbox(
        "As-of date",
        options=available_dates,
        index=current_index,
        format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"),
        key=f"{widget_key}_date_picker",
    )
    if pd.Timestamp(date_choice) != selected_date:
        st.session_state[state_key] = pd.Timestamp(date_choice)
        selected_date = pd.Timestamp(date_choice)
    current_index = available_dates.index(selected_date)
    if col_next.button(
        "→",
        disabled=current_index == len(available_dates) - 1,
        key=f"{widget_key}_next",
        use_container_width=True,
    ):
        st.session_state[state_key] = available_dates[current_index + 1]
        st.rerun()
    if col_latest.button(
        "Latest",
        key=f"{widget_key}_latest",
        use_container_width=True,
    ):
        st.session_state[state_key] = latest_date
        st.rerun()
    return pd.Timestamp(st.session_state[state_key])


def regime_counts(data: pd.DataFrame) -> dict[str, int]:
    regimes = ["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"]
    counts = data["regime"].value_counts().to_dict()
    return {regime: int(counts.get(regime, 0)) for regime in regimes}


def show_group_kpis(
    data: pd.DataFrame,
    group_column: str,
    selected_date: pd.Timestamp,
) -> None:
    counts = regime_counts(data)
    median_strength = (
        data["strength_score"].median()
        if "strength_score" in data.columns
        else float("nan")
    )
    median_breadth_50 = (
        data["pct_above_50"].median()
        if "pct_above_50" in data.columns
        else float("nan")
    )
    total_members = (
        data["members"].sum()
        if "members" in data.columns
        else 0
    )
    cols = st.columns(6)
    cols[0].metric("As of", selected_date.strftime("%d %b %Y"))
    cols[1].metric("Groups", fmt_int(data[group_column].nunique()))
    cols[2].metric("Members", fmt_int(total_members))
    cols[3].metric("Median strength", fmt_2dp(median_strength))
    cols[4].metric("Median above 50 DMA", fmt_1dp_pct(median_breadth_50))
    cols[5].metric(
        "Strong / Emerging",
        f"{counts['Strong']} / {counts['Emerging']}",
    )


def prepare_group_table(
    data: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    columns = [
        group_column,
        "regime",
        "members",
        "strength_score",
        "eq_ret_1d",
        "eq_ret_5d",
        "eq_ret_20d",
        "eq_ret_60d",
        "pct_above_20",
        "pct_above_50",
        "pct_above_200",
        "acc_minus_dist",
        "breakout_count",
        "vcp_ready_count",
        "median_dist_52w_high",
    ]
    selected = [c for c in columns if c in data.columns]
    table = data[selected].copy()
    rename_map = {
        group_column: (
            "Basic Industry" if group_column == "basic_industry" else "Industry"
        ),
        "regime": "Regime",
        "members": "Members",
        "strength_score": "Strength",
        "eq_ret_1d": "ret_1d",
        "eq_ret_5d": "ret_5d",
        "eq_ret_20d": "ret_20d",
        "eq_ret_60d": "ret_60d",
        "pct_above_20": "above_20",
        "pct_above_50": "above_50",
        "pct_above_200": "above_200",
        "acc_minus_dist": "ad_balance",
        "breakout_count": "breakouts",
        "vcp_ready_count": "vcp_ready",
        "median_dist_52w_high": "dist_52w_high",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})
    if "Strength" in table.columns:
        table = table.sort_values("Strength", ascending=False)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table.reset_index(drop=True)


def prepare_stock_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "stock_rank_in_basic_industry",
        "symbol",
        "stock_strength_score",
        "stock_strength_percentile_in_basic_industry",
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
    ]
    selected = [c for c in columns if c in data.columns]
    table = data[selected].copy()
    rename_map = {
        "stock_rank_in_basic_industry": "Rank",
        "symbol": "Symbol",
        "stock_strength_score": "Strength",
        "stock_strength_percentile_in_basic_industry": "StrengthPct",
        "close": "Close",
        "ret_1d": "ret_1d",
        "ret_5d": "ret_5d",
        "ret_20d": "ret_20d",
        "ret_60d": "ret_60d",
        "above_20": "above_20",
        "above_50": "above_50",
        "above_200": "above_200",
        "dist_52w_high": "dist_52w_high",
        "trend_template_pass": "TrendTemplate",
        "acc_day": "Accumulation",
        "dist_day": "Distribution",
        "breakout_55": "Breakouts",
        "vcp_ready": "VCPReady",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})
    if "Rank" in table.columns:
        table = table.sort_values(["Rank", "Symbol"])
    return table.reset_index(drop=True)


def format_group_display_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["Regime"] = out["Regime"].apply(regime_label)
    out["StrengthBand"] = out["Strength"].apply(strength_label)
    numeric_map = {
        "Strength": fmt_2dp,
        "Members": fmt_int,
        "ret_1d": fmt_2dp_pct,
        "ret_5d": fmt_2dp_pct,
        "ret_20d": fmt_2dp_pct,
        "ret_60d": fmt_2dp_pct,
        "above_20": fmt_1dp_pct,
        "above_50": fmt_1dp_pct,
        "above_200": fmt_1dp_pct,
        "ad_balance": fmt_int,
        "breakouts": fmt_int,
        "vcp_ready": fmt_int,
        "dist_52w_high": fmt_2dp_pct,
    }
    for col, fmt in numeric_map.items():
        if col in out.columns:
            out[col] = out[col].apply(fmt)
    return out


def format_stock_display_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["StrengthBand"] = out["Strength"].apply(strength_label)

    def fmt_strength_pct(x):
        if x is None or pd.isna(x):
            return "—"
        v = float(x)
        # If stored as 0–1, scale to percent
        if 0 <= v <= 1.01:
            return f"{v * 100:,.2f}%"
        # If stored as 0–100, keep as is
        if v <= 1000:
            return f"{v:,.2f}%"
        # If stored as 0–10000 style, scale down
        return f"{v / 100:,.2f}%"

    numeric_map = {
        "Rank": fmt_int,
        "Strength": fmt_2dp,
        "StrengthPct": fmt_strength_pct,
        "Close": fmt_2dp,
        "ret_1d": fmt_2dp_pct,
        "ret_5d": fmt_2dp_pct,
        "ret_20d": fmt_2dp_pct,
        "ret_60d": fmt_2dp_pct,
        "above_20": fmt_1dp_pct,
        "above_50": fmt_1dp_pct,
        "above_200": fmt_1dp_pct,
        "dist_52w_high": fmt_2dp_pct,
        "Accumulation": fmt_int,
        "Distribution": fmt_int,
        "Breakouts": fmt_int,
        "VCPReady": fmt_int,
    }
    for col, fmt in numeric_map.items():
        if col in out.columns:
            out[col] = out[col].apply(fmt)
    out["TrendTemplate"] = out["TrendTemplate"].apply(
        lambda x: "Pass" if x else "Fail" if pd.notna(x) else "—"
    )
    return out


def show_group_history_chart(
    history: pd.DataFrame,
    group_column: str,
    selected_group: str,
    title: str,
) -> None:
    st.markdown(f"#### {title}")
    metrics = {
        "Strength score": "strength_score",
        "20D return (%)": "eq_ret_20d",
        "60D return (%)": "eq_ret_60d",
        "Above 50 DMA (%)": "pct_above_50",
        "Above 200 DMA (%)": "pct_above_200",
        "A/D balance": "acc_minus_dist",
        "Breakout count": "breakout_count",
    }
    available_metrics = {
        label: col for label, col in metrics.items() if col in history.columns
    }
    chosen_metric = st.selectbox(
        "Group chart metric",
        options=list(available_metrics.keys()),
        key=f"group_chart_metric_{group_column}",
    )
    metric_column = available_metrics[chosen_metric]
    chart_data = history[history[group_column] == selected_group][
        ["date", metric_column]
    ].copy()
    chart_data = chart_data.sort_values("date").set_index("date")
    st.line_chart(chart_data, use_container_width=True, height=280)


def show_stock_history_chart(
    stock_history: pd.DataFrame,
    selected_group: str,
    selected_stocks: pd.DataFrame,
) -> None:
    st.markdown("#### Stock comparison history")
    symbols = sorted(selected_stocks["symbol"].unique().tolist())
    default_symbols = symbols[: min(4, len(symbols))]
    selected_symbols = st.multiselect(
        "Stocks to compare",
        options=symbols,
        default=default_symbols,
        max_selections=6,
        key="stock_chart_symbols",
    )
    if not selected_symbols:
        st.info("Choose at least one stock for comparison.")
        return
    metrics = {
        "Composite stock strength": "stock_strength_score",
        "20D return (%)": "ret_20d",
        "60D return (%)": "ret_60d",
        "Distance from 52W high": "dist_52w_high",
        "Close": "close",
    }
    available_metrics = {
        label: col for label, col in metrics.items() if col in stock_history.columns
    }
    selected_metric = st.selectbox(
        "Stock chart metric",
        options=list(available_metrics.keys()),
        key="stock_chart_metric",
    )
    metric_column = available_metrics[selected_metric]
    chart_data = stock_history[
        (stock_history["basic_industry"] == selected_group)
        & (stock_history["symbol"].isin(selected_symbols))
    ][["date", "symbol", metric_column]].copy()
    if metric_column == "dist_52w_high":
        chart_data[metric_column] = chart_data[metric_column] * 100
    chart_data = chart_data.pivot(
        index="date",
        columns="symbol",
        values=metric_column,
    ).sort_index()
    st.line_chart(chart_data, use_container_width=True, height=280)


def basic_industry_view(
    basic_history: pd.DataFrame,
    stock_history: pd.DataFrame,
) -> None:
    st.subheader("Basic Industry Leadership")
    st.caption(
        "Ranked view of granular NSE market leadership, breadth and participation. "
        "Select a Basic Industry to view its constituent stocks."
    )
    basic_history = ensure_group_columns(basic_history, "basic_industry")
    stock_history = ensure_stock_columns(stock_history)
    available_dates = get_trading_dates(basic_history)
    selected_date = date_navigation(available_dates, "basic_industry_leadership")
    selected = basic_history[basic_history["date"] == selected_date].copy()
    filter_1, filter_2, filter_3 = st.columns([1.4, 0.9, 0.9])
    all_regimes = ["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"]
    chosen_regimes = filter_1.multiselect(
        "Regime filter",
        options=all_regimes,
        default=all_regimes,
        key="basic_industry_regime_filter",
    )
    min_members = filter_2.number_input(
        "Minimum members",
        min_value=1,
        value=1,
        step=1,
        key="basic_industry_min_members",
    )
    sort_mode = filter_3.selectbox(
        "Ranking",
        options=["Highest strength", "Lowest strength"],
        key="basic_industry_sort",
    )
    if chosen_regimes:
        selected = selected[selected["regime"].isin(chosen_regimes)]
    if "members" in selected.columns:
        selected = selected[selected["members"] >= min_members]
    show_group_kpis(selected, "basic_industry", selected_date)
    table = prepare_group_table(selected, "basic_industry")
    if sort_mode == "Lowest strength" and "Strength" in table.columns:
        table = table.sort_values("Strength", ascending=True).reset_index(drop=True)
        table["Rank"] = range(1, len(table) + 1)
    display_table = format_group_display_table(table)
    group_options = display_table["Basic Industry"].tolist()
    selected_group = st.selectbox(
        "Selected Basic Industry",
        options=group_options,
        index=0,
        key="basic_industry_selector",
    )
    st.dataframe(
        display_table[
            [
                "Rank",
                "Basic Industry",
                "Regime",
                "StrengthBand",
                "Strength",
                "Members",
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "ret_60d",
                "above_50",
                "above_200",
                "ad_balance",
                "breakouts",
                "vcp_ready",
                "dist_52w_high",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=360,
    )
    st.download_button(
        "Download group table",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"basic_industry_{selected_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="basic_industry_download",
    )
    st.markdown("---")
    st.subheader(f"Constituent stocks: {selected_group}")
    stock_dates = set(pd.to_datetime(stock_history["date"]).unique().tolist())
    if selected_date not in stock_dates:
        earliest_stock_date = pd.Timestamp(stock_history["date"].min())
        st.info(
            "Stock-level drill-down is available from "
            f"{earliest_stock_date.strftime('%d %b %Y')} onward. "
            "Group-level history remains available for earlier dates."
        )
        show_group_history_chart(
            basic_history,
            "basic_industry",
            selected_group,
            "Basic Industry strength history",
        )
        return
    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (stock_history["basic_industry"] == selected_group)
    ].copy()
    if stocks.empty:
        st.warning("No stock-level data is available for this group/date.")
        return
    stock_table = prepare_stock_table(stocks)
    stock_display = format_stock_display_table(stock_table)
    st.dataframe(
        stock_display[
            [
                "Rank",
                "Symbol",
                "Strength",
                "StrengthPct",
                "StrengthBand",
                "Close",
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "ret_60d",
                "above_50",
                "above_200",
                "dist_52w_high",
                "TrendTemplate",
                "Accumulation",
                "Distribution",
                "Breakouts",
                "VCPReady",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=380,
    )
    st.download_button(
        "Download stock ranking",
        data=stock_table.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{selected_group.replace('/', '_')}_"
            f"stocks_{selected_date.strftime('%Y%m%d')}.csv"
        ),
        mime="text/csv",
        key="drilldown_stock_download",
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        show_group_history_chart(
            basic_history,
            "basic_industry",
            selected_group,
            "Basic Industry strength history",
        )
    with chart_right:
        show_stock_history_chart(stock_history, selected_group, stocks)


def industry_view(industry_history: pd.DataFrame) -> None:
    st.subheader("Industry Leadership")
    st.caption(
        "Higher-level industry leadership, breadth and rotation confirmation."
    )
    industry_history = ensure_group_columns(industry_history, "industry")
    available_dates = get_trading_dates(industry_history)
    selected_date = date_navigation(available_dates, "industry_leadership")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    filter_1, filter_2, filter_3 = st.columns([1.4, 0.9, 0.9])
    all_regimes = ["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"]
    chosen_regimes = filter_1.multiselect(
        "Regime filter",
        options=all_regimes,
        default=all_regimes,
        key="industry_regime_filter",
    )
    min_members = filter_2.number_input(
        "Minimum members",
        min_value=1,
        value=1,
        step=1,
        key="industry_min_members",
    )
    sort_mode = filter_3.selectbox(
        "Ranking",
        options=["Highest strength", "Lowest strength"],
        key="industry_sort",
    )
    if chosen_regimes:
        selected = selected[selected["regime"].isin(chosen_regimes)]
    if "members" in selected.columns:
        selected = selected[selected["members"] >= min_members]
    show_group_kpis(selected, "industry", selected_date)
    table = prepare_group_table(selected, "industry")
    if sort_mode == "Lowest strength" and "Strength" in table.columns:
        table = table.sort_values("Strength", ascending=True).reset_index(drop=True)
        table["Rank"] = range(1, len(table) + 1)
    display_table = format_group_display_table(table)
    st.dataframe(
        display_table[
            [
                "Rank",
                "Industry",
                "Regime",
                "StrengthBand",
                "Strength",
                "Members",
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "ret_60d",
                "above_50",
                "above_200",
                "ad_balance",
                "breakouts",
                "vcp_ready",
                "dist_52w_high",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=360,
    )
    st.download_button(
        "Download industry table",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"industry_{selected_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="industry_download",
    )


def overview_view(
    basic_history: pd.DataFrame,
    industry_history: pd.DataFrame,
    metadata: dict,
) -> None:
    st.subheader("Market Breadth Overview")
    st.caption(
        "High-level participation, leadership and risk across the classified NSE universe."
    )
    basic_history = ensure_group_columns(basic_history, "basic_industry")
    industry_history = ensure_group_columns(industry_history, "industry")
    latest_date = max(
        basic_history["date"].max(),
        industry_history["date"].max(),
    )
    basic_latest = basic_history[basic_history["date"] == latest_date].copy()
    industry_latest = industry_history[industry_history["date"] == latest_date].copy()
    counts = regime_counts(basic_latest)
    metrics = st.columns(5)
    metrics[0].metric("Latest data", latest_date.strftime("%d %b %Y"))
    metrics[1].metric(
        "Basic industries", fmt_int(basic_latest["basic_industry"].nunique())
    )
    metrics[2].metric(
        "Industries", fmt_int(industry_latest["industry"].nunique())
    )
    metrics[3].metric(
        "Strong / Emerging",
        f"{counts['Strong']} / {counts['Emerging']}",
    )
    metrics[4].metric(
        "Weakening / Exhausted",
        f"{counts['Weakening']} / {counts['Exhausted']}",
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Regime distribution")
        regime_table = pd.DataFrame({"Groups": counts})
        st.bar_chart(regime_table, height=280)
    with right:
        st.markdown("#### Breadth snapshot")
        breadth_columns = [
            c for c in ["pct_above_20", "pct_above_50", "pct_above_200"]
            if c in basic_latest.columns
        ]
        if breadth_columns:
            breadth = pd.DataFrame(
                {
                    "Median breadth %": [
                        basic_latest[c].median() for c in breadth_columns
                    ]
                },
                index=[
                    c.replace("pct_above_", "Above ").replace("_", " ") + " DMA"
                    for c in breadth_columns
                ],
            )
            st.bar_chart(breadth, height=280)
    st.markdown("#### Current leadership")
    leaders = prepare_group_table(basic_latest, "basic_industry").head(12)
    leaders_disp = format_group_display_table(leaders)
    st.dataframe(
        leaders_disp,
        use_container_width=True,
        hide_index=True,
        height=320,
    )
    if metadata:
        coverage = metadata.get("basic_industry", {})
        st.caption(
            "Group-level history: "
            f"{coverage.get('start_date', '—')} to "
            f"{coverage.get('latest_date', '—')}. "
            "Stock-level drill-down uses the most recent 400 trading days."
        )


def methodology_view() -> None:
    st.subheader("Methodology")
    st.caption(
        "All calculations are completed by the scheduled EOD workflow. "
        "The dashboard reads precomputed files only."
    )
    st.markdown(
        """
### Group strength framework

The daily group-strength framework combines:

- **Trend:** 20D and 60D return leadership plus participation above the 50 DMA.
- **Breadth:** percentage of stocks above the 20, 50 and 200 DMA.
- **Relative strength:** cross-group ranking of equal-weighted returns.
- **Volume behaviour:** accumulation days minus distribution days.
- **Breakout participation:** number of volume-confirmed breakouts and VCP-ready setups.
- **Risk / extension penalty:** distance from 52-week highs and short-term overextension.

### Stock rank within Basic Industry

Stocks are ranked within their selected Basic Industry using a precomputed composite:

- 20D return contribution.
- 60D return contribution.
- Above-50-DMA condition.
- Above-200-DMA condition.
- Breakout status.
- VCP-ready status.

### Regime interpretation

| Regime | Meaning |
|---|---|
| Strong | High composite strength and broad 50-DMA participation |
| Emerging | Improving momentum and breadth |
| Bottoming | Not yet a confirmed leadership or weakness state |
| Weakening | Weak short-term breadth with negative 20D return |
| Exhausted | Broad short-term extension close to 52-week highs |

This dashboard is a market-structure research tool and not investment advice.
        """
    )


def main() -> None:
    required_files = [
        BASIC_HISTORY_FILE,
        INDUSTRY_HISTORY_FILE,
        STOCK_HISTORY_FILE,
    ]
    missing_files = [
        str(file_path.relative_to(ROOT))
        for file_path in required_files
        if not file_path.exists()
    ]
    if missing_files:
        st.error(
            "Required dashboard files are missing. Run the EOD workflow first."
        )
        st.code("\n".join(missing_files))
        st.stop()
    basic_history = load_parquet(str(BASIC_HISTORY_FILE))
    industry_history = load_parquet(str(INDUSTRY_HISTORY_FILE))
    stock_history = load_parquet(str(STOCK_HISTORY_FILE))
    metadata = load_metadata(str(METADATA_FILE))
    basic_history = ensure_group_columns(basic_history, "basic_industry")
    industry_history = ensure_group_columns(industry_history, "industry")
    stock_history = ensure_stock_columns(stock_history)
    st.title("NSE Sectoral Breadth")
    sync_text = "Not available"
    if SYNC_FILE.exists():
        sync_text = SYNC_FILE.read_text(encoding="utf-8").strip()
    last_update = sync_text.split("T")[0] if "T" in sync_text else sync_text
    st.caption(
        f"Data as of: {last_update} | Last updated: {sync_text.replace('T', ' ').replace('Z', ' IST')}"
    )
    tabs = st.tabs(
        [
            "Basic Industry",
            "Industry",
            "Overview",
            "Methodology",
        ]
    )
    with tabs[0]:
        basic_industry_view(basic_history, stock_history)
    with tabs[1]:
        industry_view(industry_history)
    with tabs[2]:
        overview_view(basic_history, industry_history, metadata)
    with tabs[3]:
        methodology_view()


if __name__ == "__main__":
    main()
