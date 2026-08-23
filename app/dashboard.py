from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = (
    PROCESSED / "dashboard_basic_industry_history.parquet"
)
INDUSTRY_HISTORY_FILE = (
    PROCESSED / "dashboard_industry_history.parquet"
)
STOCK_HISTORY_FILE = (
    PROCESSED / "dashboard_stock_history.parquet"
)
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


def fmt_number(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_percent(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}%"


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def regime_color(regime: object) -> str:
    palette = {
        "Strong": "#0f9d58",
        "Emerging": "#2563eb",
        "Bottoming": "#a16207",
        "Weakening": "#ea580c",
        "Exhausted": "#dc2626",
    }
    return palette.get(clean_text(regime), "#64748b")


def regime_badge(regime: object) -> str:
    label = clean_text(regime)
    color = regime_color(label)

    return (
        "<span style='display:inline-block;"
        "padding:3px 9px;"
        "border-radius:999px;"
        f"background:{color};"
        "color:#ffffff;"
        "font-size:0.76rem;"
        "font-weight:700;'>"
        f"{label}"
        "</span>"
    )


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


def ensure_group_columns(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
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

    for column in [
        "symbol",
        "industry",
        "basic_industry",
        "sector",
    ]:
        if column not in data.columns:
            data[column] = "Unclassified"

        data[column] = data[column].map(clean_text)

    return data


def apply_table_style(table: pd.DataFrame):
    formatters: dict[str, str] = {}

    percentage_columns = [
        "1D %",
        "5D %",
        "20D %",
        "60D %",
        "Above 20 DMA %",
        "Above 50 DMA %",
        "Above 200 DMA %",
        "From 52W High %",
        "Strength %",
    ]

    for column in percentage_columns:
        if column in table.columns:
            formatters[column] = "{:.1f}%"

    for column in [
        "Strength",
        "Stock Strength",
    ]:
        if column in table.columns:
            formatters[column] = "{:.1f}"

    for column in [
        "Rank",
        "Members",
        "A/D Balance",
        "Breakouts",
        "VCP Ready",
        "Trend Template",
        "Accumulation",
        "Distribution",
    ]:
        if column in table.columns:
            formatters[column] = "{:,.0f}"

    return table.style.format(formatters, na_rep="—")


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
        [0.7, 2.2, 0.7, 1.0]
    )

    if col_previous.button(
        "← Previous",
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
        format_func=lambda value: (
            pd.Timestamp(value).strftime("%d %b %Y")
        ),
        key=f"{widget_key}_date_picker",
    )

    if pd.Timestamp(date_choice) != selected_date:
        st.session_state[state_key] = pd.Timestamp(date_choice)
        selected_date = pd.Timestamp(date_choice)

    current_index = available_dates.index(selected_date)

    if col_next.button(
        "Next →",
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
    regimes = [
        "Strong",
        "Emerging",
        "Bottoming",
        "Weakening",
        "Exhausted",
    ]

    counts = data["regime"].value_counts().to_dict()

    return {
        regime: int(counts.get(regime, 0))
        for regime in regimes
    }


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

    columns = st.columns(6)

    columns[0].metric(
        "As of",
        selected_date.strftime("%d %b %Y"),
    )
    columns[1].metric(
        "Groups",
        fmt_int(data[group_column].nunique()),
    )
    columns[2].metric(
        "Members",
        fmt_int(total_members),
    )
    columns[3].metric(
        "Median strength",
        fmt_number(median_strength),
    )
    columns[4].metric(
        "Median above 50 DMA",
        fmt_percent(median_breadth_50),
    )
    columns[5].metric(
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

    selected = [
        column
        for column in columns
        if column in data.columns
    ]

    table = data[selected].copy()

    table = table.rename(
        columns={
            group_column: "Basic Industry"
            if group_column == "basic_industry"
            else "Industry",
            "regime": "Regime",
            "members": "Members",
            "strength_score": "Strength",
            "eq_ret_1d": "1D %",
            "eq_ret_5d": "5D %",
            "eq_ret_20d": "20D %",
            "eq_ret_60d": "60D %",
            "pct_above_20": "Above 20 DMA %",
            "pct_above_50": "Above 50 DMA %",
            "pct_above_200": "Above 200 DMA %",
            "acc_minus_dist": "A/D Balance",
            "breakout_count": "Breakouts",
            "vcp_ready_count": "VCP Ready",
            "median_dist_52w_high": "From 52W High %",
        }
    )

    if "From 52W High %" in table.columns:
        table["From 52W High %"] = (
            table["From 52W High %"] * 100
        )

    if "Strength" in table.columns:
        table = table.sort_values(
            "Strength",
            ascending=False,
        )

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

    selected = [
        column
        for column in columns
        if column in data.columns
    ]

    table = data[selected].copy()

    table = table.rename(
        columns={
            "stock_rank_in_basic_industry": "Rank",
            "symbol": "Symbol",
            "stock_strength_score": "Stock Strength",
            "stock_strength_percentile_in_basic_industry": "Strength %",
            "close": "Close",
            "ret_1d": "1D %",
            "ret_5d": "5D %",
            "ret_20d": "20D %",
            "ret_60d": "60D %",
            "above_20": "Above 20 DMA",
            "above_50": "Above 50 DMA",
            "above_200": "Above 200 DMA",
            "dist_52w_high": "From 52W High %",
            "trend_template_pass": "Trend Template",
            "acc_day": "Accumulation",
            "dist_day": "Distribution",
            "breakout_55": "Breakouts",
            "vcp_ready": "VCP Ready",
        }
    )

    if "From 52W High %" in table.columns:
        table["From 52W High %"] = (
            table["From 52W High %"] * 100
        )

    if "Rank" in table.columns:
        table = table.sort_values(
            ["Rank", "Symbol"],
        )

    return table.reset_index(drop=True)


def group_leadership_view(
    history: pd.DataFrame,
    group_column: str,
    title: str,
    description: str,
    key_prefix: str,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    data = ensure_group_columns(history, group_column)
    available_dates = get_trading_dates(data)

    st.subheader(title)
    st.caption(description)

    selected_date = date_navigation(
        available_dates,
        key_prefix,
    )

    selected = data[
        data["date"] == selected_date
    ].copy()

    filter_1, filter_2, filter_3 = st.columns(
        [1.5, 1.0, 1.0]
    )

    all_regimes = [
        "Strong",
        "Emerging",
        "Bottoming",
        "Weakening",
        "Exhausted",
    ]

    chosen_regimes = filter_1.multiselect(
        "Regime filter",
        options=all_regimes,
        default=all_regimes,
        key=f"{key_prefix}_regime_filter",
    )

    min_members = filter_2.number_input(
        "Minimum members",
        min_value=1,
        value=1,
        step=1,
        key=f"{key_prefix}_min_members",
    )

    sort_mode = filter_3.selectbox(
        "Ranking",
        options=["Highest strength", "Lowest strength"],
        key=f"{key_prefix}_sort",
    )

    if chosen_regimes:
        selected = selected[
            selected["regime"].isin(chosen_regimes)
        ]

    if "members" in selected.columns:
        selected = selected[
            selected["members"] >= min_members
        ]

    show_group_kpis(
        selected,
        group_column,
        selected_date,
    )

    table = prepare_group_table(selected, group_column)

    if sort_mode == "Lowest strength" and "Strength" in table.columns:
        table = table.sort_values(
            "Strength",
            ascending=True,
        ).reset_index(drop=True)

        table["Rank"] = range(1, len(table) + 1)

    st.dataframe(
        apply_table_style(table),
        use_container_width=True,
        hide_index=True,
        height=540,
    )

    st.download_button(
        "Download selected table",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{group_column}_{selected_date.strftime('%Y%m%d')}.csv"
        ),
        mime="text/csv",
        key=f"{key_prefix}_download",
    )

    return selected_date, selected


def basic_industry_drilldown(
    basic_history: pd.DataFrame,
    stock_history: pd.DataFrame,
) -> None:
    st.subheader("Basic Industry Drill-Down")
    st.caption(
        "Select a Basic Industry to view its constituent stocks "
        "ranked by precomputed composite stock strength."
    )

    basic_history = ensure_group_columns(
        basic_history,
        "basic_industry",
    )
    stock_history = ensure_stock_columns(stock_history)

    available_dates = get_trading_dates(basic_history)

    selected_date = date_navigation(
        available_dates,
        "drilldown",
    )

    group_data = basic_history[
        basic_history["date"] == selected_date
    ].copy()

    group_options = sorted(
        group_data["basic_industry"].unique().tolist()
    )

    selected_group = st.selectbox(
        "Basic Industry",
        options=group_options,
        key="drilldown_basic_industry",
    )

    selected_group_row = group_data[
        group_data["basic_industry"] == selected_group
    ].head(1)

    if not selected_group_row.empty:
        row = selected_group_row.iloc[0]

        metrics = st.columns(6)

        metrics[0].metric(
            "Regime",
            clean_text(row.get("regime")),
        )
        metrics[1].metric(
            "Strength",
            fmt_number(row.get("strength_score")),
        )
        metrics[2].metric(
            "Members",
            fmt_int(row.get("members")),
        )
        metrics[3].metric(
            "20D return",
            fmt_percent(row.get("eq_ret_20d")),
        )
        metrics[4].metric(
            "60D return",
            fmt_percent(row.get("eq_ret_60d")),
        )
        metrics[5].metric(
            "Above 50 DMA",
            fmt_percent(row.get("pct_above_50")),
        )

    stock_dates = set(
        pd.to_datetime(stock_history["date"]).unique().tolist()
    )

    if selected_date not in stock_dates:
        earliest_stock_date = pd.Timestamp(
            stock_history["date"].min()
        )

        st.info(
            "Stock-level drill-down is available from "
            f"{earliest_stock_date.strftime('%d %b %Y')} onward. "
            "Group-level history remains available for earlier dates."
        )

        show_group_history_chart(
            basic_history,
            "basic_industry",
            selected_group,
            "Group strength history",
        )
        return

    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (
            stock_history["basic_industry"]
            == selected_group
        )
    ].copy()

    if stocks.empty:
        st.warning(
            "No stock-level data is available for this group/date."
        )
        return

    stock_table = prepare_stock_table(stocks)

    st.markdown("#### Constituent stock ranking")

    st.dataframe(
        apply_table_style(stock_table),
        use_container_width=True,
        hide_index=True,
        height=440,
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
        show_stock_history_chart(
            stock_history,
            selected_group,
            stocks,
        )


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
        label: column
        for label, column in metrics.items()
        if column in history.columns
    }

    chosen_metric = st.selectbox(
        "Group chart metric",
        options=list(available_metrics.keys()),
        key=f"group_chart_metric_{group_column}",
    )

    metric_column = available_metrics[chosen_metric]

    chart_data = history[
        history[group_column] == selected_group
    ][["date", metric_column]].copy()

    chart_data = chart_data.sort_values("date").set_index("date")

    st.line_chart(
        chart_data,
        use_container_width=True,
        height=300,
    )


def show_stock_history_chart(
    stock_history: pd.DataFrame,
    selected_group: str,
    selected_stocks: pd.DataFrame,
) -> None:
    st.markdown("#### Stock comparison history")

    symbols = sorted(
        selected_stocks["symbol"].unique().tolist()
    )

    default_symbols = symbols[: min(3, len(symbols))]

    selected_symbols = st.multiselect(
        "Stocks to compare",
        options=symbols,
        default=default_symbols,
        max_selections=8,
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
        label: column
        for label, column in metrics.items()
        if column in stock_history.columns
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
        chart_data[metric_column] = (
            chart_data[metric_column] * 100
        )

    chart_data = chart_data.pivot(
        index="date",
        columns="symbol",
        values=metric_column,
    ).sort_index()

    st.line_chart(
        chart_data,
        use_container_width=True,
        height=300,
    )


def overview_view(
    basic_history: pd.DataFrame,
    industry_history: pd.DataFrame,
    metadata: dict,
) -> None:
    st.subheader("Market Breadth Overview")
    st.caption(
        "High-level participation, leadership and risk across "
        "the classified NSE universe."
    )

    basic_history = ensure_group_columns(
        basic_history,
        "basic_industry",
    )
    industry_history = ensure_group_columns(
        industry_history,
        "industry",
    )

    latest_date = max(
        basic_history["date"].max(),
        industry_history["date"].max(),
    )

    basic_latest = basic_history[
        basic_history["date"] == latest_date
    ].copy()

    industry_latest = industry_history[
        industry_history["date"] == latest_date
    ].copy()

    counts = regime_counts(basic_latest)

    metrics = st.columns(5)

    metrics[0].metric(
        "Latest data",
        latest_date.strftime("%d %b %Y"),
    )
    metrics[1].metric(
        "Basic industries",
        fmt_int(basic_latest["basic_industry"].nunique()),
    )
    metrics[2].metric(
        "Industries",
        fmt_int(industry_latest["industry"].nunique()),
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

        regime_table = pd.DataFrame(
            {
                "Groups": counts,
            }
        )

        st.bar_chart(
            regime_table,
            height=320,
        )

    with right:
        st.markdown("#### Breadth snapshot")

        breadth_columns = [
            column
            for column in [
                "pct_above_20",
                "pct_above_50",
                "pct_above_200",
            ]
            if column in basic_latest.columns
        ]

        if breadth_columns:
            breadth = pd.DataFrame(
                {
                    "Median breadth %": [
                        basic_latest[column].median()
                        for column in breadth_columns
                    ]
                },
                index=[
                    column.replace("pct_above_", "Above ")
                    .replace("_", " ")
                    + " DMA"
                    for column in breadth_columns
                ],
            )

            st.bar_chart(
                breadth,
                height=320,
            )

    st.markdown("#### Current leadership")

    leaders = prepare_group_table(
        basic_latest,
        "basic_industry",
    ).head(12)

    st.dataframe(
        apply_table_style(leaders),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    if metadata:
        coverage = metadata.get("basic_industry", {})

        st.caption(
            "Group-level history: "
            f"{coverage.get('start_date', '—')} to "
            f"{coverage.get('latest_date', '—')}. "
            "Stock-level drill-down uses the most recent "
            "400 trading days."
        )


def industry_view(
    industry_history: pd.DataFrame,
) -> None:
    group_leadership_view(
        history=industry_history,
        group_column="industry",
        title="Industry Leadership",
        description=(
            "Higher-level industry leadership, breadth and "
            "rotation confirmation."
        ),
        key_prefix="industry_leadership",
    )


def methodology_view() -> None:
    st.subheader("Methodology")
    st.caption(
        "All calculations are completed by the scheduled EOD "
        "workflow. The dashboard reads precomputed files only."
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
            "Required dashboard files are missing. "
            "Run the EOD workflow first."
        )
        st.code("\n".join(missing_files))
        st.stop()

    basic_history = load_parquet(str(BASIC_HISTORY_FILE))
    industry_history = load_parquet(str(INDUSTRY_HISTORY_FILE))
    stock_history = load_parquet(str(STOCK_HISTORY_FILE))
    metadata = load_metadata(str(METADATA_FILE))

    basic_history = ensure_group_columns(
        basic_history,
        "basic_industry",
    )
    industry_history = ensure_group_columns(
        industry_history,
        "industry",
    )
    stock_history = ensure_stock_columns(stock_history)

    st.title("NSE Sectoral Breadth")
    st.caption(
        "Basic Industry leadership, constituent strength and "
        "historical rotation — precomputed daily."
    )

    sync_text = "Not available"

    if SYNC_FILE.exists():
        sync_text = SYNC_FILE.read_text(
            encoding="utf-8"
        ).strip()

    st.markdown(
        f"""
<div style="
    padding:8px 12px;
    border:1px solid #e2e8f0;
    border-radius:8px;
    background:#f8fafc;
    color:#475569;
    font-size:0.86rem;
    margin-bottom:14px;
">
    Data refresh: {sync_text}
    &nbsp;&nbsp;•&nbsp;&nbsp;
    Group history: Apr 2018 onward
    &nbsp;&nbsp;•&nbsp;&nbsp;
    Stock drill-down: recent 400 trading days
</div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Basic Industry",
            "Group Drill-Down",
            "Overview",
            "Industry",
            "Methodology",
        ]
    )

    with tabs[0]:
        group_leadership_view(
            history=basic_history,
            group_column="basic_industry",
            title="Basic Industry Leadership",
            description=(
                "Ranked view of granular NSE market leadership, "
                "breadth and participation."
            ),
            key_prefix="basic_industry_leadership",
        )

    with tabs[1]:
        basic_industry_drilldown(
            basic_history=basic_history,
            stock_history=stock_history,
        )

    with tabs[2]:
        overview_view(
            basic_history=basic_history,
            industry_history=industry_history,
            metadata=metadata,
        )

    with tabs[3]:
        industry_view(industry_history)

    with tabs[4]:
        methodology_view()


if __name__ == "__main__":
    main()
