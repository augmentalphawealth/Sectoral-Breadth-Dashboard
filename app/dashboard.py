from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_LATEST_FILE = (
    PROCESSED / "dashboard_basic_industry_latest.parquet"
)
INDUSTRY_LATEST_FILE = (
    PROCESSED / "dashboard_industry_latest.parquet"
)
BASIC_HISTORY_FILE = (
    PROCESSED / "dashboard_basic_industry_history.parquet"
)
INDUSTRY_HISTORY_FILE = (
    PROCESSED / "dashboard_industry_history.parquet"
)
METADATA_FILE = PROCESSED / "dashboard_metadata.json"
SYNC_FILE = PROCESSED / "last_sync.txt"


st.set_page_config(
    page_title="Sectoral Breadth",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clean_name(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    return str(value).strip() or "Unclassified"


def fmt_number(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_percent(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}%"


def fmt_integer(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def regime_color(regime: str) -> str:
    colors = {
        "Strong": "#16a34a",
        "Emerging": "#2563eb",
        "Bottoming": "#ca8a04",
        "Weakening": "#ea580c",
        "Exhausted": "#dc2626",
    }
    return colors.get(str(regime), "#64748b")


def regime_badge(regime: object) -> str:
    label = clean_name(regime)
    color = regime_color(label)

    return (
        f"<span style='display:inline-block;"
        f"padding:3px 8px;"
        f"border-radius:999px;"
        f"background:{color};"
        f"color:white;"
        f"font-size:0.78rem;"
        f"font-weight:700;'>"
        f"{label}"
        f"</span>"
    )


@st.cache_data(show_spinner=False)
def load_parquet(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data(show_spinner=False)
def load_metadata(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    required_files = [
        BASIC_LATEST_FILE,
        INDUSTRY_LATEST_FILE,
        BASIC_HISTORY_FILE,
        INDUSTRY_HISTORY_FILE,
    ]

    missing = [
        str(file_path.relative_to(ROOT))
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing:
        st.error(
            "Dashboard data is not available yet. "
            "Run the EOD workflow first."
        )
        st.code("\n".join(missing))
        st.stop()

    basic_latest = load_parquet(str(BASIC_LATEST_FILE))
    industry_latest = load_parquet(str(INDUSTRY_LATEST_FILE))
    basic_history = load_parquet(str(BASIC_HISTORY_FILE))
    industry_history = load_parquet(str(INDUSTRY_HISTORY_FILE))
    metadata = load_metadata(str(METADATA_FILE))

    return (
        basic_latest,
        industry_latest,
        basic_history,
        industry_history,
        metadata,
    )


def standardize_groups(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    data = df.copy()

    if group_column not in data.columns:
        data[group_column] = "Unclassified"

    data[group_column] = data[group_column].map(clean_name)

    if "regime" not in data.columns:
        data["regime"] = "Unclassified"

    data["regime"] = data["regime"].map(clean_name)

    return data


def selected_date_frame(
    history: pd.DataFrame,
    group_column: str,
    selected_date: pd.Timestamp,
) -> pd.DataFrame:
    data = history.copy()
    data = data[data["date"] == selected_date].copy()
    data = standardize_groups(data, group_column)

    return data


def display_kpis(
    data: pd.DataFrame,
    selected_date: pd.Timestamp,
    group_label: str,
) -> None:
    total_groups = data[group_label].nunique()

    total_members = (
        int(data["members"].sum())
        if "members" in data.columns
        else 0
    )

    strong = int((data["regime"] == "Strong").sum())
    emerging = int((data["regime"] == "Emerging").sum())
    weak = int((data["regime"] == "Weakening").sum())
    exhausted = int((data["regime"] == "Exhausted").sum())

    median_score = (
        data["strength_score"].median()
        if "strength_score" in data.columns
        else float("nan")
    )

    columns = st.columns(6)

    columns[0].metric(
        "As of",
        selected_date.strftime("%d %b %Y"),
    )
    columns[1].metric(
        f"{group_label.replace('_', ' ').title()} groups",
        f"{total_groups:,}",
    )
    columns[2].metric(
        "Constituent memberships",
        f"{total_members:,}",
    )
    columns[3].metric(
        "Median strength",
        fmt_number(median_score),
    )
    columns[4].metric(
        "Strong / Emerging",
        f"{strong} / {emerging}",
    )
    columns[5].metric(
        "Weakening / Exhausted",
        f"{weak} / {exhausted}",
    )


def prepare_leadership_table(
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

    available = [
        column
        for column in columns
        if column in data.columns
    ]

    table = data[available].copy()

    rename_map = {
        group_column: "Group",
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
        "median_dist_52w_high": "Median from 52W High %",
    }

    table = table.rename(columns=rename_map)

    if "Median from 52W High %" in table.columns:
        table["Median from 52W High %"] = (
            table["Median from 52W High %"] * 100
        )

    if "Strength" in table.columns:
        table = table.sort_values(
            "Strength",
            ascending=False,
        )

    return table.reset_index(drop=True)


def style_leadership_table(table: pd.DataFrame):
    formatters = {}

    percent_columns = [
        "1D %",
        "5D %",
        "20D %",
        "60D %",
        "Above 20 DMA %",
        "Above 50 DMA %",
        "Above 200 DMA %",
        "Median from 52W High %",
    ]

    for column in percent_columns:
        if column in table.columns:
            formatters[column] = "{:.1f}%"

    if "Strength" in table.columns:
        formatters["Strength"] = "{:.1f}"

    for column in [
        "Members",
        "A/D Balance",
        "Breakouts",
        "VCP Ready",
    ]:
        if column in table.columns:
            formatters[column] = "{:,.0f}"

    styled = table.style.format(formatters, na_rep="—")

    if "Regime" in table.columns:
        styled = styled.map(
            lambda value: (
                f"color: {regime_color(str(value))}; "
                "font-weight: 700;"
            ),
            subset=["Regime"],
        )

    if "Strength" in table.columns:
        styled = styled.background_gradient(
            subset=["Strength"],
            cmap="RdYlGn",
        )

    return styled


def leadership_view(
    history: pd.DataFrame,
    group_column: str,
    title: str,
    subtitle: str,
) -> None:
    history = standardize_groups(history, group_column)

    all_dates = sorted(
        history["date"].dropna().unique(),
        reverse=True,
    )

    if not all_dates:
        st.warning("No historical dates are available.")
        return

    latest_date = pd.Timestamp(all_dates[0])

    st.subheader(title)
    st.caption(subtitle)

    left, middle, right, far_right = st.columns(
        [1.35, 1.15, 1.15, 1.15]
    )

    selected_date = left.selectbox(
        "Historical date",
        options=all_dates,
        index=0,
        format_func=lambda value: (
            pd.Timestamp(value).strftime("%d %b %Y")
        ),
        key=f"{group_column}_date",
    )

    regime_options = sorted(
        history["regime"].dropna().unique().tolist()
    )

    selected_regimes = middle.multiselect(
        "Regime",
        options=regime_options,
        default=regime_options,
        key=f"{group_column}_regimes",
    )

    min_members = right.number_input(
        "Minimum members",
        min_value=1,
        max_value=1000,
        value=1,
        step=1,
        key=f"{group_column}_min_members",
    )

    rank_order = far_right.selectbox(
        "Display",
        options=["Top strength", "Bottom strength"],
        key=f"{group_column}_order",
    )

    selected_date = pd.Timestamp(selected_date)

    data = selected_date_frame(
        history,
        group_column,
        selected_date,
    )

    if selected_regimes:
        data = data[data["regime"].isin(selected_regimes)]

    if "members" in data.columns:
        data = data[data["members"] >= min_members]

    display_kpis(data, selected_date, group_column)

    table = prepare_leadership_table(data, group_column)

    if rank_order == "Bottom strength" and "Strength" in table.columns:
        table = table.sort_values(
            "Strength",
            ascending=True,
        ).reset_index(drop=True)

    st.dataframe(
        style_leadership_table(table),
        use_container_width=True,
        hide_index=True,
        height=580,
    )

    csv_data = table.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download selected leadership table",
        data=csv_data,
        file_name=(
            f"{group_column}_leadership_"
            f"{selected_date.strftime('%Y%m%d')}.csv"
        ),
        mime="text/csv",
        key=f"{group_column}_download",
    )


def historical_rotation_view(
    basic_history: pd.DataFrame,
    industry_history: pd.DataFrame,
) -> None:
    st.subheader("Historical Rotation")
    st.caption(
        "Charts use precomputed daily group metrics only. "
        "No indicator or ranking calculations run in Streamlit."
    )

    view_type = st.radio(
        "Group level",
        options=["Basic Industry", "Industry"],
        horizontal=True,
    )

    if view_type == "Basic Industry":
        data = standardize_groups(
            basic_history,
            "basic_industry",
        )
        group_column = "basic_industry"
    else:
        data = standardize_groups(
            industry_history,
            "industry",
        )
        group_column = "industry"

    metrics = {
        "Strength score": "strength_score",
        "20-day return (%)": "eq_ret_20d",
        "60-day return (%)": "eq_ret_60d",
        "Breadth above 50 DMA (%)": "pct_above_50",
        "Breadth above 200 DMA (%)": "pct_above_200",
        "Accumulation / distribution balance": "acc_minus_dist",
        "Breakout count": "breakout_count",
        "VCP-ready count": "vcp_ready_count",
    }

    available_metrics = {
        label: column
        for label, column in metrics.items()
        if column in data.columns
    }

    group_names = sorted(data[group_column].unique().tolist())

    controls_left, controls_right = st.columns([1, 2])

    metric_label = controls_left.selectbox(
        "Metric",
        options=list(available_metrics.keys()),
    )

    selected_groups = controls_right.multiselect(
        "Groups to compare",
        options=group_names,
        default=group_names[: min(3, len(group_names))],
        max_selections=8,
    )

    if not selected_groups:
        st.info("Choose at least one group to display history.")
        return

    min_date = data["date"].min().date()
    max_date = data["date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = min_date
        end_date = max_date

    metric_column = available_metrics[metric_label]

    chart_data = data[
        (data[group_column].isin(selected_groups))
        & (data["date"].dt.date >= start_date)
        & (data["date"].dt.date <= end_date)
    ][["date", group_column, metric_column]].copy()

    chart_data = chart_data.pivot(
        index="date",
        columns=group_column,
        values=metric_column,
    )

    st.line_chart(
        chart_data,
        use_container_width=True,
        height=450,
    )

    latest = data[
        data[group_column].isin(selected_groups)
    ].sort_values("date").groupby(group_column).tail(1)

    latest_table = prepare_leadership_table(
        latest,
        group_column,
    )

    st.markdown("#### Latest selected-group snapshot")

    st.dataframe(
        style_leadership_table(latest_table),
        use_container_width=True,
        hide_index=True,
    )


def overview_view(
    basic_history: pd.DataFrame,
    industry_history: pd.DataFrame,
    metadata: dict,
) -> None:
    basic_history = standardize_groups(
        basic_history,
        "basic_industry",
    )
    industry_history = standardize_groups(
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

    st.subheader("Market Breadth Overview")
    st.caption(
        "A compact view of leadership, participation, and "
        "risk across the classified NSE universe."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest data",
        latest_date.strftime("%d %b %Y"),
    )

    col2.metric(
        "Basic industries",
        f"{basic_latest['basic_industry'].nunique():,}",
    )

    col3.metric(
        "Industries",
        f"{industry_latest['industry'].nunique():,}",
    )

    col4.metric(
        "Coverage",
        "2,553 classified stocks",
    )

    st.markdown("#### Basic-industry regime distribution")

    regime_counts = (
        basic_latest["regime"]
        .value_counts()
        .reindex(
            [
                "Strong",
                "Emerging",
                "Bottoming",
                "Weakening",
                "Exhausted",
            ],
            fill_value=0,
        )
        .rename_axis("Regime")
        .reset_index(name="Groups")
    )

    regime_counts = regime_counts.set_index("Regime")

    st.bar_chart(regime_counts, height=300)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Leading basic industries")

        leaders = prepare_leadership_table(
            basic_latest,
            "basic_industry",
        ).head(10)

        st.dataframe(
            style_leadership_table(leaders),
            use_container_width=True,
            hide_index=True,
            height=390,
        )

    with right:
        st.markdown("#### Weakest basic industries")

        weakest = prepare_leadership_table(
            basic_latest,
            "basic_industry",
        )

        if "Strength" in weakest.columns:
            weakest = weakest.sort_values(
                "Strength",
                ascending=True,
            ).head(10)

        st.dataframe(
            style_leadership_table(weakest),
            use_container_width=True,
            hide_index=True,
            height=390,
        )

    if metadata:
        st.caption(
            "History coverage: "
            f"{metadata.get('basic_industry', {}).get('start_date', '—')} "
            "to "
            f"{metadata.get('basic_industry', {}).get('latest_date', '—')}. "
            "All market measures are precomputed by the EOD workflow."
        )


def methodology_view() -> None:
    st.subheader("Methodology")
    st.caption(
        "All measures are calculated by the scheduled EOD workflow. "
        "The Streamlit app only displays prepared tables."
    )

    st.markdown(
        """
### Strength framework

The group-level strength score combines:

- **Trend:** 20-day and 60-day return leadership plus breadth above the 50 DMA.
- **Breadth:** percentage of member stocks above the 20, 50, and 200 DMA.
- **Relative strength:** peer ranking of 20-day and 60-day equal-weighted returns.
- **Volume behaviour:** accumulation days minus distribution days.
- **Breakout participation:** member breakouts and VCP-ready setups.
- **Penalty:** damage relative to 52-week highs and overextended short-term breadth.

### Regimes

| Regime | Interpretation |
|---|---|
| Strong | High composite strength with broad 50-DMA participation |
| Emerging | Improving momentum and breadth |
| Bottoming | Neither a confirmed leadership nor weakness condition |
| Weakening | Weak short-term breadth with negative 20-day return |
| Exhausted | Very broad short-term extension close to 52-week highs |

### Data policy

- Historical prices remain in the Situational-Awareness repository.
- This app uses only compact, precomputed Industry and Basic Industry history.
- It does not download raw price data or calculate technical indicators.
- Regimes and scores are research tools, not investment recommendations.
        """
    )


def main() -> None:
    (
        basic_latest,
        industry_latest,
        basic_history,
        industry_history,
        metadata,
    ) = load_data()

    basic_history = standardize_groups(
        basic_history,
        "basic_industry",
    )
    industry_history = standardize_groups(
        industry_history,
        "industry",
    )

    st.title("NSE Sectoral Breadth")
    st.caption(
        "Precomputed breadth, trend health, and leadership across "
        "classified NSE Basic Industries and Industries."
    )

    sync_text = "Not available"

    if SYNC_FILE.exists():
        sync_text = SYNC_FILE.read_text(
            encoding="utf-8"
        ).strip()

    with st.sidebar:
        st.markdown("### Data Status")
        st.caption(f"Last EOD build: {sync_text}")

        st.markdown("---")
        st.markdown(
            """
**Data architecture**

- Raw historical prices: shared source
- Calculations: EOD GitHub workflow
- Streamlit: prepared tables only
            """
        )

    tabs = st.tabs(
        [
            "Basic Industry",
            "Overview",
            "Industry",
            "Historical Rotation",
            "Methodology",
        ]
    )

    with tabs[0]:
        leadership_view(
            history=basic_history,
            group_column="basic_industry",
            title="Basic Industry Leadership",
            subtitle=(
                "Default view: granular leadership and breadth "
                "across NSE basic industries."
            ),
        )

    with tabs[1]:
        overview_view(
            basic_history=basic_history,
            industry_history=industry_history,
            metadata=metadata,
        )

    with tabs[2]:
        leadership_view(
            history=industry_history,
            group_column="industry",
            title="Industry Leadership",
            subtitle=(
                "Higher-level industry grouping for broader "
                "rotation and trend confirmation."
            ),
        )

    with tabs[3]:
        historical_rotation_view(
            basic_history=basic_history,
            industry_history=industry_history,
        )

    with tabs[4]:
        methodology_view()


if __name__ == "__main__":
    main()
