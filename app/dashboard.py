# app/dashboard.py
# NSE Industry Momentum Monitor
#
# Purpose:
# 1. Show which Basic Industries are improving or weakening.
# 2. Show Top 20 individual stock setups across the whole market.
# 3. Show Industry Rank / Industry Leadership as context only.
#
# Important:
# - No Pandas Styler: avoids Streamlit Styler errors.
# - No dataframe merge for industry context: avoids duplicate-column errors.
# - No treemap/scatter: avoids unreadable clutter.
# - Handles scores stored either as 0-1 or 0-100.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# FILE PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = PROCESSED / "dashboard_basic_industry_history.parquet"
INDUSTRY_HISTORY_FILE = PROCESSED / "dashboard_industry_history.parquet"
STOCK_HISTORY_FILE = PROCESSED / "dashboard_stock_history.parquet"
SYNC_FILE = PROCESSED / "last_sync.txt"

SMALL_GROUP_LIMIT = 5
TOP_INDUSTRIES = 12
TOP_STOCKS = 20


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

INK = "#0F172A"
MUTED = "#64748B"
GREEN = "#15803D"
RED = "#B91C1C"
BLUE = "#1D4ED8"
AMBER = "#B45309"

REGIME_COLORS = {
    "Fresh Leader (HUNT)": GREEN,
    "Extended Leader (WAIT)": AMBER,
    "Speculative Coil (AVOID)": "#7C3AED",
    "Dead (AVOID)": RED,
    "Neutral Transition": MUTED,
}


st.set_page_config(
    page_title="NSE Industry Momentum Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1380px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    h1 {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #0F172A;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #0F172A;
        font-size: 1.55rem;
        font-weight: 750;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    .note {
        color: #64748B;
        font-size: 0.88rem;
        margin-top: -0.35rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SAFE HELPERS
# =============================================================================

def clean_text(value: object) -> str:
    """Return safe text for missing group/symbol values."""
    if value is None or pd.isna(value):
        return "Unclassified"

    text = str(value).strip()
    return text if text else "Unclassified"


def numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return a numeric series even if the original column is missing."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(default)


def normalize_score(values: pd.Series) -> pd.Series:
    """
    Scores should be shown as 0-100.

    If the upstream pipeline stored a score as 0.00 to 1.00,
    convert it to 0 to 100. Otherwise leave it unchanged.
    """
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()

    if len(valid) > 0 and valid.max() <= 1.5:
        return series * 100.0

    return series


def format_number(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"

        return f"{float(value):,.{decimals}f}"

    except (TypeError, ValueError):
        return "—"


def format_integer(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"

        return f"{int(round(float(value))):,}"

    except (TypeError, ValueError):
        return "—"


def format_return(value: object) -> str:
    """
    For return fields stored as fractions:
    0.075 becomes 7.5%.
    """
    try:
        if value is None or pd.isna(value):
            return "—"

        return f"{float(value) * 100.0:,.1f}%"

    except (TypeError, ValueError):
        return "—"


def apply_chart_style(
    figure: go.Figure,
    height: int,
) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=8, r=25, t=42, b=10),
        font=dict(
            family="Inter, -apple-system, Segoe UI, sans-serif",
            size=12,
            color=INK,
        ),
        title_font=dict(size=14, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )

    return figure


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(
            frame["date"]
        ).dt.normalize()

    return frame


def prepare_group_data(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Prepare Basic Industry or Industry history data."""
    data = frame.copy()

    if group_column not in data.columns:
        data[group_column] = "Unclassified"

    if "regime" not in data.columns:
        data["regime"] = "Neutral Transition"

    if "leadership_score" not in data.columns:
        if "strength_score" in data.columns:
            data["leadership_score"] = data["strength_score"]
        else:
            data["leadership_score"] = 0.0

    if "actionability_score" not in data.columns:
        data["actionability_score"] = 0.0

    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)

    data["leadership_score"] = normalize_score(
        data["leadership_score"]
    ).fillna(0.0)

    data["actionability_score"] = normalize_score(
        data["actionability_score"]
    ).fillna(0.0)

    return data


def prepare_stock_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare stock data and guarantee the setup flags exist."""
    data = frame.copy()

    for column in [
        "symbol",
        "basic_industry",
        "industry",
        "sector",
    ]:
        if column not in data.columns:
            data[column] = "Unclassified"

        data[column] = data[column].map(clean_text)

    for flag in [
        "established_buy_setup",
        "ipo_buy_setup",
    ]:
        if flag not in data.columns:
            data[flag] = 0

        data[flag] = pd.to_numeric(
            data[flag],
            errors="coerce",
        ).fillna(0).astype(int)

    return data


def get_trading_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if "date" not in frame.columns:
        return []

    return sorted(
        pd.Timestamp(value)
        for value in pd.to_datetime(
            frame["date"].dropna().unique()
        )
    )


# =============================================================================
# DATE PICKER
# =============================================================================

def date_picker(
    all_dates: list[pd.Timestamp],
    key: str,
) -> pd.Timestamp:
    """
    This function is included in the replacement.
    It prevents the earlier: NameError: date_picker is not defined.
    """
    latest = all_dates[-1]
    state_key = f"{key}_selected_date"

    if state_key not in st.session_state:
        st.session_state[state_key] = latest

    selected = pd.Timestamp(st.session_state[state_key])

    if selected not in all_dates:
        selected = latest
        st.session_state[state_key] = latest

    selected_index = all_dates.index(selected)

    left, right = st.columns([5.8, 2.2])

    with left:
        st.subheader("Analysis Date")

    with right:
        previous, calendar, next_button = st.columns(
            [0.4, 1.6, 0.4]
        )

        with previous:
            if st.button(
                "‹",
                key=f"{key}_previous",
                disabled=selected_index == 0,
                use_container_width=True,
            ):
                st.session_state[state_key] = all_dates[
                    selected_index - 1
                ]
                st.rerun()

        with calendar:
            requested_date = st.date_input(
                "Date",
                value=selected.date(),
                min_value=all_dates[0].date(),
                max_value=latest.date(),
                key=f"{key}_calendar",
                label_visibility="collapsed",
            )

        with next_button:
            if st.button(
                "›",
                key=f"{key}_next",
                disabled=selected_index == len(all_dates) - 1,
                use_container_width=True,
            ):
                st.session_state[state_key] = all_dates[
                    selected_index + 1
                ]
                st.rerun()

    valid_dates = [
        date
        for date in all_dates
        if date <= pd.Timestamp(requested_date)
    ]

    resolved_date = valid_dates[-1] if valid_dates else all_dates[0]

    if resolved_date != selected:
        st.session_state[state_key] = resolved_date
        st.rerun()

    st.caption(
        f"Data as of {resolved_date.strftime('%d %b %Y')}"
    )

    return resolved_date


# =============================================================================
# SAFE STREAMLIT TABLE
# =============================================================================

def show_table(
    data: pd.DataFrame,
    height: int,
    chart_links: bool = False,
    progress_columns: list[str] | None = None,
) -> None:
    """
    Do not use Pandas Styler here.

    Earlier dashboard errors came from:
    - Pandas Styler
    - Duplicate column names
    - PyArrow serialization

    This function removes duplicate columns before Streamlit displays data.
    """
    view = data.copy()

    view.columns = [str(column) for column in view.columns]
    view = view.loc[:, ~view.columns.duplicated(keep="first")]

    config = {}

    if chart_links and "Chart" in view.columns:
        config["Chart"] = st.column_config.LinkColumn(
            "Chart",
            display_text="Open ↗",
        )

    for column in progress_columns or []:
        if column in view.columns:
            config[column] = st.column_config.ProgressColumn(
                column,
                min_value=0,
                max_value=100,
                format="%.0f",
            )

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=config,
    )


# =============================================================================
# INDUSTRY MOVEMENT CALCULATIONS
# =============================================================================

def current_snapshot(
    history: pd.DataFrame,
    selected_date: pd.Timestamp,
    group_column: str,
) -> pd.DataFrame:
    """Get one latest row per group for the selected date."""
    data = history[
        history["date"] == selected_date
    ].copy()

    data = data[
        numeric_column(data, "members") >= SMALL_GROUP_LIMIT
    ].copy()

    data = data.drop_duplicates(
        group_column,
        keep="last",
    )

    data["Leadership Score"] = normalize_score(
        data["leadership_score"]
    ).fillna(0.0)

    data["Actionability %"] = normalize_score(
        data["actionability_score"]
    ).fillna(0.0)

    return data


def add_5_session_change(
    history: pd.DataFrame,
    current: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    5D Leadership Change =
    Current Leadership Score - Leadership Score five sessions earlier.
    """
    if current.empty:
        return current

    current_date = current["date"].iloc[0]

    history_dates = sorted(
        pd.Timestamp(value)
        for value in pd.to_datetime(
            history["date"].dropna().unique()
        )
    )

    earlier_dates = [
        date
        for date in history_dates
        if date < current_date
    ]

    if len(earlier_dates) < 5:
        current["5D Leadership Change"] = 0.0
        current["5D Actionability Change"] = 0.0
        return current

    prior_date = earlier_dates[-5]

    prior = history[
        history["date"] == prior_date
    ].copy()

    prior = prior.drop_duplicates(
        group_column,
        keep="last",
    )

    prior["Prior Leadership"] = normalize_score(
        prior["leadership_score"]
    ).fillna(0.0)

    prior["Prior Actionability"] = normalize_score(
        prior["actionability_score"]
    ).fillna(0.0)

    prior = prior[
        [
            group_column,
            "Prior Leadership",
            "Prior Actionability",
        ]
    ]

    result = current.merge(
        prior,
        on=group_column,
        how="left",
    )

    result["Prior Leadership"] = result[
        "Prior Leadership"
    ].fillna(result["Leadership Score"])

    result["Prior Actionability"] = result[
        "Prior Actionability"
    ].fillna(result["Actionability %"])

    result["5D Leadership Change"] = (
        result["Leadership Score"]
        - result["Prior Leadership"]
    )

    result["5D Actionability Change"] = (
        result["Actionability %"]
        - result["Prior Actionability"]
    )

    return result


def make_movement_chart(
    data: pd.DataFrame,
    title: str,
) -> None:
    """Plot a clean industry movement chart."""
    chart_data = data.sort_values(
        "5D Leadership Change"
    ).copy()

    colors = [
        GREEN if value >= 0 else RED
        for value in chart_data["5D Leadership Change"]
    ]

    figure = go.Figure(
        go.Bar(
            x=chart_data["5D Leadership Change"],
            y=chart_data["basic_industry"],
            orientation="h",
            marker_color=colors,
            text=chart_data[
                "5D Leadership Change"
            ].round(1),
            textposition="outside",
            hovertemplate=(
                "%{y}<br>"
                "5-session leadership change: %{x:.1f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=title,
        xaxis_title="Leadership Score change over 5 sessions",
        yaxis_title=None,
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=max(len(chart_data) - 0.5, 0),
                line=dict(color="#94A3B8", width=1),
            )
        ],
    )

    st.plotly_chart(
        apply_chart_style(
            figure,
            max(300, 55 + 29 * len(chart_data)),
        ),
        use_container_width=True,
    )


# =============================================================================
# INDUSTRY MONITOR TAB
# =============================================================================

def industry_monitor_tab(
    basic_history: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> None:
    st.subheader("Industry Momentum Monitor")

    st.markdown(
        """
        <p class="note">
        This screen answers one main question:
        Which Basic Industries are improving or weakening over the last
        five available trading sessions?
        </p>
        """,
        unsafe_allow_html=True,
    )

    current = current_snapshot(
        basic_history,
        selected_date,
        "basic_industry",
    )

    if current.empty:
        st.warning(
            "No Basic Industry data is available for the selected date."
        )
        return

    current = add_5_session_change(
        basic_history,
        current,
        "basic_industry",
    )

    improving = int(
        (current["5D Leadership Change"] > 0).sum()
    )

    weakening = int(
        (current["5D Leadership Change"] < 0).sum()
    )

    active_setups = int(
        (current["Actionability %"] > 0).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Industries Tracked", format_integer(len(current)))
    c2.metric("Leadership Improving", format_integer(improving))
    c3.metric("Leadership Weakening", format_integer(weakening))
    c4.metric("Industries With Setups", format_integer(active_setups))

    st.markdown("### Leadership Moving Up")

    st.markdown(
        """
        <p class="note">
        Top Basic Industries gaining the most Leadership Score
        versus five trading sessions earlier.
        </p>
        """,
        unsafe_allow_html=True,
    )

    make_movement_chart(
        current.nlargest(
            TOP_INDUSTRIES,
            "5D Leadership Change",
        ),
        "Largest 5-session leadership improvement",
    )

    st.markdown("### Leadership Moving Down")

    st.markdown(
        """
        <p class="note">
        These are the groups losing leadership fastest.
        Avoid chasing stocks in these industries unless the individual
        stock setup is exceptionally strong.
        </p>
        """,
        unsafe_allow_html=True,
    )

    make_movement_chart(
        current.nsmallest(
            TOP_INDUSTRIES,
            "5D Leadership Change",
        ),
        "Largest 5-session leadership deterioration",
    )

    st.markdown("### Current Setup Density")

    active = current[
        current["Actionability %"] > 0
    ].nlargest(
        TOP_INDUSTRIES,
        "Actionability %",
    ).sort_values("Actionability %")

    if active.empty:
        st.info(
            "Actionability is zero for every Basic Industry in the "
            "processed file on this date. The dashboard is correctly "
            "hiding the useless all-zero chart. This needs an upstream "
            "data-calculation check."
        )

    else:
        figure = go.Figure(
            go.Bar(
                x=active["Actionability %"],
                y=active["basic_industry"],
                orientation="h",
                marker_color=BLUE,
                text=active[
                    "Actionability %"
                ].round(1),
                textposition="outside",
                hovertemplate=(
                    "%{y}<br>"
                    "Actionability: %{x:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

        figure.update_layout(
            title="Industries with the most currently actionable stocks",
            xaxis_title="Actionability (%)",
            yaxis_title=None,
        )

        st.plotly_chart(
            apply_chart_style(
                figure,
                max(300, 55 + 29 * len(active)),
            ),
            use_container_width=True,
        )

    st.markdown("### Industry Decision Board")

    board = current.sort_values(
        "5D Leadership Change",
        ascending=False,
    ).copy().reset_index(drop=True)

    board.insert(
        0,
        "Rank",
        range(1, len(board) + 1),
    )

    board["Direction"] = board[
        "5D Leadership Change"
    ].map(
        lambda value: (
            "Improving"
            if value > 0.25
            else (
                "Weakening"
                if value < -0.25
                else "Flat"
            )
        )
    )

    board = board.rename(
        columns={
            "basic_industry": "Basic Industry",
            "regime": "Regime",
            "members": "Stocks",
        }
    )

    keep_columns = [
        "Rank",
        "Basic Industry",
        "Direction",
        "Leadership Score",
        "5D Leadership Change",
        "Actionability %",
        "5D Actionability Change",
        "Regime",
        "Stocks",
    ]

    board = board[
        [
            column
            for column in keep_columns
            if column in board.columns
        ]
    ]

    for column in [
        "Leadership Score",
        "5D Leadership Change",
        "Actionability %",
        "5D Actionability Change",
    ]:
        if column in board.columns:
            board[column] = board[column].round(1)

    if "Stocks" in board.columns:
        board["Stocks"] = board["Stocks"].map(
            format_integer
        )

    show_table(
        board,
        max(360, min(760, len(board) * 35 + 60)),
        progress_columns=[
            "Leadership Score",
            "Actionability %",
        ],
    )


# =============================================================================
# TOP SETUPS TAB
# =============================================================================

def percentile_rank(
    frame: pd.DataFrame,
    column: str,
    ascending: bool,
) -> pd.Series:
    values = numeric_column(
        frame,
        column,
        float("nan"),
    )

    return values.rank(
        pct=True,
        ascending=ascending,
    ).fillna(0.5)


def stock_setups_tab(
    basic_history: pd.DataFrame,
    stock_history: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> None:
    st.subheader("Top Individual Setups")

    st.markdown(
        """
        <p class="note">
        No industry hard filter. Stocks are selected using their individual
        upstream setup flags. Industry Rank, Leadership and Regime are
        shown as context only.
        </p>
        """,
        unsafe_allow_html=True,
    )

    basic = basic_history[
        basic_history["date"] == selected_date
    ].drop_duplicates(
        "basic_industry",
        keep="last",
    ).copy()

    stocks = stock_history[
        stock_history["date"] == selected_date
    ].copy()

    if stocks.empty:
        st.warning("No stock data is available for this date.")
        return

    basic = basic.sort_values(
        "leadership_score",
        ascending=False,
    ).reset_index(drop=True)

    basic["Industry Rank"] = range(1, len(basic) + 1)

    lookup = basic.set_index("basic_industry")[
        [
            "Industry Rank",
            "leadership_score",
            "regime",
        ]
    ]

    established = stocks[
        stocks["established_buy_setup"] == 1
    ].copy()

    ipo = stocks[
        stocks["ipo_buy_setup"] == 1
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Established Qualified",
        format_integer(len(established)),
    )

    c2.metric(
        "IPO Qualified",
        format_integer(len(ipo)),
    )

    c3.metric(
        "Scan Date",
        selected_date.strftime("%d %b %Y"),
    )

    st.markdown("### Top 20 Established Setups")

    if established.empty:
        st.info(
            "No established stocks pass the upstream setup gate "
            "on this date."
        )

    else:
        established["Priority Score"] = (
            0.30 * percentile_rank(
                established,
                "tight_3d_range",
                ascending=True,
            )
            + 0.25 * percentile_rank(
                established,
                "vol_ratio_50",
                ascending=True,
            )
            + 0.20 * percentile_rank(
                established,
                "gain_6m",
                ascending=False,
            )
            + 0.15 * percentile_rank(
                established,
                "up_down_ratio",
                ascending=False,
            )
            + 0.10 * percentile_rank(
                established,
                "stock_strength_score",
                ascending=False,
            )
        ) * 100.0

        established["Industry Rank"] = established[
            "basic_industry"
        ].map(lookup["Industry Rank"])

        established["Industry Leadership"] = normalize_score(
            established["basic_industry"].map(
                lookup["leadership_score"]
            )
        )

        established["Industry Regime"] = established[
            "basic_industry"
        ].map(lookup["regime"])

        established = established.sort_values(
            "Priority Score",
            ascending=False,
        ).head(TOP_STOCKS).reset_index(drop=True)

        established.insert(
            0,
            "Rank",
            range(1, len(established) + 1),
        )

        established["Chart"] = (
            "https://in.tradingview.com/chart/?symbol=NSE:"
            + established["symbol"].astype(str)
        )

        chart_data = established.sort_values(
            "Priority Score"
        )

        figure = go.Figure(
            go.Bar(
                x=chart_data["Priority Score"],
                y=chart_data["symbol"],
                orientation="h",
                marker_color=GREEN,
                text=chart_data[
                    "Priority Score"
                ].round(1),
                textposition="outside",
            )
        )

        figure.update_layout(
            title="Individual setup ranking",
            xaxis=dict(
                range=[0, 110],
                title="Priority Score",
            ),
            yaxis_title=None,
        )

        st.plotly_chart(
            apply_chart_style(
                figure,
                max(260, 29 * len(chart_data) + 60),
            ),
            use_container_width=True,
        )

        view = established.rename(
            columns={
                "symbol": "Symbol",
                "basic_industry": "Basic Industry",
                "close": "Close",
                "tight_3d_range": "Tightness (3D)",
                "vol_ratio_50": "Vol vs 50D",
                "gain_6m": "6M Gain",
                "nearest_ema_tag": "EMA Zone",
                "momentum_badge": "Momentum",
            }
        )

        keep_columns = [
            "Rank",
            "Symbol",
            "Chart",
            "Basic Industry",
            "Industry Rank",
            "Industry Leadership",
            "Industry Regime",
            "Close",
            "Priority Score",
            "Tightness (3D)",
            "Vol vs 50D",
            "6M Gain",
            "EMA Zone",
            "Momentum",
        ]

        view = view[
            [
                column
                for column in keep_columns
                if column in view.columns
            ]
        ]

        for column in [
            "Close",
            "Priority Score",
            "Vol vs 50D",
        ]:
            if column in view.columns:
                view[column] = view[column].map(
                    lambda value: format_number(value, 2)
                )

        for column in [
            "Tightness (3D)",
            "6M Gain",
        ]:
            if column in view.columns:
                view[column] = view[column].map(
                    format_return
                )

        if "Industry Rank" in view.columns:
            view["Industry Rank"] = view[
                "Industry Rank"
            ].map(format_integer)

        if "Industry Leadership" in view.columns:
            view["Industry Leadership"] = view[
                "Industry Leadership"
            ].round(1)

        show_table(
            view,
            560,
            chart_links=True,
            progress_columns=["Industry Leadership"],
        )

    st.markdown("### Top IPO Setups")

    if ipo.empty:
        st.info(
            "No IPO stocks pass the upstream IPO setup gate "
            "on this date."
        )

    else:
        ipo["Avg Turnover (Cr)"] = (
            numeric_column(
                ipo,
                "ipo_turnover_avg",
            )
            / 10_000_000.0
        )

        if "ipo_setup_score" in ipo.columns:
            ipo["Setup Score"] = numeric_column(
                ipo,
                "ipo_setup_score",
            )

        else:
            ipo["Setup Score"] = (
                0.25 * percentile_rank(
                    ipo,
                    "tight_3d_range",
                    ascending=True,
                )
                + 0.20 * percentile_rank(
                    ipo,
                    "vol_ratio_50",
                    ascending=True,
                )
                + 0.20 * percentile_rank(
                    ipo,
                    "vwap_premium",
                    ascending=False,
                )
                + 0.20 * percentile_rank(
                    ipo,
                    "retracement_from_listing_high",
                    ascending=True,
                )
                + 0.15 * percentile_rank(
                    ipo,
                    "hh_hl_count",
                    ascending=False,
                )
            ) * 100.0

        ipo["Industry Rank"] = ipo[
            "basic_industry"
        ].map(lookup["Industry Rank"])

        ipo["Industry Leadership"] = normalize_score(
            ipo["basic_industry"].map(
                lookup["leadership_score"]
            )
        )

        ipo["Industry Regime"] = ipo[
            "basic_industry"
        ].map(lookup["regime"])

        ipo = ipo.sort_values(
            "Setup Score",
            ascending=False,
        ).head(TOP_STOCKS).reset_index(drop=True)

        ipo.insert(
            0,
            "Rank",
            range(1, len(ipo) + 1),
        )

        ipo["Chart"] = (
            "https://in.tradingview.com/chart/?symbol=NSE:"
            + ipo["symbol"].astype(str)
        )

        view = ipo.rename(
            columns={
                "symbol": "Symbol",
                "basic_industry": "Basic Industry",
                "close": "Close",
                "days_listed": "Days Listed",
                "ipo_phase": "Phase",
                "vwap_premium": "Above VWAP",
                "retracement_from_listing_high": "Off High",
            }
        )

        keep_columns = [
            "Rank",
            "Symbol",
            "Chart",
            "Basic Industry",
            "Industry Rank",
            "Industry Leadership",
            "Industry Regime",
            "Close",
            "Days Listed",
            "Phase",
            "Setup Score",
            "Above VWAP",
            "Off High",
            "Avg Turnover (Cr)",
        ]

        view = view[
            [
                column
                for column in keep_columns
                if column in view.columns
            ]
        ]

        for column in [
            "Close",
            "Setup Score",
            "Avg Turnover (Cr)",
        ]:
            if column in view.columns:
                view[column] = view[column].map(
                    lambda value: format_number(value, 2)
                )

        for column in [
            "Above VWAP",
            "Off High",
        ]:
            if column in view.columns:
                view[column] = view[column].map(
                    format_return
                )

        if "Days Listed" in view.columns:
            view["Days Listed"] = view[
                "Days Listed"
            ].map(format_integer)

        if "Industry Rank" in view.columns:
            view["Industry Rank"] = view[
                "Industry Rank"
            ].map(format_integer)

        if "Industry Leadership" in view.columns:
            view["Industry Leadership"] = view[
                "Industry Leadership"
            ].round(1)

        show_table(
            view,
            max(260, 40 * len(view) + 60),
            chart_links=True,
            progress_columns=["Industry Leadership"],
        )


# =============================================================================
# BASIC INDUSTRY / INDUSTRY DETAIL TAB
# =============================================================================

def group_detail_tab(
    history: pd.DataFrame,
    stock_history: pd.DataFrame,
    selected_date: pd.Timestamp,
    group_column: str,
    title: str,
) -> None:
    current = current_snapshot(
        history,
        selected_date,
        group_column,
    )

    if current.empty:
        st.warning(
            f"No {title} data is available for this date."
        )
        return

    current = add_5_session_change(
        history,
        current,
        group_column,
    )

    current = current.sort_values(
        "Leadership Score",
        ascending=False,
    ).reset_index(drop=True)

    current.insert(
        0,
        "Rank",
        range(1, len(current) + 1),
    )

    display_group_name = (
        "Basic Industry"
        if group_column == "basic_industry"
        else "Industry"
    )

    table = current.rename(
        columns={
            group_column: display_group_name,
            "regime": "Regime",
            "members": "Stocks",
        }
    )

    keep_columns = [
        "Rank",
        display_group_name,
        "Leadership Score",
        "5D Leadership Change",
        "Actionability %",
        "5D Actionability Change",
        "Regime",
        "Stocks",
    ]

    table = table[
        [
            column
            for column in keep_columns
            if column in table.columns
        ]
    ]

    for column in [
        "Leadership Score",
        "5D Leadership Change",
        "Actionability %",
        "5D Actionability Change",
    ]:
        if column in table.columns:
            table[column] = table[column].round(1)

    if "Stocks" in table.columns:
        table["Stocks"] = table["Stocks"].map(
            format_integer
        )

    show_table(
        table,
        max(360, min(760, 35 * len(table) + 60)),
        progress_columns=[
            "Leadership Score",
            "Actionability %",
        ],
    )

    st.markdown(f"### {title} Constituents")

    selected_group = st.selectbox(
        title,
        table[display_group_name].tolist(),
        key=f"{group_column}_selector",
    )

    stock_group_column = (
        "basic_industry"
        if group_column == "basic_industry"
        else "industry"
    )

    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (
            stock_history[stock_group_column]
            == selected_group
        )
    ].copy()

    if stocks.empty:
        st.info(
            "No constituent stock records are available."
        )
        return

    if "ret_20d" in stocks.columns:
        stocks = stocks.sort_values(
            "ret_20d",
            ascending=False,
        )

    stocks = stocks.head(30).reset_index(drop=True)

    stocks.insert(
        0,
        "Rank",
        range(1, len(stocks) + 1),
    )

    stocks["Chart"] = (
        "https://in.tradingview.com/chart/?symbol=NSE:"
        + stocks["symbol"].astype(str)
    )

    view = stocks.rename(
        columns={
            "symbol": "Symbol",
            "close": "Close",
            "ret_20d": "20D Return",
            "ret_60d": "60D Return",
            "gain_6m": "6M Gain",
            "stock_strength_score": "Strength",
        }
    )

    keep_columns = [
        "Rank",
        "Symbol",
        "Chart",
        "Close",
        "20D Return",
        "60D Return",
        "6M Gain",
        "Strength",
        "established_buy_setup",
        "ipo_buy_setup",
    ]

    view = view[
        [
            column
            for column in keep_columns
            if column in view.columns
        ]
    ]

    for column in [
        "20D Return",
        "60D Return",
        "6M Gain",
    ]:
        if column in view.columns:
            view[column] = view[column].map(
                format_return
            )

    for column in [
        "Close",
        "Strength",
    ]:
        if column in view.columns:
            view[column] = view[column].map(
                lambda value: format_number(value, 2)
            )

    show_table(
        view,
        420,
        chart_links=True,
    )


# =============================================================================
# METHODOLOGY TAB
# =============================================================================

def methodology_tab() -> None:
    st.subheader("How to Read the Monitor")

    st.markdown(
        """
### Industry Momentum

- **Leadership Score** is the current relative strength score for the Basic Industry.

- **5D Leadership Change** is:

  Current Leadership Score minus Leadership Score from five available trading sessions earlier.

- Positive change means the industry is improving relative to the rest of the market.

- Negative change means the industry is weakening.

### Actionability

- **Actionability %** is the percentage of stocks in that Basic Industry which pass the upstream stock-setup conditions.

- It is separate from Leadership Score.

- A group may have high Leadership but no clean setups because stocks are already extended.

### Top Individual Setups

- Established and IPO stocks are selected using the upstream setup flags.

- Industry Leadership is shown beside each stock as context only.

- Industry score does not filter out a technically strong individual stock.

### Workflow

1. Review Leadership Moving Up.
2. Check whether that group has usable Actionability.
3. Open Top Individual Setups.
4. Review the TradingView chart.
5. Define entry, invalidation/stop, and position size.

This dashboard is a research tool, not a trade recommendation.
        """
    )


# =============================================================================
# APP START
# =============================================================================

def main() -> None:
    required_files = [
        BASIC_HISTORY_FILE,
        INDUSTRY_HISTORY_FILE,
        STOCK_HISTORY_FILE,
    ]

    missing = [
        str(path.relative_to(ROOT))
        for path in required_files
        if not path.exists()
    ]

    if missing:
        st.error(
            "Required dashboard data files are missing. "
            "Run the data workflow first."
        )
        st.code("\n".join(missing))
        st.stop()

    basic_history = prepare_group_data(
        load_parquet(str(BASIC_HISTORY_FILE)),
        "basic_industry",
    )

    industry_history = prepare_group_data(
        load_parquet(str(INDUSTRY_HISTORY_FILE)),
        "industry",
    )

    stock_history = prepare_stock_data(
        load_parquet(str(STOCK_HISTORY_FILE))
    )

    all_dates = sorted(
        set(
            get_trading_dates(basic_history)
            + get_trading_dates(industry_history)
            + get_trading_dates(stock_history)
        )
    )

    if not all_dates:
        st.error(
            "No valid trading dates were found in the processed data."
        )
        st.stop()

    st.title("NSE Industry Momentum Monitor")

    sync_text = (
        SYNC_FILE.read_text(encoding="utf-8").strip()
        if SYNC_FILE.exists()
        else "Not available"
    )

    st.caption(
        "Prepared market data: "
        + sync_text.replace("T", " ").replace("Z", " IST")
    )

    tabs = st.tabs(
        [
            "Industry Monitor",
            "Top Setups",
            "Basic Industry",
            "Industry",
            "Methodology",
        ]
    )

    with tabs[0]:
        industry_monitor_tab(
            basic_history,
            date_picker(all_dates, "monitor"),
        )

    with tabs[1]:
        stock_setups_tab(
            basic_history,
            stock_history,
            date_picker(all_dates, "setups"),
        )

    with tabs[2]:
        group_detail_tab(
            basic_history,
            stock_history,
            date_picker(all_dates, "basic"),
            "basic_industry",
            "Basic Industry",
        )

    with tabs[3]:
        group_detail_tab(
            industry_history,
            stock_history,
            date_picker(all_dates, "industry"),
            "industry",
            "Industry",
        )

    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
