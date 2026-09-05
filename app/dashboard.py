# COMPLETE REPLACEMENT FOR: app/dashboard.py
# NSE Industry Momentum Monitor
# Fixes:
# - missing date_picker NameError
# - duplicate dataframe column errors
# - pandas Styler / PyArrow errors
# - 0-to-1 versus 0-to-100 score scaling
# - cluttered charts and raw Excel-like tables
#
# Main purpose:
# - Show industries whose leadership is improving or weakening
# - Show setup density only when the data has meaningful non-zero values
# - Show Top 20 stock setups without filtering out stocks due to industry rank

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# FILE LOCATIONS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = PROCESSED / "dashboard_basic_industry_history.parquet"
INDUSTRY_HISTORY_FILE = PROCESSED / "dashboard_industry_history.parquet"
STOCK_HISTORY_FILE = PROCESSED / "dashboard_stock_history.parquet"
SYNC_FILE = PROCESSED / "last_sync.txt"

SMALL_GROUP_LIMIT = 5
TOP_N = 12
TOP_SETUPS = 20


# =============================================================================
# DESIGN SETTINGS
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
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        color: #0F172A;
    }

    h2, h3 {
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 750;
        color: #0F172A;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 650;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    .section-note {
        color: #64748B;
        font-size: 0.86rem;
        margin-top: -0.35rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SMALL HELPER FUNCTIONS
# =============================================================================

def clean(value: object) -> str:
    """Convert blank or missing text fields to a safe readable value."""
    if value is None or pd.isna(value):
        return "Unclassified"

    value = str(value).strip()
    return value if value else "Unclassified"


def num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column safely, even if it is missing."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)

    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def score_100(values: pd.Series) -> pd.Series:
    """
    Fixes the common storage mismatch:
    Some scores are stored as 0.00–1.00,
    while the UI expects 0–100.
    """
    series = pd.to_numeric(values, errors="coerce")
    non_null = series.dropna()

    if len(non_null) and non_null.max() <= 1.5:
        return series * 100.0

    return series


def fmt_num(value: object, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return "—"


def fmt_int(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}"
    except (ValueError, TypeError):
        return "—"


def fmt_pct_fraction(value: object) -> str:
    """
    Use only for return-like fields stored as fractions:
    0.125 becomes 12.5%.
    """
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100.0:,.1f}%"
    except (ValueError, TypeError):
        return "—"


def fig_style(fig: go.Figure, height: int) -> go.Figure:
    """Apply a clean institutional-style Plotly layout."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=25, t=42, b=10),
        font=dict(
            family="Inter, -apple-system, Segoe UI, sans-serif",
            size=12,
            color=INK,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14, color=INK),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    return frame


def prepare_groups(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Prepare Basic Industry or Industry data.
    Also fixes score scale if stored as 0–1 instead of 0–100.
    """
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

    data[group_column] = data[group_column].map(clean)
    data["regime"] = data["regime"].map(clean)

    data["leadership_score"] = score_100(
        data["leadership_score"]
    ).fillna(0.0)

    data["actionability_score"] = score_100(
        data["actionability_score"]
    ).fillna(0.0)

    return data


def prepare_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare stock data and guarantee safe stock setup flags."""
    data = frame.copy()

    for column in ["symbol", "basic_industry", "industry", "sector"]:
        if column not in data.columns:
            data[column] = "Unclassified"

        data[column] = data[column].map(clean)

    for flag in ["established_buy_setup", "ipo_buy_setup"]:
        if flag not in data.columns:
            data[flag] = 0

        data[flag] = (
            pd.to_numeric(data[flag], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return data


def get_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Get sorted unique trading dates from a data table."""
    if "date" not in frame.columns:
        return []

    return sorted(
        pd.Timestamp(value)
        for value in pd.to_datetime(frame["date"].dropna().unique())
    )


# =============================================================================
# DATE NAVIGATOR
# =============================================================================

def date_picker(all_dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    """
    Date selector with previous/next buttons.
    This function is included specifically to avoid the earlier NameError.
    """
    latest = all_dates[-1]
    state_key = f"{key}_selected_date"

    if state_key not in st.session_state:
        st.session_state[state_key] = latest

    selected = pd.Timestamp(st.session_state[state_key])

    if selected not in all_dates:
        selected = latest
        st.session_state[state_key] = latest

    index = all_dates.index(selected)

    left, right = st.columns([5.8, 2.2])

    with left:
        st.subheader("Analysis Date")

    with right:
        previous, calendar, next_button = st.columns([0.4, 1.6, 0.4])

        with previous:
            if st.button(
                "‹",
                key=f"{key}_previous",
                disabled=index == 0,
                use_container_width=True,
            ):
                st.session_state[state_key] = all_dates[index - 1]
                st.rerun()

        with calendar:
            chosen = st.date_input(
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
                disabled=index == len(all_dates) - 1,
                use_container_width=True,
            ):
                st.session_state[state_key] = all_dates[index + 1]
                st.rerun()

    valid_dates = [
        date for date in all_dates
        if date <= pd.Timestamp(chosen)
    ]

    resolved = valid_dates[-1] if valid_dates else all_dates[0]

    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()

    st.caption(f"Data as of {resolved.strftime('%d %b %Y')}")
    return resolved


# =============================================================================
# SAFE TABLE DISPLAY
# =============================================================================

def show_table(
    data: pd.DataFrame,
    height: int,
    links: bool = False,
    progress_columns: list[str] | None = None,
) -> None:
    """
    Safe Streamlit table output:
    - Removes duplicate columns
    - Does not use Pandas Styler
    - Avoids previous Arrow/PyArrow errors
    """
    data = data.copy()

    data.columns = [str(column) for column in data.columns]
    data = data.loc[:, ~data.columns.duplicated(keep="first")]

    config = {}

    if links and "Chart" in data.columns:
        config["Chart"] = st.column_config.LinkColumn(
            "Chart",
            display_text="Open ↗",
        )

    for column in progress_columns or []:
        if column in data.columns:
            config[column] = st.column_config.ProgressColumn(
                column,
                min_value=0,
                max_value=100,
                format="%.0f",
            )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=config,
    )


# =============================================================================
# INDUSTRY MOVEMENT CALCULATIONS
# =============================================================================

def snapshot(
    history: pd.DataFrame,
    selected_date: pd.Timestamp,
    group_column: str,
) -> pd.DataFrame:
    """Take the current-date snapshot for either industry level."""
    data = history[history["date"] == selected_date].copy()

    data = data[
        num(data, "members") >= SMALL_GROUP_LIMIT
    ].copy()

    data = data.drop_duplicates(group_column, keep="last")

    data["Leadership Score"] = score_100(
        data["leadership_score"]
    ).fillna(0.0)

    data["Actionability %"] = score_100(
        data["actionability_score"]
    ).fillna(0.0)

    return data


def add_5_session_change(
    history: pd.DataFrame,
    current: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Calculate score movement:
    current score minus score from five available trading sessions earlier.
    """
    if current.empty:
        return current

    selected_date = current["date"].iloc[0]

    available_dates = sorted(
        pd.Timestamp(value)
        for value in pd.to_datetime(history["date"].dropna().unique())
    )

    earlier_dates = [
        date for date in available_dates
        if date < selected_date
    ]

    if len(earlier_dates) < 5:
        current["5D Leadership Δ"] = 0.0
        current["5D Actionability Δ"] = 0.0
        return current

    prior_date = earlier_dates[-5]

    prior = history[
        history["date"] == prior_date
    ].copy()

    prior = prior.drop_duplicates(group_column, keep="last")

    prior = prior[
        [group_column, "leadership_score", "actionability_score"]
    ].copy()

    prior["Prior Leadership"] = score_100(
        prior["leadership_score"]
    ).fillna(0.0)

    prior["Prior Actionability"] = score_100(
        prior["actionability_score"]
    ).fillna(0.0)

    prior = prior[
        [
            group_column,
            "Prior Leadership",
            "Prior Actionability",
        ]
    ]

    current = current.merge(
        prior,
        on=group_column,
        how="left",
    )

    current["Prior Leadership"] = current[
        "Prior Leadership"
    ].fillna(current["Leadership Score"])

    current["Prior Actionability"] = current[
        "Prior Actionability"
    ].fillna(current["Actionability %"])

    current["5D Leadership Δ"] = (
        current["Leadership Score"]
        - current["Prior Leadership"]
    )

    current["5D Actionability Δ"] = (
        current["Actionability %"]
        - current["Prior Actionability"]
    )

    return current


def movement_chart(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
) -> None:
    """Clean horizontal bar chart for rising/falling industry leadership."""
    data = frame.sort_values(value_column).copy()

    colors = [
        GREEN if value >= 0 else RED
        for value in data[value_column]
    ]

    fig = go.Figure(
        go.Bar(
            x=data[value_column],
            y=data["basic_industry"],
            orientation="h",
            marker_color=colors,
            text=data[value_column].round(1),
            textposition="outside",
            hovertemplate=(
                "%{y}<br>"
                "Change: %{x:.1f} points"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Leadership Score change over 5 sessions",
        yaxis_title=None,
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=max(len(data) - 0.5, 0),
                line=dict(color="#94A3B8", width=1),
            )
        ],
    )

    st.plotly_chart(
        fig_style(fig, max(300, 55 + 29 * len(data))),
        use_container_width=True,
    )


# =============================================================================
# TAB 1 — INDUSTRY MOMENTUM MONITOR
# =============================================================================

def industry_monitor_tab(
    basic_history: pd.DataFrame,
    industry_history: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> None:
    st.subheader("Industry Momentum Monitor")

    st.markdown(
        """
        <p class="section-note">
        Focus on change: which Basic Industries are improving,
        weakening, and developing usable setups?
        </p>
        """,
        unsafe_allow_html=True,
    )

    current = snapshot(
        basic_history,
        selected_date,
        "basic_industry",
    )

    if current.empty:
        st.warning("No Basic Industry data is available for this date.")
        return

    current = add_5_session_change(
        basic_history,
        current,
        "basic_industry",
    )

    rising_count = int(
        (current["5D Leadership Δ"] > 0).sum()
    )

    falling_count = int(
        (current["5D Leadership Δ"] < 0).sum()
    )

    active_setup_count = int(
        (current["Actionability %"] > 0).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Industries Tracked", fmt_int(len(current)))
    c2.metric("Leadership Improving", fmt_int(rising_count))
    c3.metric("Leadership Weakening", fmt_int(falling_count))
    c4.metric("Industries With Setups", fmt_int(active_setup_count))

    st.markdown("### Leadership Moving Up")

    st.markdown(
        """
        <p class="section-note">
        Top 12 Basic Industries with the biggest positive Leadership Score
        change compared with five available trading sessions ago.
        </p>
        """,
        unsafe_allow_html=True,
    )

    movement_chart(
        current.nlargest(TOP_N, "5D Leadership Δ"),
        "5D Leadership Δ",
        "Largest 5-session leadership improvement",
    )

    st.markdown("### Leadership Moving Down")

    st.markdown(
        """
        <p class="section-note">
        Top 12 Basic Industries losing leadership fastest.
        This helps avoid late entries into fading groups.
        </p>
        """,
        unsafe_allow_html=True,
    )

    movement_chart(
        current.nsmallest(TOP_N, "5D Leadership Δ"),
        "5D Leadership Δ",
        "Largest 5-session leadership deterioration",
    )

    st.markdown("### Current Setup Density")

    st.markdown(
        """
        <p class="section-note">
        Only industries with a positive Actionability value are shown.
        A wall of zero-value bars is deliberately hidden.
        </p>
        """,
        unsafe_allow_html=True,
    )

    active = current[
        current["Actionability %"] > 0
    ].nlargest(TOP_N, "Actionability %").sort_values(
        "Actionability %"
    )

    if active.empty:
        st.info(
            "The processed Basic Industry data has Actionability = 0 "
            "for every industry on this date. This is an upstream "
            "data/calculation issue, not a chart issue. The dashboard "
            "is hiding the meaningless zero-bar chart."
        )
    else:
        fig = go.Figure(
            go.Bar(
                x=active["Actionability %"],
                y=active["basic_industry"],
                orientation="h",
                marker_color=BLUE,
                text=active["Actionability %"].round(1),
                textposition="outside",
                hovertemplate=(
                    "%{y}<br>"
                    "Actionability: %{x:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Industries with the most currently actionable stocks",
            xaxis_title="Actionability (%)",
            yaxis_title=None,
        )

        st.plotly_chart(
            fig_style(
                fig,
                max(300, 55 + 29 * len(active)),
            ),
            use_container_width=True,
        )

    st.markdown("### Industry Decision Board")

    board = current.sort_values(
        "5D Leadership Δ",
        ascending=False,
    ).copy().reset_index(drop=True)

    board.insert(0, "Rank", range(1, len(board) + 1))

    board["Direction"] = board["5D Leadership Δ"].map(
        lambda value:
        "↑ Improving"
        if value > 0.25
        else (
            "↓ Weakening"
            if value < -0.25
            else "→ Flat"
        )
    )

    board = board.rename(
        columns={
            "basic_industry": "Basic Industry",
            "regime": "Regime",
            "members": "Stocks",
        }
    )

    board_columns = [
        "Rank",
        "Basic Industry",
        "Direction",
        "Leadership Score",
        "5D Leadership Δ",
        "Actionability %",
        "5D Actionability Δ",
        "Regime",
        "Stocks",
    ]

    board = board[
        [
            column for column in board_columns
            if column in board.columns
        ]
    ]

    for column in [
        "Leadership Score",
        "5D Leadership Δ",
        "Actionability %",
        "5D Actionability Δ",
    ]:
        if column in board.columns:
            board[column] = board[column].round(1)

    if "Stocks" in board.columns:
        board["Stocks"] = board["Stocks"].map(fmt_int)

    show_table(
        board,
        max(360, min(760, len(board) * 35 + 60)),
        progress_columns=[
            "Leadership Score",
            "Actionability %",
        ],
    )


# =============================================================================
# STOCK SETUP CALCULATIONS
# =============================================================================

def percentile_rank(
    frame: pd.DataFrame,
    column: str,
    ascending: bool,
) -> pd.Series:
    """
    Percentile rank among qualified candidates.
    For tightness/volume ratio, ascending=True makes lower values better.
    """
    values = num(frame, column, float("nan"))

    return values.rank(
        pct=True,
        ascending=ascending,
    ).fillna(0.5)


# =============================================================================
# TAB 2 — TOP INDIVIDUAL STOCK SETUPS
# =============================================================================

def stock_setups_tab(
    basic_history: pd.DataFrame,
    stock_history: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> None:
    st.subheader("Top Individual Setups")

    st.markdown(
        """
        <p class="section-note">
        Selection is stock-level only. Industry score and regime are shown
        as context; they do not filter out an individual setup.
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

    buy = stocks[
        stocks["established_buy_setup"] == 1
    ].copy()

    ipo = stocks[
        stocks["ipo_buy_setup"] == 1
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("Established Qualified", fmt_int(len(buy)))
    c2.metric("IPO Qualified", fmt_int(len(ipo)))
    c3.metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top 20 Established Setups")

    if buy.empty:
        st.info(
            "No established stocks pass the upstream setup gate "
            "on this date."
        )

    else:
        buy["Priority Score"] = (
            0.30 * percentile_rank(
                buy,
                "tight_3d_range",
                ascending=True,
            )
            + 0.25 * percentile_rank(
                buy,
                "vol_ratio_50",
                ascending=True,
            )
            + 0.20 * percentile_rank(
                buy,
                "gain_6m",
                ascending=False,
            )
            + 0.15 * percentile_rank(
                buy,
                "up_down_ratio",
                ascending=False,
            )
            + 0.10 * percentile_rank(
                buy,
                "stock_strength_score",
                ascending=False,
            )
        ) * 100.0

        buy["Industry Rank"] = buy["basic_industry"].map(
            lookup["Industry Rank"]
        )

        buy["Industry Leadership"] = score_100(
            buy["basic_industry"].map(
                lookup["leadership_score"]
            )
        )

        buy["Industry Regime"] = buy["basic_industry"].map(
            lookup["regime"]
        )

        buy = buy.sort_values(
            "Priority Score",
            ascending=False,
        ).head(TOP_SETUPS).reset_index(drop=True)

        buy.insert(0, "Rank", range(1, len(buy) + 1))

        buy["Chart"] = (
            "https://in.tradingview.com/chart/?symbol=NSE:"
            + buy["symbol"].astype(str)
        )

        ranked = buy.sort_values("Priority Score")

        fig = go.Figure(
            go.Bar(
                x=ranked["Priority Score"],
                y=ranked["symbol"],
                orientation="h",
                marker_color=GREEN,
                text=ranked["Priority Score"].round(1),
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Individual setup ranking",
            xaxis=dict(
                range=[0, 110],
                title="Priority Score",
            ),
            yaxis_title=None,
        )

        st.plotly_chart(
            fig_style(
                fig,
                max(260, 29 * len(ranked) + 60),
            ),
            use_container_width=True,
        )

        view = buy.rename(
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

        view_columns = [
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
                column for column in view_columns
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
                    lambda value: fmt_num(value, 2)
                )

        for column in [
            "Tightness (3D)",
            "6M Gain",
        ]:
            if column in view.columns:
                view[column] = view[column].map(
                    fmt_pct_fraction
                )

        if "Industry Rank" in view.columns:
            view["Industry Rank"] = view[
                "Industry Rank"
            ].map(fmt_int)

        if "Industry Leadership" in view.columns:
            view["Industry Leadership"] = view[
                "Industry Leadership"
            ].round(1)

        show_table(
            view,
            560,
            links=True,
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
            num(ipo, "ipo_turnover_avg")
            / 10_000_000.0
        )

        if "ipo_setup_score" in ipo.columns:
            ipo["Setup Score"] = num(
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

        ipo["Industry Rank"] = ipo["basic_industry"].map(
            lookup["Industry Rank"]
        )

        ipo["Industry Leadership"] = score_100(
            ipo["basic_industry"].map(
                lookup["leadership_score"]
            )
        )

        ipo["Industry Regime"] = ipo["basic_industry"].map(
            lookup["regime"]
        )

        ipo = ipo.sort_values(
            "Setup Score",
            ascending=False,
        ).head(TOP_SETUPS).reset_index(drop=True)

        ipo.insert(0, "Rank", range(1, len(ipo) + 1))

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

        view_columns = [
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
                column for column in view_columns
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
                    lambda value: fmt_num(value, 2)
                )

        for column in [
            "Above VWAP",
            "Off High",
        ]:
            if column in view.columns:
                view[column] = view[column].map(
                    fmt_pct_fraction
                )

        if "Days Listed" in view.columns:
            view["Days Listed"] = view[
                "Days Listed"
            ].map(fmt_int)

        if "Industry Rank" in view.columns:
            view["Industry Rank"] = view[
                "Industry Rank"
            ].map(fmt_int)

        if "Industry Leadership" in view.columns:
            view["Industry Leadership"] = view[
                "Industry Leadership"
            ].round(1)

        show_table(
            view,
            max(260, 40 * len(view) + 60),
            links=True,
            progress_columns=["Industry Leadership"],
        )


# =============================================================================
# TAB 3 AND TAB 4 — GROUP DETAILS
# =============================================================================

def group_detail_tab(
    history: pd.DataFrame,
    stock_history: pd.DataFrame,
    selected_date: pd.Timestamp,
    group_column: str,
    title: str,
) -> None:
    current = snapshot(
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

    current.insert(0, "Rank", range(1, len(current) + 1))

    display_name = (
        "Basic Industry"
        if group_column == "basic_industry"
        else "Industry"
    )

    table = current.rename(
        columns={
            group_column: display_name,
            "regime": "Regime",
            "members": "Stocks",
        }
    )

    table_columns = [
        "Rank",
        display_name,
        "Leadership Score",
        "5D Leadership Δ",
        "Actionability %",
        "5D Actionability Δ",
        "Regime",
        "Stocks",
    ]

    table = table[
        [
            column for column in table_columns
            if column in table.columns
        ]
    ]

    for column in [
        "Leadership Score",
        "5D Leadership Δ",
        "Actionability %",
        "5D Actionability Δ",
    ]:
        if column in table.columns:
            table[column] = table[column].round(1)

    if "Stocks" in table.columns:
        table["Stocks"] = table["Stocks"].map(fmt_int)

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
        table[display_name].tolist(),
        key=f"{group_column}_selector",
    )

    stock_group_column = (
        "basic_industry"
        if group_column == "basic_industry"
        else "industry"
    )

    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (stock_history[stock_group_column] == selected_group)
    ].copy()

    if stocks.empty:
        st.info("No constituent stock records are available.")
        return

    if "ret_20d" in stocks.columns:
        stocks = stocks.sort_values(
            "ret_20d",
            ascending=False,
        )

    stocks = stocks.head(30).reset_index(drop=True)

    stocks.insert(0, "Rank", range(1, len(stocks) + 1))

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

    view_columns = [
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
            column for column in view_columns
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
                fmt_pct_fraction
            )

    for column in [
        "Close",
        "Strength",
    ]:
        if column in view.columns:
            view[column] = view[column].map(
                lambda value: fmt_num(value, 2)
            )

    show_table(
        view,
        420,
        links=True,
    )


# =============================================================================
# METHODOLOGY TAB
# =============================================================================

def methodology_tab() -> None:
    st.subheader("How to Read the Monitor")

    st.markdown(
        """
- **Leadership Score:** Current relative strength of an industry, based on upstream price velocity, breadth/EMA alignment, and institutional-volume calculations.

- **5D Leadership Δ:** Current score minus the score five available trading sessions ago. Positive values mean improving leadership. Negative values mean deterioration.

- **Actionability %:** Percentage of constituents currently passing the upstream stock-setup rules. This is different from Leadership Score.

- **Top Individual Setups:** Ranked across all industries. Industry leadership is context only; it does not filter out a strong individual stock.

- **Zero Actionability:** If every industry has Actionability = 0 in the processed data, the dashboard hides the zero-value chart. That is an upstream pipeline/data issue, not a UI issue.

Suggested workflow: first review industries moving up; second check whether setup density is non-zero; third inspect Top Individual Setups; finally open the TradingView chart and define entry, stop/invalidation, and position size.
        """
    )


# =============================================================================
# APP ENTRY POINT
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
        st.code("
".join(missing))
        st.stop()

    basic = prepare_groups(
        load_parquet(str(BASIC_HISTORY_FILE)),
        "basic_industry",
    )

    industry = prepare_groups(
        load_parquet(str(INDUSTRY_HISTORY_FILE)),
        "industry",
    )

    stocks = prepare_stocks(
        load_parquet(str(STOCK_HISTORY_FILE))
    )

    all_dates = sorted(
        set(
            get_dates(basic)
            + get_dates(industry)
            + get_dates(stocks)
        )
    )

    if not all_dates:
        st.error(
            "No valid trading dates were found in the processed data."
        )
        st.stop()

    st.title("NSE Industry Momentum Monitor")

    sync = (
        SYNC_FILE.read_text(encoding="utf-8").strip()
        if SYNC_FILE.exists()
        else "Not available"
    )

    st.caption(
        "Prepared market data: "
        + sync.replace("T", " ").replace("Z", " IST")
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
            basic,
            industry,
            date_picker(all_dates, "monitor"),
        )

    with tabs[1]:
        stock_setups_tab(
            basic,
            stocks,
            date_picker(all_dates, "setups"),
        )

    with tabs[2]:
        group_detail_tab(
            basic,
            stocks,
            date_picker(all_dates, "basic"),
            "basic_industry",
            "Basic Industry",
        )

    with tabs[3]:
        group_detail_tab(
            industry,
            stocks,
            date_picker(all_dates, "industry"),
            "industry",
            "Industry",
        )

    with tabs[4]:
        methodology_tab()


if __name__ == "__main__":
    main()
