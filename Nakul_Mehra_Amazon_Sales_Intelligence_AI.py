"""
Amazon Sales Intelligence & AI Demand Forecasting System
=========================================================
A professional Streamlit dashboard for analysing Amazon India sales data
(March–June 2022) with ML-powered demand forecasting.

Author : IBM Internship Project
Dataset: Amazon Sale Report.csv
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import io
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Sales Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root variables ── */
:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2d3142;
    --text: #e8eaf0;
    --muted: #8b90a0;
    --accent: #4f8ef7;
    --green: #2ecc71;
    --red: #e74c3c;
    --yellow: #f39c12;
    --purple: #9b59b6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12151f !important;
    border-right: 1px solid var(--border);
}

/* KPI cards */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: var(--accent); }
.kpi-value  { font-size: 2rem; font-weight: 700; color: var(--accent); margin: 6px 0 2px; }
.kpi-label  { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.kpi-delta  { font-size: .82rem; margin-top: 4px; }
.kpi-delta.pos { color: var(--green); }
.kpi-delta.neg { color: var(--red); }

/* Section headers */
.section-header {
    font-size: 1.4rem; font-weight: 700;
    border-left: 4px solid var(--accent);
    padding-left: 12px; margin-bottom: 18px;
    color: var(--text);
}

/* Insight cards */
.insight-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.insight-card h4 { color: var(--accent); margin: 0 0 6px; font-size: .95rem; }
.insight-card p  { color: var(--text); margin: 0; font-size: .88rem; line-height: 1.6; }
.insight-card.warn  { border-left-color: var(--yellow); }
.insight-card.warn h4 { color: var(--yellow); }
.insight-card.danger { border-left-color: var(--red); }
.insight-card.danger h4 { color: var(--red); }
.insight-card.success { border-left-color: var(--green); }
.insight-card.success h4 { color: var(--green); }

/* Quality table */
.quality-table { font-size: .84rem; }

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH = Path("Amazon Sale Report.csv")

STATE_NORMALISE: dict[str, str] = {
    # abbreviations
    "RJ": "RAJASTHAN", "PB": "PUNJAB", "NL": "NAGALAND",
    "AR": "ARUNACHAL PRADESH",
    # typos / alternate spellings
    "RAJSHTHAN": "RAJASTHAN", "RAJSTHAN": "RAJASTHAN",
    "ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY",
    "NEW DELHI": "DELHI", "PUNJAB/MOHALI/ZIRAKPUR": "PUNJAB",
    "APO": "UNKNOWN",
}


@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw_df, clean_df)."""
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    df  = raw.copy()

    # ── Column name normalisation ──
    df.columns = df.columns.str.strip()
    df.rename(columns={"Sales Channel": "Sales_Channel"}, inplace=True)

    # ── Drop junk column ──
    df.drop(columns=["Unnamed: 22"], errors="ignore", inplace=True)

    # ── Date parsing ──
    df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%y", errors="coerce")

    # ── ship-state normalisation ──
    df["ship-state"] = (
        df["ship-state"]
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
        .replace(STATE_NORMALISE)
    )

    # ── ship-city normalisation ──
    df["ship-city"] = df["ship-city"].fillna("UNKNOWN").str.strip().str.upper()

    # ── Courier Status fill ──
    df["Courier Status"] = df["Courier Status"].fillna("Unknown")

    # ── fulfilled-by: derive from Fulfilment ──
    df["fulfilled-by"] = df["fulfilled-by"].fillna(
        df["Fulfilment"].apply(lambda x: "Amazon FBA" if x == "Amazon" else "Easy Ship")
    )

    # ── promotion-ids: null → "None" ──
    df["promotion-ids"] = df["promotion-ids"].fillna("None")

    # ── currency: null → "INR" ──
    df["currency"] = df["currency"].fillna("INR")

    # ── Feature engineering ──
    df["Month"]       = df["Date"].dt.month
    df["MonthName"]   = df["Date"].dt.strftime("%b %Y")
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]   = df["Date"].dt.day_name()
    df["is_cancelled"] = df["Status"].str.startswith("Cancelled").astype(int)
    df["is_B2B"]       = df["B2B"].astype(int)
    df["is_shipped"]   = df["Status"].str.startswith("Shipped").astype(int)

    # Revenue: only rows with valid Amount & Qty > 0
    df["Revenue"] = np.where((df["Amount"].notna()) & (df["Qty"] > 0), df["Amount"], np.nan)

    return raw, df


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════════

# Shared layout kwargs applied to every figure
_DARK_BG = "rgba(0,0,0,0)"
COLOR_SEQ = px.colors.qualitative.Vivid


def _apply_dark_theme(fig: go.Figure) -> None:
    """Apply the shared dark-theme layout settings in-place."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font_color="#e8eaf0",
        legend=dict(bgcolor=_DARK_BG),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def styled_fig(fig: go.Figure) -> go.Figure:
    _apply_dark_theme(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: KPI CARD HTML
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, delta: str = "", delta_pos: bool = True) -> str:
    delta_class = "pos" if delta_pos else "neg"
    delta_html  = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    with st.sidebar:
        st.markdown("## 📦 Amazon Intelligence")
        st.markdown("---")

        pages = [
            "🏠 Executive Overview",
            "📊 Data Quality",
            "📈 Sales Analytics",
            "🗺️ Geographic Analytics",
            "📦 Order & Fulfilment",
            "🛍️ Product Analytics",
            "🤖 AI Demand Forecasting",
            "💡 AI Business Insights",
        ]
        page = st.radio("Navigation", pages, label_visibility="collapsed")
        selected_page: str = page if isinstance(page, str) else pages[0]

        st.markdown("---")
        st.markdown("### 🔧 Global Filters")

        # Date range
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Category
        categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category", categories)
        sel_cat_str: str = str(sel_cat) if sel_cat is not None else "All"

        # Fulfilment
        fulfil_opts = ["All"] + sorted(df["Fulfilment"].dropna().unique().tolist())
        sel_fulfil = st.selectbox("Fulfilment", fulfil_opts)
        sel_fulfil_str: str = str(sel_fulfil) if sel_fulfil is not None else "All"

        # B2B
        b2b_opts = ["All", "B2B", "B2C"]
        sel_b2b = st.selectbox("Order Type", b2b_opts)
        sel_b2b_str: str = str(sel_b2b) if sel_b2b is not None else "All"

        st.markdown("---")
        st.markdown(
            "<small style='color:#666'>Data: Amazon India (Mar–Jun 2022)<br>"
            "IBM Internship Project</small>",
            unsafe_allow_html=True,
        )

    # ── Apply filters ──
    filtered: pd.DataFrame = df.copy()
    if len(date_range) == 2:
        mask_date = (
            (filtered["Date"] >= pd.Timestamp(date_range[0])) &
            (filtered["Date"] <= pd.Timestamp(date_range[1]))
        )
        filtered = filtered.loc[mask_date]
    if sel_cat_str != "All":
        filtered = filtered.loc[filtered["Category"] == sel_cat_str]
    if sel_fulfil_str != "All":
        filtered = filtered.loc[filtered["Fulfilment"] == sel_fulfil_str]
    if sel_b2b_str == "B2B":
        filtered = filtered.loc[filtered["B2B"] == True]
    elif sel_b2b_str == "B2C":
        filtered = filtered.loc[filtered["B2B"] == False]

    return selected_page, filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">🏠 Executive Overview</div>', unsafe_allow_html=True)

    total_orders: int    = int(df["Order ID"].nunique())  # type: ignore[arg-type]
    total_revenue: float = float(df["Revenue"].sum())  # type: ignore[arg-type]
    total_qty: int       = int(df.loc[df["Qty"] > 0, "Qty"].sum())  # type: ignore[arg-type]
    _ord_rev: pd.Series  = cast(pd.Series, df.groupby("Order ID")["Revenue"].sum())
    avg_order_value: float = float(_ord_rev.mean())  # type: ignore[arg-type]
    cancel_rate: float   = float(df["is_cancelled"].mean()) * 100  # type: ignore[arg-type]
    _cat_rev_s: pd.Series = cast(pd.Series, df.groupby("Category")["Revenue"].sum())
    top_category: str   = str(_cat_rev_s.idxmax()) if not df.empty else "N/A"

    cols = st.columns(6)
    cards = [
        ("Total Orders",       f"{total_orders:,}",          "", True),
        ("Total Revenue",      f"₹{total_revenue/1e6:.2f}M", "", True),
        ("Total Quantity",     f"{total_qty:,}",              "", True),
        ("Avg Order Value",    f"₹{avg_order_value:.0f}",    "", True),
        ("Cancellation Rate",  f"{cancel_rate:.1f}%",         "⚠ Monitor" if cancel_rate > 15 else "✓ Healthy", cancel_rate <= 15),
        ("Top Category",       top_category,                  "", True),
    ]
    for col, (label, value, delta, pos) in zip(cols, cards):
        with col:
            st.markdown(kpi_card(label, value, delta, pos), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Daily revenue trend sparkline ──
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### 📅 Daily Revenue Trend")
        _daily_s: pd.Series = cast(pd.Series, df.groupby("Date")["Revenue"].sum())
        daily_rev: pd.DataFrame = _daily_s.reset_index()
        daily_rev = daily_rev.rename(columns={"Revenue": "Revenue (INR)"})
        fig = px.area(
            daily_rev, x="Date", y="Revenue (INR)",
            color_discrete_sequence=["#4f8ef7"],
            title="Daily Revenue (INR)",
        )
        fig.update_traces(fill="tozeroy", line_width=1.5)
        st.plotly_chart(styled_fig(fig), width="stretch")

    with col2:
        st.markdown("#### 🏷️ Revenue by Category")
        _cat_s: pd.Series = cast(pd.Series, df.groupby("Category")["Revenue"].sum())
        cat_rev: pd.DataFrame = _cat_s.sort_values(ascending=True).reset_index()
        fig2 = px.bar(
            cat_rev, x="Revenue", y="Category",
            orientation="h", color="Revenue",
            color_continuous_scale="Blues",
            title="Category Revenue (INR)",
        )
        st.plotly_chart(styled_fig(fig2), width="stretch")

    # ── Status breakdown ──
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("#### 📦 Order Status")
        status_cnt = df["Status"].value_counts().reset_index()
        status_cnt.columns = pd.Index(["Status", "Count"])
        fig3 = px.pie(status_cnt, names="Status", values="Count",
                      color_discrete_sequence=COLOR_SEQ, hole=0.4)
        st.plotly_chart(styled_fig(fig3), width="stretch")

    with col4:
        st.markdown("#### 📐 Size Distribution")
        size_order = ["XS","S","M","L","XL","XXL","3XL","4XL","5XL","6XL","Free"]
        _sz_s: pd.Series = cast(pd.Series, df.groupby("Size")["Qty"].sum())
        size_cnt: pd.DataFrame = _sz_s.reindex(size_order, fill_value=0).reset_index()
        fig4 = px.bar(size_cnt, x="Size", y="Qty",
                      color_discrete_sequence=["#9b59b6"],
                      title="Qty by Size")
        st.plotly_chart(styled_fig(fig4), width="stretch")

    with col5:
        st.markdown("#### 🚚 Fulfilment Split")
        ful_cnt = df["Fulfilment"].value_counts().reset_index()
        ful_cnt.columns = pd.Index(["Fulfilment", "Count"])
        fig5 = px.pie(ful_cnt, names="Fulfilment", values="Count",
                      color_discrete_sequence=["#4f8ef7","#2ecc71"], hole=0.4)
        st.plotly_chart(styled_fig(fig5), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def page_data_quality(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">📊 Data Quality Analysis</div>', unsafe_allow_html=True)

    # Basic info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",     f"{raw.shape[0]:,}")
    c2.metric("Total Columns",  f"{raw.shape[1]:,}")
    c3.metric("Duplicate Rows", f"{raw.duplicated().sum():,}")
    c4.metric("Duplicate Order IDs", f"{raw['Order ID'].duplicated().sum():,}")

    st.markdown("---")
    tabs = st.tabs(["Missing Values", "Data Types", "Unique Values",
                    "Numerical Stats", "Outliers", "Inconsistencies"])

    # ── Missing values ──
    with tabs[0]:
        missing: pd.Series = raw.isnull().sum()
        missing_pct: pd.Series = (missing / len(raw) * 100).round(2)
        mv_df = pd.DataFrame({
            "Column": list(missing.index),
            "Missing": missing.to_numpy(),
            "Missing %": missing_pct.to_numpy(),
            "Present": (len(raw) - missing).to_numpy(),
        })
        mv_df = mv_df.loc[mv_df["Missing"] > 0].sort_values("Missing %", ascending=False)
        fig = px.bar(mv_df, x="Column", y="Missing %",
                     color="Missing %", color_continuous_scale="Reds",
                     title="Missing Value % per Column")
        st.plotly_chart(styled_fig(fig), width="stretch")
        st.dataframe(mv_df, width="stretch")

    # ── Data types ──
    with tabs[1]:
        dt_df = pd.DataFrame({
            "Column": list(raw.dtypes.index),
            "Type":   raw.dtypes.astype(str).to_numpy(),
            "Non-Null Count": raw.notnull().sum().to_numpy(),
            "Sample Value": [
                str(raw[c].dropna().iloc[0]) if bool(raw[c].notna().any()) else "N/A"
                for c in raw.columns
            ],
        })
        st.dataframe(dt_df, width="stretch")

    # ── Unique values ──
    with tabs[2]:
        uv_df = pd.DataFrame({
            "Column": raw.columns,
            "Unique Values": [raw[c].nunique() for c in raw.columns],
        }).sort_values("Unique Values", ascending=False)
        fig2 = px.bar(uv_df.head(20), x="Column", y="Unique Values",
                      color_discrete_sequence=["#4f8ef7"],
                      title="Unique Value Count per Column (Top 20)")
        st.plotly_chart(styled_fig(fig2), width="stretch")

    # ── Numerical stats ──
    with tabs[3]:
        st.markdown("#### Qty & Amount Distributions")
        num_df = raw[["Qty", "Amount"]].describe().T
        st.dataframe(num_df, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(raw, x="Amount", nbins=80,
                                color_discrete_sequence=["#4f8ef7"],
                                title="Amount Distribution")
            st.plotly_chart(styled_fig(fig3), width="stretch")
        with col2:
            fig4 = px.histogram(raw[raw["Qty"] > 0], x="Qty", nbins=20,
                                color_discrete_sequence=["#9b59b6"],
                                title="Qty Distribution (excl. zero)")
            st.plotly_chart(styled_fig(fig4), width="stretch")

    # ── Outliers ──
    with tabs[4]:
        st.markdown("#### Box-plots for numeric columns")
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.box(raw, y="Amount", color_discrete_sequence=["#4f8ef7"],
                          title="Amount — Outlier Detection")
            st.plotly_chart(styled_fig(fig5), width="stretch")
        with col2:
            fig6 = px.box(raw[raw["Qty"] > 0], y="Qty",
                          color_discrete_sequence=["#9b59b6"],
                          title="Qty — Outlier Detection")
            st.plotly_chart(styled_fig(fig6), width="stretch")

        # IQR stats
        for col_name in ["Amount", "Qty"]:
            s = raw[col_name].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
            st.info(f"**{col_name}** — IQR: {iqr:.2f} | Outliers (IQR rule): {n_out:,} rows ({n_out/len(s)*100:.1f}%)")

    # ── Inconsistencies ──
    with tabs[5]:
        st.markdown("#### State Name Inconsistencies (Before Cleaning)")
        states = raw["ship-state"].dropna().unique()
        st.write(f"Unique raw state values: **{len(states)}**")
        st.dataframe(pd.DataFrame(sorted(states), columns=["Raw State Value"]),
                     width="stretch", height=300)

        st.markdown("#### Status Values")
        sv = raw["Status"].value_counts().reset_index()
        sv.columns = ["Status", "Count"]
        st.dataframe(sv, width="stretch")

        st.markdown("#### Zero / Null Revenue Rows")
        st.write(f"- Null Amount: **{raw['Amount'].isnull().sum():,}** rows")
        st.write(f"- Zero Amount: **{(raw['Amount']==0).sum():,}** rows")
        st.write(f"- Zero Qty:    **{(raw['Qty']==0).sum():,}** rows")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: SALES ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_sales(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">📈 Sales Analytics</div>', unsafe_allow_html=True)

    granularity = st.radio("Trend Granularity", ["Daily", "Weekly", "Monthly"],
                           horizontal=True)

    if granularity == "Daily":
        trend = df.groupby("Date").agg(
            Orders=("Order ID", "count"),
            Revenue=("Revenue", "sum"),
            Qty=("Qty", "sum"),
        ).reset_index()
        x_label = "Date"
    elif granularity == "Weekly":
        df["YearWeek"] = df["Date"].dt.strftime("%Y-W%V")
        trend = df.groupby("YearWeek").agg(
            Orders=("Order ID", "count"),
            Revenue=("Revenue", "sum"),
            Qty=("Qty", "sum"),
        ).reset_index()
        x_label = "YearWeek"
    else:
        trend = df.groupby("MonthName").agg(
            Orders=("Order ID", "count"),
            Revenue=("Revenue", "sum"),
            Qty=("Qty", "sum"),
        ).reset_index()
        # Sort by month
        trend["_sort"] = pd.to_datetime(trend["MonthName"], format="%b %Y")
        trend = trend.sort_values("_sort").drop(columns="_sort")
        x_label = "MonthName"

    # Revenue + Orders dual-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=trend[x_label], y=trend["Revenue"],
                         name="Revenue (INR)", marker_color="#4f8ef7"), secondary_y=False)
    fig.add_trace(go.Scatter(x=trend[x_label], y=trend["Orders"],
                             name="Orders", line=dict(color="#f39c12", width=2),
                             mode="lines+markers"), secondary_y=True)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
        font_color="#e8eaf0",
        title=f"{granularity} Revenue & Orders",
        legend=dict(bgcolor=_DARK_BG),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(title_text="Revenue (INR)", secondary_y=False)
    fig.update_yaxes(title_text="Order Count",   secondary_y=True)
    st.plotly_chart(fig, width="stretch")

    # Qty trend
    fig_qty = px.line(trend, x=x_label, y="Qty",
                      color_discrete_sequence=["#2ecc71"],
                      title=f"{granularity} Quantity Sold",
                      markers=True)
    st.plotly_chart(styled_fig(fig_qty), width="stretch")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Category-wise Revenue")
        cat_rev: pd.DataFrame = (
            df.groupby("Category")
            .agg(Revenue=("Revenue", "sum"), Orders=("Order ID", "count"), Qty=("Qty", "sum"))
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )
        fig_cat = px.bar(cat_rev, x="Category", y="Revenue",
                         color="Category", color_discrete_sequence=COLOR_SEQ,
                         title="Revenue by Category")
        st.plotly_chart(styled_fig(fig_cat), width="stretch")

    with col2:
        st.markdown("#### Size-wise Quantity")
        size_order = ["XS","S","M","L","XL","XXL","3XL","4XL","5XL","6XL","Free"]
        _sq_s: pd.Series = cast(pd.Series, df.groupby("Size")["Qty"].sum())
        size_qty: pd.DataFrame = _sq_s.reindex(size_order, fill_value=0).reset_index()
        fig_sz = px.bar(size_qty, x="Size", y="Qty",
                        color="Qty", color_continuous_scale="Viridis",
                        title="Quantity by Size")
        st.plotly_chart(styled_fig(fig_sz), width="stretch")

    st.markdown("#### 🏆 Top 20 SKUs by Revenue")
    top_sku: pd.DataFrame = (
        df[df["Revenue"].notna()]
        .groupby(["SKU","Category"])
        .agg(Revenue=("Revenue","sum"), Qty=("Qty","sum"), Orders=("Order ID","count"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(20)
    )
    fig_sku = px.bar(top_sku, x="SKU", y="Revenue", color="Category",
                     color_discrete_sequence=COLOR_SEQ,
                     title="Top 20 SKUs by Revenue")
    fig_sku.update_xaxes(tickangle=45)
    st.plotly_chart(styled_fig(fig_sku), width="stretch")

    # Day-of-week heatmap
    st.markdown("#### 📅 Day-of-Week Revenue Heatmap")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    _dow_s: pd.Series = cast(pd.Series, df.groupby(["DayOfWeek","Category"])["Revenue"].sum())
    heatmap_data: pd.DataFrame = cast(pd.DataFrame, _dow_s.unstack(fill_value=0))
    heatmap_data = heatmap_data.reindex(
        [d for d in dow_order if d in heatmap_data.index]
    )
    fig_heat = px.imshow(heatmap_data, color_continuous_scale="Blues",
                         title="Revenue Heatmap: Day of Week × Category",
                         aspect="auto")
    st.plotly_chart(styled_fig(fig_heat), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: GEOGRAPHIC ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_geo(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">🗺️ Geographic Analytics</div>', unsafe_allow_html=True)

    state_df: pd.DataFrame = (
        df.groupby("ship-state")
        .agg(Revenue=("Revenue","sum"), Orders=("Order ID","count"), Qty=("Qty","sum"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .rename(columns={"ship-state": "State"})
    )
    state_df = state_df.loc[state_df["State"] != "UNKNOWN"]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### State-wise Revenue (Top 20)")
        top20 = state_df.head(20)
        fig = px.bar(top20, x="Revenue", y="State", orientation="h",
                     color="Revenue", color_continuous_scale="Blues",
                     title="Top 20 States by Revenue")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_fig(fig), width="stretch")

    with col2:
        st.markdown("#### State Revenue Share (Top 10)")
        fig2 = px.pie(state_df.head(10), names="State", values="Revenue",
                      color_discrete_sequence=COLOR_SEQ, hole=0.4,
                      title="Top 10 States — Revenue Share")
        st.plotly_chart(styled_fig(fig2), width="stretch")

    # Orders by state
    st.markdown("#### Orders vs Revenue by State (Top 20)")
    fig3 = px.scatter(
        state_df.head(20), x="Orders", y="Revenue",
        size="Qty", text="State", color="State",
        color_discrete_sequence=COLOR_SEQ,
        title="Orders vs Revenue (bubble size = Qty)",
    )
    fig3.update_traces(textposition="top center")
    st.plotly_chart(styled_fig(fig3), width="stretch")

    # City-wise
    st.markdown("#### Top 25 Cities by Revenue")
    city_df: pd.DataFrame = (
        df.groupby("ship-city")
        .agg(Revenue=("Revenue","sum"), Orders=("Order ID","count"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .rename(columns={"ship-city": "City"})
    )
    city_df = city_df.loc[city_df["City"] != "UNKNOWN"].head(25)
    fig4 = px.bar(city_df, x="City", y="Revenue",
                  color="Revenue", color_continuous_scale="Purples",
                  title="Top 25 Cities by Revenue")
    fig4.update_xaxes(tickangle=45)
    st.plotly_chart(styled_fig(fig4), width="stretch")

    # Full table
    with st.expander("📋 Full State-wise Table"):
        st.dataframe(state_df, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: ORDER & FULFILMENT
# ═══════════════════════════════════════════════════════════════════════════════

def page_orders(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">📦 Order & Fulfilment Analytics</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        status_df = df["Status"].value_counts().reset_index()
        status_df.columns = pd.Index(["Status", "Count"])
        fig1 = px.pie(status_df, names="Status", values="Count",
                      color_discrete_sequence=COLOR_SEQ, hole=0.35,
                      title="Order Status Distribution")
        st.plotly_chart(styled_fig(fig1), width="stretch")

    with col2:
        ful_df = df["Fulfilment"].value_counts().reset_index()
        ful_df.columns = pd.Index(["Fulfilment", "Count"])
        fig2 = px.pie(ful_df, names="Fulfilment", values="Count",
                      color_discrete_sequence=["#4f8ef7","#2ecc71"], hole=0.35,
                      title="Fulfilment Type")
        st.plotly_chart(styled_fig(fig2), width="stretch")

    with col3:
        courier_df = df["Courier Status"].value_counts().reset_index()
        courier_df.columns = pd.Index(["Courier Status", "Count"])
        fig3 = px.pie(courier_df, names="Courier Status", values="Count",
                      color_discrete_sequence=COLOR_SEQ, hole=0.35,
                      title="Courier Status")
        st.plotly_chart(styled_fig(fig3), width="stretch")

    st.markdown("---")

    # Cancelled vs Shipped daily
    st.markdown("#### 📅 Daily: Shipped vs Cancelled Orders")
    _ds_grp: pd.Series = cast(pd.Series, df.groupby(["Date","is_cancelled"]).size())
    daily_status: pd.DataFrame = cast(pd.DataFrame, _ds_grp.reset_index(name="Count"))
    daily_status["Type"] = daily_status["is_cancelled"].apply(
        lambda v: "Cancelled" if int(v) == 1 else "Shipped/Other"
    )
    fig4 = px.bar(daily_status, x="Date", y="Count", color="Type",
                  color_discrete_map={"Cancelled":"#e74c3c","Shipped/Other":"#4f8ef7"},
                  barmode="stack", title="Daily Orders — Shipped vs Cancelled")
    st.plotly_chart(styled_fig(fig4), width="stretch")

    # Service level
    col4, col5 = st.columns(2)
    with col4:
        svc_df = df["ship-service-level"].value_counts().reset_index()
        svc_df.columns = pd.Index(["Service Level", "Count"])
        fig5 = px.bar(svc_df, x="Service Level", y="Count",
                      color="Service Level", color_discrete_sequence=COLOR_SEQ,
                      title="Service Level Distribution")
        st.plotly_chart(styled_fig(fig5), width="stretch")

    with col5:
        b2b_df = df["B2B"].value_counts().reset_index()
        b2b_df.columns = pd.Index(["B2B", "Count"])
        b2b_df["Type"] = b2b_df["B2B"].apply(lambda v: "B2B" if bool(v) else "B2C")
        fig6 = px.pie(b2b_df, names="Type", values="Count",
                      color_discrete_sequence=["#f39c12","#9b59b6"], hole=0.4,
                      title="B2B vs B2C Orders")
        st.plotly_chart(styled_fig(fig6), width="stretch")

    # Cancellation rate by category
    st.markdown("#### Cancellation Rate by Category")
    can_cat = (
        df.groupby("Category")
        .agg(Total=("Order ID","count"), Cancelled=("is_cancelled","sum"))
        .reset_index()
    )
    can_cat["Cancel Rate %"] = (can_cat["Cancelled"] / can_cat["Total"] * 100).round(1)
    fig7 = px.bar(can_cat.sort_values("Cancel Rate %", ascending=False),
                  x="Category", y="Cancel Rate %",
                  color="Cancel Rate %", color_continuous_scale="Reds",
                  title="Cancellation Rate by Category (%)")
    st.plotly_chart(styled_fig(fig7), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: PRODUCT ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_products(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">🛍️ Product Analytics</div>', unsafe_allow_html=True)

    revenue_df = df[df["Revenue"].notna() & (df["Revenue"] > 0)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 15 Best-Selling Products (by Revenue)")
        top_prod: pd.DataFrame = (
            revenue_df.groupby(["Style","Category"])
            .agg(Revenue=("Revenue","sum"), Qty=("Qty","sum"), Orders=("Order ID","count"))
            .reset_index()
            .sort_values("Revenue", ascending=False)
            .head(15)
        )
        fig1 = px.bar(top_prod, x="Revenue", y="Style", orientation="h",
                      color="Category", color_discrete_sequence=COLOR_SEQ,
                      title="Top 15 Products by Revenue")
        fig1.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_fig(fig1), width="stretch")

    with col2:
        st.markdown("#### 📉 Bottom 15 Products (by Revenue)")
        bottom_prod: pd.DataFrame = (
            revenue_df.groupby(["Style","Category"])
            .agg(Revenue=("Revenue","sum"), Qty=("Qty","sum"))
            .reset_index()
            .sort_values("Revenue", ascending=True)
            .head(15)
        )
        fig2 = px.bar(bottom_prod, x="Revenue", y="Style", orientation="h",
                      color="Category", color_discrete_sequence=COLOR_SEQ,
                      title="Bottom 15 Products by Revenue")
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_fig(fig2), width="stretch")

    st.markdown("#### 🔢 Category Performance Summary")
    cat_perf = (
        df.groupby("Category")
        .agg(
            Revenue=("Revenue","sum"),
            Orders=("Order ID","count"),
            Qty=("Qty","sum"),
            CancelRate=("is_cancelled","mean"),
            AvgOrderValue=("Revenue","mean"),
        )
        .round(2)
        .reset_index()
    )
    cat_perf["CancelRate"] = (cat_perf["CancelRate"] * 100).round(1)
    st.dataframe(
        cat_perf.sort_values("Revenue", ascending=False).style
        .format({"Revenue":"₹{:,.0f}", "CancelRate":"{:.1f}%",
                 "AvgOrderValue":"₹{:,.0f}", "Qty":"{:,.0f}"}),
        width="stretch",
    )

    # Category × Size heatmap (Qty)
    st.markdown("#### 📊 Category × Size Quantity Heatmap")
    size_order = ["XS","S","M","L","XL","XXL","3XL","4XL","5XL","6XL","Free"]
    _pv_s: pd.Series = cast(pd.Series, df.groupby(["Category","Size"])["Qty"].sum())
    pivot: pd.DataFrame = cast(pd.DataFrame, _pv_s.unstack(fill_value=0))
    pivot = cast(pd.DataFrame, pivot[[s for s in size_order if s in pivot.columns]])
    fig_h = px.imshow(pivot, color_continuous_scale="YlOrRd",
                      title="Category × Size Heatmap (Total Qty)",
                      aspect="auto")
    st.plotly_chart(styled_fig(fig_h), width="stretch")

    st.markdown("#### Top 30 SKUs Table")
    top_sku_tbl: pd.DataFrame = (
        revenue_df.groupby(["SKU","Style","Category"])
        .agg(Revenue=("Revenue","sum"), Qty=("Qty","sum"), Orders=("Order ID","count"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(30)
    )
    st.dataframe(
        top_sku_tbl.style.format({"Revenue":"₹{:,.0f}", "Qty":"{:,.0f}"}),
        width="stretch"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7: AI DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Training model…")
def train_forecast_model(
    df_json: str,
    category: str,
    model_name: str,
) -> tuple[Any, ...]:
    """Train a regression model on daily quantity data and return artifacts."""
    _raw: pd.DataFrame = pd.read_json(io.StringIO(df_json), orient="split")
    _raw["Date"] = pd.to_datetime(_raw["Date"])

    if category != "All":
        _raw = _raw.loc[_raw["Category"] == category]

    df_fc: pd.DataFrame = _raw

    # Daily aggregation
    daily: pd.DataFrame = (
        df_fc.loc[df_fc["Qty"] > 0]
        .groupby("Date")
        .agg(Qty=("Qty","sum"), Revenue=("Revenue","sum"))
        .reset_index()
        .sort_values("Date")
    )

    if len(daily) < 14:
        return None, None, None, None, None, None, None

    # Feature engineering
    daily["day_of_week"]   = daily["Date"].dt.dayofweek
    daily["day_of_month"]  = daily["Date"].dt.day
    daily["week_of_year"]  = daily["Date"].dt.isocalendar().week.astype(int)
    daily["month"]         = daily["Date"].dt.month
    daily["is_weekend"]    = (daily["day_of_week"] >= 5).astype(int)

    for lag in [1, 3, 7, 14]:
        daily[f"lag_{lag}"] = daily["Qty"].shift(lag)
    for win in [3, 7, 14]:
        daily[f"rolling_mean_{win}"] = daily["Qty"].shift(1).rolling(win).mean()
        daily[f"rolling_std_{win}"]  = daily["Qty"].shift(1).rolling(win).std()

    daily.dropna(inplace=True)

    feature_cols = (
        ["day_of_week","day_of_month","week_of_year","month","is_weekend"]
        + [f"lag_{l}" for l in [1,3,7,14]]
        + [f"rolling_mean_{w}" for w in [3,7,14]]
        + [f"rolling_std_{w}"  for w in [3,7,14]]
    )
    X = daily[feature_cols].values
    y = daily["Qty"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=4)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
    }

    test_dates = daily["Date"].iloc[split:].values

    return model, daily, feature_cols, test_dates, y_test, y_pred, metrics


def page_forecasting(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">🤖 AI Demand Forecasting</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
        _fc_cat = st.selectbox("Select Category", categories, key="fc_cat")
        fc_cat: str = str(_fc_cat) if _fc_cat is not None else "All"
    with col_b:
        _fc_model = st.selectbox("Model", ["Random Forest", "Gradient Boosting"], key="fc_model")
        fc_model: str = str(_fc_model) if _fc_model is not None else "Random Forest"
    with col_c:
        _fc_horizon = st.selectbox("Forecast Horizon (days)", [7, 14, 30], key="fc_horizon")
        fc_horizon: int = int(_fc_horizon) if _fc_horizon is not None else 7

    if st.button("🚀 Train & Forecast", type="primary"):
        _json_out = df[["Date","Category","Qty","Revenue"]].to_json(orient="split")
        df_json: str = _json_out if isinstance(_json_out, str) else ""

        result = train_forecast_model(df_json, fc_cat, fc_model)
        if result[0] is None:
            st.error("Not enough data for the selected category/date range. Please select a broader filter.")
            return

        model, daily, feature_cols, test_dates, y_test, y_pred, metrics = result
        horizon = fc_horizon

        # ── Metrics ──
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{metrics['MAE']:.2f}",  help="Mean Absolute Error")
        m2.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Root Mean Squared Error")
        m3.metric("R² Score", f"{metrics['R2']:.3f}", help="Coefficient of Determination")

        st.markdown("---")

        # ── Actual vs Predicted ──
        st.markdown("#### Actual vs Predicted Daily Quantity (Test Set)")
        comp_df = pd.DataFrame({
            "Date": pd.to_datetime(test_dates),
            "Actual": y_test,
            "Predicted": np.round(y_pred, 1),
        })
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=comp_df["Date"], y=comp_df["Actual"],
                                      name="Actual", line=dict(color="#4f8ef7", width=2)))
        fig_comp.add_trace(go.Scatter(x=comp_df["Date"], y=comp_df["Predicted"],
                                      name="Predicted", line=dict(color="#f39c12", width=2, dash="dot")))
        fig_comp.update_layout(
            template="plotly_dark",
            paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
            font_color="#e8eaf0",
            title="Actual vs Predicted Quantity",
            legend=dict(bgcolor=_DARK_BG),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_comp, width="stretch")

        # ── Full history + forecast ──
        st.markdown(f"#### 🔮 {horizon}-Day Future Demand Forecast")

        # Iterative prediction
        last_known = daily["Qty"].values.tolist()
        last_date  = daily["Date"].max()

        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)
        future_qty   = []

        for fd in future_dates:
            feats = [
                fd.dayofweek,
                fd.day,
                fd.isocalendar().week,
                fd.month,
                int(fd.dayofweek >= 5),
                last_known[-1],
                last_known[-3] if len(last_known) >= 3  else np.mean(last_known),
                last_known[-7] if len(last_known) >= 7  else np.mean(last_known),
                last_known[-14] if len(last_known) >= 14 else np.mean(last_known),
                np.mean(last_known[-3:])  if len(last_known) >= 3  else np.mean(last_known),
                np.mean(last_known[-7:])  if len(last_known) >= 7  else np.mean(last_known),
                np.mean(last_known[-14:]) if len(last_known) >= 14 else np.mean(last_known),
                np.std(last_known[-3:])   if len(last_known) >= 3  else 0,
                np.std(last_known[-7:])   if len(last_known) >= 7  else 0,
                np.std(last_known[-14:])  if len(last_known) >= 14 else 0,
            ]
            pred = float(model.predict(np.array(feats).reshape(1,-1))[0])
            pred = max(0, round(pred, 1))
            future_qty.append(pred)
            last_known.append(pred)

        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast Qty": future_qty})

        # Combined chart
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Qty"],
            name="Historical", line=dict(color="#4f8ef7", width=2)))
        fig_fc.add_trace(go.Bar(
            x=forecast_df["Date"], y=forecast_df["Forecast Qty"],
            name=f"{horizon}-Day Forecast", marker_color="#f39c12", opacity=0.8))
        fig_fc.update_layout(
            template="plotly_dark",
            paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
            font_color="#e8eaf0",
            title=f"Historical + {horizon}-Day Forecast",
            legend=dict(bgcolor=_DARK_BG),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_fc, width="stretch")

        # Forecast table
        st.markdown("#### Forecasted Quantities")
        st.dataframe(forecast_df.style.format({"Forecast Qty":"{:.0f}"}),
                     width="stretch")

        # Feature importance
        st.markdown("#### 🎯 Feature Importance")
        importance = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_ if hasattr(model, "feature_importances_") else np.zeros(len(feature_cols)),
        }).sort_values("Importance", ascending=False)
        fig_imp = px.bar(importance.head(12), x="Importance", y="Feature",
                         orientation="h", color="Importance",
                         color_continuous_scale="Blues",
                         title="Top Feature Importances")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_fig(fig_imp), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 8: AI BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_insights(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">💡 AI Business Insights</div>', unsafe_allow_html=True)
    st.caption("Automatically generated data-driven insights from the entire filtered dataset.")

    revenue_df: pd.DataFrame = df.loc[df["Revenue"].notna() & (df["Revenue"] > 0)]

    # ── Category analytics ──
    cat_stats: pd.DataFrame = cast(
        pd.DataFrame,
        revenue_df.groupby("Category")
        .agg(Revenue=("Revenue","sum"), Qty=("Qty","sum"),
             Orders=("Order ID","count"), AvgOrderValue=("Revenue","mean"))
        .round(1),
    )
    _cancel_s: pd.Series = cast(pd.Series, df.groupby("Category")["is_cancelled"].mean())
    cat_stats["CancelRate"] = (_cancel_s * 100).round(1)
    _cat_rev: pd.Series = cast(pd.Series, cat_stats["Revenue"])
    _cat_cr: pd.Series  = cast(pd.Series, cat_stats["CancelRate"])
    top_cat: str         = str(_cat_rev.idxmax())
    worst_cat: str       = str(_cat_rev.idxmin())
    high_cancel_cat: str = str(_cat_cr.idxmax())

    # ── SKU analytics ──
    sku_stats: pd.DataFrame = (
        revenue_df.groupby(["SKU","Category"])
        .agg(Revenue=("Revenue","sum"), Qty=("Qty","sum"))
        .reset_index()
    )
    top_sku: pd.Series = sku_stats.sort_values("Revenue", ascending=False).iloc[0]
    low_sku: pd.Series = sku_stats.sort_values("Revenue", ascending=True).iloc[0]

    # ── State analytics ──
    state_stats: pd.DataFrame = (
        revenue_df.groupby("ship-state")
        .agg(Revenue=("Revenue","sum"), Orders=("Order ID","count"))
        .reset_index()
    )
    state_stats = state_stats.loc[state_stats["ship-state"] != "UNKNOWN"]
    top_state: pd.Series = state_stats.sort_values("Revenue", ascending=False).iloc[0]
    low_state: pd.Series = state_stats.sort_values("Revenue", ascending=True).iloc[0]

    # ── Time analytics ──
    _dr_s: pd.Series = cast(pd.Series, revenue_df.groupby("Date")["Revenue"].sum())
    daily_rev: pd.DataFrame = _dr_s.reset_index()
    daily_rev = daily_rev.sort_values("Date")
    trend_pct: float = 0.0
    trend_dir: str   = "stable"
    if len(daily_rev) >= 14:
        first_half  = float(cast(pd.Series, daily_rev.head(len(daily_rev)//2)["Revenue"]).mean())
        second_half = float(cast(pd.Series, daily_rev.tail(len(daily_rev)//2)["Revenue"]).mean())
        if first_half != 0:
            trend_pct = (second_half - first_half) / first_half * 100
        trend_dir = "upward 📈" if trend_pct > 0 else "downward 📉"

    cancel_rate: float    = float(df["is_cancelled"].mean()) * 100  # type: ignore[arg-type]
    _b2b_rev = float(revenue_df.loc[revenue_df["B2B"] == True, "Revenue"].sum())  # type: ignore[arg-type]
    _tot_rev  = float(revenue_df["Revenue"].sum())  # type: ignore[arg-type]
    b2b_rev_pct: float    = (_b2b_rev / _tot_rev * 100) if _tot_rev != 0 else 0.0
    amazon_fba_pct: float = float((df["Fulfilment"] == "Amazon").mean()) * 100  # type: ignore[arg-type]

    _dow_rev: pd.Series = cast(pd.Series, revenue_df.groupby("DayOfWeek")["Revenue"].sum())
    best_day: str  = str(_dow_rev.idxmax())
    worst_day: str = str(_dow_rev.idxmin())

    promotion_orders: float = float((revenue_df["promotion-ids"] != "None").mean()) * 100

    # ── Render insights ──
    insights = [
        ("success", "🏆 Top Revenue Category",
         f"<b>{top_cat}</b> is the highest-revenue category, generating "
         f"₹{cat_stats.loc[top_cat,'Revenue']/1e6:.2f}M. "
         f"Focus on inventory, visibility, and promotions for this segment."),

        ("warn", "⚠️ High Cancellation Category",
         f"<b>{high_cancel_cat}</b> has the highest cancellation rate at "
         f"{cat_stats.loc[high_cancel_cat,'CancelRate']:.1f}%. "
         "Investigate size/fit issues, quality feedback, and listing accuracy."),

        ("success", "🌍 Top Performing Region",
         f"<b>{str(top_state['ship-state']).title()}</b> leads with "
         f"₹{float(top_state['Revenue'])/1e6:.2f}M in revenue across {int(top_state['Orders']):,} orders. "  # type: ignore[arg-type]
         "Consider region-specific promotions and faster delivery commitments here."),

        ("", "📈 Overall Sales Trend",
         f"Revenue shows a <b>{trend_dir}</b> trend "
         f"({'+' if trend_pct>=0 else ''}{trend_pct:.1f}% comparing first vs second half). "
         + ("Momentum is strong — capitalise with seasonal campaigns." if trend_pct > 0
            else "Investigate potential causes: seasonality, competition, or stock-outs.")),

        ("", "🏅 Best-Selling SKU",
         f"SKU <b>{top_sku['SKU']}</b> (Category: {top_sku['Category']}) is the #1 revenue driver "
         f"with ₹{top_sku['Revenue']:,.0f}. Ensure consistent stock levels for this product."),

        ("danger", "📉 Low-Performing SKU",
         f"SKU <b>{low_sku['SKU']}</b> generated only ₹{low_sku['Revenue']:,.0f}. "
         "Consider phasing it out, discounting, or improving the product listing."),

        ("", f"📅 Best Day: {best_day}",
         f"<b>{best_day}</b> consistently drives the highest revenue. "
         f"Schedule flash sales, new launches, and ad boosts on this day. "
         f"<b>{worst_day}</b> is the weakest — consider targeted promotions to lift it."),

        ("warn" if cancel_rate > 15 else "success",
         f"🚫 Overall Cancellation Rate: {cancel_rate:.1f}%",
         ("Cancellation rate exceeds 15%. This is costing significant revenue and logistics overhead. "
          "Prioritise root-cause analysis on top cancelled categories and states."
          if cancel_rate > 15 else
          "Cancellation rate is within healthy bounds. Continue monitoring for any spikes.")),

        ("", "📦 Fulfilment Insight",
         f"Amazon FBA handles <b>{amazon_fba_pct:.1f}%</b> of orders. "
         "FBA orders typically yield faster delivery and higher customer satisfaction. "
         "Consider migrating more Merchant-fulfilled products to FBA for better metrics."),

        ("success" if b2b_rev_pct > 5 else "",
         f"🏢 B2B Revenue Contribution: {b2b_rev_pct:.1f}%",
         ("B2B segment is contributing meaningfully. Nurture key accounts with volume discounts "
          "and dedicated account management."
          if b2b_rev_pct > 5 else
          "B2B contribution is low. There is significant opportunity to grow the wholesale/bulk channel.")),

        ("", f"🎁 Promotion Usage: {promotion_orders:.1f}%",
         f"{promotion_orders:.1f}% of orders used a promotion. "
         + ("High promotion dependency — evaluate margin impact and test pricing without promotions."
            if promotion_orders > 50 else
            "Promotion usage is moderate. Targeted promotions on slow-moving SKUs could increase conversion.")),

        ("", "📍 Geographic Concentration Risk",
         f"Top 3 states account for a large share of revenue. "
         f"Least performing major state: <b>{str(low_state['ship-state']).title()}</b>. "
         "Diversifying geographic reach through targeted regional campaigns can reduce concentration risk."),
    ]

    for card_class, title, body in insights:
        st.markdown(
            f'<div class="insight-card {card_class}">'
            f'<h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    # Summary recommendation table
    st.markdown("---")
    st.markdown("#### 📋 Quick Recommendation Summary")
    recs = pd.DataFrame([
        {"Priority":"🔴 High",   "Area":"Cancellations",  "Action":f"Reduce {high_cancel_cat} cancellations ({cat_stats.loc[high_cancel_cat,'CancelRate']:.1f}%)"},
        {"Priority":"🔴 High",   "Area":"Top SKU Stock",  "Action":f"Ensure stock for SKU {top_sku['SKU']}"},
        {"Priority":"🟡 Medium", "Area":"Geography",      "Action":f"Launch campaigns in under-served states"},
        {"Priority":"🟡 Medium", "Area":"Day-of-week",    "Action":f"Focus ad spend on {best_day}"},
        {"Priority":"🟡 Medium", "Area":"B2B Growth",     "Action":"Activate wholesale/bulk pricing for B2B segment"},
        {"Priority":"🟢 Low",    "Area":"Low SKUs",       "Action":f"Phase out / discount {low_sku['SKU']}"},
        {"Priority":"🟢 Low",    "Area":"Promotions",     "Action":"Optimise promotion targeting to protect margins"},
    ])
    st.dataframe(recs, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH.resolve()}\nPlease place 'Amazon Sale Report.csv' in the project directory.")
        st.stop()

    raw, df = load_data()
    page, filtered = render_sidebar(df)

    # ── Page routing ──
    if page == "🏠 Executive Overview":
        page_overview(filtered)
    elif page == "📊 Data Quality":
        page_data_quality(raw, df)
    elif page == "📈 Sales Analytics":
        page_sales(filtered)
    elif page == "🗺️ Geographic Analytics":
        page_geo(filtered)
    elif page == "📦 Order & Fulfilment":
        page_orders(filtered)
    elif page == "🛍️ Product Analytics":
        page_products(filtered)
    elif page == "🤖 AI Demand Forecasting":
        page_forecasting(filtered)
    elif page == "💡 AI Business Insights":
        page_insights(filtered)


if __name__ == "__main__":
    main()
