"""
pages/3_Smart_Visualizations.py
--------------------------------
Business KPI Dashboard

Auto-detects revenue, quantity, category, country, date, and status
columns to render a proper analyst-grade KPI dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="KPI Dashboard · Data Analyst",
    page_icon="📊",
    layout="wide",
)

from utils.ui import inject_custom_css, sidebar_brand
inject_custom_css()
sidebar_brand()

st.title("📊 Business KPI Dashboard")
st.markdown("Automatically computed KPIs and charts from your dataset. No manual setup required.")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first from the **Upload Data** page.")
    st.stop()

# Always prefer cleaned dataset
df = st.session_state.get("clean_df", st.session_state["df"]).copy()
if "clean_df" in st.session_state:
    st.caption("✅ Using **Cleaned Dataset**")

# Drop fully empty cols
df = df.dropna(axis=1, how="all")

# ── Helper: detect columns by keyword ─────────────────────────────────────────
def find_col(df, keywords, exclude_keywords=None):
    """Return first column whose name contains any keyword (case-insensitive)."""
    for col in df.columns:
        col_l = col.lower()
        if any(kw in col_l for kw in keywords):
            if exclude_keywords and any(ek in col_l for ek in exclude_keywords):
                continue
            return col
    return None

# ── Auto-detect business columns ──────────────────────────────────────────────

numeric_cols = [c for c in df.select_dtypes(include="number").columns
                if not c.endswith(("_Is_Missing", "_Is_Outlier", "_Is_Invalid"))
                and not any(kw in c.lower() for kw in ["id", "zip", "phone", "postal", "code", "index"])]

cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique() <= 50]

revenue_col   = find_col(df[numeric_cols] if numeric_cols else df.iloc[:,:0], ["revenue", "sales", "total_price", "total_amount", "income", "earnings"])
price_col     = find_col(df[numeric_cols] if numeric_cols else df.iloc[:,:0], ["unit_price", "price", "cost"], exclude_keywords=["total"])
qty_col       = find_col(df[numeric_cols] if numeric_cols else df.iloc[:,:0], ["quantity", "qty", "units", "volume"])
category_col  = find_col(df, ["category", "product_category", "segment", "type", "department"])
country_col   = find_col(df, ["country", "region", "location", "territory", "market"])
product_col   = find_col(df, ["product", "item", "sku", "product_name"])
status_col    = find_col(df, ["status", "order_status", "state"])
customer_col  = find_col(df, ["customer", "client", "customer_id"])

# Date column: prefer already-parsed, then try string detection
dt_parsed = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
date_col = dt_parsed[0] if dt_parsed else None
if not date_col:
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(30), format="mixed", errors="raise")
            date_col = col
            break
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOP KPI METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📌 Key Performance Indicators")

kpi_cols = st.columns(5)

# KPI 1: Total Revenue
with kpi_cols[0]:
    if revenue_col:
        total_rev = df[revenue_col].sum()
        st.metric("💰 Total Revenue", f"{total_rev:,.0f}")
    else:
        st.metric("📦 Total Records", f"{len(df):,}")

# KPI 2: Total Orders / Records
with kpi_cols[1]:
    total_orders = len(df)
    if customer_col:
        unique_customers = df[customer_col].nunique()
        st.metric("👥 Unique Customers", f"{unique_customers:,}")
    else:
        st.metric("📋 Total Orders", f"{total_orders:,}")

# KPI 3: Average Order Value
with kpi_cols[2]:
    if revenue_col:
        aov = df[revenue_col].mean()
        st.metric("🧾 Avg Order Value", f"{aov:,.2f}")
    elif price_col:
        avg_price = df[price_col].mean()
        st.metric("💵 Avg Unit Price", f"{avg_price:,.2f}")
    else:
        st.metric("📋 Total Orders", f"{total_orders:,}")

# KPI 4: Cancellation / Return Rate
with kpi_cols[3]:
    if status_col:
        cancel_keywords = ["cancel", "return", "refund", "reject", "failed"]
        cancel_mask = df[status_col].astype(str).str.lower().str.contains("|".join(cancel_keywords), na=False)
        cancel_rate = cancel_mask.mean() * 100
        st.metric("❌ Cancellation Rate", f"{cancel_rate:.1f}%")
    elif qty_col:
        total_qty = df[qty_col].sum()
        st.metric("📦 Total Quantity Sold", f"{total_qty:,.0f}")
    else:
        missing_pct = df.isnull().mean().mean() * 100
        st.metric("🔍 Missing Data", f"{missing_pct:.1f}%")

# KPI 5: Date range
with kpi_cols[4]:
    if date_col:
        try:
            parsed_dates = pd.to_datetime(df[date_col], format="mixed", errors="coerce").dropna()
            span = (parsed_dates.max() - parsed_dates.min()).days
            st.metric("📅 Date Span", f"{span} days")
        except Exception:
            st.metric("📅 Date Column", date_col)
    elif category_col:
        n_cats = df[category_col].nunique()
        st.metric("🗂️ Categories", f"{n_cats}")
    else:
        st.metric("🔢 Numeric Cols", f"{len(numeric_cols)}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REVENUE / METRIC BY CATEGORY & COUNTRY
# ══════════════════════════════════════════════════════════════════════════════
metric_col = revenue_col or (numeric_cols[0] if numeric_cols else None)
metric_label = revenue_col or (numeric_cols[0] if numeric_cols else "Count")

col_a, col_b = st.columns(2)

with col_a:
    if metric_col and category_col:
        st.subheader(f"📦 {metric_label} by {category_col}")
        cat_summary = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(
            cat_summary, x=metric_col, y=category_col,
            orientation="h",
            color=metric_col,
            color_continuous_scale="Blues",
            labels={metric_col: metric_label, category_col: ""},
        )
        fig.update_layout(showlegend=False, margin=dict(t=30, b=10), height=400,
                          coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    elif category_col:
        st.subheader(f"📦 Record Count by {category_col}")
        cat_summary = df[category_col].value_counts().head(15).reset_index()
        cat_summary.columns = [category_col, "Count"]
        fig = px.bar(cat_summary, x="Count", y=category_col, orientation="h",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, margin=dict(t=30, b=10), height=400,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category column detected. Rename a column to include 'category', 'type', or 'segment'.")

with col_b:
    if metric_col and country_col:
        st.subheader(f"🌍 {metric_label} by {country_col}")
        country_summary = df.groupby(country_col)[metric_col].sum().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(
            country_summary, x=country_col, y=metric_col,
            color=metric_col,
            color_continuous_scale="Teal",
            labels={metric_col: metric_label},
        )
        fig.update_layout(showlegend=False, margin=dict(t=30, b=10), height=400,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    elif country_col:
        st.subheader(f"🌍 Orders by {country_col}")
        c_summary = df[country_col].value_counts().head(15).reset_index()
        c_summary.columns = [country_col, "Count"]
        fig = px.bar(c_summary, x=country_col, y="Count", color="Count", color_continuous_scale="Teal")
        fig.update_layout(showlegend=False, margin=dict(t=30, b=10), height=400,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fall back to the best categorical column
        best_cat = sorted(cat_cols, key=lambda c: df[c].nunique())[0] if cat_cols else None
        if best_cat and metric_col:
            st.subheader(f"📊 {metric_label} by {best_cat}")
            summary = df.groupby(best_cat)[metric_col].sum().sort_values(ascending=False).head(12).reset_index()
            fig = px.pie(summary, names=best_cat, values=metric_col, hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(t=30, b=10), height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No country/region column detected.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MONTHLY TREND
# ══════════════════════════════════════════════════════════════════════════════
if date_col and metric_col:
    st.subheader(f"📈 Monthly {metric_label} Trend")
    try:
        df_trend = df.copy()
        df_trend["__date__"] = pd.to_datetime(df_trend[date_col], format="mixed", errors="coerce", dayfirst=True)
        df_trend = df_trend.dropna(subset=["__date__"])
        df_trend["__month__"] = df_trend["__date__"].dt.to_period("M").astype(str)
        monthly = df_trend.groupby("__month__")[metric_col].sum().reset_index()
        monthly.columns = ["Month", metric_label]
        monthly = monthly.sort_values("Month")

        fig = px.line(
            monthly, x="Month", y=metric_label,
            markers=True,
            color_discrete_sequence=["#6366F1"],
        )
        fig.update_traces(line_width=2.5, marker_size=7)
        fig.update_layout(
            margin=dict(t=30, b=10), height=350,
            xaxis_title="Month", yaxis_title=metric_label,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render time trend: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TOP PRODUCTS & STATUS BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns(2)

with col_c:
    if product_col and metric_col:
        st.subheader(f"🏆 Top 10 {product_col}s by {metric_label}")
        top_products = df.groupby(product_col)[metric_col].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(
            top_products, x=metric_col, y=product_col,
            orientation="h",
            color=metric_col,
            color_continuous_scale="Purples",
        )
        fig.update_layout(showlegend=False, margin=dict(t=30, b=10), height=400,
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    elif numeric_cols:
        st.subheader("📊 Numeric Column Distribution")
        best_num = numeric_cols[0]
        fig = px.histogram(df, x=best_num, nbins=40,
                           color_discrete_sequence=["#6366F1"], marginal="box")
        fig.update_layout(margin=dict(t=30, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

with col_d:
    if status_col:
        st.subheader(f"🔄 Order Status Breakdown")
        status_counts = df[status_col].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(
            status_counts, names="Status", values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(margin=dict(t=30, b=10), height=400)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    elif category_col and metric_col:
        st.subheader(f"🍩 {metric_label} Share by {category_col}")
        share = df.groupby(category_col)[metric_col].sum().reset_index()
        fig = px.pie(share, names=category_col, values=metric_col, hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(margin=dict(t=30, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)