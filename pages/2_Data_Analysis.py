"""
pages/2_Data_Analysis.py
------------------------
Automated Data Profiling Report

This page has been upgraded from a basic summary/correlation view into a
full automated data-profiling report. All heavy computation is delegated to
utils/data_profiler.py; this file handles only Streamlit layout and rendering.

Sections
--------
1. Top-level KPI metrics (rows, cols, missing, duplicates)
2. Plain-language Dataset Summary (computed, not LLM-generated)
3. Per-column Profile Table with expandable detail panels
4. Data Quality Warnings (critical / warning / info)
5. Distribution Grid — histograms for numeric, bar charts for categorical (3-per-row)
6. Correlation Heatmap with high-corr pair annotations
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Make utils importable when Streamlit runs pages from the project root ──────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_profiler import (
    build_full_profile,
    build_plain_language_summary,
    build_quality_warnings,
)

# ─── Page guard ───────────────────────────────────────────────────────────────
st.title("📊 Data Analysis — Automated Profiling Report")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first from the **Upload Data** page.")
    st.stop()

df = st.session_state["df"]


# ─── Profile computation (cached to avoid recomputing on every widget change) ─

@st.cache_data(show_spinner="Building dataset profile…")
def get_profile(_df: pd.DataFrame, dataset_id: str):
    """
    Run the full profiling pipeline and return results.

    The leading underscore on `_df` tells st.cache_data to skip hashing
    the DataFrame (DataFrames are unhashable). We pass `dataset_id` so the 
    cache correctly busts when a new dataset is uploaded.
    """
    profile  = build_full_profile(_df)
    warnings = build_quality_warnings(_df, profile)
    summary  = build_plain_language_summary(_df, profile, warnings)
    return profile, warnings, summary

# Get dataset_id to use as cache key (fallback to "default" if missing)
dataset_id = st.session_state.get("dataset_id", "default")
profile, quality_warnings, summary_text = get_profile(df, dataset_id)


# Store the summary and full profile in session state so 4_AI_Insights.py
# can use real computed stats instead of a raw df.describe() dump.
st.session_state["dataset_profile_summary"] = summary_text
st.session_state["dataset_profile"]         = profile


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Top-level KPI Metrics
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("📌 Dataset at a Glance")

total_rows, total_cols = df.shape
total_missing = int(df.isnull().sum().sum())
dup_count     = int(df.duplicated().sum())
missing_pct   = (
    round(total_missing / (total_rows * total_cols) * 100, 1)
    if total_rows * total_cols > 0 else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Rows",            f"{total_rows:,}")
kpi2.metric("Columns",         total_cols)
kpi3.metric(
    "Missing Values",
    f"{total_missing:,}",
    delta=f"{missing_pct}% of all cells",
    delta_color="inverse",
)
kpi4.metric(
    "Duplicate Rows",
    f"{dup_count:,}",
    delta=f"{round(dup_count / total_rows * 100, 1)}% of rows" if total_rows > 0 else "0%",
    delta_color="inverse",
)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Plain-language Dataset Summary
# ════════════════════════════════════════════════════════════════════════════════

st.info(f"📝 **Dataset Summary**\n\n{summary_text}")

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Per-column Profile Table
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("🔎 Column-by-Column Profile")

# Build a flat summary table row for each column
table_rows = []
for entry in profile:
    row = {
        "Column":      entry["col_name"],
        "Type":        entry["col_type"],
        "Dtype":       entry["dtype_str"],
        "Missing %":   f"{entry['missing_pct']}%",
        "Unique":      "—",
        "Cardinality": "—",
        "Key Stats":   "",
    }

    if entry["col_type"] == "numeric" and entry["numeric_stats"]:
        s = entry["numeric_stats"]
        row["Key Stats"] = (
            f"min={s['min']} | max={s['max']} | mean={s['mean']} | "
            f"std={s['std']} | skew={s['skewness']} | outliers={s['outlier_count']}"
        )

    elif entry["col_type"] == "categorical" and entry["cat_stats"]:
        c = entry["cat_stats"]
        row["Unique"]      = str(c["unique_count"])
        row["Cardinality"] = c["cardinality"] + (" 🪪 ID?" if c["is_id_column"] else "")
        top = c["top_values"]
        row["Key Stats"]   = " | ".join(f"{v} ({p}%)" for v, _, p in top[:3])

    elif entry["col_type"] == "datetime" and entry["dt_stats"]:
        d = entry["dt_stats"]
        row["Key Stats"] = (
            f"{d['min_date']} → {d['max_date']} "
            f"({d['span_days']} days, {d['gap_count']} gap(s))"
        )

    table_rows.append(row)

profile_df = pd.DataFrame(table_rows)


# Add a raw numeric column for the progress bar, then display with column_config
profile_df["_missing_num"] = pd.to_numeric(
    profile_df["Missing %"].str.replace("%", "", regex=False), errors="coerce"
).fillna(0)

st.dataframe(
    profile_df.drop(columns=["_missing_num"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Missing %": st.column_config.ProgressColumn(
            "Missing %",
            help="Percentage of missing values in this column",
            format="%s",
            min_value=0,
            max_value=100,
        ),
    },
)


# ── Expandable per-column detail panels ───────────────────────────────────────
st.markdown("**🔍 Expand a column below for full detail:**")

for entry in profile:
    with st.expander(
        f"📌 **{entry['col_name']}**  "
        f"· {entry['col_type']}  "
        f"· {entry['missing_pct']}% missing"
    ):
        left, right = st.columns(2)

        left.markdown(f"**Dtype:** `{entry['dtype_str']}`")
        left.markdown(
            f"**Missing:** {entry['missing_count']:,} rows "
            f"({entry['missing_pct']}%)"
        )

        if entry["col_type"] == "numeric" and entry["numeric_stats"]:
            s = entry["numeric_stats"]
            left.markdown(f"**Min:** {s['min']}  ·  **Max:** {s['max']}")
            left.markdown(f"**Mean:** {s['mean']}  ·  **Median:** {s['median']}")
            left.markdown(f"**Std Dev:** {s['std']}  ·  **Skewness:** {s['skewness']}")
            right.metric("Outliers (IQR method)", s["outlier_count"])

        elif entry["col_type"] == "categorical" and entry["cat_stats"]:
            c = entry["cat_stats"]
            left.markdown(f"**Unique Values:** {c['unique_count']:,}")
            left.markdown(f"**Cardinality:** {c['cardinality']}")
            left.markdown(
                f"**Likely ID Column:** {'⚠️ Yes' if c['is_id_column'] else '✅ No'}"
            )
            if c["top_values"]:
                tv_df = pd.DataFrame(
                    c["top_values"], columns=["Value", "Count", "Pct %"]
                )
                right.markdown("**Top 5 Values:**")
                right.dataframe(tv_df, hide_index=True)

        elif entry["col_type"] == "datetime" and entry["dt_stats"]:
            d = entry["dt_stats"]
            left.markdown(f"**Range:** {d['min_date']} → {d['max_date']}")
            left.markdown(f"**Span:** {d['span_days']} days")
            right.metric("Date Gaps Detected", d["gap_count"])

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Data Quality Warnings
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("⚠️ Data Quality Warnings")

if not quality_warnings:
    st.success("✅ No significant data quality issues detected!")
else:
    error_w = [w for w in quality_warnings if w["level"] == "error"]
    warn_w  = [w for w in quality_warnings if w["level"] == "warning"]
    info_w  = [w for w in quality_warnings if w["level"] == "info"]

    if error_w:
        st.markdown("#### 🔴 Critical Issues")
        for w in error_w:
            st.error(w["message"])

    if warn_w:
        st.markdown("#### 🟡 Warnings")
        for w in warn_w:
            st.warning(w["message"])

    if info_w:
        st.markdown("#### 🔵 Informational")
        for w in info_w:
            st.info(w["message"])

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Distribution Grid
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Distribution Visualisations")

numeric_col_names = [e["col_name"] for e in profile if e["col_type"] == "numeric"]
cat_col_names     = [e["col_name"] for e in profile if e["col_type"] == "categorical"]

# ── Numeric histograms — rendered 3 per row ────────────────────────────────────
if numeric_col_names:
    st.markdown("#### Numeric Columns — Histograms")
    # Split column list into rows of 3
    for chunk_start in range(0, len(numeric_col_names), 3):
        chunk = numeric_col_names[chunk_start: chunk_start + 3]
        grid  = st.columns(len(chunk))
        for widget_col, col_name in zip(grid, chunk):
            fig = px.histogram(
                df, x=col_name, nbins=30,
                title=col_name,
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=38, b=10),
                height=260,
                showlegend=False,
                title_font_size=13,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            widget_col.plotly_chart(fig, use_container_width=True)

# ── Categorical bar charts — rendered 3 per row ────────────────────────────────
if cat_col_names:
    st.markdown("#### Categorical Columns — Top Category Frequency")
    for chunk_start in range(0, len(cat_col_names), 3):
        chunk = cat_col_names[chunk_start: chunk_start + 3]
        grid  = st.columns(len(chunk))
        for widget_col, col_name in zip(grid, chunk):
            vc = df[col_name].value_counts().head(10).reset_index()
            vc.columns = [col_name, "Count"]
            fig = px.bar(
                vc, x=col_name, y="Count",
                title=col_name,
                color_discrete_sequence=["#f59e0b"],
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=38, b=10),
                height=260,
                showlegend=False,
                title_font_size=13,
                xaxis_tickangle=-30,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            widget_col.plotly_chart(fig, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════════

st.subheader("🔗 Correlation Matrix")

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr()
    fig  = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(
        height=max(400, 80 * numeric_df.shape[1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annotate highly correlated pairs surfaced by quality warnings
    high_corr_warnings = [
        w for w in quality_warnings if "correlated" in w["message"]
    ]
    if high_corr_warnings:
        st.markdown("**Highly correlated pairs (|r| > 0.85) — potential redundancy:**")
        for w in high_corr_warnings:
            st.info(w["message"])
else:
    st.info(
        "ℹ️ At least 2 numeric columns are required to display a correlation matrix."
    )