import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Interactive Dashboard")

# Check dataset
if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"]

# --------------------------
# Sidebar Filters
# --------------------------

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

# Category Filter
cat_cols = df.select_dtypes(include="object").columns.tolist()

if len(cat_cols) > 0:

    category = st.sidebar.selectbox(
        "Select Category Column",
        ["None"] + cat_cols
    )

    if category != "None":

        values = st.sidebar.multiselect(
            "Select Values",
            options=df[category].unique(),
            default=df[category].unique()
        )

        filtered_df = filtered_df[
            filtered_df[category].isin(values)
        ]

# --------------------------
# KPI Cards
# --------------------------

st.subheader("Key Performance Indicators")

num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

col1, col2, col3 = st.columns(3)

col1.metric("Rows", filtered_df.shape[0])

col2.metric("Columns", filtered_df.shape[1])

if len(num_cols) > 0:
    col3.metric(
        "Total",
        round(filtered_df[num_cols[0]].sum(),2)
    )

st.divider()

# --------------------------
# Line Chart
# --------------------------

st.subheader("Trend Analysis")

x_col = st.selectbox(
    "X Axis",
    filtered_df.columns
)

y_col = st.selectbox(
    "Y Axis",
    num_cols
)

fig = px.line(
    filtered_df,
    x=x_col,
    y=y_col,
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Bar Chart
# --------------------------

st.subheader("Bar Chart")

x_bar = st.selectbox(
    "Category",
    filtered_df.columns,
    key="barx"
)

y_bar = st.selectbox(
    "Numeric Column",
    num_cols,
    key="bary"
)

bar = px.bar(
    filtered_df,
    x=x_bar,
    y=y_bar,
    color=x_bar
)

st.plotly_chart(bar, use_container_width=True)

# --------------------------
# Pie Chart
# --------------------------

if len(cat_cols) > 0:

    st.subheader("Category Distribution")

    pie_col = st.selectbox(
        "Select Category",
        cat_cols
    )

    pie = filtered_df[pie_col].value_counts().reset_index()

    pie.columns = [pie_col, "Count"]

    fig = px.pie(
        pie,
        names=pie_col,
        values="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Scatter Plot
# --------------------------

if len(num_cols) >= 2:

    st.subheader("Scatter Plot")

    x = st.selectbox(
        "X",
        num_cols,
        key="scatterx"
    )

    y = st.selectbox(
        "Y",
        num_cols,
        key="scattery"
    )

    scatter = px.scatter(
        filtered_df,
        x=x,
        y=y,
        color=x
    )

    st.plotly_chart(scatter, use_container_width=True)