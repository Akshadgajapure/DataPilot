import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Data Analysis")

# Check if dataset is uploaded
if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"]

# Dataset Information
st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

# Data Types
st.subheader("Column Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})

st.dataframe(dtype_df)

# Summary Statistics
st.subheader("Summary Statistics")

st.dataframe(df.describe())

# Missing Values
st.subheader("Missing Values")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    st.success("No Missing Values Found!")
else:
    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x":"Column","y":"Missing Values"},
        title="Missing Values by Column"
    )

    st.plotly_chart(fig, use_container_width=True)

# Correlation Matrix
st.subheader("Correlation Matrix")

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

# Distribution Analysis
st.subheader("Distribution Analysis")

numeric_columns = numeric_df.columns.tolist()

selected = st.selectbox(
    "Select Numeric Column",
    numeric_columns
)

fig = px.histogram(
    df,
    x=selected,
    nbins=30,
    title=f"{selected} Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# Outlier Detection
st.subheader("Outlier Detection")

selected_box = st.selectbox(
    "Select Column",
    numeric_columns,
    key="box"
)

fig = px.box(
    df,
    y=selected_box,
    title=f"{selected_box} Box Plot"
)

st.plotly_chart(fig, use_container_width=True)