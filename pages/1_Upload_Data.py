import streamlit as st
import pandas as pd

st.title("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read the uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Store dataset for other pages
    st.session_state["df"] = df

    st.success("✅ Dataset uploaded successfully!")

    # Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Dataset Metrics
    st.subheader("📊 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # Column Information
    st.subheader("📝 Column Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(info)