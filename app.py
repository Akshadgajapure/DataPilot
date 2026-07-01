import streamlit as st

st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI-Powered Data Analyst Assistant")

st.markdown(
"""
### Analyze Any Dataset in Minutes 🚀

Upload your CSV or Excel dataset and let AI help you perform professional data analysis.

### Features

- 📂 Upload CSV / Excel datasets
- 📊 Automatic Exploratory Data Analysis (EDA)
- 📈 Interactive Dashboard
- 🤖 AI-Powered Business Insights
- 📋 Executive Summary Generation

---

### Tech Stack

**Python • Pandas • Plotly • Streamlit • Gemini AI**

👈 **Use the sidebar to navigate through the application.**
"""
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📂 File Support", "CSV / Excel")
col2.metric("📊 Charts", "Interactive")
col3.metric("🤖 AI", "Gemini")
col4.metric("📈 Dashboard", "Live")

st.divider()

st.info(
    "💡 Start by opening **Upload Data** from the left sidebar and upload your dataset."
)

st.divider()

st.caption("Developed by Ashika Srivastava | AI-Powered Data Analyst Assistant")