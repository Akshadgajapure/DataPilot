import streamlit as st

st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys, os
if os.path.dirname(__file__) + "/.." not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
from utils.ui import inject_custom_css, sidebar_brand
inject_custom_css()
sidebar_brand()

st.title("🤖 AI-Powered Data Analyst Assistant")

st.markdown(
"""
### Interactive Data Intelligence 🚀

Upload your raw datasets (CSV/Excel) and let our pipeline clean, analyze, and profile your data for you.

Our platform helps you identify data quality issues, visualize trends, and even chat with your data using natural language SQL generation.

---

### Tech Stack

**Python • Pandas • Plotly • DuckDB • Streamlit • Groq AI**

👈 **Use the sidebar to navigate through the application.**
"""
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📂 File Support", "CSV / Excel")
col2.metric("🛡️ Data Integrity", "Strict Pipeline")
col3.metric("🤖 AI Engine", "Groq LLM")
col4.metric("📄 Export", "PDF / JSON / CSV")

st.divider()

st.info(
    "💡 Start by opening **Upload Data** from the left sidebar and upload your dataset."
)

st.divider()

st.caption("Developed by Akshad Gajapure | AI-Powered Data Analyst Assistant")
