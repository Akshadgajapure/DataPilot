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
### Industry-Grade Data Intelligence 🚀

Upload your raw, messy datasets (CSV/Excel) and let our 3-pass AI engine transform them into strictly audited, production-ready assets.

Our platform is engineered for data integrity. We prioritize defensible logic, robust statistical imputation, and completely transparent audit logging so you can trust your data pipelines.

---

### Tech Stack

**Python • Pandas • Plotly • ReportLab • Streamlit • Groq AI**

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