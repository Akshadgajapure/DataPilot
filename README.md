# 📊 DataIQ – AI-Powered Data Analytics & Business Intelligence Platform

DataIQ is a general-purpose AI-powered analytics platform that transforms raw structured datasets into **clean, validated, visualized, and actionable business insights**.

It automates **data profiling, data cleaning, exploratory data analysis (EDA), domain-adaptive KPI generation, interactive dashboards, AI-powered business insights, natural language SQL querying, and report generation**.

Built with **Python, Streamlit, Pandas, Plotly, DuckDB, Groq AI (Llama 3.3 70B), Whisper, ReportLab, and gTTS**.

---

# ✨ Features

### 📂 Data Upload
- Upload CSV, TSV, and Excel files
- Automatic encoding & delimiter detection
- Multi-sheet Excel support
- Automatic schema detection

---

### 🧹 Intelligent Data Cleaning

Multi-stage auditable cleaning pipeline:

```
Profile
   ↓
Structural Cleaning
   ↓
Missing Value Handling
   ↓
Validity Checks
   ↓
Outlier Detection
   ↓
Validation
```

Features include:

- Missing value handling
- Duplicate removal
- Invalid value detection
- Logical range validation
- Outlier detection & flagging
- Cleaning audit log (JSON)
- Flagged outlier export (CSV)

---

### 📊 Automated EDA

Automatically generates:

- Dataset summary
- Column profiling
- Missing value analysis
- Numerical statistics
- Category distributions
- Correlation heatmaps
- Interactive Plotly visualizations
- Data quality warnings

---

### 📈 Smart KPI Dashboard

Automatically detects meaningful KPIs for different dataset domains:

- HR Analytics
- E-commerce
- Finance
- Healthcare
- Marketing
- Manufacturing
- Education

Falls back to generic statistical KPIs for unknown datasets.

---

### 🤖 AI Business Insights

Powered by **Groq Llama-3.3-70B**.

Generates:

- Executive summary
- Statistical findings
- Evidence-based insights
- Business recommendations
- Downloadable PDF & TXT reports

Designed to:

- Base insights on actual statistics
- Distinguish facts from interpretations
- Avoid unsupported causal claims
- Classify correlation strength

---

### 💬 Natural Language → SQL

Query your dataset using plain English.

Example:

```
Show the top 5 employees
```

↓

Automatically generates SQL using **DuckDB** and executes it on the uploaded dataset.

---

### 🎙️ Voice Assistant

Supports conversational analytics using:

- 🎤 Whisper (Speech-to-Text)
- 🤖 Groq LLM
- 🔊 gTTS (Text-to-Speech)

---

# 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- DuckDB
- Groq SDK
- Llama 3.3 70B
- Whisper
- gTTS
- ReportLab
- OpenPyXL
- Matplotlib

---

# 📂 Project Structure

```text
DataIQ/
│
├── app.py
├── requirements.txt
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
│
├── utils/
│   ├── data_profiler.py
│   ├── dashboard_helpers.py
│   ├── data_cleaning_engine.py
│   ├── pdf_report.py
│   └── ui.py
│
├── pages/
│   ├── 1_Upload_Data.py
│   ├── 2_Data_Analysis.py
│   ├── 3_Data_Cleaning.py
│   ├── 4_Smart_Visualizations.py
│   ├── 5_AI_Insights.py
│   └── 6_Text_to_SQL.py
│
└── datasets/
```

---

# ⚙️ Installation

```bash
git clone https://github.com/Akshadgajapure/DataIQ.git
cd DataIQ

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

# 🔑 Configure API Key

Create:

```
.streamlit/secrets.toml
```

```toml
GROQ_API_KEY="your_api_key"
```

---

# ▶️ Run

```bash
streamlit run app.py
```

Runs locally at:

```
http://localhost:8501
```

---

# 🌐 Deployment

Deploy easily using **Streamlit Community Cloud**:

1. Push repository to GitHub
2. Connect GitHub repository
3. Select `app.py`
4. Add `GROQ_API_KEY` in Secrets
5. Deploy

---

# 🚀 Future Enhancements

- PostgreSQL / MySQL connectivity
- Multi-file analytics
- Automated dataset comparison
- ML-based anomaly detection
- Custom KPI configuration
- Persistent analytical sessions
- Advanced statistical testing

---

# 👨‍💻 Author

**Akshad Gajapure**

B.Tech Chemical Engineering  
National Institute of Technology Raipur

---

## ⭐ Workflow

```
Upload Dataset
      ↓
Data Profiling
      ↓
Data Cleaning
      ↓
EDA
      ↓
Smart KPI Dashboard
      ↓
AI Business Insights
      ↓
Natural Language → SQL
      ↓
Voice Assistant
      ↓
Reports & Export
```

---

⭐ **If you find this project useful, consider giving it a Star!**
