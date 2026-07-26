# 🤖 AI-Powered Data Analyst Assistant

An AI-powered analytics app that automates **EDA**, a **strict auditable data-cleaning pipeline**, an **interactive KPI dashboard**, and **AI-generated business insights with a voice-enabled chatbot** from structured datasets.

Built with **Python, Streamlit, Pandas, Plotly, ReportLab, and Groq AI (`llama-3.3-70b-versatile`)**.

---

## ✨ Features

- **Data Upload** — CSV/TSV/Excel, auto-detects encoding & delimiter, multi-sheet support
- **Strict Data Cleaning Pipeline** — 5-stage engine (profile → structural → missing values → validity → outliers), fully logged, downloadable change log (JSON) and flagged outliers (CSV)
- **Automated EDA** — per-column profiling, data quality warnings, distribution grids, correlation heatmap
- **Business KPI Dashboard** — auto-detects revenue/quantity/category/date columns, no manual setup
- **AI Insights & Chatbot** — Groq-generated evidence-based business report (downloadable as PDF/txt) + a chat assistant with **voice input (Whisper)** and **voice output (gTTS)**

---

## 🛠️ Tech Stack

Python · Streamlit · Pandas/NumPy · Plotly · OpenPyXL · Groq SDK (LLM + Whisper) · ReportLab + Matplotlib (PDF) · audio-recorder-streamlit · gTTS

---

## 📂 Project Structure

```text
AI-Powered-Data-Analyst-Assistant
├── app.py                            # Entry point
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml                # GROQ_API_KEY (gitignored)
├── utils/
│   ├── data_profiler.py
│   ├── dashboard_helpers.py
│   ├── data_cleaning_engine.py
│   ├── pdf_report.py
│   └── ui.py
├── pages/
│   ├── 1_Upload_Data.py
│   ├── 2_Data_Analysis.py
│   ├── 3_Data_Cleaning.py
│   ├── 4_Smart_Visualizations.py
│   └── 5_AI_Insights.py
└── datasets/
    └── ecommerce_dataset_updated.csv
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Akshadgajapure/DataPilot.git
cd DataPilot
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
pip install reportlab matplotlib   # required for PDF export, missing from requirements.txt
```

---

## ▶️ Run

```bash
streamlit run app.py
```

Runs at `http://localhost:8501`

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push to GitHub (`.streamlit/secrets.toml` stays gitignored — only that file, not the whole `.streamlit/` folder, so `config.toml` still deploys)
2. Connect repo to Streamlit Community Cloud
3. Set **`app.py`** as the entry point
4. Add `GROQ_API_KEY` under App Settings → Secrets

---

## 🚀 Future Enhancements

- Add `reportlab`/`matplotlib` to `requirements.txt`
- Database connectivity (PostgreSQL/MySQL)
- Multi-file dataset analysis
- Custom AI prompts

---

## 👨‍💻 Author

**Akshad Gajapure**
