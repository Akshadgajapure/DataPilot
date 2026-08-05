"""
utils/pdf_report.py
-------------------
Industry-Grade PDF Report Generator using ReportLab.

Generates a comprehensive, multi-section data analysis report:
  - Cover Page
  - Table of Contents
  - Executive Summary
  - Dataset Overview
  - Statistical Summary (numeric + categorical)
  - Data Quality Analysis
  - Correlation Analysis
  - Distribution Analysis (top numeric columns)
  - AI-Generated Business Insights (if available)
  - Appendix: Full Column Reference
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Colour Palette ────────────────────────────────────────────────────────────
PRIMARY       = colors.HexColor("#1A1A2E")  # Dark navy
ACCENT        = colors.HexColor("#E94560")  # Red accent
ACCENT2       = colors.HexColor("#0F3460")  # Mid navy
LIGHT_GREY    = colors.HexColor("#F4F4F4")
MID_GREY      = colors.HexColor("#CCCCCC")
DARK_GREY     = colors.HexColor("#555555")
WHITE         = colors.white
SUCCESS_GREEN = colors.HexColor("#27AE60")
WARNING_AMBER = colors.HexColor("#F39C12")
DANGER_RED    = colors.HexColor("#E74C3C")


# ── Page Layout ───────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm

def _hex(c): return c.hexval() if hasattr(c, "hexval") else str(c)


# ═════════════════════════════════════════════════════════════════════════════
# Canvas callbacks (headers + footers)
# ═════════════════════════════════════════════════════════════════════════════

def _cover_canvas(canvas, doc):
    """Draw the cover-page background and decorative elements."""
    canvas.saveState()
    # Full-page navy background
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Red accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.42, PAGE_W, 0.35 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT2)
    canvas.rect(0, PAGE_H * 0.42 - 0.15 * cm, PAGE_W, 0.12 * cm, fill=1, stroke=0)
    canvas.restoreState()


def _report_canvas(canvas, doc):
    """Draw header and footer on every report page."""
    canvas.saveState()
    pw, ph = PAGE_W, PAGE_H

    # Top border
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, ph - 1.2 * cm, pw, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, ph - 1.35 * cm, pw, 0.15 * cm, fill=1, stroke=0)

    # Header title
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(WHITE)
    canvas.drawString(MARGIN, ph - 0.85 * cm, "DATA ANALYSIS REPORT")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(pw - MARGIN, ph - 0.85 * cm, doc.title)

    # Bottom footer
    canvas.setFillColor(LIGHT_GREY)
    canvas.rect(0, 0, pw, 1.0 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 1.0 * cm, pw, 0.12 * cm, fill=1, stroke=0)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(DARK_GREY)
    canvas.drawString(MARGIN, 0.35 * cm, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    canvas.drawCentredString(pw / 2, 0.35 * cm, "CONFIDENTIAL — FOR INTERNAL USE ONLY")
    canvas.drawRightString(pw - MARGIN, 0.35 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ═════════════════════════════════════════════════════════════════════════════
# Style helpers
# ═════════════════════════════════════════════════════════════════════════════

def _styles():
    base = getSampleStyleSheet()

    cover_title = ParagraphStyle(
        "CoverTitle", fontSize=36, leading=44, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
        spaceAfter=6
    )
    cover_sub = ParagraphStyle(
        "CoverSub", fontSize=14, leading=20, textColor=MID_GREY,
        fontName="Helvetica", alignment=TA_LEFT, spaceAfter=4
    )
    cover_meta = ParagraphStyle(
        "CoverMeta", fontSize=10, leading=14, textColor=MID_GREY,
        fontName="Helvetica", alignment=TA_LEFT
    )
    h1 = ParagraphStyle(
        "H1Report", fontSize=18, leading=24, textColor=PRIMARY,
        fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6,
        borderPad=(0, 0, 4, 0)
    )
    h2 = ParagraphStyle(
        "H2Report", fontSize=13, leading=18, textColor=ACCENT2,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4
    )
    h3 = ParagraphStyle(
        "H3Report", fontSize=11, leading=15, textColor=DARK_GREY,
        fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3
    )
    body = ParagraphStyle(
        "BodyReport", fontSize=9.5, leading=14, textColor=colors.black,
        fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY
    )
    body_small = ParagraphStyle(
        "BodySmall", fontSize=8.5, leading=12, textColor=DARK_GREY,
        fontName="Helvetica", spaceAfter=2
    )
    callout = ParagraphStyle(
        "Callout", fontSize=9.5, leading=14, textColor=ACCENT2,
        fontName="Helvetica-Oblique", leftIndent=12, rightIndent=12,
        borderPad=6, backColor=colors.HexColor("#EAF0FB"),
        borderColor=ACCENT2, borderWidth=0.5, spaceAfter=6
    )
    toc_entry = ParagraphStyle(
        "TOCEntry", fontSize=10, leading=16, fontName="Helvetica",
        textColor=colors.black, leftIndent=0
    )
    return dict(cover_title=cover_title, cover_sub=cover_sub,
                cover_meta=cover_meta, h1=h1, h2=h2, h3=h3,
                body=body, body_small=body_small, callout=callout,
                toc_entry=toc_entry)


_TABLE_HEADER_STYLE = TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  PRIMARY),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0),  9),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
    ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
    ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
    ("ROWHEIGHT",     (0, 0), (-1, -1), 16),
    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("LINEBELOW",     (0, 0), (-1, 0),  1.5, ACCENT),
])


def _table(data, col_widths=None, extra_style=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = list(_TABLE_HEADER_STYLE._cmds)
    if extra_style:
        style += extra_style
    t.setStyle(TableStyle(style))
    return t


def _section_rule(story, s):
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT,
                             spaceAfter=6, spaceBefore=2))


def _kpi_row(label, value, note=""):
    return [Paragraph(f"<b>{label}</b>", _styles()["body"]),
            Paragraph(str(value), _styles()["body"]),
            Paragraph(note, _styles()["body_small"])]


# ═════════════════════════════════════════════════════════════════════════════
# Chart helpers (matplotlib → ReportLab Image)
# ═════════════════════════════════════════════════════════════════════════════

def _fig_to_image(fig, width=15*cm, height=7*cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=height)


def _correlation_heatmap(df):
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, max(4, len(num_cols) * 0.7)))
    fig.patch.set_facecolor("#FAFAFA")
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < 0.7 else "white")
    ax.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    calc_height = max(5*cm, len(num_cols)*1.1*cm)
    return _fig_to_image(fig, width=16*cm, height=min(calc_height, 20*cm))


def _distribution_charts(df, max_cols=4):
    num_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    if not num_cols:
        return None
    n = len(num_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 3.5))
    fig.patch.set_facecolor("#FAFAFA")
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        data = df[col].dropna()
        ax.hist(data, bins=30, color="#0F3460", edgecolor="white", linewidth=0.4, alpha=0.85)
        mean_val = data.mean()
        ax.axvline(mean_val, color="#E94560", linewidth=1.5, linestyle="--", label=f"μ={mean_val:.2f}")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Frequency", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.set_facecolor("#F4F4F4")
    fig.suptitle("Numeric Column Distributions", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_image(fig, width=16*cm, height=5*cm)


def _top_categories_chart(df, max_cols=3):
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if 1 < df[c].nunique() <= 30][:max_cols]
    if not cat_cols:
        return None
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 3.8))
    fig.patch.set_facecolor("#FAFAFA")
    if n == 1:
        axes = [axes]
    palette = ["#1A1A2E", "#0F3460", "#E94560", "#16213E", "#533483", "#2B4865"]
    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts().head(8)
        bars = ax.barh(vc.index[::-1], vc.values[::-1],
                       color=palette[:len(vc)], edgecolor="white", linewidth=0.3)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("Count", fontsize=7)
        ax.set_facecolor("#F4F4F4")
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=7)
    fig.suptitle("Top Categorical Values", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_image(fig, width=16*cm, height=5*cm)


def _missing_values_chart(df):
    # Use plain hex strings for matplotlib — NOT ReportLab Color objects
    CLR_LOW    = "#2B4865"
    CLR_MED    = "#F39C12"
    CLR_HIGH   = "#E74C3C"

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(2, len(missing) * 0.45)))
    fig.patch.set_facecolor("#FAFAFA")
    pcts = (missing / len(df) * 100).values
    cols = [CLR_HIGH if p > 20 else CLR_MED if p > 5 else CLR_LOW for p in pcts]
    bars = ax.barh(missing.index[::-1], missing.values[::-1],
                   color=cols[::-1], edgecolor="white")
    ax.set_xlabel("Missing Count", fontsize=8)
    ax.set_title("Missing Values by Column", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.set_facecolor("#F4F4F4")
    for bar, pct in zip(bars, pcts[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=7.5)
    patches = [
        mpatches.Patch(color=CLR_LOW,  label="< 5% missing"),
        mpatches.Patch(color=CLR_MED,  label="5–20% missing"),
        mpatches.Patch(color=CLR_HIGH, label="> 20% missing"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    fig.tight_layout()
    calc_height = max(4*cm, len(missing)*0.7*cm)
    return _fig_to_image(fig, width=14*cm, height=min(calc_height, 20*cm))


# ═════════════════════════════════════════════════════════════════════════════
# Main public function
# ═════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    ai_insights: str = "",
    cleaning_fixes: list = None,
) -> bytes:
    """
    Generate an industry-grade PDF report for the given DataFrame.

    Parameters
    ----------
    df            : The (cleaned) pandas DataFrame to analyse.
    dataset_name  : Name shown on the cover page.
    ai_insights   : Optional string of AI-generated business insights.
    cleaning_fixes: Optional list of strings describing cleaning actions taken.

    Returns
    -------
    bytes: Raw PDF bytes, ready to feed into st.download_button().
    """
    buf = io.BytesIO()
    S = _styles()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.0*cm, bottomMargin=1.8*cm,
        title=dataset_name,
        author="AI-Powered Data Analyst",
        subject="Automated Data Analysis Report",
        creator="AI-Powered Data Analyst Assistant",
    )

    content_frame = Frame(
        MARGIN, 1.5*cm, PAGE_W - 2*MARGIN, PAGE_H - 3.5*cm,
        id="content"
    )
    cover_frame = Frame(
        MARGIN * 1.5, PAGE_H * 0.08,
        PAGE_W - 3*MARGIN, PAGE_H * 0.38,
        id="cover"
    )

    doc.addPageTemplates([
        PageTemplate(id="Cover",   frames=[cover_frame], onPage=_cover_canvas),
        PageTemplate(id="Report",  frames=[content_frame], onPage=_report_canvas),
    ])

    story = []
    now = datetime.now()

    # ─────────────────────────────────────────────────────────────────────────
    # COVER PAGE
    # ─────────────────────────────────────────────────────────────────────────
    story.append(NextPageTemplate("Cover"))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("DATA ANALYSIS", S["cover_sub"]))
    story.append(Paragraph("REPORT", S["cover_title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="40%", thickness=2, color=ACCENT, spaceAfter=8))
    story.append(Paragraph(dataset_name, S["cover_sub"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(f"Generated on {now.strftime('%d %B %Y at %H:%M')}", S["cover_meta"]))
    story.append(Paragraph(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns", S["cover_meta"]))
    story.append(Paragraph("Prepared by: AI-Powered Data Analyst Assistant", S["cover_meta"]))
    story.append(Paragraph("Classification: CONFIDENTIAL", S["cover_meta"]))

    story.append(NextPageTemplate("Report"))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE OF CONTENTS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Table of Contents", S["h1"]))
    _section_rule(story, S)
    toc_items = [
        ("1", "Dataset Overview"),
        ("2", "Statistical Summary"),
        ("3", "Data Quality Analysis"),
        ("4", "Correlation Analysis"),
        ("5", "Distribution Analysis"),
        ("6", "Categorical Analysis"),
        ("7", "AI-Generated Business Insights"),
        ("8", "Appendix: Full Column Reference"),
    ]
    if cleaning_fixes:
        toc_items.insert(2, ("—", "Data Cleaning Log"))
    for num, title in toc_items:
        story.append(Paragraph(f"<b>{num}.</b> &nbsp;&nbsp; {title}", S["toc_entry"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — DATASET OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Dataset Overview", S["h1"]))
    _section_rule(story, S)

    num_missing_cells = int(df.isnull().sum().sum())
    num_dupes = int(df.duplicated().sum())
    completeness = round((1 - num_missing_cells / max(df.size, 1)) * 100, 2)
    num_numeric = len(df.select_dtypes(include="number").columns)
    num_cat = len(df.select_dtypes(include=["object", "category"]).columns)
    num_date = len(df.select_dtypes(include=["datetime"]).columns)
    # ── Auto-detect datetime string columns (fix issue #6: date treated as text)
    dt_parse_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(30), format="mixed", errors="raise")
            dt_parse_cols.append(col)
        except Exception:
            pass
    num_date = len(dt_parse_cols)  # Override with the true count

    # ── Overview stats (all from the passed df — post-cleaning) ─────────────
    num_missing_cells = int(df.isnull().sum().sum())
    num_dupes = int(df.duplicated().sum())
    completeness = round((1 - num_missing_cells / max(df.size, 1)) * 100, 2)
    num_numeric = len(df.select_dtypes(include="number").columns)
    num_cat = len(df.select_dtypes(include=["object", "category"]).columns)

    overview_data = [
        ["Metric", "Value", "Notes"],
        ["Total Rows",        f"{df.shape[0]:,}",         "Current dataset"],
        ["Total Columns",     f"{df.shape[1]}",            ""],
        ["Numeric Columns",   f"{num_numeric}",            ""],
        ["Categorical Cols",  f"{num_cat}",                ""],
        ["DateTime Cols",     f"{num_date}",               f"Includes string-encoded dates"],
        ["Missing Cells",     f"{num_missing_cells:,}",    f"{100-completeness:.2f}% of data"],
        ["Duplicate Rows",    f"{num_dupes:,}",            "Based on current dataset"],
        ["Data Completeness", f"{completeness:.2f}%",      ""],
        ["Memory Usage",      f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB", ""],
    ]
    story.append(_table(overview_data, col_widths=[6*cm, 5*cm, 6*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── Nuanced completeness callout (fix issue #3: never say "production-ready" if there are still issues)
    remaining_flags = [c for c in df.columns if c.endswith(("_Is_Missing", "_Is_Invalid"))]
    has_remaining_issues = num_missing_cells > 0 or len(remaining_flags) > 0

    if completeness >= 99 and not has_remaining_issues:
        msg = (f"✅ Excellent data completeness at {completeness:.2f}%. "
               "No missing values or flagged issues remain. Dataset is ready for production use.")
    elif completeness >= 95:
        msg = (f"⚠️ Good completeness at {completeness:.2f}%, but {num_missing_cells:,} missing cells remain. "
               "Dataset quality has significantly improved but review flagged columns before production deployment.")
    elif completeness >= 80:
        msg = (f"⚠️ Moderate completeness at {completeness:.2f}%. {num_missing_cells:,} missing cells remain. "
               "Additional data collection or imputation is recommended before production use.")
    else:
        msg = (f"❌ Low completeness at {completeness:.2f}%. {num_missing_cells:,} missing cells detected. "
               "Significant data quality work is required before this dataset can be used in production.")
    story.append(Paragraph(msg, S["callout"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1B — BUSINESS KPIs (fix issue #8: add KPIs)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("1.1 Business KPIs", S["h2"]))
    story.append(Paragraph(
        "Key Performance Indicators automatically detected from numeric column names. "
        "All values are computed directly from the current dataset.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))

    kpi_keywords = {
        "Revenue / Sales": ["revenue", "sales", "income", "earnings"],
        "Cost / Price":    ["cost", "expense", "spend", "price", "unit_price"],
        "Volume / Orders": ["quantity", "qty", "units", "count", "orders"],
        "Profit / Margin": ["profit", "margin", "gain"],
    }
    kpi_rows = [["KPI Category", "Column", "Total", "Mean (Avg Order Value)", "Min", "Max"]]
    found_kpi = False
    num_df_local = df.select_dtypes(include="number")
    for kpi_label, keywords in kpi_keywords.items():
        for col in num_df_local.columns:
            if any(kw in col.lower() for kw in keywords):
                col_data = df[col].dropna()
                kpi_rows.append([
                    kpi_label, col,
                    f"{col_data.sum():,.2f}",
                    f"{col_data.mean():,.2f}",
                    f"{col_data.min():,.2f}",
                    f"{col_data.max():,.2f}",
                ])
                found_kpi = True

    # Also compute row count as an order count KPI
    kpi_rows.insert(1, ["Volume / Orders", "Total Records", f"{len(df):,}", "N/A", "N/A", "N/A"])
    found_kpi = True

    if found_kpi:
        story.append(_table(kpi_rows, col_widths=[3.5*cm, 3.5*cm, 2.5*cm, 4*cm, 1.8*cm, 1.8*cm]))
    else:
        story.append(Paragraph("No standard KPI columns detected by name. Please rename revenue/cost/quantity columns for auto-detection.", S["body"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — STATISTICAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Statistical Summary", S["h1"]))
    _section_rule(story, S)

    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        story.append(Paragraph("2.1 Numeric Columns — Descriptive Statistics", S["h2"]))
        desc = num_df.describe().T.reset_index()
        desc.columns = ["Column", "Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]
        for c in ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]:
            desc[c] = desc[c].apply(lambda x: f"{x:,.3f}" if pd.notna(x) else "N/A")

        stat_data = [desc.columns.tolist()] + desc.values.tolist()
        col_w = [3.8*cm] + [1.7*cm]*8
        story.append(_table(stat_data, col_widths=col_w))
        story.append(Spacer(1, 0.4*cm))

    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        story.append(Paragraph("2.2 Categorical Columns — Summary", S["h2"]))
        cat_rows = [["Column", "Total Rows", "Unique Values", "Top Value", "Top Count", "Missing"]]
        for col in cat_df.columns:
            vc = cat_df[col].value_counts()
            cat_rows.append([
                col,
                f"{len(df):,}",
                f"{df[col].nunique():,}",
                str(vc.index[0]) if len(vc) > 0 else "N/A",
                f"{vc.iloc[0]:,}" if len(vc) > 0 else "0",
                f"{df[col].isnull().sum():,}",
            ])
        story.append(_table(cat_rows, col_widths=[3.5*cm, 2.2*cm, 2.5*cm, 3.5*cm, 2.3*cm, 2.5*cm]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANING LOG (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if cleaning_fixes:
        story.append(Paragraph("Data Cleaning Log", S["h1"]))
        _section_rule(story, S)
        story.append(Paragraph(
            f"The following {len(cleaning_fixes)} automated cleaning operations were applied to this dataset "
            "to produce the version analysed in this report:",
            S["body"]
        ))
        story.append(Spacer(1, 0.2*cm))
        cl_data = [["#", "Cleaning Action Applied"]]
        for i, fix in enumerate(cleaning_fixes, 1):
            cl_data.append([str(i), fix])
        story.append(_table(cl_data, col_widths=[1.2*cm, 15.5*cm]))
        story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — DATA QUALITY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Data Quality Analysis", S["h1"]))
    _section_rule(story, S)

    story.append(Paragraph("3.1 Missing Values", S["h2"]))
    missing_chart = _missing_values_chart(df)
    if missing_chart:
        story.append(missing_chart)
        story.append(Spacer(1, 0.3*cm))
    else:
        story.append(Paragraph("✅ No missing values detected in this dataset.", S["callout"]))

    # Fix issue #7: Explain clearly what happened with outliers
    story.append(Paragraph("3.2 Outlier Detection (IQR Method)", S["h2"]))
    story.append(Paragraph(
        "The IQR (Interquartile Range) method flags values outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]. "
        "Note: The strict cleaning pipeline flags outliers with boolean columns rather than removing them. "
        "If 0 outliers are shown, either none were present or they were addressed in a prior cleaning step.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    
    # Only show analytical columns, exclude IDs and flag columns
    ID_KEYWORDS_OUT = ["id", "phone", "mobile", "zip", "postal", "code", "index", "row", "key"]
    outlier_cols = [
        c for c in num_df.columns
        if not any(kw in c.lower() for kw in ID_KEYWORDS_OUT)
        and not c.endswith(("_Is_Missing", "_Is_Outlier", "_Is_Invalid"))
    ]
    outlier_rows = [["Column", "Q1", "Q3", "IQR", "Lower Fence", "Upper Fence", "Outlier Count", "Outlier %"]]
    for col in outlier_cols:
        q1 = num_df[col].quantile(0.25)
        q3 = num_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())
        pct = round(n_out / max(len(df), 1) * 100, 2)
        outlier_rows.append([
            col,
            f"{q1:,.3f}", f"{q3:,.3f}", f"{iqr:,.3f}",
            f"{lower:,.3f}", f"{upper:,.3f}",
            f"{n_out:,}", f"{pct:.2f}%"
        ])
    if len(outlier_rows) > 1:
        story.append(_table(
            outlier_rows,
            col_widths=[3.5*cm, 1.7*cm, 1.7*cm, 1.7*cm, 2*cm, 2*cm, 2*cm, 1.8*cm]
        ))
    else:
        story.append(Paragraph("✅ No analytical numeric columns available for outlier analysis.", S["body"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — CORRELATION ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Correlation Analysis", S["h1"]))
    _section_rule(story, S)
    story.append(Paragraph(
        "Pearson correlation coefficients measure the linear relationship between "
        "numeric variables. Identifier columns (IDs, phone numbers, zip codes) and "
        "cleaner-generated flag columns have been excluded to show only business-meaningful relationships.",
        S["body"]
    ))
    story.append(Spacer(1, 0.3*cm))

    # Filter out non-analytical columns (fix issues #5 and #6)
    ID_KEYWORDS = ["id", "phone", "mobile", "zip", "postal", "code", "index", "row", "key"]
    analytical_num_cols = [
        c for c in num_df.columns
        if not any(kw in c.lower() for kw in ID_KEYWORDS)
        and not c.endswith(("_Is_Missing", "_Is_Outlier", "_Is_Invalid"))
    ]
    analytical_num_df = df[analytical_num_cols] if analytical_num_cols else num_df

    heatmap = _correlation_heatmap(df[analytical_num_cols] if analytical_num_cols else df)
    if heatmap:
        story.append(heatmap)
        story.append(Spacer(1, 0.3*cm))

        # Top pairs — filtered, no trivially derived columns
        corr = analytical_num_df.corr(numeric_only=True).abs()
        pairs = []
        cols_list = analytical_num_df.columns.tolist()
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                c1, c2 = cols_list[i], cols_list[j]
                r = corr.loc[c1, c2] if c1 in corr and c2 in corr.index else float("nan")
                if pd.isna(r):
                    continue
                # Skip trivially derived pairs (e.g. Total vs Total_Revenue)
                if c1.lower() in c2.lower() or c2.lower() in c1.lower():
                    continue
                pairs.append((r, c1, c2))
        pairs.sort(reverse=True)
        if pairs:
            story.append(Paragraph("Top Business-Relevant Correlated Pairs:", S["h3"]))
            pair_data = [["Column A", "Column B", "Correlation", "Strength", "Implication"]]
            for r, a, b in pairs[:8]:
                strength = "Very Strong" if r > 0.8 else "Strong" if r > 0.6 else "Moderate" if r > 0.4 else "Weak"
                impl = "Strong predictor relationship" if r > 0.7 else "Notable association" if r > 0.4 else "Weak signal"
                pair_data.append([a, b, f"{r:.4f}", strength, impl])
            story.append(_table(pair_data, col_widths=[3.5*cm, 3.5*cm, 2.5*cm, 2.5*cm, 5*cm]))
        else:
            story.append(Paragraph("No meaningful business-relevant correlations found after excluding identifier columns.", S["body"]))
    else:
        story.append(Paragraph("⚠️ Not enough analytical numeric columns for correlation analysis.", S["callout"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — DISTRIBUTION ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Distribution Analysis", S["h1"]))
    _section_rule(story, S)
    story.append(Paragraph(
        "Histograms reveal the underlying distribution shape (normal, skewed, bimodal, etc.). "
        "The red dashed line marks the mean value for each column.",
        S["body"]
    ))
    story.append(Spacer(1, 0.3*cm))
    dist_chart = _distribution_charts(df)
    if dist_chart:
        story.append(dist_chart)
    else:
        story.append(Paragraph("No numeric columns available for distribution analysis.", S["body"]))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("5.1 Skewness & Kurtosis", S["h2"]))
    story.append(Paragraph(
        "Skewness measures the asymmetry of the distribution (positive = right tail). "
        "Kurtosis measures the tail heaviness (>3 = heavier tails than normal).",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    sk_data = [["Column", "Skewness", "Kurtosis", "Assessment"]]
    for col in num_df.columns:
        sk = float(num_df[col].skew())
        kt = float(num_df[col].kurtosis())
        if abs(sk) < 0.5:
            assessment = "Approximately Normal"
        elif abs(sk) < 1.0:
            assessment = "Moderately Skewed"
        else:
            assessment = "Highly Skewed"
        sk_data.append([col, f"{sk:.4f}", f"{kt:.4f}", assessment])
    story.append(_table(sk_data, col_widths=[5*cm, 3*cm, 3*cm, 6*cm]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — CATEGORICAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("6. Categorical Analysis", S["h1"]))
    _section_rule(story, S)
    cat_chart = _top_categories_chart(df)
    if cat_chart:
        story.append(cat_chart)
        story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("6.1 Cardinality Summary", S["h2"]))
    card_data = [["Column", "Unique Values", "Most Common", "Count", "Entropy (bits)"]]
    for col in cat_df.columns:
        vc = df[col].value_counts()
        n = len(df[col].dropna())
        probs = vc / n
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        card_data.append([
            col,
            f"{df[col].nunique():,}",
            str(vc.index[0]) if len(vc) > 0 else "N/A",
            f"{vc.iloc[0]:,}" if len(vc) > 0 else "0",
            f"{entropy:.3f}"
        ])
    if len(card_data) > 1:
        story.append(_table(card_data, col_widths=[4*cm, 3*cm, 4*cm, 2.5*cm, 3*cm]))
    else:
        story.append(Paragraph("No categorical columns found.", S["body"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — AI BUSINESS INSIGHTS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("7. AI-Generated Business Insights", S["h1"]))
    _section_rule(story, S)

    if ai_insights and ai_insights.strip():
        # Split by markdown sections and render
        for line in ai_insights.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.15*cm))
            elif line.startswith("## ") or line.startswith("# "):
                heading = line.lstrip("#").strip()
                story.append(Paragraph(heading, S["h2"]))
            elif line.startswith("- ") or line.startswith("* "):
                story.append(Paragraph(f"• {line[2:]}", S["body"]))
            else:
                story.append(Paragraph(line, S["body"]))
    else:
        story.append(Paragraph(
            "No AI insights available. Please click 'Generate Insights Report' on the AI Insights page first.",
            S["callout"]
        ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 8 — APPENDIX: FULL COLUMN REFERENCE
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("8. Appendix: Full Column Reference", S["h1"]))
    _section_rule(story, S)
    story.append(Paragraph(
        "Complete metadata for every column in the dataset.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))

    app_data = [["#", "Column Name", "Data Type", "Non-Null", "Missing", "Missing %", "Unique Values"]]
    for i, col in enumerate(df.columns, 1):
        n_null = int(df[col].isnull().sum())
        n_non_null = len(df) - n_null
        pct = round(n_null / max(len(df), 1) * 100, 2)
        app_data.append([
            str(i), col,
            str(df[col].dtype),
            f"{n_non_null:,}",
            f"{n_null:,}",
            f"{pct:.2f}%",
            f"{df[col].nunique():,}"
        ])
    story.append(_table(
        app_data,
        col_widths=[0.9*cm, 4.5*cm, 2.5*cm, 2*cm, 1.8*cm, 2*cm, 2.5*cm]
    ))

    # Build PDF
    doc.build(story)
    return buf.getvalue()
