"""
report_generator1.py
Professional clean layout — no info boxes, pure typography hierarchy.
FIX 3: EDA charts now read from reports/plots_white (white background)
        instead of copying/re-rendering the dark UI versions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# ── Palette ────────────────────────────────────────────────────────────────────
INDIGO    = colors.HexColor("#4f46e5")
INDIGO_LT = colors.HexColor("#eef2ff")
CYAN      = colors.HexColor("#0ea5e9")
GREEN     = colors.HexColor("#10b981")
GREEN_LT  = colors.HexColor("#f0fdf4")
AMBER     = colors.HexColor("#f59e0b")
AMBER_LT  = colors.HexColor("#fffbeb")
PURPLE    = colors.HexColor("#a855f7")
ROSE      = colors.HexColor("#f43f5e")
SLATE_900 = colors.HexColor("#0f172a")
SLATE_800 = colors.HexColor("#1e293b")
SLATE_700 = colors.HexColor("#334155")
SLATE_600 = colors.HexColor("#475569")
SLATE_500 = colors.HexColor("#64748b")
SLATE_400 = colors.HexColor("#94a3b8")
SLATE_200 = colors.HexColor("#e2e8f0")
SLATE_100 = colors.HexColor("#f1f5f9")
SLATE_50  = colors.HexColor("#f8fafc")
WHITE     = colors.white
BLACK     = colors.HexColor("#0f172a")

BADGE = {
    "Excellent": (colors.HexColor("#d1fae5"), colors.HexColor("#065f46")),
    "Good":      (colors.HexColor("#cffafe"), colors.HexColor("#0e7490")),
    "Fair":      (colors.HexColor("#fed7aa"), colors.HexColor("#92400e")),
    "Poor":      (colors.HexColor("#fecdd3"), colors.HexColor("#881337")),
}

PLOTS_DIR       = "reports/plots_pdf"       # diagnostic/summary charts (generated here)
EDA_PLOTS_DIR   = "reports/plots_white"     # FIX 3: white-background EDA charts from eda_visualization.py
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ── Styles ─────────────────────────────────────────────────────────────────────
def _styles():
    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontName="Helvetica-Bold", fontSize=30,
            textColor=BLACK, leading=36,
            alignment=TA_CENTER, spaceAfter=6),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontName="Helvetica", fontSize=11,
            textColor=SLATE_500, leading=16,
            alignment=TA_CENTER, spaceAfter=4),
        "cover_tag": ParagraphStyle(
            "cover_tag", fontName="Helvetica", fontSize=8.5,
            textColor=INDIGO, leading=12,
            alignment=TA_CENTER, spaceAfter=0),
        "h1": ParagraphStyle(
            "h1", fontName="Helvetica-Bold", fontSize=13,
            textColor=INDIGO, leading=18,
            spaceBefore=18, spaceAfter=4),
        "h2": ParagraphStyle(
            "h2", fontName="Helvetica-Bold", fontSize=10.5,
            textColor=SLATE_700, leading=15,
            spaceBefore=10, spaceAfter=3),
        "notes_label": ParagraphStyle(
            "notes_label", fontName="Helvetica-Bold", fontSize=8.5,
            textColor=SLATE_500, leading=12,
            spaceBefore=8, spaceAfter=3,
            leftIndent=0),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=9.5,
            textColor=SLATE_700, leading=15,
            spaceAfter=4, alignment=TA_JUSTIFY),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=9,
            textColor=SLATE_600, leading=14,
            leftIndent=14, firstLineIndent=0,
            spaceAfter=3, spaceBefore=1),
        "bullet_key": ParagraphStyle(
            "bullet_key", fontName="Helvetica-Bold", fontSize=9,
            textColor=SLATE_700, leading=14,
            leftIndent=14, firstLineIndent=0,
            spaceAfter=3, spaceBefore=1),
        "sub_bullet": ParagraphStyle(
            "sub_bullet", fontName="Helvetica", fontSize=8.5,
            textColor=SLATE_500, leading=13,
            leftIndent=26, firstLineIndent=0,
            spaceAfter=2, spaceBefore=0),
        "bullet_warn": ParagraphStyle(
            "bullet_warn", fontName="Helvetica", fontSize=9,
            textColor=AMBER, leading=14,
            leftIndent=14, firstLineIndent=0,
            spaceAfter=3, spaceBefore=1),
        "bullet_good": ParagraphStyle(
            "bullet_good", fontName="Helvetica", fontSize=9,
            textColor=GREEN, leading=14,
            leftIndent=14, firstLineIndent=0,
            spaceAfter=3, spaceBefore=1),
        "caption": ParagraphStyle(
            "caption", fontName="Helvetica-Oblique", fontSize=8,
            textColor=SLATE_500, leading=11,
            alignment=TA_CENTER, spaceAfter=4),
        "footer": ParagraphStyle(
            "footer", fontName="Helvetica-Oblique", fontSize=7.5,
            textColor=SLATE_400, leading=10,
            alignment=TA_CENTER),
    }


# ── Dividers ───────────────────────────────────────────────────────────────────
def _rule(color=SLATE_200, thickness=0.5, space_before=2, space_after=8):
    return HRFlowable(
        width="100%", thickness=thickness, color=color,
        spaceAfter=space_after, spaceBefore=space_before)


def _accent_rule():
    return HRFlowable(
        width="100%", thickness=1.5, color=INDIGO,
        spaceAfter=8, spaceBefore=0)


def _gap(h=6):
    return Spacer(1, h)


# ── Notes block ────────────────────────────────────────────────────────────────
def _notes(label, points, ST, style="bullet"):
    elems = [Paragraph(label.upper(), ST["notes_label"])]
    for pt in points:
        elems.append(Paragraph(f"•  {pt}", ST[style]))
    elems.append(_gap(4))
    return elems


# ── Page template ──────────────────────────────────────────────────────────────
def _on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(WHITE)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(INDIGO)
    canvas.rect(0, h - 3, w, 3, fill=1, stroke=0)
    canvas.setStrokeColor(SLATE_200)
    canvas.setLineWidth(0.4)
    canvas.line(20*mm, 19*mm, w - 20*mm, 19*mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(SLATE_400)
    canvas.drawString(20*mm, 12*mm,
                      "Explainable AutoEDA Platform  ·  Auto-generated Report")
    canvas.drawRightString(w - 20*mm, 12*mm,
                           f"Page {doc.page}  ·  {datetime.now().strftime('%d %b %Y')}")
    canvas.restoreState()


# ── Score block ────────────────────────────────────────────────────────────────
def _score_block(score, label):
    bg, fg = BADGE.get(label, BADGE["Fair"])
    score_p = Paragraph(
        f"<font size=40 color='#4f46e5'><b>{score}</b></font>"
        f"<font size=13 color='#94a3b8'> / 100</font>",
        ParagraphStyle("sp", fontName="Helvetica-Bold", fontSize=40,
                       textColor=INDIGO, alignment=TA_CENTER, leading=48))
    label_p = Paragraph(
        "DATASET QUALITY SCORE",
        ParagraphStyle("lp", fontName="Helvetica", fontSize=7.5,
                       textColor=SLATE_400, alignment=TA_CENTER, leading=11))
    badge_p = Paragraph(
        f"<b>{label}</b>",
        ParagraphStyle("bp", fontName="Helvetica-Bold", fontSize=11,
                       textColor=fg, alignment=TA_CENTER, leading=16))
    tbl = Table([[score_p], [label_p], [badge_p]], colWidths=[100*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 1), SLATE_50),
        ("BACKGROUND",    (0, 2), (-1, 2), bg),
        ("BOX",           (0, 0), (-1, -1), 0.5, SLATE_200),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    return tbl


# ── Generic styled table ───────────────────────────────────────────────────────
def _table(rows, col_widths, header_color=INDIGO):
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8.5),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR",      (0, 1), (-1, -1), SLATE_700),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_50]),
        ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN",          (0, 0), (0,  -1), "LEFT"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("BOX",            (0, 0), (-1, -1), 0.4, SLATE_200),
        ("INNERGRID",      (0, 0), (-1, -1), 0.25, SLATE_200),
    ]))
    return tbl


# ── Chart helpers (for diagnostic/summary charts generated inside this file) ───
LIGHT = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#cbd5e1",
    "axes.labelcolor":  "#334155",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "grid.color":       "#e2e8f0",
    "text.color":       "#0f172a",
}


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _img(path, w=162, h=68):
    return Image(path, width=w*mm, height=h*mm, kind="proportional")


def _missing_heatmap(df):
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if not missing_cols:
        return None
    with plt.rc_context(LIGHT):
        sample = df[missing_cols].isnull().astype(int)
        if len(sample) > 200:
            sample = sample.sample(200, random_state=42)
        fig, ax = plt.subplots(
            figsize=(10, max(2.5, len(missing_cols) * 0.42)))
        sns.heatmap(sample.T, cmap=["#f1f5f9", "#4f46e5"],
                    cbar=False, linewidths=0,
                    ax=ax, yticklabels=True)
        ax.set_title("Missing Value Heatmap  (purple = missing)",
                     fontsize=10, pad=8)
        ax.set_xlabel("Row Index (sample)", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=6)
        plt.tight_layout()
    return _save(fig, "missing_heatmap.png")


def _outlier_chart(df):
    nc = df.select_dtypes(include=["int64", "float64"]).columns
    if not len(nc):
        return None
    counts = []
    for col in nc:
        ser = df[col].dropna()
        q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
        iqr = q3 - q1
        counts.append(
            int(((ser < q1 - 1.5*iqr) | (ser > q3 + 1.5*iqr)).sum()))
    with plt.rc_context(LIGHT):
        fig, ax = plt.subplots(
            figsize=(9, max(2.5, len(nc) * 0.38)))
        bars = ax.barh(
            list(nc), counts,
            color=["#4f46e5" if v > 0 else "#e2e8f0" for v in counts])
        ax.set_xlabel("Outlier Count (IQR Method)", fontsize=8)
        ax.set_title("Outliers Per Column", fontsize=10, pad=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_width() + 0.15,
                        bar.get_y() + bar.get_height() / 2,
                        str(val), va="center",
                        fontsize=7.5, color="#334155")
        plt.tight_layout()
    return _save(fig, "outlier_chart.png")


def _correlation_heatmap(df):
    nc = df.select_dtypes(include=["int64", "float64"]).columns
    if len(nc) < 2:
        return None
    corr = df[nc].corr()
    with plt.rc_context(LIGHT):
        sz = max(5, len(nc) * 0.92)
        fig, ax = plt.subplots(figsize=(sz, sz * 0.82))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="#e2e8f0",
                    annot_kws={"size": 7}, ax=ax,
                    cbar_kws={"shrink": 0.65})
        ax.set_title("Correlation Matrix  (lower triangle)",
                     fontsize=10, pad=8)
        ax.tick_params(axis="both", labelsize=7)
        plt.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def _before_after_chart(original_df, cleaned_df):
    metrics = ["Missing Values", "Duplicate Rows"]
    before  = [int(original_df.isnull().sum().sum()),
               int(original_df.duplicated().sum())]
    after   = [int(cleaned_df.isnull().sum().sum()),
               int(cleaned_df.duplicated().sum())]
    x = np.arange(len(metrics))
    with plt.rc_context(LIGHT):
        fig, ax = plt.subplots(figsize=(6, 3))
        w  = 0.3
        bb = ax.bar(x - w/2, before, w, label="Before", color="#94a3b8")
        ba = ax.bar(x + w/2, after,  w, label="After",  color="#4f46e5")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title("Before vs After Cleaning", fontsize=10, pad=8)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for bar in list(bb) + list(ba):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.2, str(int(h)),
                    ha="center", va="bottom", fontsize=7.5)
        plt.tight_layout()
    return _save(fig, "before_after.png")


def _scaling_chart(original_df, cleaned_df):
    nc_o = list(original_df.select_dtypes(include=["int64", "float64"]).columns)
    nc_c = list(cleaned_df.select_dtypes(include=["int64", "float64"]).columns)
    common = [c for c in nc_o if c in nc_c][:4]
    if not common:
        return None
    with plt.rc_context(LIGHT):
        fig, axes = plt.subplots(1, len(common), figsize=(4 * len(common), 3))
        if len(common) == 1:
            axes = [axes]
        for ax, col in zip(axes, common):
            ax.hist(original_df[col].dropna(), bins=20,
                    alpha=0.45, color="#94a3b8",
                    label="Before", density=True)
            ax.hist(cleaned_df[col].dropna(), bins=20,
                    alpha=0.6, color="#4f46e5",
                    label="After", density=True)
            ax.set_title(col, fontsize=8, pad=5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)
        fig.suptitle("Distribution Before vs After Scaling",
                     fontsize=9, y=1.01)
        plt.tight_layout()
    return _save(fig, "scaling_comparison.png")


# ── Data tables ────────────────────────────────────────────────────────────────
def _stats_table(df):
    nc = df.select_dtypes(include=["int64", "float64"]).columns
    if not len(nc):
        return None
    rows = [["Column", "Mean", "Median", "Std Dev", "Min", "Max", "Outliers"]]
    for col in nc:
        ser = df[col].dropna()
        q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
        iqr = q3 - q1
        out = int(((ser < q1 - 1.5*iqr) | (ser > q3 + 1.5*iqr)).sum())
        rows.append([
            col,
            f"{ser.mean():.3f}",   f"{ser.median():.3f}",
            f"{ser.std():.3f}",    f"{ser.min():.3f}",
            f"{ser.max():.3f}",    str(out),
        ])
    return _table(rows, [40*mm, 21*mm, 21*mm, 21*mm, 19*mm, 19*mm, 18*mm], INDIGO)


def _before_after_table(original_df, cleaned_df):
    def d(b, a): return f"{a - b:+d}" if a != b else "—"
    om = int(original_df.isnull().sum().sum())
    cm = int(cleaned_df.isnull().sum().sum())
    od = int(original_df.duplicated().sum())
    cd = int(cleaned_df.duplicated().sum())
    rows = [
        ["Metric",         "Before",             "After",             "Change"],
        ["Rows",           f"{len(original_df):,}", f"{len(cleaned_df):,}", d(len(original_df), len(cleaned_df))],
        ["Columns",        str(original_df.shape[1]), str(cleaned_df.shape[1]), d(original_df.shape[1], cleaned_df.shape[1])],
        ["Missing Values", f"{om:,}",            f"{cm:,}",           d(om, cm)],
        ["Duplicate Rows", f"{od:,}",            f"{cd:,}",           d(od, cd)],
        ["Missing %",
         f"{original_df.isnull().mean().mean()*100:.2f}%",
         f"{cleaned_df.isnull().mean().mean()*100:.2f}%", ""],
    ]
    return _table(rows, [52*mm, 32*mm, 32*mm, 32*mm], INDIGO)


def _encoding_table(original_df, cleaned_df):
    orig_cat  = set(original_df.select_dtypes(include=["object", "category"]).columns)
    clean_cat = set(cleaned_df.select_dtypes(include=["object", "category"]).columns)
    encoded   = orig_cat - clean_cat
    new_cols  = [c for c in cleaned_df.columns if c not in original_df.columns]
    rows = [["Column", "Action", "Result"]]
    for col in encoded:
        rows.append([col, "Label Encoding", "Integer codes"])
    for col in new_cols[:8]:
        rows.append([col, "One-Hot Encoding", "Binary column"])
    if len(new_cols) > 8:
        rows.append([f"… +{len(new_cols)-8} more", "One-Hot", "Binary columns"])
    if not encoded and not new_cols:
        rows.append(["—", "None", "No encoding performed"])
    return _table(rows, [65*mm, 50*mm, 48*mm], PURPLE)


# ── EDA chart type from filename ───────────────────────────────────────────────
_EDA_TYPES = {
    "Histogram": [
        "Groups values into bins — reveals distribution shape.",
        "KDE curve overlaid to smooth bin-level noise.",
        "Long right tail = positive skew; watch for data entry errors.",
    ],
    "Line Plot": [
        "Plots values in row order — best for sequential/time data.",
        "Spikes or flat regions can signal anomalies or batch effects.",
    ],
    "Box Plot": [
        "Shows Min, Q1, Median, Q3, Max in one view.",
        "Points beyond the whiskers are IQR-flagged outliers.",
        "Box shifted toward Q1 indicates right-skewed data.",
    ],
    "KDE Plot": [
        "Smoothed probability density — no bin-size artefacts.",
        "Multiple peaks suggest distinct sub-groups exist.",
    ],
    "Violin Plot": [
        "Box plot + KDE combined — shows full density shape.",
        "Wide sections = dense value clusters.",
    ],
}


def _chart_type_from_filename(fname):
    name = fname.replace(".png", "")
    for ct in ["Histogram", "Line_Plot", "Box_Plot", "KDE_Plot", "Violin_Plot"]:
        if name.endswith(ct):
            return ct.replace("_", " ")
    return None


# ── Recommendations ────────────────────────────────────────────────────────────
def _build_recs(original_df, cleaned_df, score, label, ST):
    nc  = cleaned_df.select_dtypes(include=["int64", "float64"]).columns
    cat = cleaned_df.select_dtypes(include=["object", "category"]).columns
    items = []

    if cleaned_df.isnull().sum().sum() > 0:
        items.append((
            f"⚠  {int(cleaned_df.isnull().sum().sum()):,} missing "
            "value(s) remain — apply imputation before modelling.",
            "bullet_warn"))

    outlier_cols = []
    for c in nc:
        ser = cleaned_df[c].dropna()
        q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0 and int(((ser < q1-1.5*iqr) | (ser > q3+1.5*iqr)).sum()) > 0:
            outlier_cols.append(c)
    if outlier_cols:
        items.append((
            f"⚠  Outliers remain in: {', '.join(outlier_cols[:4])}. "
            "Consider capping or log-transform.",
            "bullet_warn"))

    if len(cat) > 0:
        items.append((
            f"⚠  {len(cat)} categorical column(s) unencoded: "
            f"{', '.join(list(cat)[:4])}. Encode before modelling.",
            "bullet_warn"))

    skewed = [c for c in nc if abs(cleaned_df[c].dropna().skew()) > 1]
    if skewed:
        items.append((
            f"ℹ  Highly skewed: {', '.join(skewed[:4])}. "
            "Consider log or Box-Cox transformation.",
            "bullet"))

    if len(nc) >= 2:
        corr = cleaned_df[nc].corr()
        high = [f"{nc[i]} & {nc[j]}"
                for i in range(len(nc))
                for j in range(i+1, len(nc))
                if abs(corr.iloc[i, j]) >= 0.8]
        if high:
            items.append((
                f"ℹ  Strongly correlated pairs: {'; '.join(high[:3])}. "
                "Consider removing one to reduce multicollinearity.",
                "bullet"))

    if label in ("Excellent", "Good"):
        items.append((
            f"✔  Quality score {score}/100 ({label}) — "
            "dataset is in good shape for analysis.",
            "bullet_good"))
    else:
        items.append((
            f"⚠  Quality score {score}/100 ({label}) — "
            "review all cleaning steps before analysis.",
            "bullet_warn"))

    items.append((
        "✔  Next steps: feature engineering → train/test split → "
        "model selection → evaluation.",
        "bullet_good"))
    items.append((
        "✔  Save the cleaned CSV and document all preprocessing "
        "decisions for reproducibility.",
        "bullet_good"))

    return [Paragraph(text, ST[style]) for text, style in items]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(score, label, explanations, cleaning_summary,
                        original_df=None, cleaned_df=None):

    for f in os.listdir(PLOTS_DIR):
        os.remove(os.path.join(PLOTS_DIR, f))

    pdf_path = "reports/Explainable_AutoEDA_Report.pdf"
    ST = _styles()

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=22*mm, rightMargin=22*mm,
        topMargin=20*mm, bottomMargin=26*mm,
    )
    story = []

    # ══ COVER ═════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 16*mm))
    story.append(Paragraph("EXPLAINABLE AUTOEDA", ST["cover_title"]))
    story.append(Paragraph("Comprehensive Data Intelligence Report", ST["cover_sub"]))
    story.append(Paragraph(
        datetime.now().strftime("Generated on %d %B %Y at %H:%M"), ST["cover_sub"]))
    story.append(Paragraph(
        "Data Cleaning  ·  EDA  ·  Quality Scoring  ·  Preprocessing", ST["cover_tag"]))
    story.append(Spacer(1, 10*mm))
    story.append(_score_block(score, label))

    if original_df is not None and cleaned_df is not None:
        story.append(Spacer(1, 8*mm))
        ov = [
            ["Metric",        "Original",                   "Cleaned"],
            ["Rows",          f"{len(original_df):,}",      f"{len(cleaned_df):,}"],
            ["Columns",       str(original_df.shape[1]),    str(cleaned_df.shape[1])],
            ["Missing Cells", f"{int(original_df.isnull().sum().sum()):,}",
                              f"{int(cleaned_df.isnull().sum().sum()):,}"],
            ["Duplicate Rows", f"{int(original_df.duplicated().sum()):,}",
                               f"{int(cleaned_df.duplicated().sum()):,}"],
        ]
        story.append(_table(ov, [58*mm, 42*mm, 42*mm], SLATE_700))

    story.append(PageBreak())

    # ══ 01  QUALITY SCORE ═════════════════════════════════════════════════════
    story.append(Paragraph("01  Quality Score Breakdown", ST["h1"]))
    story.append(_accent_rule())

    for exp in explanations:
        story.append(Paragraph(f"•  {exp}", ST["bullet"]))

    story.append(_gap(6))
    story.extend(_notes(
        "Score methodology",
        [
            "Formula: 100 − penalties across 6 dimensions.",
            "Missing values: up to 30-point penalty based on average missing rate.",
            "Duplicate rows: up to 20-point penalty based on duplicate row percentage.",
            "IQR Outliers: up to 20-point penalty for numeric columns with outliers.",
            "Unencoded categoricals: up to 15-point penalty for text/category columns.",
            "Type inconsistency: up to 10-point penalty for mixed-type columns.",
            "Empty columns: up to 5-point penalty for entirely empty columns.",
            "Rating scale:  ≥90 Excellent  |  ≥75 Good  |  ≥50 Fair  |  <50 Poor.",
        ],
        ST))

    # ══ 02  MISSING VALUES ════════════════════════════════════════════════════
    story.append(Paragraph("02  Missing Value Analysis", ST["h1"]))
    story.append(_accent_rule())

    if original_df is not None:
        hp = _missing_heatmap(original_df)
        if hp:
            story.append(_img(hp, 162, 60))
            story.append(Paragraph(
                "Purple cells indicate missing values. Each row represents one dataset column.",
                ST["caption"]))
        else:
            story.append(Paragraph("✓  No missing values detected in the dataset.", ST["body"]))

        mc         = original_df.isnull().sum()
        miss_cols  = [c for c in original_df.columns if mc[c] > 0]
        total_miss = int(original_df.isnull().sum().sum())
        miss_pct   = original_df.isnull().mean().mean() * 100
        worst = (f"{mc.idxmax()} ({mc.max()/len(original_df)*100:.1f}% missing)"
                 if miss_cols else "None")

        story.extend(_notes(
            "Missing value summary",
            [
                f"Total missing cells: {total_miss:,} ({miss_pct:.2f}% of all data).",
                f"Affected columns ({len(miss_cols)}): {', '.join(miss_cols) if miss_cols else 'None'}.",
                f"Most affected column: {worst}.",
                "Numeric columns: Median imputation (robust to outliers).",
                "Categorical columns: Mode imputation (preserves most frequent value).",
                "Columns with >30% missing values warrant careful review — consider dropping them.",
            ],
            ST))

    # ══ 03  OUTLIER DETECTION ═════════════════════════════════════════════════
    story.append(Paragraph("03  Outlier Detection", ST["h1"]))
    story.append(_accent_rule())

    if cleaned_df is not None:
        op = _outlier_chart(cleaned_df)
        if op:
            story.append(_img(op, 162, 60))
            story.append(Paragraph(
                "Bar length = number of outliers per column (IQR method, cleaned dataset).",
                ST["caption"]))
        else:
            story.append(Paragraph("✓  No outliers detected in any numeric column.", ST["body"]))

        nc    = cleaned_df.select_dtypes(include=["int64", "float64"]).columns
        oinfo = []
        for col in nc:
            ser = cleaned_df[col].dropna()
            q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
            iqr = q3 - q1
            cnt = int(((ser < q1-1.5*iqr) | (ser > q3+1.5*iqr)).sum())
            if cnt > 0:
                oinfo.append(f"{col}: {cnt}")

        story.extend(_notes(
            "Outlier summary",
            [
                "Method: IQR — values below Q1−1.5×IQR or above Q3+1.5×IQR are flagged.",
                f"Columns with outliers: {', '.join(oinfo) if oinfo else 'None detected.'}",
                "Capping (Winsorization): replaces outliers with boundary values — preserves row count.",
                "Row removal: only appropriate for confirmed data entry errors.",
                "Outliers inflate standard deviation and distort regression coefficients.",
            ],
            ST))

    # ══ 04  COLUMN STATISTICS ═════════════════════════════════════════════════
    story.append(Paragraph("04  Column Statistics  (cleaned dataset)", ST["h1"]))
    story.append(_accent_rule())

    if cleaned_df is not None:
        tbl = _stats_table(cleaned_df)
        if tbl:
            story.append(tbl)
            story.append(_gap(6))

            nc2    = cleaned_df.select_dtypes(include=["int64", "float64"]).columns
            skewed = [
                f"{c} (skew={cleaned_df[c].dropna().skew():.2f})"
                for c in nc2
                if abs(cleaned_df[c].dropna().skew()) > 1
            ]
            story.extend(_notes(
                "Statistical notes",
                [
                    "Mean vs Median: large gap = skewed distribution; use median as central measure.",
                    "Std Dev: high relative to mean = high variability across records.",
                    "Min / Max: verify against expected domain bounds — extremes may be data errors.",
                    f"Skewed columns (|skew|>1): {'; '.join(skewed) if skewed else 'None detected.'}",
                    "Highly skewed columns may benefit from log or Box-Cox transformation before modelling.",
                ],
                ST))
        else:
            story.append(Paragraph("No numeric columns available.", ST["body"]))

    # ══ 05  CORRELATION MATRIX ════════════════════════════════════════════════
    story.append(Paragraph("05  Correlation Analysis", ST["h1"]))
    story.append(_accent_rule())

    if cleaned_df is not None:
        cp = _correlation_heatmap(cleaned_df)
        if cp:
            story.append(_img(cp, 162, 75))
            story.append(Paragraph(
                "Green = positive, Red = negative. "
                "Values range from −1 (perfect inverse) to +1 (perfect direct). Lower triangle shown.",
                ST["caption"]))

            nc3    = cleaned_df.select_dtypes(include=["int64", "float64"]).columns
            strong = []
            if len(nc3) >= 2:
                corr = cleaned_df[nc3].corr()
                for i in range(len(nc3)):
                    for j in range(i+1, len(nc3)):
                        v = corr.iloc[i, j]
                        if abs(v) >= 0.7:
                            strong.append(f"{nc3[i]} & {nc3[j]} (r = {v:.2f})")

            story.extend(_notes(
                "Correlation notes",
                [
                    "Pearson r measures linear association between numeric column pairs.",
                    "Strength:  |r|≥0.80 Strong  |  0.50–0.79 Moderate  |  0.30–0.49 Weak  |  <0.30 Negligible.",
                    f"Strong pairs (|r|≥0.70): {'; '.join(strong) if strong else 'None found.'}",
                    "High correlation (multicollinearity) inflates coefficient variance in regression models.",
                    "Pearson only captures linear relationships — supplement with scatter plots.",
                ],
                ST))
        else:
            story.append(Paragraph(
                "Insufficient numeric columns for correlation analysis.", ST["body"]))

    # ══ 06  BEFORE vs AFTER ═══════════════════════════════════════════════════
    if original_df is not None and cleaned_df is not None:
        story.append(Paragraph("06  Before vs After Cleaning", ST["h1"]))
        story.append(_accent_rule())
        story.append(_before_after_table(original_df, cleaned_df))
        story.append(_gap(8))

        bap = _before_after_chart(original_df, cleaned_df)
        if bap:
            story.append(_img(bap, 115, 52))
            story.append(Paragraph("Grey = before cleaning, Indigo = after.", ST["caption"]))

        om  = int(original_df.isnull().sum().sum())
        cm2 = int(cleaned_df.isnull().sum().sum())
        od  = int(original_df.duplicated().sum())
        cd  = int(cleaned_df.duplicated().sum())

        improvements = []
        if om - cm2 > 0:
            improvements.append(f"{om - cm2:,} missing values filled via imputation.")
        if od - cd > 0:
            improvements.append(f"{od - cd:,} duplicate rows removed.")
        if cleaned_df.shape[1] > original_df.shape[1]:
            improvements.append(
                f"Column count increased from {original_df.shape[1]} to "
                f"{cleaned_df.shape[1]} (One-Hot encoding applied).")
        if len(cleaned_df) < len(original_df):
            improvements.append(
                f"Row count reduced from {len(original_df):,} to "
                f"{len(cleaned_df):,} (outlier or duplicate removal).")
        if not improvements:
            improvements.append("No changes applied — dataset was already clean.")

        story.extend(_notes(
            "Changes applied",
            improvements + ["All modifications reflect user-approved actions only."],
            ST))

    # ══ 07  FEATURE SCALING ═══════════════════════════════════════════════════
    story.append(Paragraph("07  Feature Scaling", ST["h1"]))
    story.append(_accent_rule())

    if original_df is not None and cleaned_df is not None:
        sp = _scaling_chart(original_df, cleaned_df)
        if sp:
            story.append(_img(sp, 162, 55))
            story.append(Paragraph("Grey = before scaling, Indigo = after.", ST["caption"]))

    story.extend(_notes(
        "Scaling notes",
        [
            "Scaling brings numeric columns to a common range so no column dominates by magnitude alone.",
            "Min-Max Scaling: maps values to [0, 1]. Best for KNN, K-Means, and neural networks. Sensitive to outliers.",
            "Standard Scaling: transforms to mean=0, std=1. Robust to outliers. Required for PCA and linear models.",
            "Apply scaling after outlier treatment — outliers significantly distort Min-Max bounds.",
            "ID and name columns are automatically excluded from scaling.",
        ],
        ST))

    # ══ 08  CATEGORICAL ENCODING ══════════════════════════════════════════════
    story.append(Paragraph("08  Categorical Encoding", ST["h1"]))
    story.append(_accent_rule())

    if original_df is not None and cleaned_df is not None:
        enc_tbl = _encoding_table(original_df, cleaned_df)
        story.append(enc_tbl)
        story.append(_gap(6))

    story.extend(_notes(
        "Encoding notes",
        [
            "Encoding converts text/category columns to numbers — required by most ML algorithms.",
            "Label Encoding: assigns an integer per category (e.g. Red=0, Blue=1). Best for ordinal data.",
            "One-Hot Encoding: creates one binary column per category. Best for nominal data (no natural order).",
            "Avoid One-Hot for columns with >10 unique values — it creates too many columns (curse of dimensionality).",
            "ID and name columns (e.g. CustomerID, Name) are automatically excluded from encoding.",
        ],
        ST))

    # ══ 09  EDA CHARTS ════════════════════════════════════════════════════════
    # FIX 3: Read from plots_white — these were saved with white backgrounds by eda_visualization.py
    if os.path.exists(EDA_PLOTS_DIR):
        plot_files = sorted([
            f for f in os.listdir(EDA_PLOTS_DIR)
            if f.lower().endswith(".png")])

        if plot_files:
            story.append(Paragraph("09  Exploratory Data Analysis", ST["h1"]))
            story.append(_accent_rule())

            for img_file in plot_files:
                ct      = _chart_type_from_filename(img_file)
                caption = (img_file.replace(".png", "").replace("_", " ").title())
                img_path = os.path.join(EDA_PLOTS_DIR, img_file)  # already white — no copy needed

                story.append(_img(img_path, 162, 65))
                story.append(Paragraph(caption, ST["caption"]))
                story.append(_gap(2))

                if ct and ct in _EDA_TYPES:
                    story.extend(_notes(f"{ct} — notes", _EDA_TYPES[ct], ST))

                story.append(_gap(8))

    # ══ 10  CLEANING ACTIONS ══════════════════════════════════════════════════
    story.append(Paragraph("10  Cleaning Actions Summary", ST["h1"]))
    story.append(_accent_rule())
    story.append(Paragraph(cleaning_summary, ST["body"]))
    story.append(_gap(4))

    if original_df is not None and cleaned_df is not None:
        om2 = int(original_df.isnull().sum().sum())
        cm3 = int(cleaned_df.isnull().sum().sum())
        od2 = int(original_df.duplicated().sum())
        cd2 = int(cleaned_df.duplicated().sum())
        actions = []
        if om2 - cm3 > 0:
            actions.append(
                f"Imputed {om2 - cm3:,} missing values using median (numeric) / mode (categorical).")
        if od2 - cd2 > 0:
            actions.append(f"Removed {od2 - cd2:,} duplicate rows.")
        if cleaned_df.shape[1] != original_df.shape[1]:
            actions.append(
                f"Column count changed from {original_df.shape[1]} to "
                f"{cleaned_df.shape[1]} (encoding applied).")
        actions += [
            "Median preferred over mean: not distorted by outliers.",
            "Duplicate removal prevents inflated counts and biased model training.",
            "ID and name columns were excluded from scaling and encoding.",
            "Suggested pipeline: feature engineering → train/test split → model selection → evaluation.",
        ]
        story.extend(_notes("Actions log", actions, ST))

    # ══ 11  RECOMMENDATIONS ═══════════════════════════════════════════════════
    story.append(Paragraph("11  Recommendations & Next Steps", ST["h1"]))
    story.append(_accent_rule())

    if cleaned_df is not None:
        for elem in _build_recs(original_df, cleaned_df, score, label, ST):
            story.append(elem)

    # ══ FOOTER ════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 10*mm))
    story.append(_rule(thickness=0.4))
    story.append(Paragraph(
        "This report was automatically generated by the Explainable AutoEDA Platform. "
        "All findings are based on the uploaded dataset and user-approved preprocessing actions.",
        ST["footer"]))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return pdf_path