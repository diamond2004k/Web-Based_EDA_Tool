import pandas as pd
import numpy as np
from datetime import datetime


def _separator(char="=", width=65):
    return char * width + "\n"


def _section(number, title):
    return (
        f"\n{'=' * 65}\n"
        f"  {number}  {title}\n"
        f"{'=' * 65}\n"
    )


def _subsection(title):
    return f"\n  {'─' * 55}\n  {title}\n  {'─' * 55}\n"


# ── Column statistics ──────────────────────────────────────────────────────────
def _col_stats(df):
    lines = []
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) == 0:
        return "  No numeric columns found.\n"
    for col in numeric_cols:
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        skew_val = s.skew()
        skew_dir = ("right-skewed (positive)" if skew_val > 1
                    else "left-skewed (negative)" if skew_val < -1
                    else "approximately symmetric")
        lines.append(f"  Column   : {col}")
        lines.append(f"    Mean     : {s.mean():.4f}")
        lines.append(f"    Median   : {s.median():.4f}")
        lines.append(f"    Std Dev  : {s.std():.4f}")
        lines.append(f"    Min      : {s.min():.4f}")
        lines.append(f"    Max      : {s.max():.4f}")
        lines.append(f"    Skewness : {skew_val:.4f}  ({skew_dir})")
        lines.append(f"    Outliers : {outliers} (IQR method)")
        lines.append("")
    return "\n".join(lines)


# ── Missing value summary ──────────────────────────────────────────────────────
def _missing_summary(df):
    lines = []
    rows = len(df)
    any_missing = False
    for col in df.columns:
        mc = df[col].isnull().sum()
        if mc > 0:
            any_missing = True
            dtype = df[col].dtype
            rec = "Median imputation" if dtype in ["int64", "float64"] else "Mode imputation"
            lines.append(f"  {col:<30} {mc:>6} missing  ({mc/rows*100:.2f}%)  →  Recommended: {rec}")
    if not any_missing:
        lines.append("  No missing values detected — dataset is complete.")

    total = df.isnull().sum().sum()
    total_pct = total / df.size * 100
    lines.append(f"\n  Total missing cells : {total:,} / {df.size:,}  ({total_pct:.2f}%)")
    return "\n".join(lines)


# ── Outlier summary ────────────────────────────────────────────────────────────
def _outlier_summary(df):
    lines = []
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    found = False
    for col in numeric_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((s < lower) | (s > upper)).sum())
        if outlier_count > 0:
            found = True
            pct = outlier_count / len(s) * 100
            lines.append(
                f"  {col:<30} {outlier_count:>5} outlier(s)  ({pct:.2f}%)"
                f"  [range: {lower:.3f} – {upper:.3f}]"
            )
    if not found:
        lines.append("  No outliers detected in any numeric column.")
    lines.append(
        "\n  METHOD: IQR — values below Q1−1.5×IQR or above Q3+1.5×IQR are flagged."
    )
    return "\n".join(lines)


# ── Correlation summary ────────────────────────────────────────────────────────
def _correlation_summary(df):
    lines = []
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) < 2:
        return "  Not enough numeric columns for correlation analysis.\n"
    corr = df[numeric_cols].corr()
    reported = set()
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[c1, c2]
            pair = tuple(sorted([c1, c2]))
            if pair not in reported:
                reported.add(pair)
                if abs(val) >= 0.80:
                    strength = "STRONG"
                elif abs(val) >= 0.50:
                    strength = "MODERATE"
                elif abs(val) >= 0.30:
                    strength = "WEAK"
                else:
                    strength = "negligible"
                direction = "positive" if val > 0 else "negative"
                pairs.append(
                    f"  {c1} × {c2:<25} r = {val:+.4f}  "
                    f"({strength}, {direction})"
                )
    lines.extend(pairs if pairs else ["  No correlations computed."])
    lines.append(
        "\n  GUIDE: |r|≥0.80 Strong | 0.50–0.79 Moderate | 0.30–0.49 Weak | <0.30 Negligible"
    )
    return "\n".join(lines)


# ── Scaling summary ────────────────────────────────────────────────────────────
def _scaling_summary(original_df, cleaned_df):
    lines = []
    numeric_orig   = set(original_df.select_dtypes(include=["int64", "float64"]).columns)
    numeric_clean  = set(cleaned_df.select_dtypes(include=["int64", "float64"]).columns)
    common         = numeric_orig & numeric_clean

    scaled = []
    for col in common:
        o_min, o_max = original_df[col].min(), original_df[col].max()
        c_min, c_max = cleaned_df[col].min(),  cleaned_df[col].max()
        if abs(c_min) < 5 and abs(c_max) < 5 and abs(o_max - o_min) > 1:
            scaled.append(col)

    if scaled:
        lines.append(f"  Columns likely scaled : {', '.join(scaled)}")
        lines.append(
            "  Scaling compresses values to a common range, preventing columns with "
            "larger magnitudes from dominating analysis or model training."
        )
    else:
        lines.append("  No scaling was applied or detected.")
    return "\n".join(lines)


# ── Encoding summary ───────────────────────────────────────────────────────────
def _encoding_summary(original_df, cleaned_df):
    lines = []
    orig_cat  = set(original_df.select_dtypes(include=["object", "category"]).columns)
    clean_cat = set(cleaned_df.select_dtypes(include=["object", "category"]).columns)
    encoded   = orig_cat - clean_cat

    new_ohe_cols = [c for c in cleaned_df.columns
                    if c not in original_df.columns]

    if encoded:
        lines.append(f"  Label-encoded columns  : {', '.join(encoded)}")
    if new_ohe_cols:
        lines.append(
            f"  One-Hot encoded columns created : {len(new_ohe_cols)} new binary column(s)"
        )
        lines.append(f"  New columns : {', '.join(new_ohe_cols[:10])}"
                     + (" ..." if len(new_ohe_cols) > 10 else ""))
    if not encoded and not new_ohe_cols:
        lines.append("  No categorical encoding was applied.")
    lines.append(
        "\n  Encoding converts text/category columns to numeric form, required by "
        "most machine learning algorithms."
    )
    return "\n".join(lines)


# ── Before / After table ───────────────────────────────────────────────────────
def _before_after(original_df, cleaned_df):
    lines = []
    orig_missing  = int(original_df.isnull().sum().sum())
    clean_missing = int(cleaned_df.isnull().sum().sum())
    orig_dups     = int(original_df.duplicated().sum())
    clean_dups    = int(cleaned_df.duplicated().sum())
    orig_rows, orig_cols   = original_df.shape
    clean_rows, clean_cols = cleaned_df.shape

    def delta(b, a):
        d = a - b
        return f"{d:+d}" if d != 0 else "  0 (no change)"

    w = 32
    lines.append(f"  {'Metric':<{w}} {'Before':>10} {'After':>10}  {'Change':>12}")
    lines.append(f"  {'─'*68}")
    lines.append(f"  {'Total Rows':<{w}} {orig_rows:>10} {clean_rows:>10}  {delta(orig_rows, clean_rows):>12}")
    lines.append(f"  {'Total Columns':<{w}} {orig_cols:>10} {clean_cols:>10}  {delta(orig_cols, clean_cols):>12}")
    lines.append(f"  {'Missing Values':<{w}} {orig_missing:>10} {clean_missing:>10}  {delta(orig_missing, clean_missing):>12}")
    lines.append(f"  {'Duplicate Rows':<{w}} {orig_dups:>10} {clean_dups:>10}  {delta(orig_dups, clean_dups):>12}")
    lines.append(f"  {'Missing %':<{w}} {original_df.isnull().mean().mean()*100:>9.2f}% {cleaned_df.isnull().mean().mean()*100:>9.2f}%")
    return "\n".join(lines)


# ── Recommendations ────────────────────────────────────────────────────────────
def _recommendations(original_df, cleaned_df, score, label):
    lines = []
    numeric_cols = cleaned_df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols     = cleaned_df.select_dtypes(include=["object", "category"]).columns

    recs = []

    # Missing
    remaining_missing = cleaned_df.isnull().sum().sum()
    if remaining_missing > 0:
        recs.append(
            f"  ⚠  {remaining_missing} missing value(s) remain. "
            "Apply imputation in the cleaning step before modelling."
        )

    # Outliers
    outlier_cols = []
    for col in numeric_cols:
        s = cleaned_df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()) > 0:
            outlier_cols.append(col)
    if outlier_cols:
        recs.append(
            f"  ⚠  Outliers remain in: {', '.join(outlier_cols)}. "
            "Consider capping or log-transforming before modelling."
        )

    # Categorical still present
    if len(cat_cols) > 0:
        recs.append(
            f"  ⚠  {len(cat_cols)} categorical column(s) still present "
            f"({', '.join(list(cat_cols)[:5])}). "
            "Apply Label or One-Hot Encoding before using in a model."
        )

    # Skewed columns
    skewed = []
    for col in numeric_cols:
        s = cleaned_df[col].dropna()
        if abs(s.skew()) > 1:
            skewed.append(col)
    if skewed:
        recs.append(
            f"  ℹ  Highly skewed columns: {', '.join(skewed)}. "
            "Consider log or Box-Cox transformation to improve normality."
        )

    # Correlation
    if len(numeric_cols) >= 2:
        corr = cleaned_df[numeric_cols].corr()
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if abs(corr.iloc[i, j]) >= 0.8:
                    high_corr.append(f"{numeric_cols[i]} & {numeric_cols[j]}")
        if high_corr:
            recs.append(
                f"  ℹ  Highly correlated pairs: {'; '.join(high_corr)}. "
                "Consider removing one from each pair to reduce multicollinearity."
            )

    # Score-based
    if label in ("Poor", "Fair"):
        recs.append(
            f"  ⚠  Quality score is {label} ({score}/100). "
            "Review and apply all available cleaning steps before analysis."
        )
    else:
        recs.append(
            f"  ✔  Quality score is {label} ({score}/100). "
            "Dataset is in good shape for analysis or modelling."
        )

    if not recs:
        recs.append("  ✔  No further issues detected. Dataset is ready for analysis.")

    lines.extend(recs)
    return "\n".join(lines)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def generate_report(score, label, explanations, action_taken,
                    original_df=None, cleaned_df=None):

    lines = []
    now = datetime.now().strftime("%d %B %Y  %H:%M")

    # ── Header ─────────────────────────────────────────────────────────────────
    lines.append(_separator("═"))
    lines.append("  EXPLAINABLE AUTOEDA — DETAILED ANALYSIS REPORT\n")
    lines.append(f"  Generated  : {now}\n")
    lines.append(
        "  Platform   : Explainable AutoEDA  |  "
        "Automated Data Intelligence\n"
    )
    lines.append(_separator("═"))

    # ── 01 Quality Score ───────────────────────────────────────────────────────
    lines.append(_section("01", "DATASET QUALITY SCORE"))
    lines.append(f"  Score  : {score} / 100\n")
    lines.append(f"  Rating : {label}\n")
    lines.append("  Breakdown:\n")
    for exp in explanations:
        lines.append(f"    → {exp}")

    lines.append(
        "\n  EXPLANATION:\n"
        "  The quality score is computed across four dimensions:\n"
        "    • Missing values   (max 40-point penalty)\n"
        "    • Duplicate rows   (max 30-point penalty)\n"
        "    • Type consistency (max 20-point penalty)\n"
        "    • Column completeness (max 10-point penalty)\n"
        "  A score ≥85 = Excellent | ≥70 = Good | ≥50 = Fair | <50 = Poor."
    )

    # ── 02 Missing Values ──────────────────────────────────────────────────────
    if original_df is not None:
        lines.append(_section("02", "MISSING VALUE SUMMARY  (original dataset)"))
        lines.append(_missing_summary(original_df))
        lines.append(
            "\n  EXPLANATION:\n"
            "  Missing values are absent data points that can bias analysis.\n"
            "  Median imputation is used for numeric columns (robust to outliers).\n"
            "  Mode imputation is used for categorical columns (most frequent value).\n"
            "  Dropping rows is only recommended when very few rows are affected (<5%)."
        )

    # ── 03 Outlier Detection ───────────────────────────────────────────────────
    if cleaned_df is not None:
        lines.append(_section("03", "OUTLIER DETECTION  (cleaned dataset, IQR method)"))
        lines.append(_outlier_summary(cleaned_df))
        lines.append(
            "\n  EXPLANATION:\n"
            "  Outliers are extreme values that can distort averages and inflate\n"
            "  standard deviations. The IQR method flags values beyond 1.5×IQR\n"
            "  from Q1 or Q3. Capping (winsorization) replaces them with boundary\n"
            "  values. Removal is only appropriate when outliers are confirmed errors."
        )

    # ── 04 Column Statistics ───────────────────────────────────────────────────
    if cleaned_df is not None:
        lines.append(_section("04", "COLUMN-BY-COLUMN STATISTICS  (cleaned dataset)"))
        lines.append(_col_stats(cleaned_df))
        lines.append(
            "  EXPLANATION:\n"
            "  Mean vs Median: A large difference indicates skewness — use median\n"
            "  as the measure of central tendency for skewed columns.\n"
            "  Std Dev: High relative to mean = high variability.\n"
            "  Min/Max: Check against expected domain bounds for data errors.\n"
            "  Skewness > |1| suggests a log or Box-Cox transformation may help."
        )

    # ── 05 Correlation Summary ─────────────────────────────────────────────────
    if cleaned_df is not None:
        lines.append(_section("05", "CORRELATION SUMMARY  (cleaned dataset)"))
        lines.append(_correlation_summary(cleaned_df))
        lines.append(
            "\n  EXPLANATION:\n"
            "  Pearson r measures linear association between numeric columns.\n"
            "  Highly correlated pairs (|r|≥0.80) indicate multicollinearity,\n"
            "  which can inflate coefficient variance in regression models.\n"
            "  Consider removing or combining one column from each strong pair.\n"
            "  Note: Pearson only captures linear relationships."
        )

    # ── 06 Before vs After ─────────────────────────────────────────────────────
    if original_df is not None and cleaned_df is not None:
        lines.append(_section("06", "BEFORE vs AFTER CLEANING"))
        lines.append(_before_after(original_df, cleaned_df))
        lines.append(
            "\n  EXPLANATION:\n"
            "  This table shows the cumulative effect of all preprocessing steps\n"
            "  applied: imputation, duplicate removal, outlier treatment,\n"
            "  scaling, and encoding. Column count changes indicate encoding.\n"
            "  Row count changes indicate duplicate or outlier row removal."
        )

    # ── 07 Scaling Summary ─────────────────────────────────────────────────────
    if original_df is not None and cleaned_df is not None:
        lines.append(_section("07", "FEATURE SCALING SUMMARY"))
        lines.append(_scaling_summary(original_df, cleaned_df))
        lines.append(
            "\n  EXPLANATION:\n"
            "  Min-Max Scaling: maps values to [0,1]. Sensitive to outliers.\n"
            "  Standard Scaling: transforms to mean=0, std=1. Robust to outliers.\n"
            "  Scaling is essential for distance-based algorithms (KNN, K-Means,\n"
            "  SVM) and recommended for linear models and PCA."
        )

    # ── 08 Encoding Summary ────────────────────────────────────────────────────
    if original_df is not None and cleaned_df is not None:
        lines.append(_section("08", "CATEGORICAL ENCODING SUMMARY"))
        lines.append(_encoding_summary(original_df, cleaned_df))

    # ── 09 Cleaning Actions ────────────────────────────────────────────────────
    lines.append(_section("09", "CLEANING ACTIONS LOG"))
    if action_taken:
        lines.append(
            "  One or more cleaning/preprocessing actions were applied:\n"
            "    • Missing value imputation (median/mean/mode or row drop)\n"
            "    • Duplicate row removal\n"
            "    • Outlier capping or removal\n"
            "    • Feature scaling (Min-Max or Standard)\n"
            "    • Categorical encoding (Label or One-Hot)\n\n"
            "  All actions were performed based on user approval."
        )
    else:
        lines.append(
            "  No cleaning actions were applied.\n"
            "  The dataset was either already clean or the user did not\n"
            "  approve any of the suggested preprocessing steps."
        )

    # ── 10 Recommendations ─────────────────────────────────────────────────────
    if cleaned_df is not None:
        lines.append(_section("10", "RECOMMENDATIONS FOR NEXT STEPS"))
        lines.append(_recommendations(original_df, cleaned_df, score, label))

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines.append(f"\n{_separator('═')}")
    lines.append(
        "  This report was auto-generated by the Explainable AutoEDA Platform.\n"
        "  All findings are based on the uploaded dataset and user-approved actions.\n"
    )
    lines.append(_separator("═"))

    return "\n".join(lines)