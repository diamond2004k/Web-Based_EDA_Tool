"""
data_quality.py — Dataset Quality Scorer

WHY THE OLD SCORER GAVE 100 WITHOUT PREPROCESSING
===================================================
The old scorer only checked 4 things:
  1. Missing values      (max 40-pt penalty)
  2. Duplicate rows      (max 30-pt penalty)
  3. Type consistency    (max 20-pt penalty)
  4. Column completeness (max 10-pt penalty)

A dataset with no missing values and no duplicates scored 100
even if it was full of outliers and unencoded categories.

NEW SCORING (6 penalties, 100 pts total)
=========================================
Penalty 1 — Missing values         : up to 30 pts
Penalty 2 — Duplicate rows         : up to 20 pts
Penalty 3 — IQR Outliers           : up to 20 pts  ← NEW
Penalty 4 — Unencoded categoricals : up to 15 pts  ← NEW
Penalty 5 — Type inconsistency     : up to 10 pts
Penalty 6 — Empty columns          : up to  5 pts

Score of 100 is now ONLY possible once all cleaning steps are done:
  ✔ No missing values        (Step 03 → Missing Value Treatment)
  ✔ No duplicate rows        (Step 03 → Duplicate Row Removal)
  ✔ No IQR outliers          (Step 03 → Outlier Detection & Treatment)
  ✔ No unencoded categoricals(Step 03 → Categorical Encoding)
  ✔ No mixed-type columns
  ✔ No entirely empty columns
"""

import pandas as pd


def _outlier_column_count(df: pd.DataFrame) -> int:
    """Count numeric columns that contain at least one IQR outlier."""
    count = 0
    for col in df.select_dtypes(include=["number"]).columns:
        ser = df[col].dropna()
        if len(ser) < 4:
            continue
        q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        if ((ser < q1 - 1.5 * iqr) | (ser > q3 + 1.5 * iqr)).any():
            count += 1
    return count


def calculate_quality_score(df: pd.DataFrame):
    """
    Returns (score: float, label: str, explanations: list[str]).
    Score reaches 100 only after all preprocessing steps are complete.
    """
    if df is None or df.empty:
        return 0.0, "Poor", ["Dataset is empty or not loaded."]

    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return 0.0, "Poor", ["Dataset has no rows or columns."]

    score        = 100.0
    explanations = []

    # ── Penalty 1: Missing values (max 30 pts) ─────────────────────────────────
    missing_pct     = df.isnull().mean().mean() * 100
    missing_penalty = min(missing_pct * 0.75, 30)
    score          -= missing_penalty

    if missing_penalty == 0:
        explanations.append(
            "✔ Missing values: 0-pt penalty — no missing cells detected."
        )
    else:
        total_missing = int(df.isnull().sum().sum())
        explanations.append(
            f"Missing values: -{missing_penalty:.1f} pts "
            f"({total_missing:,} missing cells, avg {missing_pct:.2f}% per column). "
            "Fix: apply imputation in Step 03 → Missing Value Treatment."
        )

    # ── Penalty 2: Duplicate rows (max 20 pts) ─────────────────────────────────
    dup_count   = int(df.duplicated().sum())
    dup_pct     = (dup_count / rows) * 100
    dup_penalty = min(dup_pct * 0.5, 20)
    score      -= dup_penalty

    if dup_penalty == 0:
        explanations.append(
            "✔ Duplicate rows: 0-pt penalty — no duplicates detected."
        )
    else:
        explanations.append(
            f"Duplicate rows: -{dup_penalty:.1f} pts "
            f"({dup_count:,} duplicate rows = {dup_pct:.2f}% of data). "
            "Fix: apply removal in Step 03 → Duplicate Row Removal."
        )

    # ── Penalty 3: IQR Outliers (max 20 pts) ── NEW ────────────────────────────
    num_cols        = df.select_dtypes(include=["number"]).columns
    n_num           = len(num_cols)
    outlier_cols    = _outlier_column_count(df)
    outlier_ratio   = (outlier_cols / n_num) if n_num > 0 else 0
    outlier_penalty = min(outlier_ratio * 20, 20)
    score          -= outlier_penalty

    if n_num == 0:
        explanations.append(
            "— Outlier check skipped: no numeric columns present."
        )
    elif outlier_penalty == 0:
        explanations.append(
            "✔ Outliers: 0-pt penalty — no IQR outliers in any numeric column."
        )
    else:
        explanations.append(
            f"Outliers: -{outlier_penalty:.1f} pts "
            f"({outlier_cols} of {n_num} numeric column(s) contain IQR outliers). "
            "Fix: apply capping or removal in Step 03 → Outlier Detection & Treatment."
        )

    # ── Penalty 4: Unencoded categorical columns (max 15 pts) ── NEW ───────────
    cat_cols    = df.select_dtypes(include=["object", "category"]).columns
    cat_ratio   = len(cat_cols) / cols
    cat_penalty = min(cat_ratio * 15, 15)
    score      -= cat_penalty

    if cat_penalty == 0:
        explanations.append(
            "✔ Categorical encoding: 0-pt penalty — no unencoded text/category columns."
        )
    else:
        sample = list(cat_cols)[:4]
        more   = f"...+{len(cat_cols)-4} more" if len(cat_cols) > 4 else ""
        explanations.append(
            f"Unencoded categoricals: -{cat_penalty:.1f} pts "
            f"({len(cat_cols)} text/category column(s) still unencoded: "
            f"{', '.join(sample)}{more}). "
            "Fix: apply Label or One-Hot encoding in Step 03 → Categorical Encoding."
        )

    # ── Penalty 5: Mixed data types (max 10 pts) ────────────────────────────────
    mixed_cols   = sum(1 for col in df.columns if df[col].apply(type).nunique() > 1)
    type_penalty = min((mixed_cols / cols) * 10, 10)
    score       -= type_penalty

    if type_penalty == 0:
        explanations.append(
            "✔ Type consistency: 0-pt penalty — all columns have consistent types."
        )
    else:
        explanations.append(
            f"Type inconsistency: -{type_penalty:.1f} pts "
            f"({mixed_cols} column(s) contain mixed data types). "
            "Fix: ensure each column stores only one data type."
        )

    # ── Penalty 6: Empty columns (max 5 pts) ────────────────────────────────────
    non_empty     = int(df.notnull().any().sum())
    empty_count   = cols - non_empty
    empty_penalty = min((empty_count / cols) * 5, 5)
    score        -= empty_penalty

    if empty_penalty == 0:
        explanations.append(
            "✔ Column completeness: 0-pt penalty — all columns contain at least one value."
        )
    else:
        explanations.append(
            f"Empty columns: -{empty_penalty:.1f} pts "
            f"({empty_count} column(s) are entirely empty). "
            "Fix: drop or fill these columns."
        )

    # ── Final score & grade ─────────────────────────────────────────────────────
    score = round(max(0.0, min(100.0, score)), 1)

    if score >= 90:
        label = "Excellent"
    elif score >= 75:
        label = "Good"
    elif score >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return score, label, explanations