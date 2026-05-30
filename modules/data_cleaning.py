import streamlit as st
import pandas as pd
import numpy as np
from modules.explainability import explain_missing, explain_duplicates


# ── Safe numeric check ─────────────────────────────────────────────────────────
def _is_numeric(series):
    try:
        return pd.api.types.is_numeric_dtype(series) and series.dtype.kind in "iufcb"
    except Exception:
        return False


def _safe_median(series):
    try:
        return series.dropna().median()
    except TypeError:
        return None


def _safe_mean(series):
    try:
        return series.dropna().mean()
    except TypeError:
        return None


# ── ID/Name column detection ───────────────────────────────────────────────────
def _is_id_like(col_name):
    """Returns True if the column looks like an identifier or name — excluded from scaling/encoding."""
    c = col_name.lower().strip()
    return (
        c in ("id", "customerid", "customer_id", "userid", "user_id", "name",
               "fullname", "full_name", "firstname", "lastname", "email")
        or c.endswith("_id")
        or c.endswith("id")
        or c.startswith("id_")
        or c == "name"
    )


# ── Per-step applied state ─────────────────────────────────────────────────────
# STEP_KEYS is exported so app.py can reset them on demand.
STEP_KEYS = {
    "missing":    "step_missing_done",
    "duplicates": "step_duplicates_done",
    "outliers":   "step_outliers_done",
    "scaling":    "step_scaling_done",
    "encoding":   "step_encoding_done",
}


def _init_step_states():
    """Initialise step flags only if they don't already exist in session_state."""
    for key in STEP_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = False


def _mark_done(step: str):
    st.session_state[STEP_KEYS[step]] = True
    st.session_state.action_taken = True


def _is_done(step: str) -> bool:
    return st.session_state.get(STEP_KEYS[step], False)


def _applied_badge(message: str):
    """Renders a green 'Applied ✅' badge instead of the action button."""
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:0.5rem;"
        f"background:#052e1688;border:1px solid #166534;border-radius:10px;"
        f"padding:0.55rem 1.1rem;font-size:0.85rem;color:#4ade80;"
        f"margin-top:0.25rem;margin-bottom:0.5rem;'>"
        f"✅ {message}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Shared UI helpers ──────────────────────────────────────────────────────────
def _card(content_html, border_color="#6366f1", bg="#0f172a"):
    return f"""
    <div style='background:{bg};border:1px solid {border_color};border-radius:12px;
                padding:1.1rem 1.4rem;margin-bottom:1rem;'>
        {content_html}
    </div>"""


def _section_label(text, color="#6366f1"):
    return (
        f"<div style='font-family:Space Mono,monospace;font-size:0.65rem;"
        f"color:{color};letter-spacing:0.15em;text-transform:uppercase;"
        f"margin-bottom:0.6rem;'>{text}</div>"
    )


def _explain_box(title, body, border="#1e293b"):
    st.markdown(f"""
    <div style='background:#080f1e;border-left:3px solid {border};
                border-radius:0 10px 10px 0;
                padding:0.8rem 1.1rem;margin-bottom:0.75rem;'>
        <div style='color:#6366f1;font-size:0.75rem;font-weight:600;
                    margin-bottom:0.3rem;'>{title}</div>
        <div style='color:#94a3b8;font-size:0.85rem;line-height:1.55;'>{body}</div>
    </div>""", unsafe_allow_html=True)


def _sub_header(icon, gradient, title, done=False):
    done_pill = (
        "<span style='font-size:0.65rem;background:#052e16;color:#4ade80;"
        "border:1px solid #166534;border-radius:999px;padding:0.15rem 0.6rem;"
        "margin-left:0.6rem;vertical-align:middle;'>✔ Applied</span>"
        if done else ""
    )
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:0.6rem;
                margin:1.5rem 0 1rem 0;padding-bottom:0.6rem;
                border-bottom:1px solid #1e293b;'>
        <div style='background:{gradient};border-radius:6px;
                    padding:4px 8px;font-size:0.85rem;'>{icon}</div>
        <span style='color:#e2e8f0;font-weight:600;font-size:1rem;'>
            {title}{done_pill}
        </span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MISSING VALUE TREATMENT
# ══════════════════════════════════════════════════════════════════════════════
def _handle_missing(df):
    working      = st.session_state.cleaned_df.copy()
    missing_cols = [c for c in working.columns if working[c].isnull().sum() > 0]

    if not missing_cols:
        st.success("✅ No missing values in the current dataset.")
        return False

    # ── Already applied: show badge only, no button ────────────────────────────
    if _is_done("missing"):
        _applied_badge("Missing values have been treated. Use Reset Cleaning to redo.")
        return False

    rows = len(working)

    st.markdown(_card(
        _section_label("📖 What are missing values?") +
        """<div style='color:#94a3b8;font-size:0.88rem;line-height:1.6;'>
        Missing values are empty cells in your dataset — places where data was not
        recorded or is unavailable. They can cause errors in analysis and reduce
        model reliability.<br><br>
        <b style='color:#e2e8f0;'>Strategies:</b><br>
        • <b style='color:#6366f1;'>Median</b> — middle value (best for numeric with outliers)<br>
        • <b style='color:#6366f1;'>Mean</b> — average (best for normally distributed numeric)<br>
        • <b style='color:#6366f1;'>Mode</b> — most frequent value (best for categorical)<br>
        • <b style='color:#f59e0b;'>Drop rows</b> — remove rows (use sparingly)
        </div>""",
        border_color="#1e3a5f"
    ), unsafe_allow_html=True)

    st.markdown("**Missing Value Summary**")
    summary_data = []
    for col in missing_cols:
        mc  = working[col].isnull().sum()
        pct = mc / rows * 100
        rec = "Median" if _is_numeric(working[col]) else "Mode"
        summary_data.append({
            "Column": col, "Missing": mc,
            "% Missing": f"{pct:.1f}%",
            "Type": str(working[col].dtype),
            "Recommended": rec,
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    mode = st.radio(
        "Choose your cleaning mode:",
        ["🤖 Auto Clean (recommended for beginners)",
         "🔧 Manual (choose strategy per column)"],
        key="missing_mode",
    )

    # ── AUTO MODE ─────────────────────────────────────────────────────────────
    if mode.startswith("🤖"):
        _explain_box(
            "💡 What Auto Clean will do",
            "Applies <b style='color:#6366f1;'>Median</b> for numeric columns and "
            "<b style='color:#6366f1;'>Mode</b> for categorical columns. No rows dropped.",
            border="#6366f1",
        )
        preview = []
        for col in missing_cols:
            if _is_numeric(working[col]):
                fv = _safe_median(working[col])
                preview.append({"Column": col, "Strategy": "Median",
                                 "Fill Value": f"{fv:.4f}" if fv is not None else "N/A"})
            else:
                mv = working[col].mode()
                fv = mv.iloc[0] if len(mv) > 0 else "N/A"
                preview.append({"Column": col, "Strategy": "Mode",
                                 "Fill Value": str(fv)})
        st.markdown("**Preview — values that will be used:**")
        st.dataframe(pd.DataFrame(preview), use_container_width=True)

        # Button only shown if step is NOT done (guard is above, so this is always fresh)
        if st.button("🤖 Apply Auto Clean", key="auto_clean_btn"):
            result = st.session_state.cleaned_df.copy()
            for col in missing_cols:
                if col not in result.columns:
                    continue
                if _is_numeric(result[col]):
                    fv = _safe_median(result[col])
                    if fv is not None:
                        result[col] = result[col].fillna(fv)
                else:
                    mv = result[col].mode()
                    if len(mv) > 0:
                        result[col] = result[col].fillna(mv.iloc[0])
            st.session_state.cleaned_df = result
            _mark_done("missing")
            st.rerun()

    # ── MANUAL MODE ───────────────────────────────────────────────────────────
    else:
        _explain_box(
            "🔧 Manual Mode",
            "Choose a strategy per column. Click <b>Apply Manual Selections</b> when ready.",
            border="#0ea5e9",
        )
        strategies = {}
        for col in missing_cols:
            mc      = working[col].isnull().sum()
            pct     = mc / rows * 100
            is_num  = _is_numeric(working[col])
            rec_idx = 0 if is_num else 2

            with st.expander(
                f"📌 {col}  —  {mc} missing ({pct:.1f}%)  |  {working[col].dtype}",
                expanded=False,
            ):
                if is_num:
                    _explain_box("Why Median is recommended",
                                 f"'{col}' is numeric. Median resists outlier influence.",
                                 border="#6366f1")
                else:
                    _explain_box("Why Mode is recommended",
                                 f"'{col}' is categorical. Mode preserves the most common category.",
                                 border="#6366f1")
                choice = st.selectbox(
                    f"Strategy for '{col}'",
                    ["Median (fill with middle value)",
                     "Mean (fill with average)",
                     "Mode (fill with most frequent value)",
                     "Drop rows with missing values"],
                    index=rec_idx,
                    key=f"strategy_{col}",
                )
                strategies[col] = choice

        if st.button("🔧 Apply Manual Selections", key="manual_clean_btn"):
            result = st.session_state.cleaned_df.copy()
            for col, choice in strategies.items():
                if col not in result.columns:
                    continue
                if choice.startswith("Median"):
                    fv = _safe_median(result[col])
                    if fv is not None:
                        result[col] = result[col].fillna(fv)
                    else:
                        mv = result[col].mode()
                        if len(mv) > 0:
                            result[col] = result[col].fillna(mv.iloc[0])
                elif choice.startswith("Mean"):
                    fv = _safe_mean(result[col])
                    if fv is not None:
                        result[col] = result[col].fillna(fv)
                elif choice.startswith("Mode"):
                    mv = result[col].mode()
                    if len(mv) > 0:
                        result[col] = result[col].fillna(mv.iloc[0])
                elif choice.startswith("Drop"):
                    result = result.dropna(subset=[col])
            result = result.reset_index(drop=True)
            st.session_state.cleaned_df = result
            _mark_done("missing")
            st.rerun()

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DUPLICATE ROWS
# ══════════════════════════════════════════════════════════════════════════════
def _handle_duplicates():
    working   = st.session_state.cleaned_df
    dup_count = working.duplicated().sum()

    if dup_count == 0:
        st.success("✅ No duplicate rows found.")
        return False

    # ── Already applied: show badge only, no button ────────────────────────────
    if _is_done("duplicates"):
        _applied_badge("Duplicate rows have been removed. Use Reset Cleaning to redo.")
        return False

    st.markdown(_card(
        _section_label("🔁 Duplicate rows detected") +
        f"<div style='color:#94a3b8;font-size:0.88rem;line-height:1.6;'>"
        f"Found <b style='color:#f59e0b;'>{dup_count} duplicate row(s)</b>. "
        "Duplicate records inflate counts, skew averages, and bias model training.</div>",
        border_color="#f59e0b",
    ), unsafe_allow_html=True)

    if st.button("🗑️ Remove Duplicate Rows", key="remove_duplicates"):
        result = working.drop_duplicates().reset_index(drop=True)
        st.session_state.cleaned_df = result
        _mark_done("duplicates")
        st.rerun()

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — OUTLIER CAPPING
# ══════════════════════════════════════════════════════════════════════════════
def _handle_outliers():
    working      = st.session_state.cleaned_df
    numeric_cols = [c for c in working.columns if _is_numeric(working[c])]
    outlier_info = {}

    for col in numeric_cols:
        s = working[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < lower) | (s > upper)).sum())
        if count > 0:
            outlier_info[col] = {"count": count, "lower": lower, "upper": upper}

    if not outlier_info:
        st.success("✅ No outliers detected in any numeric column.")
        return False

    # ── Already applied: show badge only, no button ────────────────────────────
    if _is_done("outliers"):
        _applied_badge("Outlier treatment has been applied. Use Reset Cleaning to redo.")
        return False

    st.markdown(_card(
        _section_label("📖 What are outliers?") +
        """<div style='color:#94a3b8;font-size:0.88rem;line-height:1.6;'>
        Outliers are values unusually far from the rest of the data.
        They distort averages and mislead models.<br><br>
        <b style='color:#e2e8f0;'>IQR Method:</b> Flags values below
        <b style='color:#6366f1;'>Q1 − 1.5×IQR</b> or above
        <b style='color:#6366f1;'>Q3 + 1.5×IQR</b>.<br><br>
        <b style='color:#e2e8f0;'>Capping (Winsorization)</b> replaces outliers
        with boundary values, preserving all rows.
        </div>""",
        border_color="#0ea5e9",
    ), unsafe_allow_html=True)

    st.markdown("**Outlier Summary:**")
    st.dataframe(pd.DataFrame([
        {"Column": col, "Outlier Count": info["count"],
         "Lower Bound": f"{info['lower']:.3f}",
         "Upper Bound": f"{info['upper']:.3f}"}
        for col, info in outlier_info.items()
    ]), use_container_width=True)

    method = st.radio(
        "Choose outlier handling method:",
        ["🔒 Cap outliers to boundary values (Winsorization) — recommended",
         "🗑️ Remove rows containing outliers — use with caution"],
        key="outlier_method",
    )

    if st.button("⚡ Apply Outlier Treatment", key="outlier_btn"):
        result = st.session_state.cleaned_df.copy()
        for col, info in outlier_info.items():
            if col not in result.columns:
                continue
            if method.startswith("🔒"):
                result[col] = result[col].clip(lower=info["lower"], upper=info["upper"])
            else:
                mask = ((result[col] >= info["lower"]) & (result[col] <= info["upper"]))
                result = result[mask]
        result = result.reset_index(drop=True)
        st.session_state.cleaned_df = result
        _mark_done("outliers")
        st.rerun()

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════
def _handle_scaling():
    working     = st.session_state.cleaned_df
    all_numeric = [c for c in working.columns if _is_numeric(working[c])]

    id_cols      = [c for c in all_numeric if _is_id_like(c)]
    numeric_cols = [c for c in all_numeric if not _is_id_like(c)]

    if not numeric_cols and not id_cols:
        st.info("No numeric columns available for scaling.")
        return False

    if not numeric_cols:
        st.info("No scalable numeric columns found (all numeric columns are ID-type).")
        return False

    if id_cols:
        st.markdown(
            f"<div style='background:#0a1628;border:1px solid #f59e0b33;"
            f"border-left:3px solid #f59e0b;border-radius:0 10px 10px 0;"
            f"padding:0.6rem 1rem;margin-bottom:0.75rem;font-size:0.82rem;color:#94a3b8;'>"
            f"⚠️ <b style='color:#f59e0b;'>Auto-excluded from scaling:</b> "
            f"{', '.join(id_cols)} — identifier columns should not be scaled."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Already applied: show badge only, no button ────────────────────────────
    if _is_done("scaling"):
        _applied_badge("Feature scaling has been applied. Use Reset Cleaning to redo.")
        return False

    st.markdown(_card(
        _section_label("📖 What is feature scaling?") +
        """<div style='color:#94a3b8;font-size:0.88rem;line-height:1.6;'>
        Scaling brings all numeric columns onto a similar range so no single column
        dominates model training by magnitude alone.<br><br>
        <b style='color:#e2e8f0;'>Min-Max Scaling</b> → range
        <b style='color:#6366f1;'>[0, 1]</b>. Good for KNN, K-Means, neural networks.
        Sensitive to outliers.<br><br>
        <b style='color:#e2e8f0;'>Standard Scaling</b> →
        <b style='color:#6366f1;'>mean=0, std=1</b>. Robust to outliers.
        Required for PCA and linear models.
        </div>""",
        border_color="#16a34a",
    ), unsafe_allow_html=True)

    scale_method = st.radio(
        "Choose scaling method:",
        ["📏 Min-Max Scaling (range 0–1)",
         "📐 Standard Scaling (mean=0, std=1)"],
        key="scale_method",
    )

    cols_to_scale = st.multiselect(
        "Select columns to scale (ID/name columns excluded automatically):",
        options=numeric_cols,
        default=numeric_cols,
        key="scale_cols",
    )

    if cols_to_scale:
        preview_rows = []
        for col in cols_to_scale[:5]:
            s = working[col].dropna()
            if len(s) == 0:
                continue
            orig = round(float(s.iloc[0]), 4)
            if scale_method.startswith("📏"):
                mn, mx = float(s.min()), float(s.max())
                rng = mx - mn if mx != mn else 1
                scaled = round((float(s.iloc[0]) - mn) / rng, 4)
            else:
                mean, std = float(s.mean()), float(s.std())
                std = std if std != 0 else 1
                scaled = round((float(s.iloc[0]) - mean) / std, 4)
            preview_rows.append({"Column": col, "Original [0]": orig, "Scaled [0]": scaled})
        if preview_rows:
            st.markdown("**Scaling Preview (first row of each column):**")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

        if st.button("📏 Apply Scaling", key="scale_btn"):
            result = st.session_state.cleaned_df.copy()
            for col in cols_to_scale:
                if col not in result.columns:
                    continue
                s = result[col].astype(float)
                if scale_method.startswith("📏"):
                    mn, mx = s.min(), s.max()
                    rng = mx - mn if mx != mn else 1
                    result[col] = (s - mn) / rng
                else:
                    mean, std = s.mean(), s.std()
                    std = std if std != 0 else 1
                    result[col] = (s - mean) / std
            st.session_state.cleaned_df = result
            _mark_done("scaling")
            st.rerun()

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CATEGORICAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════
def _handle_encoding():
    working = st.session_state.cleaned_df
    all_cat = [c for c in working.columns if not _is_numeric(working[c])]

    if not all_cat:
        st.success("✅ No categorical columns found — nothing to encode.")
        return False

    rows        = len(working)
    id_cat_cols = [
        c for c in all_cat
        if _is_id_like(c) or (working[c].nunique() / max(rows, 1) > 0.8)
    ]
    cat_cols = [c for c in all_cat if c not in id_cat_cols]

    if id_cat_cols:
        st.markdown(
            f"<div style='background:#0a1628;border:1px solid #f59e0b33;"
            f"border-left:3px solid #f59e0b;border-radius:0 10px 10px 0;"
            f"padding:0.6rem 1rem;margin-bottom:0.75rem;font-size:0.82rem;color:#94a3b8;'>"
            f"⚠️ <b style='color:#f59e0b;'>Auto-excluded from encoding:</b> "
            f"{', '.join(id_cat_cols)} — identifier or near-unique columns produce meaningless codes."
            f"</div>",
            unsafe_allow_html=True,
        )

    if not cat_cols:
        st.success("✅ No encodable categorical columns found after excluding identifiers.")
        return False

    # ── Already applied: show badge only, no button ────────────────────────────
    if _is_done("encoding"):
        _applied_badge("Categorical encoding has been applied. Use Reset Cleaning to redo.")
        return False

    st.markdown(_card(
        _section_label("📖 What is categorical encoding?") +
        """<div style='color:#94a3b8;font-size:0.88rem;line-height:1.6;'>
        Most ML models require numeric input. Encoding converts text/category
        columns to numbers.<br><br>
        <b style='color:#e2e8f0;'>Label Encoding</b> — integer per category
        (Red=0, Blue=1). <b style='color:#f59e0b;'>Best for ordinal data</b>
        (Low/Medium/High).<br><br>
        <b style='color:#e2e8f0;'>One-Hot Encoding</b> — one binary column per
        category. <b style='color:#4ade80;'>Best for nominal data</b> (City, Gender).
        Avoid for &gt;10 unique values.
        </div>""",
        border_color="#a855f7",
    ), unsafe_allow_html=True)

    st.markdown("**Categorical Column Summary:**")
    st.dataframe(pd.DataFrame([
        {"Column": col,
         "Unique Values": working[col].nunique(),
         "Sample Values": ", ".join(str(v) for v in working[col].dropna().unique()[:4]),
         "Recommended": "Label Encode" if working[col].nunique() <= 10
                        else "Label Encode (high cardinality)"}
        for col in cat_cols
    ]), use_container_width=True)

    encode_method = st.radio(
        "Choose encoding method:",
        ["🏷️ Label Encoding (replaces categories with integers)",
         "🔢 One-Hot Encoding (creates binary columns — avoid >10 unique values)"],
        key="encode_method",
    )

    cols_to_encode = st.multiselect(
        "Select columns to encode (identifier/name columns excluded automatically):",
        options=cat_cols,
        default=cat_cols,
        key="encode_cols",
    )

    if cols_to_encode:
        if st.button("🔡 Apply Encoding", key="encode_btn"):
            result = st.session_state.cleaned_df.copy()
            if encode_method.startswith("🏷️"):
                for col in cols_to_encode:
                    if col not in result.columns:
                        continue
                    result[col] = result[col].astype("category").cat.codes
                    result[col] = result[col].replace(-1, np.nan)
            else:
                high_card = [c for c in cols_to_encode if working[c].nunique() > 10]
                if high_card:
                    st.warning(
                        f"⚠️ High cardinality: {', '.join(high_card)}. "
                        "One-Hot will create many columns. Proceeding anyway...")
                existing = [c for c in cols_to_encode if c in result.columns]
                if existing:
                    result = pd.get_dummies(result, columns=existing, drop_first=False)
            st.session_state.cleaned_df = result
            _mark_done("encoding")
            st.rerun()

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def clean_data(original_df):
    """
    Main entry point called by app.py.

    Session-state contract:
      - cleaned_df   : current working copy of the dataframe
      - action_taken : True if at least one step has been applied
      - STEP_KEYS    : per-step boolean flags (set to True when applied, reset
                       to False by app.py's _reset_all_step_states())
    """
    if "cleaned_df" not in st.session_state or st.session_state.cleaned_df is None:
        st.session_state.cleaned_df   = original_df.copy()
        st.session_state.action_taken = False

    _init_step_states()

    _sub_header("🔹", "linear-gradient(135deg,#6366f1,#0ea5e9)",
                "Missing Value Treatment", done=_is_done("missing"))
    _handle_missing(original_df)
    st.markdown("<br>", unsafe_allow_html=True)

    _sub_header("🔁", "linear-gradient(135deg,#f59e0b,#ef4444)",
                "Duplicate Row Removal", done=_is_done("duplicates"))
    _handle_duplicates()
    st.markdown("<br>", unsafe_allow_html=True)

    _sub_header("📊", "linear-gradient(135deg,#0ea5e9,#6366f1)",
                "Outlier Detection & Treatment", done=_is_done("outliers"))
    _handle_outliers()
    st.markdown("<br>", unsafe_allow_html=True)

    _sub_header("📏", "linear-gradient(135deg,#16a34a,#0ea5e9)",
                "Feature Scaling / Normalisation", done=_is_done("scaling"))
    _handle_scaling()
    st.markdown("<br>", unsafe_allow_html=True)

    _sub_header("🔡", "linear-gradient(135deg,#a855f7,#6366f1)",
                "Categorical Encoding", done=_is_done("encoding"))
    _handle_encoding()

    return st.session_state.cleaned_df, st.session_state.action_taken