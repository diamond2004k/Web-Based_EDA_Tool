import streamlit as st
import pandas as pd


def profile_data(df):
    rows, cols = df.shape

    # ── Summary metric cards ───────────────────────────────────────────────────
    total_missing  = int(df.isnull().sum().sum())
    total_missing_pct = round(total_missing / df.size * 100, 1)
    complete_cols  = int((df.isnull().sum() == 0).sum())
    numeric_cols   = len(df.select_dtypes(include=["int64","float64"]).columns)
    cat_cols       = len(df.select_dtypes(include=["object","category"]).columns)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Missing",    f"{total_missing:,}",  f"{total_missing_pct}% of all cells", "#f59e0b"),
        (c2, "Complete Columns", str(complete_cols),    f"out of {cols}",                     "#4ade80"),
        (c3, "Numeric Columns",  str(numeric_cols),     "int / float",                        "#6366f1"),
        (c4, "Categorical Cols", str(cat_cols),         "object / category",                  "#0ea5e9"),
    ]
    for col_widget, title, value, sub, color in cards:
        with col_widget:
            st.markdown(f"""
            <div style='background:#0f172a;border:1px solid #1e293b;border-radius:14px;
                        padding:1.2rem 1.4rem;border-top:3px solid {color};'>
                <div style='color:#64748b;font-size:0.7rem;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:0.4rem;'>{title}</div>
                <div style='font-family:Space Mono,monospace;font-size:1.7rem;
                            color:{color};font-weight:700;line-height:1;'>{value}</div>
                <div style='color:#475569;font-size:0.75rem;
                            margin-top:0.3rem;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-column profile table ───────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6366f1;
                letter-spacing:0.15em;text-transform:uppercase;
                margin-bottom:0.75rem;'>Column Overview</div>
    """, unsafe_allow_html=True)

    profile_rows = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct   = round(missing_count / rows * 100, 1)
        dtype         = str(df[col].dtype)
        unique        = df[col].nunique()

        if df[col].dtype in ["int64","float64"]:
            sample = f"min={df[col].min():.1f}, max={df[col].max():.1f}"
        else:
            top = df[col].mode()[0] if not df[col].mode().empty else "—"
            sample = f"top='{top}'"

        status = "✅" if missing_count == 0 else "⚠️"
        profile_rows.append({
            "":        status,
            "Column":  col,
            "Type":    dtype,
            "Unique":  unique,
            "Missing": missing_count,
            "Missing %": f"{missing_pct}%",
            "Sample Info": sample,
        })

    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-column detail cards ────────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6366f1;
                letter-spacing:0.15em;text-transform:uppercase;
                margin-bottom:0.75rem;'>Column Details</div>
    """, unsafe_allow_html=True)

    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    ok_cols      = [c for c in df.columns if df[c].isnull().sum() == 0]

    if missing_cols:
        st.markdown(
            "<div style='color:#f59e0b;font-size:0.85rem;"
            "font-weight:600;margin-bottom:0.5rem;'>"
            f"⚠️  {len(missing_cols)} column(s) with missing values</div>",
            unsafe_allow_html=True)

        for col in missing_cols:
            mc  = df[col].isnull().sum()
            pct = mc / rows * 100
            bar_w = min(int(pct * 2), 100)

            st.markdown(f"""
            <div style='background:#0f172a;border:1px solid #f59e0b33;
                        border-left:3px solid #f59e0b;border-radius:0 12px 12px 0;
                        padding:0.9rem 1.2rem;margin-bottom:0.6rem;'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:0.5rem;'>
                    <span style='color:#e2e8f0;font-weight:600;
                                 font-size:0.9rem;'>⚠️ {col}</span>
                    <span style='color:#f59e0b;font-family:Space Mono,monospace;
                                 font-size:0.8rem;'>{mc} missing ({pct:.1f}%)</span>
                </div>
                <div style='background:#1e293b;border-radius:4px;height:5px;
                            margin-bottom:0.5rem;'>
                    <div style='background:#f59e0b;height:5px;border-radius:4px;
                                width:{bar_w}%;'></div>
                </div>
                <div style='color:#64748b;font-size:0.78rem;'>
                    Type: {df[col].dtype}  ·
                    Unique values: {df[col].nunique()}  ·
                    Missing values can reduce analysis reliability.
                </div>
            </div>""", unsafe_allow_html=True)

    if ok_cols:
        with st.expander(
                f"✅  {len(ok_cols)} complete column(s) — no missing values",
                expanded=False):
            for col in ok_cols:
                st.markdown(f"""
                <div style='background:#0f172a;border:1px solid #4ade8033;
                            border-left:3px solid #4ade80;
                            border-radius:0 10px 10px 0;
                            padding:0.7rem 1.1rem;margin-bottom:0.5rem;'>
                    <span style='color:#4ade80;font-weight:600;
                                 font-size:0.875rem;'>✔ {col}</span>
                    <span style='color:#475569;font-size:0.78rem;
                                 margin-left:1rem;'>
                        {df[col].dtype}  ·  {df[col].nunique()} unique
                    </span>
                </div>""", unsafe_allow_html=True)