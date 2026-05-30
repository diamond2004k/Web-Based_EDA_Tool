import streamlit as st
from modules.data_upload import upload_data
from modules.data_profiling import profile_data
from modules.data_cleaning import clean_data
from modules.eda_visualization import show_eda
from modules.data_quality import calculate_quality_score
from modules.report_generator import generate_report
from modules.report_generator1 import generate_pdf_report

st.set_page_config(
    page_title="AutoEDA Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state init ─────────────────────────────────────────────────────────
for key, default in {
    "original_df":   None,
    "cleaned_df":    None,
    "action_taken":  False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #050d1a;
    color: #e2e8f0;
}
.stApp { background: #050d1a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 3rem 4rem 3rem; max-width: 1120px; }

.hero {
    background: linear-gradient(135deg,#060e1f 0%,#0d1535 40%,#080f1e 100%);
    border: 1px solid #1e3a5f; border-radius: 24px;
    padding: 3.5rem 4rem; margin: 2rem 0 2.5rem 0;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:320px; height:320px;
    background:radial-gradient(circle,#6366f125 0%,transparent 70%);
    border-radius:50%;
}
.hero-tag {
    font-family:'Space Mono',monospace; font-size:0.65rem;
    color:#6366f1; letter-spacing:0.3em; text-transform:uppercase;
    margin-bottom:1.1rem;
}
.hero h1 {
    font-size:3rem; font-weight:600;
    background:linear-gradient(100deg,#e2e8f0 20%,#818cf8 70%,#38bdf8 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 0.9rem 0; line-height:1.15;
}
.hero p { color:#475569; font-size:1rem; margin:0; font-weight:300; line-height:1.7; }
.hero-badges { display:flex; gap:0.6rem; margin-top:1.5rem; flex-wrap:wrap; }
.badge {
    background:#0f172a; border:1px solid #1e3a5f; border-radius:999px;
    padding:0.25rem 0.85rem; font-size:0.72rem; color:#64748b;
    font-family:'Space Mono',monospace;
}

.step-header {
    display:flex; align-items:center; gap:1rem;
    margin:2.5rem 0 1.5rem 0; padding:1rem 1.4rem;
    background:#0a1628; border:1px solid #1e293b;
    border-left:3px solid #6366f1; border-radius:0 14px 14px 0;
}
.step-num  { font-family:'Space Mono',monospace; font-size:0.6rem; color:#6366f1; letter-spacing:0.2em; text-transform:uppercase; }
.step-title { font-size:1.1rem; font-weight:600; color:#e2e8f0; margin:0; }
.step-icon {
    width:38px; height:38px;
    background:linear-gradient(135deg,#6366f1,#0ea5e9);
    border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:1.1rem; flex-shrink:0;
}

.stat-card {
    background:#0a1628; border:1px solid #1e293b;
    border-radius:14px; padding:1.3rem 1.5rem; text-align:center;
}

.score-wrapper {
    background:linear-gradient(135deg,#090f1e,#0f1a35);
    border:1px solid #1e293b; border-radius:20px;
    padding:2.5rem; text-align:center; margin-bottom:1rem;
}
.score-number {
    font-family:'Space Mono',monospace; font-size:4.5rem; font-weight:700;
    background:linear-gradient(135deg,#818cf8,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1;
}
.score-label {
    color:#334155; margin-top:0.5rem; text-transform:uppercase;
    letter-spacing:0.15em; font-family:'Space Mono',monospace; font-size:0.65rem;
}
.score-badge {
    display:inline-block; margin-top:1rem; padding:0.4rem 1.2rem;
    border-radius:999px; font-size:0.78rem; font-weight:600; letter-spacing:0.06em;
}
.badge-excellent {background:#052e16;color:#4ade80;border:1px solid #166534;}
.badge-good      {background:#042f2e;color:#2dd4bf;border:1px solid #0f766e;}
.badge-fair      {background:#431407;color:#fb923c;border:1px solid #9a3412;}
.badge-poor      {background:#3b0764;color:#e879f9;border:1px solid #7e22ce;}

.expl-item {
    background:#080f1e; border-left:2px solid #6366f1;
    border-radius:0 8px 8px 0; padding:0.75rem 1.1rem;
    margin-bottom:0.5rem; color:#64748b; font-size:0.875rem; line-height:1.6;
}

.dl-card {
    background:#0a1628; border:1px solid #1e293b; border-radius:16px;
    padding:1.6rem; text-align:center; height:100%;
}
.dl-card-icon  { font-size:2rem; margin-bottom:0.75rem; }
.dl-card-title { color:#e2e8f0; font-weight:600; font-size:0.95rem; margin-bottom:0.3rem; }
.dl-card-sub   { color:#475569; font-size:0.78rem; margin-bottom:1rem; }

.stButton > button {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-family:'DM Sans',sans-serif !important;
    font-weight:500 !important; padding:0.55rem 1.2rem !important;
    transition:opacity 0.2s !important; width:100%;
}
.stButton > button:hover { opacity:0.85 !important; }

.stDownloadButton > button {
    background:#0f172a !important; color:#6366f1 !important;
    border:1px solid #6366f1 !important; border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
    width:100% !important; transition:all 0.2s !important;
}
.stDownloadButton > button:hover {
    background:#6366f115 !important; border-color:#818cf8 !important;
}

.stSelectbox > div > div {
    background:#0a1628 !important; border:1px solid #1e3a5f !important;
    border-radius:10px !important; color:#e2e8f0 !important;
}
.stFileUploader {
    background:#0a1628 !important; border:2px dashed #1e3a5f !important;
    border-radius:16px !important; padding:1.5rem !important;
}
.stCheckbox label { color:#64748b !important; font-size:0.875rem !important; }
div[data-testid="stMetricValue"] {
    font-family:'Space Mono',monospace !important;
    font-size:2.2rem !important; color:#6366f1 !important;
}
div[data-testid="stAlert"] { border-radius:12px !important; border:none !important; }
.stDataFrame { border-radius:12px !important; overflow:hidden !important; }
hr { border-color:#1e293b !important; margin:2rem 0 !important; }
.stRadio label { color:#94a3b8 !important; font-size:0.875rem !important; }
details {
    background:#0a1628 !important; border:1px solid #1e293b !important;
    border-radius:12px !important; padding:0.25rem 0 !important;
}
details summary { color:#94a3b8 !important; font-size:0.875rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper ─────────────────────────────────────────────────────────────────────
def step_header(icon, step_num, title, accent="#6366f1"):
    st.markdown(f"""
    <div class="step-header" style="border-left-color:{accent};">
        <div class="step-icon"
             style="background:linear-gradient(135deg,{accent},{accent}99);">
            {icon}
        </div>
        <div>
            <div class="step-num">Step {step_num}</div>
            <div class="step-title">{title}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _reset_all_step_states():
    """
    Fully resets all cleaning step states, cleaned_df, and action_taken.
    Called both on new file upload and on the Reset button.
    """
    from modules.data_cleaning import STEP_KEYS
    for key in STEP_KEYS.values():
        st.session_state[key] = False
    st.session_state.cleaned_df   = st.session_state.original_df.copy()
    st.session_state.action_taken = False


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Explainable AutoEDA Platform</div>
    <h1>Automated Data<br>Intelligence</h1>
    <p>Upload a dataset and get instant profiling, explainable cleaning,
    quality scoring, interactive visualisations, and a full downloadable report.</p>
    <div class="hero-badges">
        <span class="badge">⚡ Auto Cleaning</span>
        <span class="badge">📊 EDA Charts</span>
        <span class="badge">🎯 Quality Score</span>
        <span class="badge">📄 PDF Report</span>
        <span class="badge">🔡 Encoding</span>
        <span class="badge">📏 Scaling</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
step_header("📂", "01", "Upload Dataset", "#6366f1")
df = upload_data()

if df is not None:

    # ── Detect a genuinely NEW file upload and reset everything ────────────────
    # We compare by shape + column names + a hash of the first few values.
    # This avoids resetting on every rerun when the same file is loaded.
    def _df_fingerprint(d):
        try:
            return (d.shape, tuple(d.columns), tuple(d.iloc[0]) if len(d) > 0 else ())
        except Exception:
            return None

    current_fp  = _df_fingerprint(df)
    stored_fp   = _df_fingerprint(st.session_state.original_df) if st.session_state.original_df is not None else None

    if current_fp != stored_fp:
        # Genuinely new file — reset everything including step states
        st.session_state.original_df = df.copy()
        _reset_all_step_states()

    rows, cols_count  = df.shape
    missing_total     = int(df.isnull().sum().sum())
    dup_total         = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    for widget, lbl, val, color in [
        (c1, "Rows",          f"{rows:,}",          "#6366f1"),
        (c2, "Columns",       f"{cols_count:,}",    "#0ea5e9"),
        (c3, "Missing Cells", f"{missing_total:,}", "#f59e0b"),
        (c4, "Duplicates",    f"{dup_total:,}",     "#ef4444"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='color:#334155;font-size:0.68rem;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:0.35rem;'>{lbl}</div>
                <div style='font-family:Space Mono,monospace;font-size:1.9rem;
                            color:{color};font-weight:700;line-height:1;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — PROFILING
    # ══════════════════════════════════════════════════════════════════════════
    step_header("🔍", "02", "Data Profiling & Validation", "#0ea5e9")
    profile_data(st.session_state.original_df)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — CLEANING
    # ══════════════════════════════════════════════════════════════════════════
    step_header("🧹", "03", "Explainable Data Cleaning", "#a855f7")

    orig = st.session_state.original_df
    curr = st.session_state.cleaned_df
    orig_miss  = int(orig.isnull().sum().sum())
    curr_miss  = int(curr.isnull().sum().sum())
    orig_dups  = int(orig.duplicated().sum())
    curr_dups  = int(curr.duplicated().sum())

    if st.session_state.action_taken:
        st.info(
            f"🔄 **Cleaning in progress** — "
            f"Missing: {orig_miss} → {curr_miss}  |  "
            f"Duplicates: {orig_dups} → {curr_dups}  |  "
            f"Rows: {len(orig):,} → {len(curr):,}  |  "
            f"Columns: {orig.shape[1]} → {curr.shape[1]}"
        )

    # ── Reset button — resets cleaned_df AND all step states ──────────────────
    if st.session_state.action_taken:
        if st.button("🔄 Reset All Cleaning (start over)", key="reset_btn"):
            _reset_all_step_states()
            st.rerun()

    clean_data(st.session_state.original_df)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4 — QUALITY SCORE
    # ══════════════════════════════════════════════════════════════════════════
    step_header("🎯", "04", "Dataset Quality Score", "#4ade80")

    cleaned_df = st.session_state.cleaned_df
    score, label, explanations = calculate_quality_score(cleaned_df)

    badge_class = {
        "Excellent": "badge-excellent", "Good": "badge-good",
        "Fair": "badge-fair",           "Poor": "badge-poor",
    }.get(label, "badge-fair")

    qa, qb = st.columns([1, 1.6])
    with qa:
        st.markdown(f"""
        <div class="score-wrapper">
            <div class="score-number">{score}</div>
            <div class="score-label">out of 100</div>
            <div><span class="score-badge {badge_class}">{label}</span></div>
        </div>""", unsafe_allow_html=True)

        orig_df   = st.session_state.original_df
        clean_df  = st.session_state.cleaned_df
        o_miss    = int(orig_df.isnull().sum().sum())
        c_miss    = int(clean_df.isnull().sum().sum())
        o_dups    = int(orig_df.duplicated().sum())
        c_dups    = int(clean_df.duplicated().sum())
        m_color   = "#4ade80" if c_miss < o_miss else "#f59e0b"
        d_color   = "#4ade80" if c_dups < o_dups else "#f59e0b"

        st.markdown(f"""
        <div style='background:#080f1e;border:1px solid #1e293b;
                    border-radius:12px;padding:1rem 1.2rem;'>
            <div style='font-family:Space Mono,monospace;font-size:0.6rem;
                        color:#334155;letter-spacing:0.15em;text-transform:uppercase;
                        margin-bottom:0.75rem;'>Quick Comparison</div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:0.4rem;text-align:center;font-size:0.72rem;
                        color:#475569;margin-bottom:0.4rem;'>
                <span></span>
                <span style='color:#64748b;'>Before</span>
                <span style='color:#64748b;'>After</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:0.4rem;text-align:center;font-size:0.8rem;
                        margin-bottom:0.3rem;'>
                <span style='color:#475569;text-align:left;'>Missing</span>
                <span style='color:#f59e0b;font-family:Space Mono,monospace;'>{o_miss}</span>
                <span style='color:{m_color};font-family:Space Mono,monospace;'>{c_miss}</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:0.4rem;text-align:center;font-size:0.8rem;
                        margin-bottom:0.3rem;'>
                <span style='color:#475569;text-align:left;'>Duplicates</span>
                <span style='color:#f59e0b;font-family:Space Mono,monospace;'>{o_dups}</span>
                <span style='color:{d_color};font-family:Space Mono,monospace;'>{c_dups}</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:0.4rem;text-align:center;font-size:0.8rem;'>
                <span style='color:#475569;text-align:left;'>Rows</span>
                <span style='color:#64748b;font-family:Space Mono,monospace;'>
                    {len(orig_df):,}</span>
                <span style='color:#64748b;font-family:Space Mono,monospace;'>
                    {len(clean_df):,}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with qb:
        st.markdown("""
        <div style='font-family:Space Mono,monospace;font-size:0.6rem;
                    color:#334155;letter-spacing:0.15em;text-transform:uppercase;
                    margin-bottom:0.75rem;'>Score Breakdown</div>
        """, unsafe_allow_html=True)
        for exp in explanations:
            st.markdown(f'<div class="expl-item">→ {exp}</div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5 — EDA
    # ══════════════════════════════════════════════════════════════════════════
    step_header("📈", "05", "Exploratory Data Analysis", "#0ea5e9")
    show_eda(st.session_state.cleaned_df)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6 — DOWNLOADS
    #  PDF/TXT/CSV are generated here from the FINAL cleaned_df so they always
    #  reflect every preprocessing step the user has applied.
    # ══════════════════════════════════════════════════════════════════════════
    step_header("⬇️", "06", "Download Reports & Data", "#f59e0b")

    st.markdown("""
    <div style='background:#080f1e;border:1px solid #1e293b;border-radius:12px;
                padding:0.9rem 1.2rem;margin-bottom:1.5rem;display:flex;
                align-items:center;gap:0.75rem;'>
        <span style='font-size:1rem;'>💡</span>
        <span style='color:#475569;font-size:0.85rem;'>
            EDA charts from Step 05 are embedded in the PDF.
            Apply all desired cleaning steps in Step 03 before downloading
            to get the fully processed dataset and report.
        </span>
    </div>
    """, unsafe_allow_html=True)

    cleaning_summary = (
        "Cleaning actions were applied based on user approval."
        if st.session_state.action_taken else
        "No cleaning actions were applied."
    )

    # Snapshot the final cleaned state right now, at render time.
    # This guarantees the PDF reflects every cleaning step applied so far.
    final_original_df = st.session_state.original_df
    final_cleaned_df  = st.session_state.cleaned_df
    final_score, final_label, final_explanations = calculate_quality_score(final_cleaned_df)

    pdf_path = generate_pdf_report(
        final_score, final_label, final_explanations, cleaning_summary,
        original_df=final_original_df,
        cleaned_df=final_cleaned_df,
    )
    report_text = generate_report(
        final_score, final_label, final_explanations,
        st.session_state.action_taken,
        original_df=final_original_df,
        cleaned_df=final_cleaned_df,
    )
    csv_data = final_cleaned_df.to_csv(index=False).encode("utf-8")

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-card-icon">📄</div>
            <div class="dl-card-title">Full PDF Report</div>
            <div class="dl-card-sub">11 sections · Charts · Explanations</div>
        </div>""", unsafe_allow_html=True)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF", data=f,
                file_name="AutoEDA_Report.pdf",
                mime="application/pdf", key="dl_pdf")

    with dl2:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-card-icon">📝</div>
            <div class="dl-card-title">Text Report</div>
            <div class="dl-card-sub">Plain text · All stats · Recommendations</div>
        </div>""", unsafe_allow_html=True)
        st.download_button(
            label="Download TXT", data=report_text,
            file_name="AutoEDA_Report.txt",
            mime="text/plain", key="dl_txt")

    with dl3:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-card-icon">📊</div>
            <div class="dl-card-title">Cleaned Dataset</div>
            <div class="dl-card-sub">CSV · All cleaning applied</div>
        </div>""", unsafe_allow_html=True)
        st.download_button(
            label="Download CSV", data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv", key="dl_csv")