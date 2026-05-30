import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

PLOTS_DIR       = "reports/plots"          # dark versions — shown in Streamlit UI
PLOTS_DIR_WHITE = "reports/plots_white"    # white versions — embedded in PDF
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR_WHITE, exist_ok=True)

# ── Per-chart explanations ─────────────────────────────────────────────────────
CHART_EXPLANATIONS = {
    "Histogram": {
        "what":   "A histogram groups values into bins and shows how many records fall in each range.",
        "use":    "Use it to understand the overall distribution — whether data is symmetric, skewed left/right, or has multiple peaks (bimodal).",
        "tip":    "The overlaid KDE curve smooths the distribution so you can spot the shape even when bins are noisy.",
        "watch":  "Watch for long tails (skew) or gaps between bars — these may indicate outliers or data-entry errors.",
    },
    "Line Plot": {
        "what":   "A line plot connects each data point in row order, showing how values change across the dataset.",
        "use":    "Best for ordered or time-series data where the sequence of records matters.",
        "tip":    "Sudden spikes or flat stretches can indicate anomalies, sensor errors, or batch effects in the data.",
        "watch":  "If your data has no meaningful row order, a histogram or box plot will be more informative.",
    },
    "Box Plot": {
        "what":   "A box plot summarises the distribution using five statistics: minimum, Q1 (25th percentile), median, Q3 (75th percentile), and maximum.",
        "use":    "Use it to quickly spot the spread, skew, and potential outliers (dots beyond the whiskers).",
        "tip":    "The box covers the middle 50% of data (IQR). A median line closer to Q1 means the data is right-skewed.",
        "watch":  "Points plotted beyond the whiskers are flagged as potential outliers — investigate them individually.",
    },
    "KDE Plot": {
        "what":   "A Kernel Density Estimate (KDE) is a smoothed, continuous version of a histogram that estimates the probability density of the data.",
        "use":    "Use it to understand the shape of the distribution without the noise of bin-size choices in histograms.",
        "tip":    "Multiple peaks (modes) in the KDE suggest the column may contain data from distinct sub-groups.",
        "watch":  "KDE can extend beyond the actual data range (e.g. below zero for counts). Always check the axis limits.",
    },
    "Violin Plot": {
        "what":   "A violin plot combines a box plot with a mirrored KDE, showing both the summary statistics and the full distribution shape.",
        "use":    "Ideal when you want to see where data is densest, not just the quartiles.",
        "tip":    "Wide sections of the violin mean many values cluster there; narrow sections mean few values.",
        "watch":  "Very thin violins can indicate low variance or a small number of unique values — worth checking with value_counts().",
    },
}


def _explanation_html(chart_type):
    e = CHART_EXPLANATIONS.get(chart_type, {})
    rows = [
        ("🔎 What it shows", e.get("what", "")),
        ("✅ When to use",    e.get("use",  "")),
        ("💡 Tip",            e.get("tip",  "")),
        ("⚠️ Watch out for", e.get("watch", "")),
    ]
    items = "".join(
        f"""<div style='display:flex;gap:0.6rem;margin-bottom:0.55rem;align-items:flex-start;'>
              <span style='color:#6366f1;font-weight:600;min-width:160px;font-size:0.82rem;'>{label}</span>
              <span style='color:#94a3b8;font-size:0.82rem;line-height:1.55;'>{text}</span>
           </div>"""
        for label, text in rows
    )
    return f"""
    <div style='background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                padding:1.1rem 1.4rem;margin-top:0.75rem;margin-bottom:0.5rem;'>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6366f1;
                    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.75rem;'>
            📝 Chart Explanation
        </div>
        {items}
    </div>
    """


# ── White-background style — used for PDF charts AND individual downloads ──────
WHITE_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#cbd5e1",
    "axes.labelcolor":  "#334155",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "grid.color":       "#e2e8f0",
    "text.color":       "#0f172a",
}

# ── Dark style for Streamlit UI ────────────────────────────────────────────────
DARK_STYLE = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "text.color":       "#e2e8f0",
}


def _build_chart(ax, fig, chart_type, series, color):
    """Render the chosen chart type onto the given axes."""
    if chart_type == "Histogram":
        sns.histplot(series, kde=True, color=color, ax=ax)
    elif chart_type == "Line Plot":
        ax.plot(series.values, color=color, linewidth=1.5)
    elif chart_type == "Box Plot":
        sns.boxplot(x=series, color=color, ax=ax)
    elif chart_type == "KDE Plot":
        sns.kdeplot(series, color=color, fill=True, ax=ax, alpha=0.4)
    elif chart_type == "Violin Plot":
        sns.violinplot(x=series, color=color, ax=ax)


def show_eda(df):
    st.markdown("## 📈 Exploratory Data Analysis")

    # Clear old plots from both directories on each EDA render
    for d in (PLOTS_DIR, PLOTS_DIR_WHITE):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for EDA.")
        return

    for col in numeric_cols:
        with st.container():
            st.markdown(f"""
            <div style='padding:0.75rem 1rem;background:#0f172a;border-left:3px solid #6366f1;
                        border-radius:0 10px 10px 0;margin-bottom:1rem;'>
                <h3 style='margin:0;color:#e2e8f0;font-size:1.1rem;'>🔹 {col}</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                chart_type = st.selectbox(
                    "Chart type",
                    ["Histogram", "Line Plot", "Box Plot", "KDE Plot", "Violin Plot"],
                    key=f"{col}_chart"
                )
            with col2:
                # The user's chosen color is used for ALL versions:
                # dark UI chart, white PDF chart, and white individual download.
                color = st.color_picker("Color", "#6366f1", key=f"{col}_color")

            safe_name = col.replace(" ", "_").replace("/", "_")
            plot_name = f"{safe_name}_{chart_type.replace(' ', '_')}.png"

            # ── Dark version: shown in the Streamlit UI only ───────────────────
            with plt.rc_context(DARK_STYLE):
                fig_dark, ax_dark = plt.subplots(figsize=(8, 4))
                fig_dark.patch.set_facecolor("#0f172a")
                ax_dark.set_facecolor("#1e293b")
                ax_dark.tick_params(colors="#94a3b8")
                ax_dark.xaxis.label.set_color("#94a3b8")
                ax_dark.yaxis.label.set_color("#94a3b8")
                ax_dark.title.set_color("#e2e8f0")
                for spine in ax_dark.spines.values():
                    spine.set_edgecolor("#334155")

                _build_chart(ax_dark, fig_dark, chart_type, df[col], color)
                ax_dark.set_title(f"{chart_type} — {col}", fontsize=13, pad=12)
                plt.tight_layout()

                dark_path = os.path.join(PLOTS_DIR, plot_name)
                fig_dark.savefig(dark_path, dpi=150, bbox_inches="tight",
                                 facecolor=fig_dark.get_facecolor())
                st.pyplot(fig_dark)
                plt.close(fig_dark)

            # ── White version: used for BOTH PDF embedding AND individual download.
            #    Key fix: uses `color` (the user's picker value), NOT a hardcoded hex.
            with plt.rc_context(WHITE_STYLE):
                fig_white, ax_white = plt.subplots(figsize=(8, 4))
                fig_white.patch.set_facecolor("white")
                ax_white.set_facecolor("#f8fafc")

                _build_chart(ax_white, fig_white, chart_type, df[col], color)  # ← user color

                ax_white.set_title(f"{chart_type} — {col}", fontsize=13, pad=12,
                                   color="#0f172a")
                ax_white.tick_params(colors="#64748b")
                for spine in ax_white.spines.values():
                    spine.set_edgecolor("#cbd5e1")
                plt.tight_layout()

                # Save to plots_white so the PDF generator picks it up
                white_path = os.path.join(PLOTS_DIR_WHITE, plot_name)
                fig_white.savefig(white_path, dpi=150, bbox_inches="tight",
                                  facecolor="white")

                # Build the download buffer from the same white figure
                dl_buf = io.BytesIO()
                fig_white.savefig(dl_buf, format="png", dpi=150,
                                  bbox_inches="tight", facecolor="white")
                dl_buf.seek(0)
                plt.close(fig_white)

            # Download button: white background, user's chosen color ─────────
            st.download_button(
                label=f"⬇️ Download {col} chart",
                data=dl_buf,
                file_name=f"{col}_{chart_type.replace(' ', '_')}.png",
                mime="image/png",
                key=f"{col}_download"
            )

            st.markdown(_explanation_html(chart_type), unsafe_allow_html=True)
            st.markdown(
                "<hr style='border-color:#1e293b;margin:1.5rem 0;'>",
                unsafe_allow_html=True
            )