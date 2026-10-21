"""
app.py – Spaceship Titanic: Passenger Transportation Prediction
================================================================
Streamlit dashboard for the B.Tech CSE Data Science project.

Pipeline mirrors the notebook exactly (see preprocessing.py for details).
The notebook itself is never modified.

Run with:
    streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

from preprocessing import (
    load_and_preprocess,
    train_models,
    evaluate_models,
    get_confusion_matrices,
    preprocess_single_input,
    FEATURE_COLS,
    OBJECT_CAT_COLS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Spaceship Titanic – ML Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.92rem; }

/* ── General body ───────────────────────────────── */
h1 { color: #1e293b; font-weight: 800; }
h2 { color: #1e293b; font-weight: 700; }
h3 { color: #334155; font-weight: 700; }

/* ── KPI tile ───────────────────────────────────── */
.kpi-tile {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    border-left: 4px solid #6366f1;
    margin-bottom: 6px;
}
.kpi-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
             text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #1e293b; line-height: 1.1; }
.kpi-sub   { font-size: 0.76rem; color: #94a3b8; margin-top: 2px; }

/* ── Section label ──────────────────────────────── */
.section-tag {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6366f1; margin-bottom: 10px;
}

/* ── Verdict / result banner ────────────────────── */
.verdict-wrap {
    border-radius: 14px; padding: 28px 32px; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.09);
}
.verdict-yes {
    background: linear-gradient(135deg,#d1fae5,#a7f3d0);
    border: 2px solid #34d399;
}
.verdict-no {
    background: linear-gradient(135deg,#fee2e2,#fecaca);
    border: 2px solid #f87171;
}
.verdict-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.10em;
                 text-transform: uppercase; color: #475569; margin-bottom: 8px; }
.verdict-result { font-size: 2.1rem; font-weight: 900; line-height: 1.15; }
.verdict-sub    { font-size: 0.84rem; color: #475569; margin-top: 8px; }
.verdict-yes .verdict-result { color: #065f46; }
.verdict-no  .verdict-result { color: #7f1d1d; }

/* ── Probability bar ────────────────────────────── */
.pbar-bg   { background:#e2e8f0; border-radius:6px; height:10px;
             width:100%; overflow:hidden; margin:8px 0 4px; }
.pbar-fill-yes { background:#34d399; height:10px; border-radius:6px; }
.pbar-fill-no  { background:#f87171; height:10px; border-radius:6px; }
.pbar-labels   { display:flex; justify-content:space-between;
                 font-size:0.68rem; color:#94a3b8; }

/* ── Feature pills ──────────────────────────────── */
.pill-row  { display:flex; flex-wrap:wrap; gap:7px; margin-top:8px; }
.pill      { background:#f1f5f9; border-radius:20px; padding:4px 12px;
             font-size:0.76rem; color:#334155; border:1px solid #e2e8f0; }
.pill-hi   { background:#ede9fe; border:1px solid #c4b5fd; color:#4c1d95;
             font-weight:700; }

/* ── Model context table ────────────────────────── */
.ctx-table td { padding: 5px 10px; font-size: 0.83rem; }
.ctx-table tr:nth-child(even) td { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Cache — load once per session
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading dataset and training models — please wait…")
def load_everything():
    (X_train, X_test, y_train, y_test,
     encoders, scaler, enc_cats,
     spending_meds, all_cols) = load_and_preprocess(DATA_PATH)

    models    = train_models(X_train, y_train)
    results   = evaluate_models(models, X_test, y_test)
    conf_mats = get_confusion_matrices(models, X_test, y_test)
    raw_df    = pd.read_csv(DATA_PATH)

    return dict(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        encoders=encoders, scaler=scaler, enc_cats=enc_cats,
        spending_meds=spending_meds, all_cols=all_cols,
        models=models, results=results, conf_mats=conf_mats, raw_df=raw_df,
    )


if not os.path.exists(DATA_PATH):
    st.error(f"**Dataset not found.** Expected: `{DATA_PATH}`")
    st.stop()

D = load_everything()   # D = data dict


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚀 Spaceship Titanic")
    st.markdown("**B.Tech CSE · Data Science Project**")
    st.markdown("---")
    section = st.radio(
        "Navigate",
        [
            "🏠  Project Overview",
            "📊  EDA & Insights",
            "🤖  Model Performance",
            "🔮  Predict a Passenger",
            "📝  Findings & Conclusion",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Dataset: Kaggle Spaceship Titanic<br>"
        "Models: LR · KNN · NB · DT<br>"
        "Best model: **KNN (K=9)**</small>",
        unsafe_allow_html=True,
    )


# ── Helper: KPI tile ──────────────────────────────────────────────────────────
def kpi(label, value, sub="", accent="#6366f1"):
    st.markdown(
        f"""<div class="kpi-tile" style="border-left-color:{accent};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ① PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section == "🏠  Project Overview":

    st.title("🚀 Spaceship Titanic")
    st.markdown(
        "#### Passenger Transportation Prediction Using Machine Learning"
    )
    st.markdown(
        "> *In the year 2912, the Spaceship Titanic collided with a spacetime anomaly, "
        "transporting nearly half its passengers to an alternate dimension. "
        "This project builds and compares four ML classifiers to predict "
        "which passengers were transported.*"
    )

    st.markdown("---")

    # ── KPI row ───────────────────────────────────────────────────────────────
    raw_df = D["raw_df"]
    total        = len(raw_df)
    transported  = int(raw_df["Transported"].sum())
    not_transp   = total - transported
    missing_tot  = int(raw_df.isnull().sum().sum())
    best_acc_row = max(D["results"], key=lambda r: r["Accuracy"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total Passengers", f"{total:,}", "records in training set", "#6366f1")
    with c2:
        kpi("Transported", f"{transported:,}", f"{transported/total*100:.1f}% of total", "#10b981")
    with c3:
        kpi("Not Transported", f"{not_transp:,}", f"{not_transp/total*100:.1f}% of total", "#f59e0b")
    with c4:
        kpi("Best Model Accuracy",
            f"{best_acc_row['Accuracy']*100:.2f}%",
            f"{best_acc_row['Model']}", "#6366f1")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        with st.expander("📌 Problem Statement", expanded=True):
            st.markdown(
                """
                The **challenge** is to predict which passengers aboard the Spaceship Titanic
                were transported to an alternate dimension based on their passenger record.

                The dataset contains missing values and a mix of numerical and categorical
                features — requiring careful preprocessing before modelling.
                """
            )
        with st.expander("🎯 Project Objectives", expanded=True):
            st.markdown(
                """
                - Explore patterns in passenger transportation through EDA.
                - Handle missing values and engineer useful features.
                - Build and compare **four classification models**.
                - Evaluate using Accuracy, Precision, Recall, and F1-Score.
                - Identify and select the best-performing model.
                """
            )

    with col_r:
        st.markdown("##### 📋 Dataset Snapshot")
        st.dataframe(raw_df.head(8), use_container_width=True, height=280)
        st.caption("First 8 rows of `train.csv` — 8,693 passengers, 14 columns.")

    st.markdown("---")
    st.subheader("🔍 Missing Value Summary")

    missing_count = raw_df.isnull().sum()
    missing_pct   = (missing_count / total * 100).round(2)
    miss_df = pd.DataFrame({
        "Missing Count":   missing_count,
        "Missing %": missing_pct,
    })
    miss_df = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)

    m1, m2 = st.columns([2, 1])
    with m1:
        st.dataframe(miss_df, use_container_width=True)
    with m2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Every affected column has fewer than **2.5% missing values** — "
            "imputed with training-set median (numerical) or mode (categorical)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# ② EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊  EDA & Insights":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visual exploration of the training dataset to understand passenger patterns.")
    st.markdown("---")

    raw_df = D["raw_df"]

    # ── Chart helper ──────────────────────────────────────────────────────────
    PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#f87171", "#38bdf8"]

    def show(fig):
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    r1a, r1b = st.columns(2, gap="large")

    with r1a:
        st.subheader("1. Transportation Distribution")
        counts = raw_df["Transported"].value_counts().sort_index()
        labels = ["Not Transported", "Transported"]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        bars = ax.bar(labels, counts.values, color=[PALETTE[3], PALETTE[2]],
                      width=0.45, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Passengers")
        ax.set_title("Transportation Outcome", fontweight="bold", fontsize=11)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylim(0, max(counts.values) * 1.15)
        show(fig)
        st.caption(
            f"Dataset is nearly balanced: {counts[True]:,} transported "
            f"({counts[True]/len(raw_df)*100:.1f}%) vs {counts[False]:,} not transported "
            f"({counts[False]/len(raw_df)*100:.1f}%)."
        )

    with r1b:
        st.subheader("2. HomePlanet Distribution")
        hp = raw_df["HomePlanet"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
        ax2.bar(hp.index, hp.values, color=PALETTE[:3], width=0.45,
                edgecolor="white", linewidth=1.5)
        ax2.set_ylabel("Passengers")
        ax2.set_title("Passengers by Home Planet", fontweight="bold", fontsize=11)
        for i, v in enumerate(hp.values):
            ax2.text(i, v + 40, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax2.set_ylim(0, max(hp.values) * 1.15)
        show(fig2)
        st.caption("**Earth** has the most passengers, followed by Europa and Mars.")

    st.markdown("---")

    # ── Row 2 ──────────────────────────────────────────────────────────────────
    r2a, r2b = st.columns(2, gap="large")

    with r2a:
        st.subheader("3. Age Distribution")
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.8))
        ax3.hist(raw_df["Age"].dropna(), bins=30, color=PALETTE[0],
                 edgecolor="white", linewidth=0.8, alpha=0.85)
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Passengers")
        ax3.set_title("Age Distribution", fontweight="bold", fontsize=11)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax3.axvline(raw_df["Age"].median(), color="#f59e0b", linestyle="--",
                    linewidth=1.5, label=f"Median = {raw_df['Age'].median():.0f}")
        ax3.legend(fontsize=8)
        show(fig3)
        st.caption(
            f"Ages range 0–79, median **{raw_df['Age'].median():.0f} yrs**. "
            "Distribution is slightly right-skewed."
        )

    with r2b:
        st.subheader("4. RoomService Spending")
        rs = raw_df[raw_df["RoomService"] > 0]["RoomService"]
        fig4, ax4 = plt.subplots(figsize=(5.5, 3.8))
        ax4.hist(rs, bins=40, color=PALETTE[1], edgecolor="white",
                 linewidth=0.8, alpha=0.85)
        ax4.set_xlabel("Amount (credits)")
        ax4.set_ylabel("Passengers")
        ax4.set_title("RoomService Spending (> 0 only)", fontweight="bold", fontsize=11)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        show(fig4)
        st.caption(
            "Spending is **highly right-skewed** — most spend 0, a few spend up to ~14,000 credits."
        )

    st.markdown("---")

    # ── Row 3 ──────────────────────────────────────────────────────────────────
    r3a, r3b = st.columns(2, gap="large")

    with r3a:
        st.subheader("5. CryoSleep vs Transportation")
        cryo_t = (
            raw_df.dropna(subset=["CryoSleep"])
                  .groupby(["CryoSleep", "Transported"])
                  .size().unstack(fill_value=0)
        )
        fig5, ax5 = plt.subplots(figsize=(5.5, 3.8))
        cryo_t.plot(kind="bar", ax=ax5, color=[PALETTE[3], PALETTE[2]],
                    edgecolor="white", linewidth=1.2, width=0.6)
        ax5.set_xticklabels(["Awake (False)", "In CryoSleep (True)"], rotation=0)
        ax5.set_ylabel("Passengers")
        ax5.set_title("CryoSleep vs Transported", fontweight="bold", fontsize=11)
        ax5.legend(["Not Transported", "Transported"])
        ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        show(fig5)
        st.caption(
            "Passengers **in CryoSleep are far more likely to be transported**. "
            "This is one of the strongest predictors."
        )

    with r3b:
        st.subheader("6. Destination Distribution")
        dest = raw_df["Destination"].value_counts()
        fig6, ax6 = plt.subplots(figsize=(5.5, 3.8))
        ax6.bar(dest.index, dest.values, color=PALETTE[:3], width=0.45,
                edgecolor="white", linewidth=1.5)
        ax6.set_ylabel("Passengers")
        ax6.set_title("Passengers by Destination", fontweight="bold", fontsize=11)
        for i, v in enumerate(dest.values):
            ax6.text(i, v + 40, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
        ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax6.tick_params(axis="x", labelsize=8)
        show(fig6)
        st.caption("**TRAPPIST-1e** is by far the most popular destination.")

    st.markdown("---")
    st.subheader("7. Descriptive Statistics")
    st.dataframe(raw_df.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ③ MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🤖  Model Performance":

    st.title("🤖 Model Performance & Comparison")
    st.markdown(
        "All four models were trained on the same 80% training set and evaluated "
        "on the same 20% held-out test set using the exact preprocessing from the notebook."
    )
    st.markdown("---")

    results    = D["results"]
    models     = D["models"]
    conf_mats  = D["conf_mats"]
    results_df = pd.DataFrame(results)

    # ── KPI row ───────────────────────────────────────────────────────────────
    best_acc  = max(results, key=lambda r: r["Accuracy"])
    best_f1   = max(results, key=lambda r: r["F1-Score"])
    best_rec  = max(results, key=lambda r: r["Recall"])
    best_prec = max(results, key=lambda r: r["Precision"])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Best Accuracy",  f"{best_acc['Accuracy']*100:.2f}%",  best_acc["Model"],  "#6366f1")
    with k2:
        kpi("Best Precision", f"{best_prec['Precision']*100:.2f}%", best_prec["Model"], "#10b981")
    with k3:
        kpi("Best Recall",    f"{best_rec['Recall']*100:.2f}%",    best_rec["Model"],  "#f59e0b")
    with k4:
        kpi("Best F1-Score",  f"{best_f1['F1-Score']*100:.2f}%",  best_f1["Model"],   "#38bdf8")

    st.markdown("---")

    # ── Comparison table ──────────────────────────────────────────────────────
    st.subheader("📊 Model Comparison Table")

    def highlight_max(col):
        is_max = col == col.max()
        return [
            "background-color:#d1fae5; color:#065f46; font-weight:700;" if v else ""
            for v in is_max
        ]

    styled = (
        results_df.set_index("Model")
                  .style
                  .apply(highlight_max, axis=0)
                  .format("{:.4f}")
    )
    st.dataframe(styled, use_container_width=True)
    st.caption("✅ Green cells mark the best value for each metric.")

    st.markdown("---")

    # ── Dual chart row ────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="large")

    PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#f87171"]

    with ch1:
        st.subheader("📈 Accuracy Comparison")
        fig_bar, ax_bar = plt.subplots(figsize=(5.5, 4))
        bars = ax_bar.bar(
            results_df["Model"], results_df["Accuracy"],
            color=PALETTE, edgecolor="white", linewidth=1.5, width=0.55,
        )
        ax_bar.set_ylim(0.65, 0.84)
        ax_bar.set_ylabel("Accuracy")
        ax_bar.set_title("Test Accuracy per Model", fontweight="bold", fontsize=11)
        ax_bar.tick_params(axis="x", rotation=12, labelsize=8.5)
        for bar, row in zip(bars, results):
            ax_bar.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{row['Accuracy']:.4f}",
                ha="center", fontsize=8.5, fontweight="bold"
            )
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)
        st.info(
            f"🏆 **{best_acc['Model']}** — highest accuracy "
            f"({best_acc['Accuracy']*100:.2f}%)"
        )

    with ch2:
        st.subheader("📉 All Metrics")
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        x       = np.arange(len(results_df))
        width   = 0.19
        fig_m, ax_m = plt.subplots(figsize=(5.5, 4))
        mc = ["#6366f1", "#10b981", "#f59e0b", "#38bdf8"]
        for i, (met, c) in enumerate(zip(metrics, mc)):
            vals   = results_df[met].values
            offset = (i - 1.5) * width
            ax_m.bar(x + offset, vals, width, label=met, color=c, edgecolor="white")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(results_df["Model"], rotation=12, fontsize=8)
        ax_m.set_ylim(0.60, 0.96)
        ax_m.set_ylabel("Score")
        ax_m.set_title("All Metrics per Model", fontweight="bold", fontsize=11)
        ax_m.legend(loc="lower right", fontsize=7.5)
        ax_m.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.subheader("🗂️ Confusion Matrices")
    st.caption(
        "Rows = actual label, Columns = predicted label. "
        "Diagonal = correct predictions."
    )
    cm_cols = st.columns(4, gap="small")
    for col, (mname, cm) in zip(cm_cols, D["conf_mats"].items()):
        with col:
            fig_cm, ax_cm = plt.subplots(figsize=(3, 2.6))
            im = ax_cm.imshow(cm, cmap="Blues", vmin=0)
            ax_cm.set_title(mname, fontsize=8.5, fontweight="bold")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Not T.", "Trans."], fontsize=7)
            ax_cm.set_yticklabels(["Not T.", "Trans."], fontsize=7)
            ax_cm.set_xlabel("Predicted", fontsize=7)
            ax_cm.set_ylabel("Actual", fontsize=7)
            threshold = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, cm[i, j],
                               ha="center", va="center",
                               fontsize=11, fontweight="bold",
                               color="white" if cm[i, j] > threshold else "#1e293b")
            fig_cm.colorbar(im, ax=ax_cm, shrink=0.82)
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

    st.markdown("---")

    # ── Decision Tree ─────────────────────────────────────────────────────────
    st.subheader("🌳 Decision Tree — Top 5 Levels (max_depth = 4)")
    st.caption(
        "Each coloured node shows the dominant class at that decision split. "
        "The full unconstrained tree was used for prediction."
    )
    dt_model = models["Decision Tree"]
    fig_dt, ax_dt = plt.subplots(figsize=(22, 9))
    plot_tree(
        dt_model,
        feature_names=FEATURE_COLS,
        class_names=["Not Transported", "Transported"],
        filled=True, rounded=True, max_depth=4, fontsize=8,
        ax=ax_dt,
    )
    ax_dt.set_title("Decision Tree — Top 5 Levels", fontsize=13, fontweight="bold")
    st.pyplot(fig_dt, use_container_width=True)
    plt.close(fig_dt)

    st.markdown("---")

    # ── Key insights table ────────────────────────────────────────────────────
    st.subheader("💡 Key Metric Insights")
    insights_df = pd.DataFrame([
        {"Metric": "Accuracy",  "Best Model": best_acc["Model"],
         "Score": f"{best_acc['Accuracy']:.4f}",
         "What it means": "Fewest overall misclassifications"},
        {"Metric": "Precision", "Best Model": best_prec["Model"],
         "Score": f"{best_prec['Precision']:.4f}",
         "What it means": "Fewest false 'Transported' predictions"},
        {"Metric": "Recall",    "Best Model": best_rec["Model"],
         "Score": f"{best_rec['Recall']:.4f}",
         "What it means": "Best at catching all transported passengers"},
        {"Metric": "F1-Score",  "Best Model": best_f1["Model"],
         "Score": f"{best_f1['F1-Score']:.4f}",
         "What it means": "Best precision–recall balance"},
    ])
    st.dataframe(insights_df.set_index("Metric"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ④ PREDICT A PASSENGER
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔮  Predict a Passenger":

    st.title("🔮 Passenger Transportation Predictor")
    st.markdown(
        "Enter a passenger's details. The prediction is made using **KNN (K=9)** — "
        "the best-performing model from the evaluation "
        f"(test accuracy = {[r for r in D['results'] if r['Model']=='KNN (K=9)'][0]['Accuracy']*100:.2f}%). "
        "Full model comparison is available in the **🤖 Model Performance** section."
    )
    st.markdown("---")

    enc           = D["encoders"]
    enc_cats      = D["enc_cats"]
    spending_meds = D["spending_meds"]
    scaler        = D["scaler"]
    all_cols      = D["all_cols"]
    knn_model     = D["models"]["KNN (K=9)"]

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("pred_form"):

        st.markdown('<p class="section-tag">🌍 Travel Information</p>',
                    unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            home_planet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"],
                                       help="Planet the passenger departed from.")
        with f2:
            destination = st.selectbox("Destination",
                                       ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
                                       help="Exoplanet the passenger is travelling to.")
        with f3:
            cryo_sleep = st.selectbox("CryoSleep", ["False", "True"],
                                      help="Suspended animation — no onboard spending allowed.")
        with f4:
            vip = st.selectbox("VIP Status", ["False", "True"],
                               help="Whether the passenger paid for VIP service.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-tag">🛏️ Personal & Cabin Details</p>',
                    unsafe_allow_html=True)
        g1, g2, g3, g4, g5 = st.columns(5)
        with g1:
            age = st.number_input("Age", min_value=0, max_value=100, value=27)
        with g2:
            group_size = st.number_input("Group Size", min_value=1, max_value=15, value=1,
                                         help="Passengers travelling in the same group.")
        with g3:
            deck = st.selectbox("Cabin Deck",
                                ["A","B","C","D","E","F","G","T"], index=4)
        with g4:
            cabin_num = st.number_input("Cabin Number", min_value=0, max_value=2000, value=100)
        with g5:
            side = st.selectbox("Cabin Side", ["P","S"],
                                help="P = Port (left), S = Starboard (right).")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-tag">💳 Onboard Spending (credits)</p>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            room_service  = st.number_input("Room Service",  min_value=0.0, value=0.0, step=10.0)
        with s2:
            food_court    = st.number_input("Food Court",    min_value=0.0, value=0.0, step=10.0)
        with s3:
            shopping_mall = st.number_input("Shopping Mall", min_value=0.0, value=0.0, step=10.0)
        with s4:
            spa           = st.number_input("Spa",           min_value=0.0, value=0.0, step=10.0)
        with s5:
            vr_deck       = st.number_input("VR Deck",       min_value=0.0, value=0.0, step=10.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🚀  Predict with KNN (K=9)", use_container_width=True
        )

    # ── On submit ──────────────────────────────────────────────────────────────
    if submitted:

        user_input = {
            "HomePlanet":   home_planet,
            "CryoSleep":    cryo_sleep,
            "Destination":  destination,
            "Age":          float(age),
            "VIP":          vip,
            "RoomService":  room_service,
            "FoodCourt":    food_court,
            "ShoppingMall": shopping_mall,
            "Spa":          spa,
            "VRDeck":       vr_deck,
            "Deck":         deck,
            "CabinNumber":  float(cabin_num),
            "Side":         side,
            "GroupSize":    float(group_size),
        }

        arr_scaled = preprocess_single_input(
            user_input, enc, enc_cats, spending_meds, scaler, all_cols
        )

        pred  = int(knn_model.predict(arr_scaled)[0])
        proba = float(knn_model.predict_proba(arr_scaled)[0][1])
        transported = pred == 1

        st.markdown("---")

        # ── Result banner ──────────────────────────────────────────────────────
        if transported:
            bclass  = "verdict-yes"
            emoji   = "🌌"
            label   = "TRANSPORTED"
            sub_txt = "KNN predicts this passenger was transported to an alternate dimension."
            fill_cls = "pbar-fill-yes"
        else:
            bclass  = "verdict-no"
            emoji   = "🚢"
            label   = "NOT TRANSPORTED"
            sub_txt = "KNN predicts this passenger remained aboard the Spaceship Titanic."
            fill_cls = "pbar-fill-no"

        pct = int(proba * 100)

        st.markdown(
            f"""
            <div class="verdict-wrap {bclass}">
              <div class="verdict-label">
                Final Prediction Model · KNN (K=9) · Test Accuracy ≈
                {[r for r in D['results'] if r['Model']=='KNN (K=9)'][0]['Accuracy']*100:.2f}%
              </div>
              <div class="verdict-result">{emoji}&nbsp;&nbsp;{label}</div>
              <div class="verdict-sub">{sub_txt}</div>
              <div style="max-width:420px;margin:16px auto 0;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.74rem;color:#475569;margin-bottom:4px;">
                  <span>P(Transported)</span>
                  <span><b>{proba:.4f}</b></span>
                </div>
                <div class="pbar-bg">
                  <div class="{fill_cls}" style="width:{pct}%;"></div>
                </div>
                <div class="pbar-labels">
                  <span>0.00</span>
                  <span>Decision boundary · 0.50</span>
                  <span>1.00</span>
                </div>
              </div>
              <div style="font-size:0.72rem;color:#94a3b8;margin-top:14px;">
                ⚠️ This prediction is based on learnt statistical patterns
                and is not guaranteed to be correct.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence gauge chart ─────────────────────────────────────────────
        fig_g, ax_g = plt.subplots(figsize=(7, 2.4))
        ax_g.axis("off")
        ax_g.set_xlim(0, 1)
        ax_g.set_ylim(0, 1)

        # Gradient bar
        for i in range(200):
            t = i / 200
            r = 1 - t * 0.4
            g = 0.2 + t * 0.6
            b = 0.2
            ax_g.barh(0.5, 1/200, left=i/200, height=0.22,
                      color=(r, g, b), linewidth=0)

        # Boundary line
        ax_g.axvline(0.5, ymin=0.27, ymax=0.73, color="#475569",
                     linestyle="--", linewidth=1.2)

        # Needle
        ax_g.annotate("", xy=(proba, 0.73), xytext=(proba, 0.90),
                      arrowprops=dict(arrowstyle="-|>", color="#1e293b", lw=2.0))

        # Labels
        ax_g.text(0,    0.23, "Not Transported", ha="left",   fontsize=8.5, color="#7f1d1d")
        ax_g.text(1,    0.23, "Transported",     ha="right",  fontsize=8.5, color="#065f46")
        ax_g.text(0.5,  0.23, "0.50",            ha="center", fontsize=7.5, color="#475569")
        color_val = "#065f46" if transported else "#7f1d1d"
        ax_g.text(proba, 0.04, f"{proba:.3f}", ha="center",
                  fontsize=11, fontweight="bold", color=color_val)
        ax_g.set_title("KNN (K=9) — Confidence Score",
                       fontsize=10.5, fontweight="bold", pad=4)
        fig_g.patch.set_facecolor("#fafafa")
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

        # ── Two-column detail strip ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        dc1, dc2 = st.columns(2, gap="large")

        with dc1:
            st.markdown("##### 📋 Model Information")
            st.markdown("""
            <table class="ctx-table" width="100%">
            <tr><td><b>Selected model</b></td><td>KNN (K=9)</td></tr>
            <tr><td><b>Why selected</b></td><td>Highest test accuracy among all 4 models</td></tr>
            <tr><td><b>Algorithm</b></td><td>Majority vote of 9 nearest neighbours</td></tr>
            <tr><td><b>Features used</b></td><td>16 engineered & scaled features</td></tr>
            <tr><td><b>Training samples</b></td><td>6,954 (80% of 8,693)</td></tr>
            <tr><td><b>Test samples</b></td><td>1,739 (20% of 8,693)</td></tr>
            </table>
            """, unsafe_allow_html=True)

        with dc2:
            total_spend = room_service + food_court + shopping_mall + spa + vr_deck
            st.markdown("##### 🔧 Engineered Features Applied")
            pills = [
                ("HomePlanet",    home_planet,                       False),
                ("CryoSleep",     cryo_sleep,                        cryo_sleep == "True"),
                ("Destination",   destination,                       False),
                ("Age",           f"{age} yrs",                      False),
                ("VIP",           vip,                               vip == "True"),
                ("Deck",          deck,                              False),
                ("CabinNumber",   str(cabin_num),                    False),
                ("Side",          side,                              False),
                ("GroupSize",     str(group_size),                   False),
                ("TotalSpending", f"{total_spend:,.0f} cr",          total_spend > 0),
                ("NoSpending",    "Yes" if total_spend == 0 else "No",
                                  total_spend == 0),
            ]
            pill_html = '<div class="pill-row">'
            for lab, val, hi in pills:
                cls = "pill pill-hi" if hi else "pill"
                pill_html += f'<span class="{cls}"><b>{lab}:</b> {val}</span>'
            pill_html += "</div>"
            st.markdown(pill_html, unsafe_allow_html=True)
            st.caption(
                "🔵 Purple pills = values that strongly influence prediction "
                "(CryoSleep=True, VIP=True, spending activity, or NoSpending)."
            )

        st.markdown(
            "<br><div style='font-size:0.78rem;color:#94a3b8;text-align:center;'>"
            "Compare all four models in <b>🤖 Model Performance</b>."
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ FINDINGS & CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📝  Findings & Conclusion":

    st.title("📝 Findings & Conclusion")
    st.markdown("---")

    results    = D["results"]
    results_df = pd.DataFrame(results)
    best_acc   = max(results, key=lambda r: r["Accuracy"])
    best_f1    = max(results, key=lambda r: r["F1-Score"])
    best_rec   = max(results, key=lambda r: r["Recall"])
    best_prec  = max(results, key=lambda r: r["Precision"])

    st.subheader("🔬 Key Findings")

    findings = [
        ("Dataset Overview",
         "8,693 passengers across 14 columns. The target (`Transported`) is nearly "
         "balanced — 4,378 transported (50.4%) vs 4,315 not transported (49.6%). "
         "No duplicate rows were found."),

        ("Missing Values",
         "12 of 14 columns had missing values (each ~2%). Numerical columns were "
         "imputed with the **training-set median**; categorical columns with the "
         "**training-set mode**. Test data used the same training statistics to "
         "prevent data leakage."),

        ("Feature Engineering",
         "Four new features were created post-split: **TotalSpending** (sum of all 5 "
         "amenity columns), **NoSpending** (binary flag), plus **Deck**, **CabinNumber**, "
         "and **Side** extracted from the Cabin column. **GroupSize** was computed "
         "from PassengerId before the split."),

        ("CryoSleep is the Strongest Signal",
         "Passengers in CryoSleep are significantly more likely to be transported. "
         "Since CryoSleep passengers cannot use onboard services, their TotalSpending "
         "is 0 — directly captured by the NoSpending feature."),

        (f"Best Accuracy — {best_acc['Model']}",
         f"**{best_acc['Model']}** achieved the highest test accuracy of "
         f"**{best_acc['Accuracy']*100:.2f}%** and was therefore selected as the "
         "final prediction model. K=9 was chosen over K=6 because both achieved "
         "the same accuracy but K=9 had a higher F1-Score (0.7832 vs 0.773)."),

        (f"Best F1-Score — {best_f1['Model']}",
         f"**{best_f1['Model']}** has the best F1-Score of **{best_f1['F1-Score']:.4f}**, "
         "reflecting the best balance between precision and recall."),

        (f"Best Recall — {best_rec['Model']}",
         f"**{best_rec['Model']}** achieved the highest recall of "
         f"**{best_rec['Recall']*100:.2f}%**, meaning it identifies the most transported "
         "passengers — but at the cost of more false positives."),

        ("No Single Model Wins on All Metrics",
         "Each model has different strengths. KNN wins on accuracy and precision; "
         "Logistic Regression wins on F1-Score; Naive Bayes wins on recall. "
         "The best choice depends on the specific use-case priority."),
    ]

    for title, content in findings:
        with st.expander(f"📌 {title}", expanded=False):
            st.markdown(content)

    st.markdown("---")
    st.subheader("📊 Final Results Summary")

    def highlight_max(col):
        is_max = col == col.max()
        return [
            "background-color:#d1fae5; color:#065f46; font-weight:700;" if v else ""
            for v in is_max
        ]

    st.dataframe(
        results_df.set_index("Model")
                  .style
                  .apply(highlight_max, axis=0)
                  .format("{:.4f}"),
        use_container_width=True,
    )
    st.caption("✅ Green = best value per metric. Results match the notebook exactly.")

    st.markdown("---")
    st.subheader("🏁 Conclusion")
    st.markdown(
        f"""
        This project demonstrates a complete supervised machine learning workflow
        on the Kaggle **Spaceship Titanic** dataset:

        1. **Data understanding** — 8,693 records, nearly balanced classes, ~2% missing values.
        2. **Preprocessing** — median/mode imputation, LabelEncoding, StandardScaling.
        3. **Feature engineering** — GroupSize, TotalSpending, NoSpending, Deck/CabinNumber/Side.
        4. **Four classifiers** trained and compared on the same 80/20 stratified split.
        5. **{best_acc['Model']}** was selected as the final model with test accuracy
           **{best_acc['Accuracy']*100:.2f}%**, the highest among all four models.

        **Future scope:**
        - Hyperparameter tuning with cross-validation (GridSearchCV).
        - Ensemble methods (Random Forest, XGBoost).
        - Using the Kaggle test set for a real leaderboard submission.
        """
    )

    st.markdown("---")
    st.caption(
        "Dashboard by Pragati · B.Tech CSE · Data Science Project · "
        "Kaggle Spaceship Titanic Dataset"
    )
