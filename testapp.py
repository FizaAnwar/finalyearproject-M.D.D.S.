"""
Multimodal Stroke Severity Detection System — FYP Dashboard
--------------------------------------------------------------------
Stage 1: ViT               -> stroke / non-stroke classification
Stage 2: wav2vec2 + LSTM   -> severity classification (Low / Medium / High)

This dashboard reports real training results (confusion matrices, accuracy)
and adds an explainability layer. Two kinds of visuals appear, and are
labelled accordingly throughout the app:

  DERIVED FROM EVALUATION DATA — computed directly from your confusion
    matrices / logs.
  ILLUSTRATIVE EXAMPLE — a conceptual demonstration of what the
    explainability technique would show (e.g. Grad-CAM patch attention,
    feature attribution). Replace these with your real computed outputs
    using the uploaders in the sidebar before your defense.

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# Page config + global style
# ==========================================================================
st.set_page_config(
    page_title="Multimodal Stroke Severity Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0F2A43"
TEAL = "#1F7A6C"
MAROON = "#8C3B3B"
GOLD = "#A6791F"
INK = "#1B2733"
MUTED = "#7C8894"
RULE = "#E3E7EA"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, div {{ font-family: 'Inter', -apple-system, sans-serif; }}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
.block-container {{ padding-top: 1rem; max-width: 1180px; }}

/* ---------- Hero ---------- */
.hero {{
    background: linear-gradient(100deg, {NAVY} 0%, #163C5C 55%, {TEAL} 140%);
    padding: 42px 46px;
    color: #F2F6F8;
    margin-bottom: 6px;
    border-bottom: 3px solid {TEAL};
}}
.hero .eyebrow {{
    text-transform: uppercase; letter-spacing: 2.6px; font-size: 12px;
    color: #9FD8CE; font-weight: 600; margin-bottom: 12px;
}}
.hero h1 {{
    font-family: 'Playfair Display', serif; font-weight: 700; font-size: 33px;
    margin: 0 0 12px 0; letter-spacing: 0.2px; color: #FFFFFF;
}}
.hero p {{ font-size: 15px; color: #D7E2E9; max-width: 780px; line-height: 1.6; margin: 0; }}
.hero .tags {{ margin-top: 22px; }}
.tag {{
    display: inline-block; border: 1px solid rgba(255,255,255,0.35);
    padding: 6px 15px; font-size: 11.5px; letter-spacing: 0.6px; text-transform: uppercase;
    margin-right: 10px; color: #E7EEF1;
}}

/* ---------- Section headings ---------- */
.section-title {{
    font-family: 'Playfair Display', serif; font-size: 22px; color: {NAVY};
    margin: 8px 0 2px 0; font-weight: 700;
}}
.section-rule {{ height: 2px; width: 54px; background: {TEAL}; margin-bottom: 18px; }}
.section-sub {{ font-size: 13.5px; color: {MUTED}; margin: -8px 0 18px 0; max-width: 720px; line-height: 1.5; }}

/* ---------- Report-style panel (replaces boxed cards) ---------- */
.panel {{ border-left: 3px solid {TEAL}; padding: 4px 0 4px 18px; margin: 16px 0; }}
.panel.alt {{ border-left-color: {NAVY}; }}

/* ---------- Stat strip ---------- */
.stat-strip {{
    display: flex; flex-wrap: wrap; border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE};
    padding: 20px 0; margin: 8px 0 26px 0;
}}
.stat {{ flex: 1; min-width: 160px; padding: 0 24px; border-right: 1px solid {RULE}; }}
.stat:last-child {{ border-right: none; }}
.stat .label {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.8px; color: {MUTED}; font-weight: 600; }}
.stat .value {{ font-family: 'Playfair Display', serif; font-size: 30px; color: {NAVY}; margin-top: 7px; font-weight: 700; }}
.stat .sub {{ font-size: 12px; color: #9AA5AE; margin-top: 5px; }}

/* ---------- Timeline ---------- */
.timeline {{ display: flex; align-items: flex-start; gap: 0; margin: 22px 0 8px 0; }}
.tl-item {{ flex: 1; text-align: center; position: relative; padding: 0 12px; }}
.tl-marker {{
    width: 34px; height: 34px; border: 2px solid {NAVY}; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; margin: 0 auto 14px auto;
    font-weight: 700; color: {NAVY}; background: #fff; font-family: 'Playfair Display', serif; font-size: 14px;
}}
.tl-item:not(:last-child)::after {{
    content: ""; position: absolute; top: 17px; left: 58%; width: 84%; height: 2px; background: {RULE};
}}
.tl-title {{ font-weight: 700; color: {NAVY}; font-size: 14.5px; margin-bottom: 5px; }}
.tl-desc {{ font-size: 12.5px; color: {MUTED}; line-height: 1.5; }}

/* ---------- Architecture flow ---------- */
.flow-box {{ border: 1px solid {RULE}; padding: 14px 16px; text-align: center; font-size: 13.5px; color: {INK}; }}
.flow-rule {{ height: 18px; width: 1px; background: {RULE}; margin: 0 auto; }}

/* ---------- Attribution tags ---------- */
.attr-tag {{
    display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
    padding: 3px 0; margin-bottom: 12px; border-bottom: 2px solid;
}}
.attr-real {{ color: {TEAL}; border-color: {TEAL}; }}
.attr-demo {{ color: {GOLD}; border-color: {GOLD}; }}

.footer-note {{
    text-align: center; color: #9AA5AE; font-size: 12px; margin-top: 38px; padding-top: 18px;
    border-top: 1px solid {RULE};
}}
</style>
""", unsafe_allow_html=True)


def section_title(text, subtitle=None):
    st.markdown(f'<div class="section-title">{text}</div><div class="section-rule"></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def stat_strip(items):
    html = '<div class="stat-strip">'
    for label, value, sub in items:
        html += f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def timeline(items):
    html = '<div class="timeline">'
    for num, title, desc in items:
        html += (f'<div class="tl-item"><div class="tl-marker">{num}</div>'
                  f'<div class="tl-title">{title}</div><div class="tl-desc">{desc}</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def attribution(kind: str):
    if kind == "real":
        st.markdown('<span class="attr-tag attr-real">Derived from evaluation data</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="attr-tag attr-demo">Illustrative example — not model output</span>',
                     unsafe_allow_html=True)


# ==========================================================================
# Real results
# ==========================================================================
VIT_LABELS = ["non-stroke", "stroke"]
VIT_CM = np.array([[56, 24],
                    [11, 33]])
VIT_REPORTED_ACC = 71.46

SEV_LABELS = ["Low", "Medium", "High"]
SEV_CM = np.array([[2306, 1142, 820],
                    [2253, 1330, 992],
                    [854, 489, 791]])
LSTM_REPORTED_ACC = 44.4

DEFAULT_HISTORY = pd.DataFrame({
    "epoch": list(range(15)),
    "train_acc": [0.355, 0.357, 0.362, 0.369, 0.367, 0.378, 0.386, 0.385,
                  0.387, 0.387, 0.388, 0.391, 0.391, 0.393, 0.390],
    "val_acc":   [0.379, 0.387, 0.384, 0.370, 0.387, 0.387, 0.393, 0.406,
                  0.393, 0.392, 0.388, 0.406, 0.393, 0.384, 0.393],
    "train_loss": [1.365, 1.172, 1.150, 1.130, 1.118, 1.100, 1.088, 1.082,
                   1.080, 1.078, 1.078, 1.079, 1.080, 1.079, 1.078],
    "val_loss":   [1.086, 1.087, 1.086, 1.088, 1.083, 1.082, 1.081, 1.079,
                   1.080, 1.079, 1.080, 1.081, 1.082, 1.079, 1.078],
})

# Illustrative example values only — replace with real SHAP / permutation
# importance results via the sidebar uploader.
DEFAULT_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": ["Jitter (local)", "Shimmer (local)", "Mean Pitch (F0)", "Speech Rate",
                "Pause Ratio", "HNR", "MFCC-1 Energy", "Articulation Rate"],
    "importance": [0.22, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05],
})


# ==========================================================================
# Helpers
# ==========================================================================
def compute_metrics(cm: np.ndarray, labels: list[str]) -> tuple[float, pd.DataFrame]:
    total = cm.sum()
    accuracy = np.trace(cm) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.diag(cm) / cm.sum(axis=0)
        recall = np.diag(cm) / cm.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)
    df = pd.DataFrame({
        "Class": labels,
        "Precision": np.nan_to_num(precision),
        "Recall": np.nan_to_num(recall),
        "F1-score": np.nan_to_num(f1),
        "Support": cm.sum(axis=1),
    })
    return accuracy, df


def plot_confusion_matrix(cm, labels, title, colorscale="Blues"):
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels], y=[f"True: {l}" for l in labels],
        colorscale=colorscale, text=cm, texttemplate="%{text}",
        textfont=dict(size=15), showscale=True,
    ))
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=40, b=10),
                       yaxis_autorange="reversed", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_metric_bars(df, title):
    fig = go.Figure()
    for metric, color in zip(["Precision", "Recall", "F1-score"], [NAVY, TEAL, MAROON]):
        fig.add_trace(go.Bar(x=df["Class"], y=df[metric], name=metric, marker_color=color, opacity=0.92))
    fig.update_layout(title=title, barmode="group", height=340, yaxis_range=[0, 1],
                       margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_sankey_from_cm(cm, labels, title):
    n = len(labels)
    node_labels = [f"True: {l}" for l in labels] + [f"Pred: {l}" for l in labels]
    node_colors = [NAVY] * n + [TEAL] * n
    source, target, value, link_colors = [], [], [], []
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            source.append(i)
            target.append(n + j)
            value.append(int(cm[i, j]))
            link_colors.append("rgba(31,122,108,0.45)" if i == j else "rgba(140,59,59,0.35)")
    fig = go.Figure(go.Sankey(
        node=dict(label=node_labels, pad=22, thickness=16, color=node_colors,
                  line=dict(color="white", width=0.5)),
        link=dict(source=source, target=target, value=value, color=link_colors),
    ))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10),
                       font=dict(size=12), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(value_pct, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 50], "color": "#F3E7E7"},
                {"range": [50, 75], "color": "#F5EFDD"},
                {"range": [75, 100], "color": "#E5EFEC"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def synthetic_attention_grid(seed: int, size: int = 8):
    """Illustrative conceptual patch-attention map — not real Grad-CAM output."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(2, size - 2), rng.uniform(2, size - 2)
    sigma = rng.uniform(1.3, 2.3)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    grid = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    grid += rng.normal(0, 0.04, grid.shape)
    return np.clip(grid, 0, 1)


def synthetic_confidence_distribution(seed: int, n=400):
    """Illustrative predicted-confidence histogram — not extracted from real outputs."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.beta(5, 2, n), 0, 1)


# ==========================================================================
# Sidebar
# ==========================================================================
st.sidebar.markdown("## Project Control Panel")
st.sidebar.caption("Multimodal Stroke Severity Detection System")
st.sidebar.divider()

st.sidebar.markdown("**Replace illustrative content with real outputs**")
history_upload = st.sidebar.file_uploader(
    "Training history CSV", type=["csv"],
    help="Columns: epoch, train_acc, val_acc, train_loss, val_loss")
gradcam_upload = st.sidebar.file_uploader(
    "Real Grad-CAM / saliency image", type=["png", "jpg", "jpeg"],
    help="If an actual Grad-CAM overlay was exported from the ViT notebook, show it here instead of the demo.")
importance_upload = st.sidebar.file_uploader(
    "Feature importance CSV", type=["csv"],
    help="Columns: feature, importance (e.g. from SHAP or permutation importance)")

if history_upload is not None:
    try:
        history_df = pd.read_csv(io.BytesIO(history_upload.getvalue()))
        st.sidebar.success("Custom training history loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        history_df = DEFAULT_HISTORY
else:
    history_df = DEFAULT_HISTORY

if importance_upload is not None:
    try:
        feature_importance_df = pd.read_csv(io.BytesIO(importance_upload.getvalue()))
        st.sidebar.success("Custom feature importance loaded.")
        importance_is_real = True
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        feature_importance_df = DEFAULT_FEATURE_IMPORTANCE
        importance_is_real = False
else:
    feature_importance_df = DEFAULT_FEATURE_IMPORTANCE
    importance_is_real = False

st.sidebar.divider()
prevalence = st.sidebar.slider("Assumed stroke prevalence (for pipeline estimate)", 0.0, 1.0, 0.35, 0.05)

if "xai_seed" not in st.session_state:
    st.session_state.xai_seed = 42
if st.sidebar.button("Regenerate illustrative example"):
    st.session_state.xai_seed = np.random.randint(0, 100000)

st.sidebar.divider()
st.sidebar.text_input("Student name", placeholder="e.g. Jane Doe", key="student_name")
st.sidebar.text_input("Supervisor", placeholder="e.g. Dr. ___", key="supervisor_name")


# ==========================================================================
# Compute metrics
# ==========================================================================
vit_cm_acc, vit_df = compute_metrics(VIT_CM, VIT_LABELS)
sev_cm_acc, sev_df = compute_metrics(SEV_CM, SEV_LABELS)
sensitivity = vit_df.loc[vit_df["Class"] == "stroke", "Recall"].values[0]
specificity = vit_df.loc[vit_df["Class"] == "non-stroke", "Recall"].values[0]

# ==========================================================================
# Hero header
# ==========================================================================
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Final Year Project</div>
  <h1>Multimodal Stroke Severity Detection System</h1>
  <p>A two-stage clinical decision-support pipeline that screens brain imaging for stroke and grades
  severity from speech biomarkers, with an integrated explainability layer for clinical transparency.</p>
  <div class="tags">
    <span class="tag">Stage 1 — Vision Transformer</span>
    <span class="tag">Stage 2 — wav2vec2 + LSTM</span>
    <span class="tag">Explainability Layer</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_arch, tab_vit, tab_sev, tab_xai, tab_pipeline = st.tabs(
    ["Overview", "Architecture", "Stage 1 — ViT", "Stage 2 — Severity", "Explainability", "Combined Pipeline"]
)

# ==========================================================================
# OVERVIEW
# ==========================================================================
with tab_overview:
    stat_strip([
        ("ViT Test Accuracy", f"{VIT_REPORTED_ACC:.2f}%", "Stroke vs. non-stroke"),
        ("LSTM Best Val Accuracy", f"{LSTM_REPORTED_ACC:.1f}%", "At early stopping"),
        ("Stage 1 Samples", f"{VIT_CM.sum():,}", "Imaging test set"),
        ("Stage 2 Samples", f"{SEV_CM.sum():,}", "Speech test set"),
    ])

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge(VIT_REPORTED_ACC, "ViT Accuracy", NAVY), use_container_width=True)
    with g2:
        st.plotly_chart(gauge(LSTM_REPORTED_ACC, "Severity Model Accuracy", TEAL), use_container_width=True)

    section_title("How the Pipeline Works")
    timeline([
        ("1", "Scan intake", "A CT/MRI image is submitted to the imaging model."),
        ("2", "Stroke screening", "The Vision Transformer predicts stroke or non-stroke."),
        ("3", "Speech intake", "If stroke-positive, a speech sample is analyzed."),
        ("4", "Severity grading", "wav2vec2 + LSTM predicts Low, Medium, or High severity."),
    ])

    st.write("")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_confusion_matrix(VIT_CM, VIT_LABELS, "ViT — Confusion Matrix"),
                         use_container_width=True)
    with col_b:
        st.plotly_chart(plot_confusion_matrix(SEV_CM, SEV_LABELS, "Severity — Confusion Matrix",
                                               colorscale="Teal"), use_container_width=True)

    st.markdown(f"""
    <div class="panel">
    Confusion-matrix-derived accuracy is {vit_cm_acc*100:.2f}% for the imaging model and
    {sev_cm_acc*100:.2f}% for the severity model — close to the reported {VIT_REPORTED_ACC:.2f}% /
    {LSTM_REPORTED_ACC:.1f}%. The small gap is expected, since the reported figures were captured
    at a specific checkpoint (e.g. early stopping), which can differ slightly from this
    confusion-matrix snapshot.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================================
# ARCHITECTURE
# ==========================================================================
with tab_arch:
    section_title("Model Architecture")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Stage 1 — Vision Transformer (ViT)**")
        steps_vit = ["Input CT / MRI image", "Patch embedding (image split into patches)",
                     "Positional encoding added", "Transformer encoder blocks (multi-head self-attention)",
                     "MLP classification head", "Output — stroke / non-stroke"]
        for i, s in enumerate(steps_vit):
            if i > 0:
                st.markdown('<div class="flow-rule"></div>', unsafe_allow_html=True)
            st.markdown(f"<div class='flow-box'>{s}</div>", unsafe_allow_html=True)
    with a2:
        st.markdown("**Stage 2 — wav2vec2 + LSTM**")
        steps_sev = ["Raw speech waveform", "wav2vec2 feature extractor (pretrained encoder)",
                     "LSTM layer(s) over temporal features", "Dense layer with softmax",
                     "Output — Low / Medium / High severity"]
        for i, s in enumerate(steps_sev):
            if i > 0:
                st.markdown('<div class="flow-rule"></div>', unsafe_allow_html=True)
            st.markdown(f"<div class='flow-box'>{s}</div>", unsafe_allow_html=True)

    section_title("Dataset Snapshot", "From the evaluation runs behind the confusion matrices above.")
    stat_strip([
        ("Stage 1 Classes", "2", "non-stroke / stroke"),
        ("Stage 1 Test Samples", f"{VIT_CM.sum():,}", "from confusion matrix"),
        ("Stage 2 Classes", "3", "Low / Medium / High"),
        ("Stage 2 Test Samples", f"{SEV_CM.sum():,}", "from confusion matrix"),
    ])

    st.text_area(
        "Notes for the report (dataset source, preprocessing, train/val/test split, augmentation, etc.)",
        placeholder="e.g. Dataset: XYZ Stroke Imaging Dataset (n=...), split 70/15/15, "
                    "images resized to 224×224, speech resampled to 16kHz...",
        height=100,
    )

# ==========================================================================
# STAGE 1 — ViT
# ==========================================================================
with tab_vit:
    section_title("Stage 1 — ViT: Stroke vs. Non-Stroke")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy (reported)", f"{VIT_REPORTED_ACC:.2f}%")
    m2.metric("Accuracy (from CM)", f"{vit_cm_acc*100:.2f}%")
    m3.metric("Sensitivity (stroke recall)", f"{sensitivity*100:.1f}%")
    m4.metric("Specificity (non-stroke recall)", f"{specificity*100:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_confusion_matrix(VIT_CM, VIT_LABELS, "Confusion Matrix"), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_metric_bars(vit_df, "Per-Class Precision / Recall / F1"), use_container_width=True)

    st.plotly_chart(plot_sankey_from_cm(VIT_CM, VIT_LABELS, "Patient flow: true to predicted class"),
                     use_container_width=True)

    st.markdown("**Classification report**")
    st.dataframe(
        vit_df.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-score": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown(f"""
    <div class="panel alt">
    11 stroke cases were misclassified as non-stroke (false negatives) — the metric to watch closely
    for a clinical screening tool, since these patients never reach Stage 2.
    </div>
    """, unsafe_allow_html=True)

    section_title("Explainability — Where Does the Model Look?")
    attribution("illustrative")
    st.caption(
        "This patch-attention heatmap is a conceptual illustration of what a Grad-CAM / "
        "attention-rollout map over the ViT's image patches would look like. It is not extracted "
        "from the actual model. Upload a real exported Grad-CAM image in the sidebar to replace it."
    )
    col_x, col_y = st.columns(2)
    with col_x:
        grid = synthetic_attention_grid(st.session_state.xai_seed)
        fig_att = go.Figure(data=go.Heatmap(z=grid, colorscale="Inferno", showscale=False))
        fig_att.update_layout(title="Illustrative patch-attention map (8×8 patches)",
                               height=340, margin=dict(l=10, r=10, t=40, b=10),
                               xaxis=dict(showticklabels=False),
                               yaxis=dict(showticklabels=False, autorange="reversed"),
                               paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_att, use_container_width=True)
    with col_y:
        if gradcam_upload is not None:
            st.image(gradcam_upload, caption="Uploaded Grad-CAM output", use_container_width=True)
        else:
            st.markdown(
                "No real Grad-CAM image uploaded yet. In the full write-up, this panel would show "
                "the original scan beside the highlighted region the model attended to most when "
                "predicting stroke, giving a clinician a visual justification for the call."
            )

# ==========================================================================
# STAGE 2 — Severity
# ==========================================================================
with tab_sev:
    section_title("Stage 2 — wav2vec2 + LSTM: Severity Grading")
    m1, m2, m3 = st.columns(3)
    m1.metric("Best Val Accuracy (early stopping)", f"{LSTM_REPORTED_ACC:.1f}%")
    m2.metric("Accuracy (from CM)", f"{sev_cm_acc*100:.2f}%")
    best_epoch = int(history_df.loc[history_df["val_acc"].idxmax(), "epoch"])
    m3.metric("Best Epoch", f"{best_epoch}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_confusion_matrix(SEV_CM, SEV_LABELS, "Confusion Matrix (counts)",
                                               colorscale="Teal"), use_container_width=True)
    with col_b:
        recall_norm = SEV_CM / SEV_CM.sum(axis=1, keepdims=True)
        st.plotly_chart(plot_confusion_matrix(recall_norm, SEV_LABELS, "Confusion Matrix (per-class recall)",
                                               colorscale="Blues"), use_container_width=True)

    st.plotly_chart(plot_sankey_from_cm(SEV_CM, SEV_LABELS, "Patient flow: true to predicted severity"),
                     use_container_width=True)

    st.markdown("**Classification report**")
    st.dataframe(
        sev_df.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-score": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )

    section_title("Training Curves")
    attribution("real" if history_upload is not None else "illustrative")
    if history_upload is None:
        st.caption("Approximate reconstruction of the plotted graph — upload the real epoch log "
                   "in the sidebar for exact values.")
    col_c, col_d = st.columns(2)
    with col_c:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["train_acc"], mode="lines+markers",
                                      name="Train", line=dict(color=NAVY)))
        fig_acc.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["val_acc"], mode="lines+markers",
                                      name="Validation", line=dict(color=MAROON)))
        fig_acc.add_vline(x=best_epoch, line_dash="dot", line_color="gray",
                           annotation_text="early stopping", annotation_position="top")
        fig_acc.update_layout(title="Accuracy", height=340, xaxis_title="Epoch", yaxis_title="Accuracy",
                               margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_acc, use_container_width=True)
    with col_d:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["train_loss"], mode="lines+markers",
                                       name="Train", line=dict(color=NAVY)))
        fig_loss.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["val_loss"], mode="lines+markers",
                                       name="Validation", line=dict(color=MAROON)))
        fig_loss.add_vline(x=best_epoch, line_dash="dot", line_color="gray")
        fig_loss.update_layout(title="Loss", height=340, xaxis_title="Epoch", yaxis_title="Loss",
                                margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown(f"""
    <div class="panel">
    Validation loss plateaus almost immediately while validation accuracy stays low and noisy —
    consistent with the severity model struggling to learn a strong signal from current
    features/labels rather than classic overfitting. Class imbalance and the low Medium-class
    recall (~29%) are good next areas to investigate.
    </div>
    """, unsafe_allow_html=True)

    section_title("Explainability — Which Acoustic Features Matter?")
    attribution("real" if importance_is_real else "illustrative")
    if not importance_is_real:
        st.caption(
            "These are example values for demonstration, not computed SHAP or permutation "
            "importance from the model. Upload a real feature-importance CSV in the sidebar "
            "to show the actual results here."
        )
    fi_sorted = feature_importance_df.sort_values("importance", ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=fi_sorted["importance"], y=fi_sorted["feature"], orientation="h",
        marker=dict(color=fi_sorted["importance"], colorscale="Tealgrn"),
    ))
    fig_fi.update_layout(title="Feature importance for severity prediction", height=380,
                          margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Relative importance",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_fi, use_container_width=True)

# ==========================================================================
# EXPLAINABILITY TAB
# ==========================================================================
with tab_xai:
    section_title(
        "Explainability Overview",
        "Explainability turns each prediction from a black-box number into a story a clinician can "
        "check: which region of the scan drove the stroke call, and which speech characteristics "
        "drove the severity grade."
    )

    st.markdown("**Per-class performance — radar view**")
    attribution("real")
    fig_radar = go.Figure()
    fig_radar.update_layout(colorway=[NAVY, TEAL, MAROON])
    for _, row in sev_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Precision"], row["Recall"], row["F1-score"]],
            theta=["Precision", "Recall", "F1-score"],
            fill="toself", name=row["Class"],
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                             height=420, margin=dict(l=20, r=20, t=20, b=20),
                             legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                             paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_radar, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Illustrative prediction-confidence spread**")
        attribution("illustrative")
        st.caption("Example distribution shape only — not extracted from real per-sample outputs.")
        conf = synthetic_confidence_distribution(st.session_state.xai_seed)
        fig_hist = go.Figure(go.Histogram(x=conf, nbinsx=20, marker_color=NAVY, opacity=0.88))
        fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="Predicted confidence", yaxis_title="Count",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        st.markdown("**Walk through an example decision**")
        chosen = st.selectbox("Pick a hypothetical severity outcome to explain:", SEV_LABELS)
        top_feats = feature_importance_df.sort_values("importance", ascending=False).head(3)["feature"].tolist()
        st.markdown(f"""
        **Stage 1.** The imaging model flags the scan as stroke-positive, with confidence driven
        by the highlighted region in the patch-attention map.

        **Stage 2.** The severity model grades the case as **{chosen}**, most influenced by:
        {", ".join(f"*{f}*" for f in top_feats)}.

        **Why this matters.** Showing the top contributing features and regions behind a
        prediction, rather than only the final label, lets a clinician verify the model's
        reasoning instead of accepting it on faith.
        """)
        attribution("illustrative")

# ==========================================================================
# COMBINED PIPELINE
# ==========================================================================
with tab_pipeline:
    section_title("End-to-End Pipeline")
    timeline([
        ("1", "CT / MRI scan", "Patient imaging enters the pipeline."),
        ("2", "Stage 1 — ViT", f"Stroke screening · accuracy {VIT_REPORTED_ACC:.2f}%"),
        ("3", "Speech sample", "Collected if the scan is stroke-positive."),
        ("4", "Stage 2 — wav2vec2 + LSTM", f"Severity grading · best val. accuracy {LSTM_REPORTED_ACC:.1f}%"),
        ("5", "Report", "Diagnosis and severity grade returned."),
    ])

    st.write("")
    section_title("Illustrative End-to-End Accuracy Estimate")
    attribution("illustrative")
    st.caption(
        "Approximation only — Stage 2 was trained and evaluated on its own severity dataset, not "
        "strictly the stroke-positive subset routed from Stage 1."
    )
    stage1_acc = VIT_REPORTED_ACC / 100
    stage2_acc = LSTM_REPORTED_ACC / 100
    end_to_end = sensitivity * stage2_acc

    e1, e2, e3 = st.columns(3)
    e1.metric("Stage 1 accuracy", f"{stage1_acc*100:.2f}%")
    e2.metric("Stage 2 accuracy (given stroke reaches it)", f"{stage2_acc*100:.1f}%")
    e3.metric("Estimated end-to-end accuracy", f"{end_to_end*100:.1f}%",
              help="Approximately Stage 1 sensitivity multiplied by Stage 2 accuracy")

    section_title("Patient Funnel", "Simulated outcome for 1,000 patients under the assumed prevalence.")
    n = 1000
    n_stroke = int(n * prevalence)
    n_caught = int(n_stroke * sensitivity)
    n_missed = n_stroke - n_caught
    n_correct_severity = int(n_caught * stage2_acc)

    funnel = go.Figure(go.Funnel(
        y=["Simulated patients", "True stroke cases", "Caught by Stage 1", "Correctly graded by Stage 2"],
        x=[n, n_stroke, n_caught, n_correct_severity],
        textinfo="value+percent initial",
        marker=dict(color=[NAVY, MAROON, TEAL, GOLD]),
    ))
    funnel.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(funnel, use_container_width=True)
    st.markdown(f"""
    <div class="panel">
    With an assumed stroke prevalence of {prevalence:.0%}: out of {n_stroke} true stroke cases,
    Stage 1 sensitivity ({sensitivity*100:.1f}%) catches about {n_caught}, of which the severity
    model ({stage2_acc*100:.1f}%) grades roughly {n_correct_severity} correctly, and {n_missed}
    stroke cases are missed entirely at Stage 1.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================================
# Footer
# ==========================================================================
name = st.session_state.get("student_name") or "Student Name"
sup = st.session_state.get("supervisor_name") or "Supervisor Name"
st.markdown(f"""
<div class="footer-note">
  Final Year Project — Multimodal Stroke Severity Detection System (ViT + wav2vec2/LSTM) &nbsp;·&nbsp;
  {name} &nbsp;·&nbsp; Supervised by {sup}
</div>
""", unsafe_allow_html=True)