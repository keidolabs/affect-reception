"""
Experiment 00 — Interactive Visualizations
Date: 2026-02-24

Produces 5 Plotly HTML figures for Phase 0 exploratory analysis.
Static matplotlib plots are for papers; these are for understanding.

Figures:
  1. auroc_curve.html         — Layer-wise AUROC with CI ribbon, hover details
  2. pca_scatter.html         — PCA at best layer, hover shows actual stimulus text
  3. tsne_scatter.html        — t-SNE at best layer, hover shows actual stimulus text
  4. layer_heatmap.html       — Layer × emotion mean activation norm (where do emotions diverge?)
  5. per_class_auroc.html     — Layer × emotion OvR AUROC matrix (which emotions decode first?)

Run: uv run python experiments/00_phase0_replication/visualize.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

console = Console()

np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
OUTPUT_DIR = EXP_DIR / "outputs"

sys.path.insert(0, str(EXP_DIR.parent.parent))
from stimuli.loader import load_set_a

# Consistent emotion colors across all figures
EMOTION_COLORS = {
    "rage":        "#e74c3c",
    "grief":       "#3498db",
    "terror":      "#2c3e50",
    "ecstasy":     "#f1c40f",
    "loathing":    "#27ae60",
    "amazement":   "#9b59b6",
    "admiration":  "#e67e22",
    "vigilance":   "#1abc9c",
    "neutral":     "#bdc3c7",
}

EMOTION_ORDER = [
    "rage", "grief", "terror", "ecstasy",
    "loathing", "amazement", "admiration", "vigilance",
]

# ============================================================================
# LOAD DATA
# ============================================================================

console.print("[bold cyan]Loading activations and metadata...[/bold cyan]")

activations = np.load(ACT_DIR / "set_a_residuals.npy")   # (90, 16, 2048)
with open(ACT_DIR / "metadata.json") as f:
    metadata = json.load(f)

n_stimuli, n_layers, d_model = activations.shape

# Load full stimulus objects to get text
stimuli_by_id = {s.id: s for s in load_set_a()}

labels  = np.array([m["emotion"] for m in metadata])
ids     = [m["id"] for m in metadata]
n_tokens = [m["n_tokens"] for m in metadata]

# Truncated text for hover (keep readable)
texts_short = [
    stimuli_by_id[id_].text[:200].replace('"', "'") + "..."
    if len(stimuli_by_id[id_].text) > 200 else stimuli_by_id[id_].text.replace('"', "'")
    for id_ in ids
]

results_df = pl.read_csv(OUTPUT_DIR / "results.csv")
layer_aurocs = results_df["auroc_mean"].to_list()
layer_auroc_stds = results_df["auroc_std"].to_list()

best_layer = int(np.argmax(layer_aurocs))
best_auroc = layer_aurocs[best_layer]
chance = 1.0 / len(EMOTION_ORDER)

console.print(f"Activations: {activations.shape}, best layer: {best_layer} (AUROC={best_auroc:.4f})")

# ============================================================================
# FIGURE 1 — AUROC CURVE (with CI ribbon)
# ============================================================================

console.print("\n[bold cyan]Figure 1: AUROC curve...[/bold cyan]")

layers = list(range(n_layers))
upper = [a + s for a, s in zip(layer_aurocs, layer_auroc_stds)]
lower = [a - s for a, s in zip(layer_aurocs, layer_auroc_stds)]

fig1 = go.Figure()

# CI ribbon
fig1.add_trace(go.Scatter(
    x=layers + layers[::-1],
    y=upper + lower[::-1],
    fill="toself",
    fillcolor="rgba(52, 152, 219, 0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    showlegend=True,
    name="±1 SD",
    hoverinfo="skip",
))

# AUROC line
fig1.add_trace(go.Scatter(
    x=layers,
    y=layer_aurocs,
    mode="lines+markers",
    line=dict(color="#3498db", width=2.5),
    marker=dict(
        size=9,
        color=layer_aurocs,
        colorscale="Blues",
        cmin=0.0,
        cmax=1.0,
        line=dict(color="white", width=1.5),
    ),
    name="Mean AUROC (5-fold CV)",
    customdata=list(zip(layer_auroc_stds, [f"Layer {i}" for i in layers])),
    hovertemplate=(
        "<b>%{customdata[1]}</b><br>"
        "AUROC: %{y:.4f}<br>"
        "SD: %{customdata[0]:.4f}<br>"
        "<extra></extra>"
    ),
))

# Reference lines
fig1.add_hline(y=chance, line_dash="dot", line_color="gray",
               annotation_text=f"Chance (1/8 = {chance:.3f})", annotation_position="right")
fig1.add_hline(y=0.40, line_dash="dash", line_color="orange",
               annotation_text="Gate threshold (0.40)", annotation_position="right")
fig1.add_vline(x=best_layer, line_dash="dot", line_color="#e74c3c",
               annotation_text=f"Best: L{best_layer}", annotation_position="top right")

fig1.update_layout(
    title=dict(
        text=f"Phase 0 — Layer-wise Emotion Probe AUROC<br>"
             f"<sup>Llama-3.2-1B · Set A (n=80 emotional) · 8-class OvR · 5-fold CV · seed=42</sup>",
        x=0.5,
    ),
    xaxis=dict(title="Layer", tickmode="linear", dtick=1, gridcolor="#f0f0f0"),
    yaxis=dict(title="Macro AUROC", range=[0, 1.05], gridcolor="#f0f0f0"),
    plot_bgcolor="white",
    legend=dict(x=0.02, y=0.02),
    width=850, height=500,
)

fig1.write_html(OUTPUT_DIR / "auroc_curve.html")
console.print("[bold green]✓ auroc_curve.html[/bold green]")

# ============================================================================
# COMPUTE PCA + t-SNE (at best layer, all 90 stimuli)
# ============================================================================

console.print(f"\n[bold cyan]Computing PCA + t-SNE at layer {best_layer}...[/bold cyan]")

X_best = activations[:, best_layer, :]   # (90, 2048)

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_best)   # (90, 50)

tsne = TSNE(n_components=2, perplexity=15, max_iter=1000,
            learning_rate="auto", init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_pca)  # (90, 2)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

# Build hover text (stimulus ID + first 200 chars of text + token count)
hover_texts = [
    f"<b>{id_}</b><br>"
    f"Emotion: {lab}<br>"
    f"Tokens: {tok}<br>"
    f"<i>{txt}</i>"
    for id_, lab, tok, txt in zip(ids, labels, n_tokens, texts_short)
]

# ============================================================================
# FIGURE 2 — PCA SCATTER (hover = actual text)
# ============================================================================

console.print("[bold cyan]Figure 2: PCA scatter...[/bold cyan]")

fig2 = go.Figure()

for emotion in EMOTION_ORDER + ["neutral"]:
    mask = labels == emotion
    if not mask.any():
        continue
    color = EMOTION_COLORS.get(emotion, "#888")
    symbol = "x" if emotion == "neutral" else "circle"

    fig2.add_trace(go.Scatter(
        x=X_pca[mask, 0],
        y=X_pca[mask, 1],
        mode="markers",
        marker=dict(
            size=10 if emotion != "neutral" else 9,
            color=color,
            symbol=symbol,
            line=dict(color="white", width=0.8),
            opacity=0.85,
        ),
        name=emotion,
        text=[hover_texts[i] for i in np.where(mask)[0]],
        hoverinfo="text",
    ))

fig2.update_layout(
    title=dict(
        text=f"PCA — Residual Stream at Layer {best_layer} (AUROC={best_auroc:.3f})<br>"
             f"<sup>Llama-3.2-1B · Set A (n=90) · hover for stimulus text</sup>",
        x=0.5,
    ),
    xaxis=dict(title=f"PC1 ({var1:.1f}% var)", gridcolor="#f0f0f0", zeroline=False),
    yaxis=dict(title=f"PC2 ({var2:.1f}% var)", gridcolor="#f0f0f0", zeroline=False),
    plot_bgcolor="white",
    legend=dict(title="Emotion", itemsizing="constant"),
    width=850, height=600,
    hovermode="closest",
)

fig2.write_html(OUTPUT_DIR / "pca_scatter.html")
console.print("[bold green]✓ pca_scatter.html[/bold green]")

# ============================================================================
# FIGURE 3 — t-SNE SCATTER (hover = actual text)
# ============================================================================

console.print("[bold cyan]Figure 3: t-SNE scatter...[/bold cyan]")

fig3 = go.Figure()

for emotion in EMOTION_ORDER + ["neutral"]:
    mask = labels == emotion
    if not mask.any():
        continue
    color = EMOTION_COLORS.get(emotion, "#888")
    symbol = "x" if emotion == "neutral" else "circle"

    fig3.add_trace(go.Scatter(
        x=X_tsne[mask, 0],
        y=X_tsne[mask, 1],
        mode="markers",
        marker=dict(
            size=10 if emotion != "neutral" else 9,
            color=color,
            symbol=symbol,
            line=dict(color="white", width=0.8),
            opacity=0.85,
        ),
        name=emotion,
        text=[hover_texts[i] for i in np.where(mask)[0]],
        hoverinfo="text",
    ))

fig3.update_layout(
    title=dict(
        text=f"t-SNE — Residual Stream at Layer {best_layer} (AUROC={best_auroc:.3f})<br>"
             f"<sup>Llama-3.2-1B · Set A (n=90) · perplexity=15 · hover for stimulus text</sup>",
        x=0.5,
    ),
    xaxis=dict(title="t-SNE 1", gridcolor="#f0f0f0", zeroline=False),
    yaxis=dict(title="t-SNE 2", gridcolor="#f0f0f0", zeroline=False),
    plot_bgcolor="white",
    legend=dict(title="Emotion", itemsizing="constant"),
    width=850, height=600,
    hovermode="closest",
)

fig3.write_html(OUTPUT_DIR / "tsne_scatter.html")
console.print("[bold green]✓ tsne_scatter.html[/bold green]")

# ============================================================================
# FIGURE 4 — LAYER × EMOTION ACTIVATION NORM HEATMAP
# Question: at which layer does each emotion's activation diverge from others?
# ============================================================================

console.print("[bold cyan]Figure 4: Layer × emotion norm heatmap...[/bold cyan]")

# For each (emotion, layer): compute mean L2 norm of final-token residual
# Then subtract global per-layer mean to show relative deviation
norm_matrix = np.zeros((len(EMOTION_ORDER), n_layers))

for emo_idx, emotion in enumerate(EMOTION_ORDER):
    mask = labels == emotion
    emo_acts = activations[mask]        # (10, 16, 2048)
    for layer_idx in range(n_layers):
        norms = np.linalg.norm(emo_acts[:, layer_idx, :], axis=1)   # (10,)
        norm_matrix[emo_idx, layer_idx] = norms.mean()

# Normalize: subtract per-layer global mean, divide by per-layer std
# This shows which emotions have above/below-average activation magnitude per layer
global_mean = norm_matrix.mean(axis=0, keepdims=True)   # (1, 16)
global_std  = norm_matrix.std(axis=0, keepdims=True)    # (1, 16)
norm_z = (norm_matrix - global_mean) / (global_std + 1e-8)  # z-score per layer

fig4 = go.Figure(data=go.Heatmap(
    z=norm_z,
    x=[f"L{i}" for i in range(n_layers)],
    y=EMOTION_ORDER,
    colorscale="RdBu_r",
    zmid=0,
    text=np.round(norm_matrix, 1),
    customdata=np.round(norm_z, 3),
    hovertemplate=(
        "<b>%{y}</b> at <b>%{x}</b><br>"
        "Mean L2 norm: %{text}<br>"
        "Z-score vs other emotions: %{customdata:.3f}<br>"
        "<extra></extra>"
    ),
    colorbar=dict(title="Z-score<br>(within layer)"),
))

fig4.add_vline(x=best_layer - 0.5, line_dash="dot", line_color="#e74c3c", line_width=1.5,
               annotation_text=f"best layer ({best_layer})", annotation_position="top right")

fig4.update_layout(
    title=dict(
        text="Layer × Emotion: Mean Activation Norm (z-scored within each layer)<br>"
             "<sup>Red = above-average magnitude · Blue = below-average · "
             "Shows where emotions diverge in activation space</sup>",
        x=0.5,
    ),
    xaxis=dict(title="Layer", side="bottom"),
    yaxis=dict(title="Emotion", autorange="reversed"),
    width=900, height=420,
)

fig4.write_html(OUTPUT_DIR / "layer_heatmap.html")
console.print("[bold green]✓ layer_heatmap.html[/bold green]")

# ============================================================================
# FIGURE 5 — PER-CLASS AUROC HEATMAP (layer × emotion)
# Question: which emotions decode earliest / latest?
# ============================================================================

console.print("[bold cyan]Figure 5: Per-class AUROC heatmap (recomputing probes)...[/bold cyan]")
console.print("  Running 8 × 16 binary OvR probes (this takes ~30 sec)...")

# OvR: for each emotion class, binary probe (that class vs all others)
emotional_mask = labels != "neutral"
X_emo = activations[emotional_mask]   # (80, 16, 2048)
y_emo = labels[emotional_mask]         # (80,)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
per_class_auroc = np.zeros((n_layers, len(EMOTION_ORDER)))

for layer_idx in range(n_layers):
    X_layer = X_emo[:, layer_idx, :]

    for emo_idx, emotion in enumerate(EMOTION_ORDER):
        y_binary = (y_emo == emotion).astype(int)   # 1 = this emotion, 0 = all others

        fold_aurocs = []
        for train_idx, test_idx in cv.split(X_layer, y_binary):
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
            clf.fit(X_layer[train_idx], y_binary[train_idx])
            probs = clf.predict_proba(X_layer[test_idx])[:, 1]
            # Guard against degenerate folds (all one class in test)
            if len(np.unique(y_binary[test_idx])) < 2:
                fold_aurocs.append(0.5)
            else:
                fold_aurocs.append(roc_auc_score(y_binary[test_idx], probs))

        per_class_auroc[layer_idx, emo_idx] = np.mean(fold_aurocs)

# Build annotation text for heatmap cells
annotation_text = np.round(per_class_auroc, 2).astype(str)

fig5 = go.Figure(data=go.Heatmap(
    z=per_class_auroc.T,                      # (n_emotions, n_layers)
    x=[f"L{i}" for i in range(n_layers)],
    y=EMOTION_ORDER,
    colorscale="Viridis",
    zmin=0.5,
    zmax=1.0,
    text=annotation_text.T,
    hovertemplate=(
        "<b>%{y}</b> at <b>%{x}</b><br>"
        "OvR AUROC (5-fold CV): %{text}<br>"
        "<extra></extra>"
    ),
    colorbar=dict(title="OvR AUROC", tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
))

# Highlight chance level at 0.5 with a note
fig5.add_hline(y=-0.5, line_dash="dot", line_color="gray")   # cosmetic
fig5.add_vline(x=best_layer - 0.5, line_dash="dot", line_color="white", line_width=1.5,
               annotation_text=f"best macro layer ({best_layer})",
               annotation_font_color="white", annotation_position="top right")

fig5.update_layout(
    title=dict(
        text="Per-Class OvR AUROC by Layer — Which Emotions Decode First?<br>"
             "<sup>Llama-3.2-1B · Set A · Binary OvR per emotion · 5-fold CV · seed=42 · "
             "chance = 0.50</sup>",
        x=0.5,
    ),
    xaxis=dict(title="Layer", side="bottom"),
    yaxis=dict(title="Emotion", autorange="reversed"),
    width=900, height=420,
)

fig5.write_html(OUTPUT_DIR / "per_class_auroc.html")
console.print("[bold green]✓ per_class_auroc.html[/bold green]")

# Print per-class AUROC table (useful for journal notes)
console.print("\n[bold]Per-class OvR AUROC at best layer (L14):[/bold]")
for emo_idx, emotion in enumerate(EMOTION_ORDER):
    score = per_class_auroc[best_layer, emo_idx]
    console.print(f"  {emotion:<12} {score:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

console.print(f"\n[bold green]{'=' * 55}[/bold green]")
console.print(f"[bold green]All 5 interactive figures saved to outputs/[/bold green]")
console.print(f"[bold green]{'=' * 55}[/bold green]")
outputs = [
    "auroc_curve.html    — Layer AUROC with CI ribbon",
    "pca_scatter.html    — PCA, hover = stimulus text",
    "tsne_scatter.html   — t-SNE, hover = stimulus text",
    "layer_heatmap.html  — Layer × emotion activation norms",
    "per_class_auroc.html— Layer × emotion OvR AUROC matrix",
]
for o in outputs:
    console.print(f"  {o}")
