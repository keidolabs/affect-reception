"""
Experiment 01 — Phase 0a: Layer-wise Probes + Geometry
Model: meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
Date: 2026-02-24

Gate criterion: best layer AUROC > 0.40 (chance = 0.125 for 8-class)
Run: uv run python experiments/01_phase0a_llama3-8b/run.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
ACT_DIR = OUTPUT_DIR / "activations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

EMOTION_COLORS = {
    "rage": "#e74c3c", "grief": "#3498db", "terror": "#2c3e50",
    "ecstasy": "#f1c40f", "loathing": "#27ae60", "amazement": "#9b59b6",
    "admiration": "#e67e22", "vigilance": "#1abc9c",
}
EMOTION_ORDER = list(EMOTION_COLORS.keys())

# ============================================================================
# LOAD
# ============================================================================

console.print("[bold cyan]Loading activations...[/bold cyan]")

act_path = ACT_DIR / "set_a_residuals.npy"
if not act_path.exists():
    console.print(f"[red]ERROR: {act_path} not found. Run extract.py first.[/red]")
    sys.exit(1)

activations = np.load(act_path)
with open(ACT_DIR / "metadata.json") as f:
    metadata = json.load(f)

n_stimuli, n_layers, d_model = activations.shape
labels = np.array([m["emotion"] for m in metadata])
ids    = [m["id"] for m in metadata]
console.print(f"Activations: {activations.shape}")

# ============================================================================
# SECTION 1 — LAYER-WISE LINEAR PROBES
# ============================================================================

console.print("\n[bold cyan]Section 1: Layer-wise linear probes...[/bold cyan]")

emotional_mask = labels != "neutral"
X_emo = activations[emotional_mask]
y_emo = labels[emotional_mask]

console.print(f"Emotional stimuli: {X_emo.shape[0]}, classes: {np.unique(y_emo).tolist()}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
layer_aurocs, layer_auroc_stds = [], []

for layer_idx in range(n_layers):
    X_layer = X_emo[:, layer_idx, :]
    fold_aurocs = []
    for train_idx, test_idx in cv.split(X_layer, y_emo):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
        clf.fit(X_layer[train_idx], y_emo[train_idx])
        probs = clf.predict_proba(X_layer[test_idx])
        auroc = roc_auc_score(y_emo[test_idx], probs, multi_class="ovr",
                              average="macro", labels=clf.classes_)
        fold_aurocs.append(auroc)
    layer_aurocs.append(np.mean(fold_aurocs))
    layer_auroc_stds.append(np.std(fold_aurocs))
    console.print(f"  Layer {layer_idx:2d}: AUROC = {layer_aurocs[-1]:.4f} ± {layer_auroc_stds[-1]:.4f}")

best_layer = int(np.argmax(layer_aurocs))
best_auroc = layer_aurocs[best_layer]
chance = 1.0 / len(np.unique(y_emo))

console.print(f"\n[bold green]Best layer: {best_layer} — AUROC = {best_auroc:.4f}[/bold green]")
gate_pass = best_auroc > 0.40
console.print(f"[{'bold green' if gate_pass else 'bold red'}]Gate (>0.40): {'PASS' if gate_pass else 'FAIL'}[/{'bold green' if gate_pass else 'bold red'}]")

# AUROC curve
fig, ax = plt.subplots(figsize=(9, 5))
layers = list(range(n_layers))
ax.plot(layers, layer_aurocs, "o-", color="#3498db", linewidth=2, markersize=5)
ax.fill_between(layers,
    [a - s for a, s in zip(layer_aurocs, layer_auroc_stds)],
    [a + s for a, s in zip(layer_aurocs, layer_auroc_stds)],
    alpha=0.2, color="#3498db")
ax.axhline(chance, color="gray", linestyle="--", linewidth=1.2, label=f"Chance ({chance:.3f})")
ax.axhline(0.40, color="orange", linestyle="--", linewidth=1.2, label="Gate (0.40)")
ax.axvline(best_layer, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7)
ax.scatter([best_layer], [best_auroc], color="#e74c3c", s=80, zorder=5)
ax.annotate(f"L{best_layer}\n{best_auroc:.3f}", (best_layer, best_auroc),
            textcoords="offset points", xytext=(6, -16), fontsize=8, color="#e74c3c")
ax.set_xlabel("Layer"); ax.set_ylabel("Macro AUROC")
ax.set_title(f"Phase 0a: Layer-wise Probe — {MODEL_ID}\nSet A (n=80 emotional), 8-class OvR, 5-fold CV")
ax.set_xlim(-0.5, n_layers - 0.5); ax.set_ylim(0, 1.05)
ax.set_xticks(range(0, n_layers, 2)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "auroc_curve.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "auroc_curve.svg", bbox_inches="tight")
plt.close(fig)
console.print("✓ auroc_curve.png + .svg")

# Save results CSV
pl.DataFrame({"layer": list(range(n_layers)),
              "auroc_mean": [round(a, 6) for a in layer_aurocs],
              "auroc_std":  [round(s, 6) for s in layer_auroc_stds]}
             ).write_csv(OUTPUT_DIR / "results.csv")
console.print("✓ results.csv")

# ============================================================================
# SECTION 2 — GEOMETRY AT BEST LAYER
# ============================================================================

console.print(f"\n[bold cyan]Section 2: Geometry at layer {best_layer}...[/bold cyan]")

X_best = activations[:, best_layer, :]
pca = PCA(n_components=min(50, X_best.shape[0] - 1), random_state=42)
X_pca = pca.fit_transform(X_best)
var12 = pca.explained_variance_ratio_[:2].sum() * 100
console.print(f"PC1+PC2 variance: {var12:.1f}%")

le = LabelEncoder()
sil_score = silhouette_score(X_pca[emotional_mask, :10],
                             le.fit_transform(y_emo), metric="euclidean", random_state=42)
console.print(f"Silhouette (PCA-10, emotional): {sil_score:.4f}")

tsne = TSNE(n_components=2, perplexity=15, max_iter=1000,
            learning_rate="auto", init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_pca)

neutral_mask = labels == "neutral"

def scatter_plot(X, xlabel, ylabel, title, path_stem):
    fig, ax = plt.subplots(figsize=(9, 7))
    for emo in EMOTION_ORDER:
        mask = labels == emo
        if not mask.any(): continue
        ax.scatter(X[mask, 0], X[mask, 1], c=EMOTION_COLORS[emo], label=emo,
                   s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
    if neutral_mask.any():
        ax.scatter(X[neutral_mask, 0], X[neutral_mask, 1],
                   c="#bdc3c7", marker="x", s=60, linewidths=1.5, label="neutral")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{path_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{path_stem}.svg", bbox_inches="tight")
    plt.close(fig)

scatter_plot(X_pca,
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
    f"PCA — Layer {best_layer} (AUROC={best_auroc:.3f}, sil={sil_score:.3f})\n{MODEL_ID}",
    "pca_scatter")

scatter_plot(X_tsne, "t-SNE 1", "t-SNE 2",
    f"t-SNE — Layer {best_layer} (AUROC={best_auroc:.3f})\n{MODEL_ID}",
    "tsne_scatter")

console.print("✓ pca_scatter + tsne_scatter (png + svg)")

# ============================================================================
# SECTION 3 — README
# ============================================================================

layer_table = "| Layer | Mean AUROC | Std |\n|-------|-----------|-----|\n"
for i, (a, s) in enumerate(zip(layer_aurocs, layer_auroc_stds)):
    marker = " **← best**" if i == best_layer else ""
    layer_table += f"| {i:2d} | {a:.4f} | {s:.4f} |{marker}\n"

gate_txt = (
    "**Gate: PASS** — Emotion information is linearly decodable from the residual stream."
    if gate_pass else
    "**Gate: FAIL** — Best AUROC did not exceed 0.40."
)

readme = f"""# Phase 0a — {MODEL_ID}

**Date:** 2026-02-24
**Model:** {MODEL_ID} ({n_layers} layers, d_model={d_model})
**Stimuli:** Set A standard (n=90: 80 emotional + 10 neutral)

## Gate Decision: {"PASS ✓" if gate_pass else "FAIL ✗"}

{gate_txt}

**Best layer: {best_layer}** — AUROC = **{best_auroc:.4f}** ± {layer_auroc_stds[best_layer]:.4f}
- Chance: {chance:.3f} | Improvement: {best_auroc - chance:.4f} ({(best_auroc-chance)/chance*100:.1f}%)
- Silhouette at best layer (PCA-10): {sil_score:.4f}

## Layer-wise AUROC

{layer_table}

## Outputs

- `outputs/activations/set_a_residuals.npy` — shape (90, {n_layers}, {d_model})
- `outputs/results.csv` — per-layer AUROC
- `outputs/auroc_curve.png/.svg`
- `outputs/pca_scatter.png/.svg`
- `outputs/tsne_scatter.png/.svg`
""".strip()

(EXP_DIR / "README.md").write_text(readme)
console.print("✓ README.md")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

gate_color = "bold green" if gate_pass else "bold red"
console.print(f"\n{'='*55}")
console.print(f"Model:      {MODEL_ID}")
console.print(f"Best layer: {best_layer} / {n_layers-1}")
console.print(f"Best AUROC: {best_auroc:.4f} (chance: {chance:.3f})")
console.print(f"Silhouette: {sil_score:.4f}")
console.print(f"[{gate_color}]Gate (>0.40): {'PASS' if gate_pass else 'FAIL'}[/{gate_color}]")
console.print(f"{'='*55}")
