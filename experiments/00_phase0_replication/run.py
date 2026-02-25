"""
Experiment 00 — Phase 0 Analysis: Layer-wise Probes + Representational Geometry
Question: Do mid-layers of Llama-3.2-1B encode emotion category information
          in Set A (keyword-rich) stimuli?
Date: 2026-02-24

Gate criterion: best layer AUROC > 0.40 (chance = 0.125 for 8-class)

Sections:
  1. Layer-wise linear probes — AUROC curve across all 16 layers
  2. Representational geometry — PCA + t-SNE at best layer
  3. Report generation — README.md with findings + gate decision

Requires: extract.py must be run first to produce set_a_residuals.npy

Run: uv run python experiments/00_phase0_replication/run.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

console = Console()

# Pin random seeds everywhere for reproducibility
torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
ACT_DIR = OUTPUT_DIR / "activations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.2-1B"  # model whose activations were extracted

# Consistent emotion color map (Plutchik wheel-inspired)
EMOTION_COLORS = {
    "rage": "#e74c3c",        # red
    "grief": "#3498db",       # blue
    "terror": "#2c3e50",      # dark blue/gray
    "ecstasy": "#f1c40f",     # yellow
    "loathing": "#27ae60",    # green
    "amazement": "#9b59b6",   # purple
    "admiration": "#e67e22",  # orange
    "vigilance": "#1abc9c",   # teal
}

EMOTION_ORDER = list(EMOTION_COLORS.keys())

# ============================================================================
# LOAD ACTIVATIONS AND METADATA
# ============================================================================

console.print("[bold cyan]Loading extracted activations...[/bold cyan]")

act_path = ACT_DIR / "set_a_residuals.npy"
meta_path = ACT_DIR / "metadata.json"

if not act_path.exists():
    console.print(f"[bold red]ERROR: {act_path} not found![/bold red]")
    console.print("Run extract.py first: uv run python experiments/00_phase0_replication/extract.py")
    sys.exit(1)

activations = np.load(act_path)  # shape: (90, 16, 2048)
with open(meta_path) as f:
    metadata = json.load(f)

n_stimuli, n_layers, d_model = activations.shape
console.print(f"Activations: {activations.shape} (n_stimuli={n_stimuli}, n_layers={n_layers}, d_model={d_model})")

# Build label arrays
labels = np.array([m["emotion"] for m in metadata])
stimulus_sets = np.array([m["stimulus_set"] for m in metadata])
ids = [m["id"] for m in metadata]

console.print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ============================================================================
# SECTION 1 — LAYER-WISE LINEAR PROBES
# ============================================================================

console.print("\n[bold cyan]Section 1: Layer-wise linear probes (8-class emotion classification)[/bold cyan]")
console.print("Method: 5-fold stratified CV, LogisticRegression (OvR), macro AUROC")
console.print("Note: Excluding 10 neutral stimuli — probing 8-class emotion only\n")

# Use only emotional stimuli (exclude 10 Set A neutrals)
emotional_mask = labels != "neutral"
X_emo = activations[emotional_mask]   # (80, 16, 2048)
y_emo = labels[emotional_mask]         # (80,) — 8 classes, 10 each

console.print(f"Emotional stimuli: {X_emo.shape[0]} (excluded {(~emotional_mask).sum()} neutrals)")
console.print(f"Classes: {np.unique(y_emo).tolist()}")

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

layer_aurocs = []   # mean AUROC per layer
layer_auroc_stds = []  # std across folds

for layer_idx in range(n_layers):
    X_layer = X_emo[:, layer_idx, :]   # (80, 2048)

    fold_aurocs = []
    for train_idx, test_idx in cv.split(X_layer, y_emo):
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        clf.fit(X_layer[train_idx], y_emo[train_idx])
        probs = clf.predict_proba(X_layer[test_idx])

        auroc = roc_auc_score(
            y_emo[test_idx],
            probs,
            multi_class="ovr",
            average="macro",
            labels=clf.classes_,
        )
        fold_aurocs.append(auroc)

    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    layer_aurocs.append(mean_auroc)
    layer_auroc_stds.append(std_auroc)
    console.print(
        f"  Layer {layer_idx:2d}: AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}"
    )

best_layer = int(np.argmax(layer_aurocs))
best_auroc = layer_aurocs[best_layer]
chance_level = 1.0 / len(np.unique(y_emo))  # 1/8 = 0.125

console.print(f"\n[bold green]Best layer: {best_layer} — AUROC = {best_auroc:.4f}[/bold green]")
console.print(f"Chance level (1/8 classes): {chance_level:.3f}")
console.print(f"Improvement over chance: {best_auroc - chance_level:.4f} ({(best_auroc - chance_level) / chance_level * 100:.1f}%)")

# Gate decision
gate_pass = best_auroc > 0.40
gate_status = "PASS" if gate_pass else "FAIL"
gate_color = "bold green" if gate_pass else "bold red"
console.print(f"\n[{gate_color}]Gate check (AUROC > 0.40): {gate_status}[/{gate_color}]")

# Plot AUROC curve
fig, ax = plt.subplots(figsize=(8, 5))
layers = list(range(n_layers))
ax.plot(layers, layer_aurocs, "o-", color="#3498db", linewidth=2, markersize=6, label="Mean AUROC (5-fold CV)")
ax.fill_between(
    layers,
    [a - s for a, s in zip(layer_aurocs, layer_auroc_stds)],
    [a + s for a, s in zip(layer_aurocs, layer_auroc_stds)],
    alpha=0.2,
    color="#3498db",
    label="±1 SD",
)
ax.axhline(y=chance_level, color="gray", linestyle="--", linewidth=1.5, label=f"Chance (1/8 = {chance_level:.3f})")
ax.axhline(y=0.40, color="orange", linestyle="--", linewidth=1.5, label="Gate threshold (0.40)")
ax.axvline(x=best_layer, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Best layer ({best_layer})")

ax.scatter([best_layer], [best_auroc], color="#e74c3c", s=100, zorder=5)
ax.annotate(
    f"Layer {best_layer}\nAUROC={best_auroc:.3f}",
    (best_layer, best_auroc),
    textcoords="offset points",
    xytext=(10, -15),
    fontsize=9,
    color="#e74c3c",
)

ax.set_xlabel("Layer Index", fontsize=11)
ax.set_ylabel("Macro AUROC (8-class OvR)", fontsize=11)
ax.set_title(
    f"Phase 0: Layer-wise Emotion Probe — {best_layer if True else 'N/A'} layers, Llama-3.2-1B\n"
    f"Set A (keyword-rich), 80 emotional stimuli, 5-fold CV",
    fontsize=10,
)
ax.set_xlim(-0.5, n_layers - 0.5)
ax.set_xticks(layers)
ax.set_ylim(0.0, 1.0)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "auroc_curve.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "auroc_curve.svg", bbox_inches="tight")
plt.close(fig)
console.print(f"\n[bold green]✓ Saved: auroc_curve.png + .svg[/bold green]")

# Save per-layer results to CSV
results_data = {
    "layer": list(range(n_layers)),
    "auroc_mean": [round(a, 6) for a in layer_aurocs],
    "auroc_std": [round(s, 6) for s in layer_auroc_stds],
}
results_df = pl.DataFrame(results_data)
results_df.write_csv(OUTPUT_DIR / "results.csv")
console.print(f"[bold green]✓ Saved: results.csv[/bold green]")

# ============================================================================
# SECTION 2 — REPRESENTATIONAL GEOMETRY AT BEST LAYER
# ============================================================================

console.print(f"\n[bold cyan]Section 2: Representational geometry at best layer ({best_layer})[/bold cyan]")

# Use all 90 stimuli (including neutral) for geometry analysis
X_best = activations[:, best_layer, :]   # (90, 2048)

# --- PCA ---
console.print("Computing PCA (50 components → plot PC1 vs PC2)...")
pca = PCA(n_components=min(50, X_best.shape[0] - 1), random_state=42)
X_pca = pca.fit_transform(X_best)   # (90, 50)

var_explained = pca.explained_variance_ratio_[:2].sum() * 100
console.print(f"PC1+PC2 explain {var_explained:.1f}% of variance")

# Silhouette score on PCA-10 (excluding neutral for emotion clustering)
X_pca10 = X_pca[emotional_mask, :10]
le = LabelEncoder()
y_encoded = le.fit_transform(y_emo)
sil_score = silhouette_score(X_pca10, y_encoded, metric="euclidean", random_state=42)
console.print(f"Silhouette score on PCA-10 (emotional only): {sil_score:.4f}")
console.print("  (range: -1 to 1; >0.1 suggests separable clusters)")

# PCA scatter plot
fig, ax = plt.subplots(figsize=(9, 7))

for emotion in EMOTION_ORDER:
    mask = labels == emotion
    if mask.sum() == 0:
        continue
    color = EMOTION_COLORS.get(emotion, "#888888")
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=color, label=emotion, s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
    )

# Plot neutrals with 'x' marker
neutral_mask = labels == "neutral"
if neutral_mask.sum() > 0:
    ax.scatter(
        X_pca[neutral_mask, 0], X_pca[neutral_mask, 1],
        c="#bdc3c7", marker="x", s=60, linewidths=1.5, label="neutral (Set A)",
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
ax.set_title(
    f"PCA — Residual Stream at Layer {best_layer} (AUROC={best_auroc:.3f})\n"
    f"Llama-3.2-1B, Set A stimuli (n=90, silhouette={sil_score:.3f})",
    fontsize=10,
)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pca_scatter.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "pca_scatter.svg", bbox_inches="tight")
plt.close(fig)
console.print(f"[bold green]✓ Saved: pca_scatter.png + .svg[/bold green]")

# --- t-SNE ---
console.print("\nComputing t-SNE (from PCA-50 space, perplexity=15 for n=90)...")
# Use perplexity=15 (n/5 heuristic for n=90; standard perplexity=30 needs n>>100)
tsne = TSNE(
    n_components=2,
    perplexity=15,  # appropriate for n=90
    random_state=42,
    max_iter=1000,
    learning_rate="auto",
    init="pca",
)
X_tsne = tsne.fit_transform(X_pca)  # (90, 2)

fig, ax = plt.subplots(figsize=(9, 7))

for emotion in EMOTION_ORDER:
    mask = labels == emotion
    if mask.sum() == 0:
        continue
    color = EMOTION_COLORS.get(emotion, "#888888")
    ax.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        c=color, label=emotion, s=60, alpha=0.85, edgecolors="white", linewidths=0.5,
    )

if neutral_mask.sum() > 0:
    ax.scatter(
        X_tsne[neutral_mask, 0], X_tsne[neutral_mask, 1],
        c="#bdc3c7", marker="x", s=60, linewidths=1.5, label="neutral (Set A)",
    )

ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.set_title(
    f"t-SNE — Residual Stream at Layer {best_layer} (AUROC={best_auroc:.3f})\n"
    f"Llama-3.2-1B, Set A stimuli (n=90, perplexity=15)",
    fontsize=10,
)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "tsne_scatter.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "tsne_scatter.svg", bbox_inches="tight")
plt.close(fig)
console.print(f"[bold green]✓ Saved: tsne_scatter.png + .svg[/bold green]")

# ============================================================================
# SECTION 3 — REPORT GENERATION
# ============================================================================

console.print(f"\n[bold cyan]Section 3: Writing findings to README.md...[/bold cyan]")

# Layer-wise AUROC table for markdown
layer_table_md = "| Layer | Mean AUROC | Std |\n|-------|-----------|-----|\n"
for i, (a, s) in enumerate(zip(layer_aurocs, layer_auroc_stds)):
    marker = " **← best**" if i == best_layer else ""
    layer_table_md += f"| {i:2d} | {a:.4f} | {s:.4f} |{marker}\n"

gate_interpretation = (
    "**Gate: PASS** — Emotion category information is decodable from Llama-3.2-1B residual stream "
    "with Set A (keyword-rich) stimuli. Proceeding to Phase 1 (clinical validity test)."
    if gate_pass else
    "**Gate: FAIL** — Best layer AUROC did not exceed threshold (0.40). This may reflect:\n"
    "- Small sample size (n=80 emotional stimuli, 5-fold CV with n=16 test per fold)\n"
    "- Llama-3.2-1B (1B params) may have weaker emotion representations than larger models\n"
    "Recommendation: Re-test with Llama-3-8B before concluding pipeline failure."
)

readme_content = f"""# Phase 0 Reproduction Study — Results

**Date:** 2026-02-24
**Model:** {MODEL_ID} (Llama-3.2-1B, 16 layers, d_model=2048)
**Stimuli:** Set A standard (n=90 total: 80 emotional + 10 neutral)
**Task:** 8-class emotion classification via linear probes on residual stream activations

---

## Research Question

Do mid-layers of Llama-3.2-1B encode emotion category information that is
linearly decodable from the residual stream, when probed with keyword-rich stimuli (Set A)?

## Hypothesis

A linear probe trained on residual stream activations at mid-layers will achieve
macro AUROC > 0.40 (well above chance of 0.125), confirming that emotion category
information is encoded in the model's internal representations.

## Gate Criterion

Phase 0 passes if: **best layer AUROC > 0.40**

---

## Results

### Gate Decision: {gate_status}

{gate_interpretation}

### Section 1: Layer-wise Linear Probes

**Method:** 5-fold stratified cross-validation, LogisticRegression (OvR, max_iter=1000),
macro AUROC across 8 emotion classes.
- Stimuli: 80 emotional stimuli (8 classes × 10 each)
- Chance level: 0.125 (1/8 classes)
- Gate threshold: 0.40

**Best layer: {best_layer}** — AUROC = **{best_auroc:.4f}** ± {layer_auroc_stds[best_layer]:.4f}
- Improvement over chance: {best_auroc - chance_level:.4f} ({(best_auroc - chance_level) / chance_level * 100:.1f}% relative)

{layer_table_md}

![AUROC Curve](outputs/auroc_curve.png)

### Section 2: Representational Geometry at Layer {best_layer}

**PCA analysis:**
- PC1 + PC2 explain {var_explained:.1f}% of variance in layer {best_layer} residual stream
- Silhouette score on PCA-10 space (emotional stimuli only): **{sil_score:.4f}**
  - Range: -1 to 1; values > 0.1 suggest separable emotion clusters
- Silhouette = {sil_score:.4f}: {"clusters show moderate separability" if sil_score > 0.1 else "weak cluster structure — emotions not clearly separated in PCA space"}

**t-SNE analysis:**
- Computed from PCA-50 space (perplexity=15, n_iter=1000, seed=42)
- Visual cluster structure assessed from scatter plot

![PCA Scatter](outputs/pca_scatter.png)

![t-SNE Scatter](outputs/tsne_scatter.png)

---

## Methodology

1. **Stimulus set**: 90 Set A standard stimuli loaded from `stimuli/set-a-standard.jsonl`
   - 80 emotional: 8 Plutchik primaries × 10 stimuli each (keyword-rich, first-person narratives)
   - 10 neutral: matched controls from crowd-enVENT `no-emotion` category
2. **Activation extraction**: `experiments/00_phase0_replication/extract.py`
   - Final-token residual stream (`cache["resid_post", layer][:, -1, :]`) at all 16 layers
   - Saved as: `outputs/activations/set_a_residuals.npy` — shape (90, 16, 2048)
3. **Linear probes**: `sklearn.LogisticRegression` (OvR, lbfgs solver, max_iter=1000)
   - 5-fold stratified CV, `roc_auc_score(multi_class="ovr", average="macro")`
4. **Geometry**: PCA (50 components) → t-SNE (from PCA-50 space, perplexity=15)
   - Silhouette score on PCA-10 space, euclidean metric

## Validation Notes (from pre-experiment checks)

From `validation/token_counts.py`:
- 59/96 Set B clinical/neutral pairs exceed ±10% token count threshold
- Clinical vignettes are systematically longer than their neutral controls
- This is a confound to be addressed in Phase 1 stimulus curation

From `validation/lexical_screening.py`:
- Set A emotional: mean sentiment polarity = -0.238 (keyword-laden, mostly negative)
- Set B clinical: mean polarity = -0.071 (near-neutral — keyword control working)
- Set A vs Set B: Mann-Whitney U p=0.0003 (significantly different — as intended)
- Set B vs neutral: p=0.4022 (not significantly different — keyword control validated)

## Outputs

- `outputs/activations/set_a_residuals.npy` — raw activations, shape (90, 16, 2048)
- `outputs/activations/metadata.json` — index → stimulus metadata
- `outputs/results.csv` — per-layer AUROC mean and std
- `outputs/auroc_curve.png/.svg` — layer-wise AUROC visualization
- `outputs/pca_scatter.png/.svg` — PCA visualization at best layer
- `outputs/tsne_scatter.png/.svg` — t-SNE visualization at best layer

## Random Seeds

- `torch.manual_seed(42)`, `np.random.seed(42)` set in all scripts
- `StratifiedKFold(random_state=42)`, `LogisticRegression(random_state=42)`
- `PCA(random_state=42)`, `TSNE(random_state=42)`

## Next Steps

{"- Proceed to Phase 1: Test with Set B (keyword-free clinical vignettes)" if gate_pass else "- Consider testing with Llama-3-8B (32 layers, stronger representations)"}
- Replicate Set A probing with Set B clinical stimuli (controlled stimulus set)
- Examine which neurons/attention heads contribute most to emotion differentiation
- Ablation study: which layers are necessary vs. sufficient for emotion decoding
"""

readme_path = EXP_DIR / "README.md"
readme_path.write_text(readme_content.strip())
console.print(f"[bold green]✓ Saved: {readme_path}[/bold green]")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

console.print(f"\n{'=' * 60}")
console.print(f"Phase 0 Analysis Complete")
console.print(f"{'=' * 60}")
console.print(f"Model:       {MODEL_ID}")
console.print(f"Best layer:  {best_layer}")
console.print(f"Best AUROC:  {best_auroc:.4f} (chance: {chance_level:.3f})")
console.print(f"Silhouette:  {sil_score:.4f}")
console.print(f"Gate ({'>'}0.40): [{gate_color}]{gate_status}[/{gate_color}]")
console.print(f"{'=' * 60}")
console.print(f"\nOutputs in: {OUTPUT_DIR}")
console.print(f"Findings:   {readme_path}")
