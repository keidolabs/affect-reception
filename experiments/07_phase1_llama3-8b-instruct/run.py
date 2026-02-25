"""
Experiment 07 — Phase 1: Probes + Geometry (Llama-3-8B Instruct, Set B)
Date: 2026-02-24

Three analyses on Set B activations:
  1. 8-class probe (clinical stimuli only) — direct comparison to Phase 0
  2. Binary probe (clinical vs neutral, all 192 stimuli)
  3. Token-length sensitivity analysis (matched pairs ±10%)

Main finding: Set A AUROC (Phase 0) vs Set B AUROC (Phase 1) by layer.
Run: uv run python experiments/05_phase1_llama1b/run.py
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

EXP_DIR    = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
ACT_DIR    = OUTPUT_DIR / "activations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID     = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_COLOR  = "#e74c3c"   # red — matches Phase 0 comparison chart
MODEL_LABEL  = "Llama-3-8B (instruct)"

# Path to the corresponding Phase 0 results for comparison
PHASE0_RESULTS = Path(__file__).parent.parent / "02_phase0b_llama3-8b-instruct" / "outputs" / "results.csv"

EMOTION_COLORS = {
    "rage": "#e74c3c", "grief": "#3498db", "terror": "#2c3e50",
    "ecstasy": "#f1c40f", "loathing": "#27ae60", "amazement": "#9b59b6",
    "admiration": "#e67e22", "vigilance": "#1abc9c",
}
EMOTION_ORDER = list(EMOTION_COLORS.keys())

# ============================================================================
# SECTION 1 — LOAD ACTIVATIONS
# ============================================================================

console.print("[bold cyan]Loading activations...[/bold cyan]")

act_path = ACT_DIR / "set_b_residuals.npy"
if not act_path.exists():
    console.print(f"[red]ERROR: {act_path} not found. Run extract.py first.[/red]")
    sys.exit(1)

activations = np.load(act_path)
with open(ACT_DIR / "metadata.json") as f:
    metadata = json.load(f)

n_stimuli, n_layers, d_model = activations.shape
console.print(f"Activations: {activations.shape}")

# ============================================================================
# SECTION 2 — SPLIT CLINICAL vs NEUTRAL
# ============================================================================

stimulus_sets = np.array([m["stimulus_set"] for m in metadata])
emotions      = np.array([m["emotion"] for m in metadata])
n_tokens_arr  = np.array([m["n_tokens"] for m in metadata])
ids           = np.array([m["id"] for m in metadata])
ctrl_ids      = np.array([m.get("matched_control_id") for m in metadata])

clinical_mask = stimulus_sets == "B_clinical"
neutral_mask  = stimulus_sets == "neutral"

X_clinical      = activations[clinical_mask]    # (96, n_layers, d_model)
labels_clinical = emotions[clinical_mask]        # 8 emotion categories

X_neutral       = activations[neutral_mask]
n_tok_clinical  = n_tokens_arr[clinical_mask]
n_tok_neutral   = n_tokens_arr[neutral_mask]

console.print(f"Clinical: {clinical_mask.sum()} | Neutral: {neutral_mask.sum()}")
console.print(f"Clinical token length: mean={n_tok_clinical.mean():.1f} ± {n_tok_clinical.std():.1f}")
console.print(f"Neutral  token length: mean={n_tok_neutral.mean():.1f} ± {n_tok_neutral.std():.1f}")

# ============================================================================
# SECTION 3 — 8-CLASS PROBE (clinical only, direct Phase 0 comparison)
# ============================================================================

console.print("\n[bold cyan]Section 3: 8-class probe (clinical stimuli)...[/bold cyan]")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
layer_aurocs_8, layer_auroc_stds_8 = [], []

for layer_idx in range(n_layers):
    X_layer = X_clinical[:, layer_idx, :]
    fold_aurocs = []
    for train_idx, test_idx in cv.split(X_layer, labels_clinical):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
        clf.fit(X_layer[train_idx], labels_clinical[train_idx])
        probs = clf.predict_proba(X_layer[test_idx])
        auroc = roc_auc_score(labels_clinical[test_idx], probs, multi_class="ovr",
                              average="macro", labels=clf.classes_)
        fold_aurocs.append(auroc)
    layer_aurocs_8.append(np.mean(fold_aurocs))
    layer_auroc_stds_8.append(np.std(fold_aurocs))
    console.print(f"  Layer {layer_idx:2d}: AUROC = {layer_aurocs_8[-1]:.4f} ± {layer_auroc_stds_8[-1]:.4f}")

best_layer = int(np.argmax(layer_aurocs_8))
best_auroc = layer_aurocs_8[best_layer]
chance_8   = 1.0 / len(np.unique(labels_clinical))
gate_pass  = best_auroc > 0.40

console.print(f"\n[bold green]Phase 1 best layer: {best_layer} — AUROC = {best_auroc:.4f}[/bold green]")
console.print(f"[{'bold green' if gate_pass else 'bold red'}]Gate (>0.40): {'PASS' if gate_pass else 'FAIL'}[/{'bold green' if gate_pass else 'bold red'}]")

pl.DataFrame({
    "layer":      list(range(n_layers)),
    "auroc_mean": [round(a, 6) for a in layer_aurocs_8],
    "auroc_std":  [round(s, 6) for s in layer_auroc_stds_8],
}).write_csv(OUTPUT_DIR / "results_8class.csv")
console.print("✓ results_8class.csv")

# ============================================================================
# SECTION 4 — BINARY PROBE (clinical vs neutral, all 192)
# ============================================================================

console.print("\n[bold cyan]Section 4: Binary probe (clinical vs neutral)...[/bold cyan]")

X_all    = activations                                    # (192, n_layers, d_model)
y_binary = (stimulus_sets == "B_clinical").astype(int)   # 1=clinical, 0=neutral

layer_aurocs_bin, layer_auroc_stds_bin = [], []

for layer_idx in range(n_layers):
    X_layer = X_all[:, layer_idx, :]
    fold_aurocs = []
    for train_idx, test_idx in cv.split(X_layer, y_binary):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_layer[train_idx], y_binary[train_idx])
        proba = clf.predict_proba(X_layer[test_idx])[:, 1]
        auroc = roc_auc_score(y_binary[test_idx], proba)
        fold_aurocs.append(auroc)
    layer_aurocs_bin.append(np.mean(fold_aurocs))
    layer_auroc_stds_bin.append(np.std(fold_aurocs))
    console.print(f"  Layer {layer_idx:2d}: binary AUROC = {layer_aurocs_bin[-1]:.4f} ± {layer_auroc_stds_bin[-1]:.4f}")

best_layer_bin = int(np.argmax(layer_aurocs_bin))
console.print(f"\n[bold green]Binary best layer: {best_layer_bin} — AUROC = {layer_aurocs_bin[best_layer_bin]:.4f}[/bold green]")

pl.DataFrame({
    "layer":      list(range(n_layers)),
    "auroc_mean": [round(a, 6) for a in layer_aurocs_bin],
    "auroc_std":  [round(s, 6) for s in layer_auroc_stds_bin],
}).write_csv(OUTPUT_DIR / "results_binary.csv")
console.print("✓ results_binary.csv")

# ============================================================================
# SECTION 5 — LOAD PHASE 0 RESULTS FOR COMPARISON
# ============================================================================

console.print("\n[bold cyan]Section 5: Loading Phase 0 results for comparison...[/bold cyan]")

phase0_aurocs, phase0_stds = None, None
if PHASE0_RESULTS.exists():
    df0 = pl.read_csv(PHASE0_RESULTS)
    phase0_aurocs = df0["auroc_mean"].to_list()
    phase0_stds   = df0["auroc_std"].to_list()
    phase0_best   = max(phase0_aurocs)
    console.print(f"Phase 0 best AUROC: {phase0_best:.4f}")
    console.print(f"Phase 1 best AUROC: {best_auroc:.4f}")
    console.print(f"Drop: {phase0_best - best_auroc:.4f} ({(phase0_best - best_auroc)/phase0_best*100:.1f}%)")
else:
    console.print(f"[yellow]Phase 0 results not found at {PHASE0_RESULTS} — skipping comparison[/yellow]")
    phase0_best = None

# ============================================================================
# SECTION 6 — GEOMETRY AT BEST LAYER
# ============================================================================

console.print(f"\n[bold cyan]Section 6: Geometry at layer {best_layer}...[/bold cyan]")

X_best = X_clinical[:, best_layer, :]
pca = PCA(n_components=min(50, X_best.shape[0] - 1), random_state=42)
X_pca = pca.fit_transform(X_best)
var12 = pca.explained_variance_ratio_[:2].sum() * 100

le = LabelEncoder()
sil_score = silhouette_score(X_pca[:, :10], le.fit_transform(labels_clinical),
                             metric="euclidean", random_state=42)
console.print(f"PC1+PC2 variance: {var12:.1f}% | Silhouette (PCA-10): {sil_score:.4f}")

tsne = TSNE(n_components=2, perplexity=15, max_iter=1000,
            learning_rate="auto", init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_pca)

def scatter_plot(X, xlabel, ylabel, title, path_stem):
    fig, ax = plt.subplots(figsize=(9, 7))
    for emo in EMOTION_ORDER:
        mask = labels_clinical == emo
        if not mask.any(): continue
        ax.scatter(X[mask, 0], X[mask, 1], c=EMOTION_COLORS[emo], label=emo,
                   s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{path_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{path_stem}.svg", bbox_inches="tight")
    plt.close(fig)

scatter_plot(X_pca,
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
    f"PCA — Layer {best_layer} (AUROC={best_auroc:.3f}, sil={sil_score:.3f})\n{MODEL_ID} — Set B clinical",
    "pca_scatter")
scatter_plot(X_tsne, "t-SNE 1", "t-SNE 2",
    f"t-SNE — Layer {best_layer} (AUROC={best_auroc:.3f})\n{MODEL_ID} — Set B clinical",
    "tsne_scatter")
console.print("✓ pca_scatter + tsne_scatter (png + svg)")

# ============================================================================
# SECTION 7 — PLOTS
# ============================================================================

console.print("\n[bold cyan]Section 7: Plots...[/bold cyan]")

layers_abs = list(range(n_layers))
layers_rel = [li / (n_layers - 1) for li in layers_abs]
chance_bin = 0.5

# --- Plot 1: Set A vs Set B 8-class AUROC (normalized depth) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(layers_rel, layer_aurocs_8, "o-", color=MODEL_COLOR, linewidth=2.5,
        markersize=4, label=f"Phase 1 — Set B clinical (n=96)")
ax.fill_between(layers_rel,
    [a - s for a, s in zip(layer_aurocs_8, layer_auroc_stds_8)],
    [a + s for a, s in zip(layer_aurocs_8, layer_auroc_stds_8)],
    alpha=0.15, color=MODEL_COLOR)

if phase0_aurocs is not None:
    n0 = len(phase0_aurocs)
    layers_rel_0 = [li / (n0 - 1) for li in range(n0)]
    ax.plot(layers_rel_0, phase0_aurocs, "--", color=MODEL_COLOR, linewidth=1.5,
            alpha=0.6, label=f"Phase 0 — Set A keyword-rich (n=80)")
    ax.fill_between(layers_rel_0,
        [a - s for a, s in zip(phase0_aurocs, phase0_stds)],
        [a + s for a, s in zip(phase0_aurocs, phase0_stds)],
        alpha=0.08, color=MODEL_COLOR)

ax.axhline(chance_8, color="gray", linestyle="--", linewidth=1, label=f"Chance ({chance_8:.3f})")
ax.axhline(0.40, color="orange", linestyle=":", linewidth=1, label="Gate (0.40)")
ax.set_xlabel("Normalized layer depth (0=input, 1=output)")
ax.set_ylabel("Macro AUROC (8-class OvR, 5-fold CV)")
ax.set_title(f"Phase 0 vs Phase 1 — {MODEL_LABEL}\nSet A (keyword-rich) vs Set B (clinical vignettes)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "auroc_setA_vs_setB.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "auroc_setA_vs_setB.svg", bbox_inches="tight")
plt.close(fig)
console.print("✓ auroc_setA_vs_setB.png + .svg")

# --- Plot 2: Binary probe (clinical vs neutral) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(layers_rel, layer_aurocs_bin, "s-", color="#e67e22", linewidth=2,
        markersize=4, label="Binary: clinical vs neutral")
ax.fill_between(layers_rel,
    [a - s for a, s in zip(layer_aurocs_bin, layer_auroc_stds_bin)],
    [a + s for a, s in zip(layer_aurocs_bin, layer_auroc_stds_bin)],
    alpha=0.15, color="#e67e22")
ax.axhline(chance_bin, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
ax.set_xlabel("Normalized layer depth")
ax.set_ylabel("Binary AUROC (5-fold CV)")
ax.set_title(f"Binary Probe: Emotional vs Neutral — {MODEL_LABEL}\nSet B clinical (n=96) vs neutral (n=96)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "auroc_binary.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "auroc_binary.svg", bbox_inches="tight")
plt.close(fig)
console.print("✓ auroc_binary.png + .svg")

# ============================================================================
# SECTION 8 — TOKEN LENGTH SENSITIVITY ANALYSIS
# ============================================================================

console.print("\n[bold cyan]Section 8: Token length sensitivity analysis...[/bold cyan]")

# Build lookup: id → n_tokens for all stimuli
id_to_ntok = {m["id"]: m["n_tokens"] for m in metadata}

# For each clinical stimulus, find its matched neutral and check length ratio
clinical_meta = [m for m in metadata if m["stimulus_set"] == "B_clinical"]
matched_indices = []   # indices into X_clinical for well-matched pairs
length_ratios = []

for i, m in enumerate(clinical_meta):
    ctrl_id = m.get("matched_control_id")
    if ctrl_id and ctrl_id in id_to_ntok:
        n_clin = m["n_tokens"]
        n_ctrl = id_to_ntok[ctrl_id]
        ratio  = abs(n_clin - n_ctrl) / max(n_clin, n_ctrl)
        length_ratios.append({
            "clinical_id":  m["id"],
            "neutral_id":   ctrl_id,
            "n_clinical":   n_clin,
            "n_neutral":    n_ctrl,
            "ratio":        round(ratio, 4),
            "matched":      ratio <= 0.10,
        })
        if ratio <= 0.10:
            matched_indices.append(i)

n_matched = len(matched_indices)
console.print(f"Pairs within ±10% token length: {n_matched} / {len(length_ratios)} (of 96 pairs)")

pl.DataFrame(length_ratios).write_csv(OUTPUT_DIR / "token_lengths.csv")
console.print("✓ token_lengths.csv")

# Run sensitivity probe only if we have at least 24 matched clinical stimuli (3 per class min)
if n_matched >= 24:
    X_matched = X_clinical[matched_indices]
    y_matched = labels_clinical[matched_indices]
    console.print(f"\n[dim]Sensitivity probe on {n_matched} matched pairs (LOW POWER — supplementary only):[/dim]")

    class_counts = {c: int((y_matched == c).sum()) for c in np.unique(y_matched)}
    min_class_count = min(class_counts.values())
    n_splits_sens = min(3, min_class_count)
    console.print(f"  Class counts in matched subset: {class_counts}")
    console.print(f"  Min class size = {min_class_count} → using {n_splits_sens}-fold CV")

    if n_splits_sens < 2:
        console.print(f"[yellow]  Sensitivity skipped — min class has only {min_class_count} sample(s).[/yellow]")
        best_sens = None
    else:
        sens_aurocs = []
        cv_sens = StratifiedKFold(n_splits=n_splits_sens, shuffle=True, random_state=42)

        for layer_idx in range(n_layers):
            X_layer = X_matched[:, layer_idx, :]
            fold_aurocs = []
            for train_idx, test_idx in cv_sens.split(X_layer, y_matched):
                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                clf.fit(X_layer[train_idx], y_matched[train_idx])
                probs = clf.predict_proba(X_layer[test_idx])
                try:
                    auroc = roc_auc_score(y_matched[test_idx], probs, multi_class="ovr",
                                          average="macro", labels=clf.classes_)
                except ValueError:
                    auroc = float("nan")
                fold_aurocs.append(auroc)
            sens_aurocs.append(np.nanmean(fold_aurocs))

        best_sens = max(s for s in sens_aurocs if not np.isnan(s))
        console.print(f"Sensitivity best AUROC: {best_sens:.4f} (n={n_matched}, {n_layers} layers)")
        pl.DataFrame({
            "layer":       list(range(n_layers)),
            "auroc_mean":  [round(a, 6) if not np.isnan(a) else None for a in sens_aurocs],
        }).write_csv(OUTPUT_DIR / "results_sensitivity.csv")
        console.print("✓ results_sensitivity.csv")
else:
    console.print(f"[yellow]Only {n_matched} matched pairs — too few for a stable 5-fold probe. Skipping.[/yellow]")
    best_sens = None

# ============================================================================
# SECTION 9 — WRITE README
# ============================================================================

console.print("\n[bold cyan]Section 9: Writing README...[/bold cyan]")

p0_line = (
    f"- **Phase 0 (Set A):** best AUROC = {phase0_best:.4f} (layer {int(np.argmax(phase0_aurocs))}/{n_layers-1})"
    if phase0_best else "- Phase 0 results not available for comparison"
)
sens_line = (
    f"- **Sensitivity (matched pairs, n={n_matched}):** best AUROC = {best_sens:.4f} (LOW POWER)"
    if best_sens else f"- Sensitivity analysis skipped (only {n_matched} matched pairs)"
)
drop_line = (
    f"- **Decodability drop:** {phase0_best - best_auroc:.4f} ({(phase0_best - best_auroc)/phase0_best*100:.1f}%)"
    if phase0_best else ""
)

layer_table = "| Layer | AUROC (8-class) | Std | Binary AUROC | Std |\n|-------|----------------|-----|-------------|-----|\n"
for i, (a8, s8, ab, sb) in enumerate(zip(layer_aurocs_8, layer_auroc_stds_8,
                                          layer_aurocs_bin, layer_auroc_stds_bin)):
    marker = " **← best**" if i == best_layer else ""
    layer_table += f"| {i:2d} | {a8:.4f} | {s8:.4f} | {ab:.4f} | {sb:.4f} |{marker}\n"

readme = f"""# Phase 1 — {MODEL_ID}

**Date:** 2026-02-24
**Model:** {MODEL_ID} ({n_layers} layers, d_model={d_model})
**Stimuli:** Set B (96 clinical vignettes + 96 matched neutral controls)

## Gate Decision: {"PASS ✓" if gate_pass else "FAIL ✗"}

**Best layer: {best_layer}** (norm: {best_layer/(n_layers-1):.1%}) — AUROC = **{best_auroc:.4f}** ± {layer_auroc_stds_8[best_layer]:.4f}
{p0_line}
{drop_line}
- Chance: {chance_8:.3f} | Gate threshold: 0.40

## Binary Probe (clinical vs neutral)
- Best layer: {best_layer_bin} — AUROC = **{layer_aurocs_bin[best_layer_bin]:.4f}**

## Silhouette at Best Layer
- {sil_score:.4f} (PCA-10, emotional stimuli only)

## Token Length Note
Clinical vignettes are longer than neutral controls on average
(clinical: {n_tok_clinical.mean():.0f} ± {n_tok_clinical.std():.0f} tokens vs neutral: {n_tok_neutral.mean():.0f} ± {n_tok_neutral.std():.0f}).
{n_matched}/96 pairs are within ±10% token length threshold.
{sens_line}
See `outputs/token_lengths.csv` for per-pair breakdown.

## Layer-wise AUROC

{layer_table}

## Outputs
- `outputs/activations/set_b_residuals.npy` — shape (192, {n_layers}, {d_model})
- `outputs/results_8class.csv` — 8-class AUROC by layer
- `outputs/results_binary.csv` — binary AUROC by layer
- `outputs/token_lengths.csv` — per-pair token length comparison
- `outputs/auroc_setA_vs_setB.png/.svg` — Phase 0 vs Phase 1 comparison
- `outputs/auroc_binary.png/.svg` — binary probe
- `outputs/pca_scatter.png/.svg` + `outputs/tsne_scatter.png/.svg`
""".strip()

(EXP_DIR / "README.md").write_text(readme)
console.print("✓ README.md")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

gate_color = "bold green" if gate_pass else "bold red"
console.print(f"\n{'='*60}")
console.print(f"Model:           {MODEL_ID}")
console.print(f"Phase 1 best:    L{best_layer} AUROC={best_auroc:.4f}")
if phase0_best:
    console.print(f"Phase 0 best:    AUROC={phase0_best:.4f}")
    console.print(f"Drop:            {phase0_best - best_auroc:.4f} ({(phase0_best-best_auroc)/phase0_best*100:.1f}%)")
console.print(f"Binary best:     L{best_layer_bin} AUROC={layer_aurocs_bin[best_layer_bin]:.4f}")
console.print(f"Silhouette:      {sil_score:.4f}")
console.print(f"Matched pairs:   {n_matched}/96")
console.print(f"[{gate_color}]Gate (>0.40): {'PASS' if gate_pass else 'FAIL'}[/{gate_color}]")
console.print(f"{'='*60}")
