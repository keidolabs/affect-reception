"""
Experiment 17 — v2 Representational Geometry
Question: Do Set A and Set B emotions occupy the same representational space?
Date: 2026-02-25

Analyses:
  1. Joint PCA: concat Set A + Set B h activations at peak layer, PCA to 2D
     - Color by emotion, shape by set (circle=A, triangle=B)
     - Silhouette score by emotion (should be high) vs by set (should be low)
  2. Cosine similarity matrix: centroids per (emotion × set)
     - Within-emotion cross-set similarity vs cross-emotion similarity
     - If same-emotion cross-set similarity > cross-emotion: shared space
  3. Cross-topic clustering (Set B only):
     - For each emotion, measure within-emotion domain similarity vs cross-emotion
     - Permutation test: shuffle emotion labels, compute expected similarity under null
     - Confirms: emotion > topic structure in Set B

Prerequisite:
  - Run experiments/11_v2_extract_seta/extract_llama1b_inst.py
  - Run experiments/12_v2_extract_setb/extract_llama1b_inst.py
  - Run experiments/13_v2_probes/run.py (to get peak layer)

Run: uv run python experiments/17_v2_geometry/run.py
"""

import sys
import json
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
from rich.console import Console
from rich.table import Table

console = Console()
np.random.seed(42)

EXP_DIR  = Path(__file__).parent
ROOT     = EXP_DIR.parent.parent
OUT_DIR  = EXP_DIR / "outputs"
PROBES_DIR = ROOT / "experiments" / "13_v2_probes" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_CONFIGS = {
    "llama1b_inst": {"n_layers": 16},
    "llama1b_base": {"n_layers": 16},
    "llama8b_base": {"n_layers": 32},
    "llama8b_inst": {"n_layers": 32},
    "gemma9b_base": {"n_layers": 42},
    "gemma9b_inst": {"n_layers": 42},
}

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "llama1b_inst"
assert MODEL_KEY in MODEL_CONFIGS, f"Unknown MODEL_KEY: {MODEL_KEY}. Options: {list(MODEL_CONFIGS)}"
N_LAYERS  = MODEL_CONFIGS[MODEL_KEY]["n_layers"]

SETA_DIR = ROOT / "experiments" / "11_v2_extract_seta" / "outputs" / "activations" / MODEL_KEY
SETB_DIR = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations" / MODEL_KEY

EMOTIONS = ["rage", "grief", "terror", "ecstasy", "loathing", "amazement", "admiration", "vigilance"]
EMOTION_COLORS = {
    "rage": "#e74c3c", "grief": "#95a5a6", "terror": "#8e44ad",
    "ecstasy": "#f39c12", "loathing": "#27ae60", "amazement": "#2980b9",
    "admiration": "#d35400", "vigilance": "#16a085",
}

# ============================================================================
# DETERMINE PEAK LAYER (from Exp 13 probe results)
# ============================================================================

peak_layer = None
probe_csv = PROBES_DIR / f"probe_results_{MODEL_KEY}_seta_h.csv"
if probe_csv.exists():
    df = pl.read_csv(probe_csv)
    peak_layer = int(df["auroc_mean"].to_numpy().argmax())
    console.print(f"[bold cyan]Peak layer from Exp 13: L{peak_layer}[/bold cyan]")
else:
    peak_layer = 9  # Default: ~60% depth for 16-layer model
    console.print(f"[yellow]Exp 13 results not found — using default peak L{peak_layer}[/yellow]")

# ============================================================================
# LOAD ACTIVATIONS
# ============================================================================

def load_h_acts(act_dir, filter_sets=None):
    """Load h activations from all stimuli in a directory."""
    manifest = pl.read_csv(act_dir / "manifest.csv")
    acts, labels, sets, domains = [], [], [], []
    for row in manifest.iter_rows(named=True):
        if filter_sets and row["stimulus_set"] not in filter_sets:
            continue
        npz_path = act_dir / f"{row['id']}.npz"
        if not npz_path.exists(): continue
        npz = np.load(npz_path, allow_pickle=True)
        h = npz["h"]  # (n_layers, d_model)
        if not np.all(np.isfinite(h)):
            h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        acts.append(h)
        labels.append(row["emotion"])
        sets.append(row["stimulus_set"])
        domains.append(str(row.get("domain", "")))
    if not acts:
        return None, None, None, None
    return np.stack(acts), np.array(labels), np.array(sets), np.array(domains)


console.print(f"\n[bold cyan]Loading activations...[/bold cyan]")
seta_acts, seta_labels, seta_sets, seta_domains = load_h_acts(SETA_DIR)
setb_acts, setb_labels, setb_sets, setb_domains = load_h_acts(SETB_DIR)

# Clinical only for 8-class analysis
setb_clin_mask = setb_sets == "B_clinical"
setb_clin_acts   = setb_acts[setb_clin_mask]
setb_clin_labels = setb_labels[setb_clin_mask]
setb_clin_domains = setb_domains[setb_clin_mask]

# Emotional Set A only
seta_emo_mask = seta_labels != "neutral"
seta_emo_acts   = seta_acts[seta_emo_mask]
seta_emo_labels = seta_labels[seta_emo_mask]

console.print(f"Set A emotional: {len(seta_emo_acts)} stimuli")
console.print(f"Set B clinical:  {len(setb_clin_acts)} stimuli")

# Extract peak layer representations
seta_peak = seta_emo_acts[:, peak_layer, :]     # (80, d_model)
setb_peak = setb_clin_acts[:, peak_layer, :]    # (96, d_model)

# ============================================================================
# 1. JOINT PCA VISUALIZATION
# ============================================================================

console.print(f"\n[bold cyan]1. Joint PCA (Set A + Set B at L{peak_layer})...[/bold cyan]")

all_peak = np.concatenate([seta_peak, setb_peak], axis=0)
all_labels = np.concatenate([seta_emo_labels, setb_clin_labels])
all_sets   = np.concatenate([np.full(len(seta_emo_labels), "A"), np.full(len(setb_clin_labels), "B")])

# Normalize before PCA
mu = all_peak.mean(axis=0, keepdims=True)
std = all_peak.std(axis=0, keepdims=True) + 1e-8
all_peak_n = (all_peak - mu) / std

pca = PCA(n_components=2, random_state=42)
pca_embed = pca.fit_transform(all_peak_n)
var_exp = pca.explained_variance_ratio_
console.print(f"  PCA variance explained: PC1={var_exp[0]:.3f}, PC2={var_exp[1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Joint PCA — Set A + Set B at L{peak_layer} — {MODEL_KEY}", fontsize=12)

# Panel 1: Color by emotion
ax = axes[0]
for emotion in EMOTIONS:
    color = EMOTION_COLORS.get(emotion, "gray")
    for set_id, marker in [("A", "o"), ("B", "^")]:
        mask = (all_labels == emotion) & (all_sets == set_id)
        if mask.sum() > 0:
            ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1],
                       c=color, marker=marker, s=60 if set_id == "A" else 40,
                       alpha=0.8, edgecolors="white", linewidths=0.5,
                       label=f"{emotion} ({'A' if set_id == 'A' else 'B'})" if set_id == "A" else "")

legend_patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in EMOTIONS]
legend_shapes = [plt.scatter([], [], c="gray", marker="o", label="Set A", s=60),
                 plt.scatter([], [], c="gray", marker="^", label="Set B", s=40)]
ax.legend(handles=legend_patches + legend_shapes, fontsize=7, ncol=2, loc="upper right")
ax.set_title("Color by emotion"); ax.set_xlabel(f"PC1 ({var_exp[0]:.2%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.2%})")
ax.grid(True, alpha=0.2)

# Panel 2: Color by set (A=red, B=blue)
ax = axes[1]
for set_id, color, label, marker in [("A", "#e74c3c", "Set A (keyword-rich)", "o"),
                                       ("B", "#2980b9", "Set B (clinical)", "^")]:
    mask = all_sets == set_id
    ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1],
               c=color, marker=marker, s=50, alpha=0.7, label=label)
ax.legend(fontsize=9); ax.set_title("Color by set")
ax.set_xlabel(f"PC1 ({var_exp[0]:.2%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.2%})")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / f"pca_joint_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"pca_joint_{MODEL_KEY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ pca_joint_{MODEL_KEY}.png/.svg saved[/bold green]")

# ============================================================================
# 2. SILHOUETTE SCORES — by emotion vs by set
# ============================================================================

console.print(f"\n[bold cyan]2. Silhouette scores (emotion vs set clustering)...[/bold cyan]")

# Encode labels for silhouette
emotion_int = np.array([EMOTIONS.index(l) if l in EMOTIONS else -1 for l in all_labels])
valid_mask = emotion_int >= 0
set_int = (all_sets == "B").astype(int)

sil_emotion = silhouette_score(all_peak_n[valid_mask], emotion_int[valid_mask])
sil_set     = silhouette_score(all_peak_n, set_int)

console.print(f"  Silhouette by emotion: {sil_emotion:.4f}")
console.print(f"  Silhouette by set:     {sil_set:.4f}")

if sil_emotion > sil_set:
    console.print(f"  [bold green]✓ Emotion structure > set structure ({sil_emotion:.4f} vs {sil_set:.4f})[/bold green]")
    console.print(f"  Supports: joint representations are organized by emotion, not by set")
else:
    console.print(f"  [yellow]Set structure dominates ({sil_set:.4f} vs {sil_emotion:.4f})[/yellow]")

silhouette_rows = [
    {"model_key": MODEL_KEY, "layer": peak_layer, "grouping": "emotion", "silhouette": sil_emotion},
    {"model_key": MODEL_KEY, "layer": peak_layer, "grouping": "set",     "silhouette": sil_set},
]

# ============================================================================
# 3. COSINE SIMILARITY MATRIX — centroids per (emotion × set)
# ============================================================================

console.print(f"\n[bold cyan]3. Cosine similarity matrix (centroids)...[/bold cyan]")

centroids = {}
for emotion in EMOTIONS:
    for set_id, peak_acts, peak_lbs in [("A", seta_peak, seta_emo_labels),
                                          ("B", setb_peak, setb_clin_labels)]:
        mask = peak_lbs == emotion
        if mask.sum() > 0:
            centroid = peak_acts[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[(emotion, set_id)] = centroid / (norm + 1e-8)

# Build similarity matrix
keys = sorted(centroids.keys(), key=lambda x: (x[0], x[1]))
n = len(keys)
sim_matrix = np.zeros((n, n))
for i, ki in enumerate(keys):
    for j, kj in enumerate(keys):
        sim_matrix[i, j] = float(np.dot(centroids[ki], centroids[kj]))

# Statistics: within-emotion cross-set vs cross-emotion
within_emo_sims = []
cross_emo_sims  = []
for i, (emo_i, set_i) in enumerate(keys):
    for j, (emo_j, set_j) in enumerate(keys):
        if i >= j: continue
        if emo_i == emo_j and set_i != set_j:
            within_emo_sims.append(sim_matrix[i, j])
        elif emo_i != emo_j:
            cross_emo_sims.append(sim_matrix[i, j])

console.print(f"  Within-emotion cross-set similarity: {np.mean(within_emo_sims):.4f} ± {np.std(within_emo_sims):.4f}")
console.print(f"  Cross-emotion similarity:            {np.mean(cross_emo_sims):.4f} ± {np.std(cross_emo_sims):.4f}")

if np.mean(within_emo_sims) > np.mean(cross_emo_sims):
    console.print(f"  [bold green]✓ Same-emotion cross-set similarity > cross-emotion: shared emotional space[/bold green]")
else:
    console.print(f"  [yellow]Cross-emotion similarity comparable to within-emotion cross-set[/yellow]")

silhouette_rows.append({
    "model_key": MODEL_KEY, "layer": peak_layer, "grouping": "within_emo_cross_set_cosine",
    "silhouette": float(np.mean(within_emo_sims))
})
silhouette_rows.append({
    "model_key": MODEL_KEY, "layer": peak_layer, "grouping": "cross_emo_cosine",
    "silhouette": float(np.mean(cross_emo_sims))
})

pl.DataFrame(silhouette_rows).write_csv(OUT_DIR / f"silhouette_scores_{MODEL_KEY}.csv")
console.print(f"[bold green]✓ silhouette_scores_{MODEL_KEY}.csv saved[/bold green]")

# Plot cosine similarity matrix
fig, ax = plt.subplots(figsize=(10, 8))
key_labels = [f"{e}\n({s})" for e, s in keys]
im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n)); ax.set_xticklabels(key_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(key_labels, fontsize=8)
ax.set_title(f"Cosine Similarity — Emotion × Set Centroids at L{peak_layer}", fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT_DIR / f"cosine_similarity_matrix_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"cosine_similarity_matrix_{MODEL_KEY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ cosine_similarity_matrix_{MODEL_KEY}.png/.svg saved[/bold green]")

# ============================================================================
# 4. CROSS-TOPIC CLUSTERING (Set B only)
# ============================================================================

console.print(f"\n[bold cyan]4. Cross-topic clustering (Set B only)...[/bold cyan]")

# For each emotion, measure: within-emotion cross-domain similarity vs cross-emotion similarity
cross_topic_rows = []
rng = np.random.default_rng(42)

for emotion in EMOTIONS:
    emo_mask = setb_clin_labels == emotion
    emo_acts = setb_peak[emo_mask]
    emo_domains = setb_clin_domains[emo_mask]

    if len(emo_acts) < 4:
        continue

    # Within-emotion centroid cosine similarity (same emotion, different domains)
    within_emo_sims_b = []
    for d1 in ["1", "2", "3"]:
        for d2 in ["1", "2", "3"]:
            if d1 >= d2: continue
            acts_d1 = emo_acts[emo_domains == d1]
            acts_d2 = emo_acts[emo_domains == d2]
            if len(acts_d1) == 0 or len(acts_d2) == 0: continue
            c1 = acts_d1.mean(axis=0); c2 = acts_d2.mean(axis=0)
            n1 = np.linalg.norm(c1) + 1e-8; n2 = np.linalg.norm(c2) + 1e-8
            within_emo_sims_b.append(float(np.dot(c1/n1, c2/n2)))

    # Cross-emotion similarity (this emotion vs all others)
    cross_emo_sims_b = []
    emo_centroid = emo_acts.mean(axis=0)
    emo_centroid /= np.linalg.norm(emo_centroid) + 1e-8
    for other_emo in EMOTIONS:
        if other_emo == emotion: continue
        other_mask = setb_clin_labels == other_emo
        if other_mask.sum() == 0: continue
        other_centroid = setb_peak[other_mask].mean(axis=0)
        other_centroid /= np.linalg.norm(other_centroid) + 1e-8
        cross_emo_sims_b.append(float(np.dot(emo_centroid, other_centroid)))

    mean_within = float(np.mean(within_emo_sims_b)) if within_emo_sims_b else float("nan")
    mean_cross  = float(np.mean(cross_emo_sims_b)) if cross_emo_sims_b else float("nan")

    # Permutation test: shuffle Set B labels, measure expected within-emotion similarity
    null_sims = []
    for _ in range(500):
        shuffled = rng.permutation(setb_clin_labels)
        shuf_mask = shuffled == emotion
        if shuf_mask.sum() < 2: continue
        null_centroid = setb_peak[shuf_mask].mean(axis=0)
        null_centroid /= np.linalg.norm(null_centroid) + 1e-8
        null_sims.append(float(np.dot(emo_centroid, null_centroid)))

    null_mean = float(np.mean(null_sims)) if null_sims else float("nan")
    p_val = float(np.mean([s >= mean_within for s in null_sims])) if null_sims else float("nan")

    cross_topic_rows.append({
        "emotion": emotion, "within_emo_cross_domain_sim": mean_within,
        "cross_emo_sim": mean_cross, "null_sim": null_mean, "p_permutation": p_val,
        "emotion_dominates": mean_within > mean_cross,
    })

    console.print(f"  {emotion:12s}: within={mean_within:.4f}, cross-emo={mean_cross:.4f}, null={null_mean:.4f}, p={p_val:.3f}")

if cross_topic_rows:
    ct_df = pl.DataFrame(cross_topic_rows)
    ct_df.write_csv(OUT_DIR / f"permutation_test_{MODEL_KEY}.csv")
    n_emo_dominant = sum(r["emotion_dominates"] for r in cross_topic_rows)
    console.print(f"\n  Emotion > topic: {n_emo_dominant}/{len(cross_topic_rows)} emotions")
    console.print(f"[bold green]✓ permutation_test_{MODEL_KEY}.csv saved[/bold green]")

    # Plot cross-topic clustering
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cross_topic_rows))
    emotions_sorted = [r["emotion"] for r in cross_topic_rows]
    within_vals = [r["within_emo_cross_domain_sim"] for r in cross_topic_rows]
    cross_vals  = [r["cross_emo_sim"] for r in cross_topic_rows]

    ax.bar(x - 0.2, within_vals, 0.4, label="Within-emotion (cross-domain)", color="#27ae60", alpha=0.8)
    ax.bar(x + 0.2, cross_vals,  0.4, label="Cross-emotion",                  color="#e74c3c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(emotions_sorted, rotation=30, ha="right")
    ax.set_ylabel("Centroid cosine similarity"); ax.set_xlabel("Emotion")
    ax.set_title(f"Cross-Topic Clustering (Set B) — L{peak_layer}", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cross_topic_clustering_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
    plt.savefig(OUT_DIR / f"cross_topic_clustering_{MODEL_KEY}.svg", bbox_inches="tight")
    plt.close()
    console.print(f"[bold green]✓ cross_topic_clustering_{MODEL_KEY}.png/.svg saved[/bold green]")

# ============================================================================
# README
# ============================================================================

readme = f"""# Experiment 17 — v2 Representational Geometry

**Date:** 2026-02-25
**Model:** {MODEL_KEY} (Llama-3.2-1B-Instruct)
**Peak layer:** L{peak_layer} (from Exp 13 probe results)

## Research Questions

1. Do Set A and Set B emotions cluster together in representation space?
2. Is the structure organized by emotion (shared across sets) or by stimulus set?
3. In Set B, does emotion dominate over topic (domain)?

## Key Findings

| Metric | Value |
|--------|-------|
| Silhouette by emotion | {sil_emotion:.4f} |
| Silhouette by set | {sil_set:.4f} |
| Emotion > set structure | {'✓' if sil_emotion > sil_set else '✗'} |
| Within-emo cross-set cosine similarity | {np.mean(within_emo_sims):.4f} |
| Cross-emotion cosine similarity | {np.mean(cross_emo_sims):.4f} |
| Emotions dominating topic (Set B) | {sum(r['emotion_dominates'] for r in cross_topic_rows) if cross_topic_rows else '??'}/{len(cross_topic_rows) if cross_topic_rows else '??'} |

## Methodology

1. **Joint PCA**: Normalize Set A + Set B peak-layer activations, PCA to 2D
2. **Silhouette**: emotion vs set grouping silhouette scores
3. **Cosine similarity**: centroids per (emotion × set), compare same-emotion cross-set vs cross-emotion
4. **Cross-topic clustering**: within-emotion cross-domain vs cross-emotion similarity in Set B
5. **Permutation test**: 500 random label shuffles → empirical p-value

## Outputs

- `outputs/pca_joint_{MODEL_KEY}.png/.svg` — joint PCA visualization
- `outputs/cosine_similarity_matrix_{MODEL_KEY}.png/.svg` — centroid similarity heatmap
- `outputs/cross_topic_clustering_{MODEL_KEY}.png/.svg` — emotion vs topic comparison
- `outputs/silhouette_scores_{MODEL_KEY}.csv` — silhouette and cosine metrics
- `outputs/permutation_test_{MODEL_KEY}.csv` — permutation test results
"""

(EXP_DIR / "README.md").write_text(readme)
console.print(f"[bold green]✓ README.md written[/bold green]")
console.print(f"\n[bold green]✓ Experiment 17 complete![/bold green]")
