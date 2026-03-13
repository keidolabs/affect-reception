"""
Figure 6 — Representational Geometry: Instruction Tuning Effect
================================================================
Question: How does instruction tuning reorganize emotional representations?

Shows two panels:
  Panel A: PCA scatter of activations at peak layer (Llama-8B base vs instruct)
           Points colored by emotion, shaped by set (Set A = ○, Set B = △)
           Base: clusters by set. Instruct: clusters by emotion.
  Panel B: Silhouette score comparison — by-emotion vs by-set, for base vs instruct

Key message: Alignment training doesn't create emotion encoding — it reorganizes
geometry from set-organized to emotion-organized.

Data sources:
  Panel A: .npz activation files from experiments/11_v2_extract_seta/ and
           experiments/12_v2_extract_setb/ at peak layer for llama8b_base and inst
  Panel B: experiments/17_v2_geometry/outputs/silhouette_scores_{model}.csv

Output: outputs/fig06_geometry.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import polars as pl
from sklearn.decomposition import PCA
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_LABELS, COLORS,
    EMOTION_COLORS, EMOTION_MARKERS, save_fig
)

console = Console()
apply_style()
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

# Plutchik high-intensity names used in the data → display labels in EMOTION_COLORS
PLUTCHIK_MAP = {
    "rage":       "anger",
    "loathing":   "disgust",
    "terror":     "fear",
    "grief":      "grief",
    "ecstasy":    "happiness",
    "admiration": "love",
    "amazement":  "surprise",
    "vigilance":  "neutral",
}

EXP_DIR   = Path(__file__).parent.parent.parent / "experiments"
GEO_DIR   = EXP_DIR / "17_v2_geometry" / "outputs"
ACTS_SETA = EXP_DIR / "11_v2_extract_seta" / "outputs" / "activations"
ACTS_SETB = EXP_DIR / "12_v2_extract_setb" / "outputs" / "activations"

# Peak layers from summary_table (residual stream, h):
#   llama8b_base: layer 17 (norm_depth 0.5625)
#   llama8b_inst: layer 31 (norm_depth 1.000)
PEAK_LAYERS = {
    "llama8b_base": 17,
    "llama8b_inst": 31,
}

MODELS_PANEL_A = ["llama8b_base", "llama8b_inst"]
PANEL_TITLES   = ["Llama-3-8B Base", "Llama-3-8B Instruct"]

# ============================================================================
# PANEL A — PCA SCATTER
# Load .npz activation files, extract at peak layer, run PCA jointly
# ============================================================================

def load_activations_for_model(model_key, peak_layer):
    """
    Load all .npz activation files for a model across Set A and Set B.
    Returns (embeddings, emotions, sets) where embeddings is (n, d_model).
    """
    embeddings = []
    emotions   = []
    sets       = []

    for set_name, acts_dir in [("A", ACTS_SETA), ("B", ACTS_SETB)]:
        model_dir = acts_dir / model_key
        if not model_dir.exists():
            console.print(f"[yellow]  Missing dir: {model_dir}[/yellow]")
            continue

        # Use manifest.csv for ground-truth emotion labels
        manifest_path = model_dir / "manifest.csv"
        id_to_emotion = {}
        if manifest_path.exists():
            mf = pl.read_csv(manifest_path)
            for row in mf.iter_rows(named=True):
                raw_emo = row["emotion"].lower()
                # Map Plutchik name → display label; keep as-is if already a display label
                id_to_emotion[row["id"]] = PLUTCHIK_MAP.get(raw_emo, raw_emo)

        # Only use Set A (emotional) files for Panel A — skip neutral controls (N- prefix)
        npz_files = sorted(f for f in model_dir.glob("*.npz")
                           if not f.stem.startswith("N-"))
        console.print(f"  {model_key} Set {set_name}: {len(npz_files)} emotional files")

        for fpath in npz_files:
            data = np.load(fpath)
            keys = list(data.files)
            key = keys[0] if len(keys) == 1 else "activations" if "activations" in keys else keys[0]
            arr = data[key]

            if arr.ndim == 3:
                layer_act = arr[peak_layer].mean(axis=0)
            elif arr.ndim == 2:
                layer_act = arr[peak_layer]
            else:
                console.print(f"[yellow]  Unexpected shape {arr.shape} in {fpath.name}[/yellow]")
                continue

            embeddings.append(layer_act.astype(np.float32))
            emotion = id_to_emotion.get(fpath.stem, "unknown")
            emotions.append(emotion)
            sets.append(set_name)

    if not embeddings:
        return None, None, None

    return np.stack(embeddings), emotions, sets


console.print("[bold cyan]Loading activations for Panel A...[/bold cyan]")

pca_data = {}
can_do_panel_a = True

for mk in MODELS_PANEL_A:
    peak_layer = PEAK_LAYERS[mk]
    embs, emos, stim_sets = load_activations_for_model(mk, peak_layer)
    if embs is None:
        console.print(f"[yellow]  No activations for {mk} — Panel A will be skipped[/yellow]")
        can_do_panel_a = False
        break
    pca_data[mk] = (embs, emos, stim_sets)
    console.print(f"  {mk}: {embs.shape[0]} stimuli × {embs.shape[1]} dims (layer {peak_layer})")

# Run PCA jointly across both models? No — separately per panel for clarity.
if can_do_panel_a:
    for mk in MODELS_PANEL_A:
        embs, emos, stim_sets = pca_data[mk]
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embs)
        var_explained = pca.explained_variance_ratio_
        pca_data[mk] = (coords, emos, stim_sets, var_explained)
        console.print(f"  {mk} PCA: PC1={var_explained[0]:.1%} PC2={var_explained[1]:.1%}")

# ============================================================================
# PANEL B — SILHOUETTE SCORES
# Load silhouette_scores_{model}.csv for llama8b_base and llama8b_inst
# ============================================================================

console.print("[bold cyan]Loading silhouette scores for Panel B...[/bold cyan]")

sil_records = {}  # mk -> {"emotion": float, "set": float}

for mk in MODELS_PANEL_A:
    path = GEO_DIR / f"silhouette_scores_{mk}.csv"
    if not path.exists():
        console.print(f"[yellow]  Missing: {path.name}[/yellow]")
        continue
    df = pl.read_csv(path)
    emo_sil = df.filter(pl.col("grouping") == "emotion")["silhouette"]
    set_sil = df.filter(pl.col("grouping") == "set")["silhouette"]
    sil_records[mk] = {
        "emotion": float(emo_sil.mean()),
        "set":     float(set_sil.mean()),
    }
    console.print(f"  {mk}: by-emotion={sil_records[mk]['emotion']:.4f}  by-set={sil_records[mk]['set']:.4f}")

# ============================================================================
# BUILD FIGURE
# Layout: top row = Panel A (PCA scatters, 1×2)
#         bottom row = Panel B (silhouette bars)
# ============================================================================

console.print("[bold cyan]Building figure...[/bold cyan]")

fig = plt.figure(figsize=(FULL_WIDTH, 6.2))
# GridSpec: 2 rows, Panel A taller
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2.2, 1.0], hspace=0.85, wspace=0.35)

ax_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_b  = fig.add_subplot(gs[1, :])   # Panel B spans full width

# ── Panel A: PCA scatter ──────────────────────────────────────────────────

if can_do_panel_a:
    for pi, mk in enumerate(MODELS_PANEL_A):
        ax = ax_a[pi]
        coords, emos, stim_sets, var_exp = pca_data[mk]

        for emo in EMOTION_COLORS:
            for stim_set, marker in [("A", "o"), ("B", "^")]:
                mask = [
                    e == emo and s == stim_set
                    for e, s in zip(emos, stim_sets)
                ]
                if not any(mask):
                    continue
                idx = np.where(mask)[0]
                ax.scatter(
                    coords[idx, 0], coords[idx, 1],
                    color=EMOTION_COLORS[emo],
                    marker=marker,
                    s=22 if stim_set == "A" else 28,
                    alpha=0.85,
                    edgecolors="white" if stim_set == "B" else "none",
                    linewidths=0.5,
                    zorder=3,
                )

        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=8)
        ax.set_title(PANEL_TITLES[pi], fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    # Shared legend for Panel A: emotions
    legend_emo = [
        mpatches.Patch(color=EMOTION_COLORS[e], label=e.capitalize())
        for e in EMOTION_COLORS
    ]
    legend_set = [
        mlines.Line2D([], [], marker="o", color="#666666", linestyle="None",
                      markersize=5, label="Set A (keyword-rich)"),
        mlines.Line2D([], [], marker="^", color="#666666", linestyle="None",
                      markersize=5, label="Set B (clinical)"),
    ]
    # Place legend below both Panel A charts, above Panel B
    fig.legend(
        handles=legend_emo + legend_set,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.42),
        fontsize=6.5, ncol=5,
        frameon=True, framealpha=0.9, edgecolor="#dddddd",
    )

    # Panel A label
    fig.text(0.02, 0.98, "A", fontsize=11, fontweight="bold", va="top")
else:
    for ax in ax_a:
        ax.text(0.5, 0.5, "Activation .npz files\nnot found locally.\nRun extraction first.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#999999", style="italic")
        ax.set_xticks([])
        ax.set_yticks([])

# ── Panel B: Silhouette bars ──────────────────────────────────────────────

fig.text(0.02, 0.38, "B", fontsize=11, fontweight="bold", va="top")

if sil_records:
    n_models = len(MODELS_PANEL_A)
    x = np.arange(n_models)
    bar_width = 0.35

    emo_vals = [sil_records.get(mk, {}).get("emotion", np.nan) for mk in MODELS_PANEL_A]
    set_vals = [sil_records.get(mk, {}).get("set",     np.nan) for mk in MODELS_PANEL_A]

    bars_emo = ax_b.bar(x - bar_width/2, emo_vals, bar_width,
                        color=COLORS["orange"], label="By emotion", zorder=3)
    bars_set = ax_b.bar(x + bar_width/2, set_vals, bar_width,
                        color=COLORS["sky_blue"], label="By stimulus set", zorder=3)

    # Annotate bars
    for bar in bars_emo + bars_set:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax_b.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                  f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax_b.axhline(0, color="#888888", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([PANEL_TITLES[i] for i in range(n_models)], fontsize=8.5)
    ax_b.set_ylabel("Silhouette score", fontsize=9)
    ax_b.set_ylim(top=max(filter(lambda v: not np.isnan(v), emo_vals + set_vals)) * 1.30)
    ax_b.set_title(
        "Geometry: instruction tuning flips organisation from set-structured to emotion-structured",
        fontsize=8.5, pad=12
    )
    ax_b.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        fontsize=8, ncol=2, frameon=True, framealpha=0.9, edgecolor="#dddddd",
    )
    # Add zero line annotation
    ax_b.text(x[-1] + 0.5, 0.002, "random = 0", fontsize=7, color="#888888")
else:
    ax_b.text(0.5, 0.5, "Silhouette CSV files not found.\nCheck experiments/17_v2_geometry/outputs/",
              ha="center", va="center", transform=ax_b.transAxes,
              fontsize=9, color="#999999", style="italic")
    ax_b.set_xticks([])
    ax_b.set_yticks([])

plt.suptitle(
    "Representational geometry: how instruction tuning reorganises emotion encoding",
    fontsize=9, y=1.01
)

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig06_geometry")
console.print("[bold green]✓ Fig 6 complete[/bold green]")
plt.close()
