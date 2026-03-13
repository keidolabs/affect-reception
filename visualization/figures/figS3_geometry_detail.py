"""
Figure S3 — Cross-Topic Permutation Detail (Supplementary)
===========================================================
Section: 7.3

Shows: Per-emotion cross-topic clustering: within-emotion cross-domain cosine
similarity vs. cross-emotion cosine similarity, with permutation test p-values.

Key message: Grief shows the most consistent cross-topic convergence across
multiple models. The power limitation is visually apparent — most emotions fail
significance despite observable trends.

Data source:
- experiments/17_v2_geometry/outputs/permutation_test_{model}.csv
  Columns: emotion, within_emo_cross_domain_sim, cross_emo_sim, null_sim,
           p_permutation, emotion_dominates

Output: outputs/figS3_geometry_detail.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import polars as pl
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_ORDER, MODEL_LABELS,
    EMOTION_COLORS, COLORS, save_fig
)

console = Console()
apply_style()

GEO_DIR = Path(__file__).parent.parent.parent / "experiments" / "17_v2_geometry" / "outputs"

# Show representative models (avoid too many panels)
MODELS_SHOWN = ["llama8b_base", "llama8b_inst", "gemma9b_base", "gemma9b_inst"]

EMOTIONS = ["anger", "disgust", "fear", "grief", "happiness", "love", "neutral", "surprise"]
EMO_DISPLAY = [e.capitalize() for e in EMOTIONS]

ALPHA_BONFERRONI = 0.05 / 8   # Bonferroni correction for 8 emotions
ALPHA_FDR        = 0.05        # BH FDR threshold (applied per model)

# ============================================================================
# LOAD PERMUTATION TEST DATA
# ============================================================================

console.print("[bold cyan]Loading permutation test data...[/bold cyan]")

perm_data = {}
for mk in MODELS_SHOWN:
    path = GEO_DIR / f"permutation_test_{mk}.csv"
    if not path.exists():
        console.print(f"[yellow]  Missing: {path.name}[/yellow]")
        continue
    df = pl.read_csv(path)
    perm_data[mk] = df
    n_sig = int(df.filter(pl.col("emotion_dominates") == True).height)
    console.print(f"  {mk}: {n_sig}/{len(df)} emotions dominate (p < threshold)")

if not perm_data:
    console.print("[bold red]No permutation test data found![/bold red]")
    raise SystemExit(1)

# ============================================================================
# BUILD FIGURE — 2×2 grid (one panel per model shown)
# ============================================================================

console.print("[bold cyan]Building permutation detail figure...[/bold cyan]")

n_rows = 2
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(FULL_WIDTH, 6.5),
                          sharex=True, sharey=False)
axes_flat = axes.flatten()

x = np.arange(len(EMOTIONS))
bar_width = 0.38

for pi, mk in enumerate(MODELS_SHOWN):
    if pi >= len(axes_flat):
        break
    ax = axes_flat[pi]

    if mk not in perm_data:
        ax.text(0.5, 0.5, "Data\nnot found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#999999")
        continue

    df = perm_data[mk]
    # Align to standard emotion order
    emo_map = {
        row["emotion"]: row for row in df.iter_rows(named=True)
    }

    within_vals = []
    cross_vals  = []
    p_vals      = []
    dominates   = []
    for emo in EMOTIONS:
        row = emo_map.get(emo, {})
        within_vals.append(float(row.get("within_emo_cross_domain_sim", np.nan)))
        cross_vals.append(float(row.get("cross_emo_sim",                np.nan)))
        p_vals.append(float(row.get("p_permutation",                    1.0)))
        dominates.append(bool(row.get("emotion_dominates",               False)))

    # Bars: within-emotion similarity (blue) and cross-emotion (grey)
    bars_w = ax.bar(x - bar_width/2, within_vals, bar_width,
                    color=COLORS["blue"], alpha=0.85, label="Within-emo\ncross-domain",
                    edgecolor="white", linewidth=0.4, zorder=3)
    bars_c = ax.bar(x + bar_width/2, cross_vals, bar_width,
                    color="#bbbbbb", alpha=0.85, label="Cross-emo\nmean sim.",
                    edgecolor="white", linewidth=0.4, zorder=3)

    # P-value significance markers above bars
    for i, (p, dom) in enumerate(zip(p_vals, dominates)):
        max_bar = max(within_vals[i], cross_vals[i])
        if np.isnan(max_bar):
            continue
        y_marker = max_bar + 0.015
        if p < ALPHA_BONFERRONI:
            ax.text(x[i], y_marker + 0.01, "**", ha="center", fontsize=9,
                    color=COLORS["vermillion"], fontweight="bold")
        elif p < ALPHA_FDR:
            ax.text(x[i], y_marker, "*", ha="center", fontsize=9,
                    color=COLORS["orange"])
        # Emotion-specific color dots for emotions that dominate
        if dom:
            ax.scatter(x[i], y_marker + 0.03, marker="^",
                       color=EMOTION_COLORS.get(EMOTIONS[i], COLORS["blue"]),
                       s=25, zorder=5)

    # Highlight grief (known convergent emotion)
    grief_idx = EMOTIONS.index("grief")
    ax.axvspan(grief_idx - 0.5, grief_idx + 0.5, color="#fffde7", alpha=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(EMO_DISPLAY, rotation=40, ha="right", fontsize=7.5,
                        visible=(pi >= n_cols))
    ax.set_ylabel("Cosine similarity", fontsize=8)
    ax.set_ylim(0, min(1.0, max(np.nanmax(within_vals + cross_vals) * 1.25 + 0.05, 0.3)))
    ax.set_title(MODEL_LABELS[mk].replace("\n", " "), fontsize=8.5)

    if pi == 0:
        ax.legend(fontsize=7.5, loc="upper right")

    # p-value legend text
    ax.text(0.02, 0.97,
            "* p<0.05 FDR\n** p<0.006 Bonferroni",
            transform=ax.transAxes, fontsize=6.5, va="top", color="#555555")

# Hide unused panels
for pi in range(len(MODELS_SHOWN), len(axes_flat)):
    axes_flat[pi].set_visible(False)

fig.suptitle(
    "S3 — Cross-topic emotion clustering: within-emotion vs. cross-emotion cosine similarity\n"
    "(grief shows most consistent convergence; shaded column)",
    fontsize=8.5, y=1.01
)
plt.tight_layout()

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "figS3_geometry_detail")
console.print("[bold green]✓ Fig S3 complete[/bold green]")
plt.close()
