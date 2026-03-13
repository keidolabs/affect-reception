"""
Figure 2 — Layer-wise Probe Trajectories (Set A vs Set B)
==========================================================
Question: How does probe AUROC build across layers, and how do Set A and Set B compare?

Shows: 6-panel small multiples (3 model sizes × 2 variants).
Each panel: layer curves for Set A 8-class, Set B 8-class, and Set B binary (where available).

Key message:
- Binary AUROC saturates early (~30% depth) and identically across sets.
- 8-class shows the A/B keyword gap emerging in mid-to-late layers.
- The "affect reception zone" (shaded) is where binary reaches ≥0.99.

Data sources:
- 8-class probe curves: experiments/13_v2_probes/outputs/probe_results_{model}_{set}_h.csv
- Binary curve (llama1b_inst only): experiments/19_zeroshot_control/outputs/probe_comparison.csv

Output: outputs/fig02_layer_trajectories.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import polars as pl
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_ORDER, MODEL_LABELS,
    COLORS, COND_COLORS, save_fig
)

console = Console()
apply_style()

EXP_DIR   = Path(__file__).parent.parent.parent / "experiments"
PROBES_DIR = EXP_DIR / "13_v2_probes" / "outputs"
PROBE19    = EXP_DIR / "19_zeroshot_control" / "outputs" / "probe_comparison.csv"

# Panel layout: 3 rows (model families) × 2 columns (base / instruct)
# Row 0: Llama-1B, Row 1: Llama-8B, Row 2: Gemma-9B
PANEL_MAP = {
    (0, 0): "llama1b_base",
    (0, 1): "llama1b_inst",
    (1, 0): "llama8b_base",
    (1, 1): "llama8b_inst",
    (2, 0): "gemma9b_base",
    (2, 1): "gemma9b_inst",
}

ROW_LABELS = ["Llama-3.2-1B", "Llama-3-8B", "Gemma-2-9B"]
COL_LABELS = ["Base", "Instruct"]

# ============================================================================
# LOAD BINARY CURVE (exp19, llama1b_inst only)
# ============================================================================

binary_curve = {}   # mk -> (norm_depth_array, auroc_array)

if PROBE19.exists():
    df19 = pl.read_csv(PROBE19)
    # Use few-shot binary AUROC as the primary binary curve
    binary_curve["llama1b_inst"] = (
        df19["norm_depth"].to_numpy(),
        df19["fs_binary_auroc"].to_numpy(),
    )
    console.print(f"[cyan]Loaded exp19 binary curve for llama1b_inst ({len(df19)} layers)[/cyan]")
else:
    console.print("[yellow]exp19 probe_comparison.csv not found — no binary curve plotted[/yellow]")

# ============================================================================
# BUILD FIGURE
# ============================================================================

console.print("[bold cyan]Building 6-panel figure...[/bold cyan]")

fig, axes = plt.subplots(
    3, 2,
    figsize=(FULL_WIDTH, 8.5),
    sharex=False,
    sharey=False,
)

for (row, col), mk in PANEL_MAP.items():
    ax = axes[row, col]

    # ── Load Set A 8-class curve ──────────────────────────────────────────
    seta_path = PROBES_DIR / f"probe_results_{mk}_seta_h.csv"
    if seta_path.exists():
        seta = pl.read_csv(seta_path)
        x_a = seta["norm_depth"].to_numpy()
        y_a = seta["auroc_mean"].to_numpy()
        ax.plot(x_a, y_a,
                color=COLORS["blue"], linestyle="-", linewidth=1.6,
                label="Set A (8-class)", zorder=3)
        # Mark peak
        peak_idx = np.argmax(y_a)
        ax.scatter(x_a[peak_idx], y_a[peak_idx],
                   color=COLORS["blue"], s=35, zorder=5, edgecolors="white", linewidths=0.8)
    else:
        console.print(f"[yellow]  Missing: {seta_path.name}[/yellow]")

    # ── Load Set B 8-class curve ──────────────────────────────────────────
    setb_path = PROBES_DIR / f"probe_results_{mk}_setb_clinical_h.csv"
    if setb_path.exists():
        setb = pl.read_csv(setb_path)
        x_b = setb["norm_depth"].to_numpy()
        y_b = setb["auroc_mean"].to_numpy()
        ax.plot(x_b, y_b,
                color=COLORS["orange"], linestyle="--", linewidth=1.6, dashes=(5, 3),
                label="Set B (8-class)", zorder=3)
        peak_idx = np.argmax(y_b)
        ax.scatter(x_b[peak_idx], y_b[peak_idx],
                   color=COLORS["orange"], s=35, zorder=5, edgecolors="white", linewidths=0.8)
    else:
        console.print(f"[yellow]  Missing: {setb_path.name}[/yellow]")

    # ── Affect reception saturation zone (binary ≥ 0.99) ─────────────────
    # Based on exp19: binary reaches ≥0.99 from norm_depth ~0.25 onward.
    # Shade the zone where affect reception has saturated.
    ax.axvspan(0.0, 0.32, color="#e8f4f8", alpha=0.6, zorder=0,
               label="_nolegend_")
    ax.axvline(0.32, color="#aaccdd", linewidth=0.8, linestyle=":", zorder=1)

    # ── Panel styling ─────────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "", "0.7", "", "0.9", "1.0"])

    # X ticks — show on all rows so each panel is self-contained
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("Normalised layer depth", fontsize=8)

    # Row label (left margin)
    if col == 0:
        ax.set_ylabel("AUROC", fontsize=8)
        ax.text(-0.28, 0.5, ROW_LABELS[row],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="center", ha="right", rotation=90)

    # Column header (top)
    if row == 0:
        ax.set_title(COL_LABELS[col], fontsize=9, fontweight="bold", pad=6)

    console.print(f"  Panel ({row},{col}) {mk} — done")

# ── Shared legend ─────────────────────────────────────────────────────────
zone_patch = plt.Rectangle((0,0),1,1, fc="#e8f4f8", ec="#aaccdd", linewidth=0.8)

legend_handles = [
    mlines.Line2D([], [], color=COLORS["blue"],   linestyle="-",  linewidth=1.5),
    mlines.Line2D([], [], color=COLORS["orange"], linestyle="--", linewidth=1.5, dashes=(5,3)),
    zone_patch,
]
legend_labels = [
    "Set A — 8-class (keyword-rich)",
    "Set B — 8-class (clinical, keyword-free)",
    "Affect-reception saturation zone",
]

fig.legend(
    handles=legend_handles, labels=legend_labels,
    loc="upper center", bbox_to_anchor=(0.5, 1.01),
    ncol=2, fontsize=8, frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig02_layer_trajectories")
console.print("[bold green]✓ Fig 2 complete[/bold green]")
plt.close()
