"""
Figure 1 — The Core Dissociation (Hero Figure)
===============================================
Question: How large is the binary vs. 8-class AUROC gap on Set B (keyword-free)?

Shows: Binary AUROC vs. 8-class AUROC on Set B, across all 6 models.
The gap between the two dots IS the dissociation — perfect affect reception,
imperfect emotion categorization. Only visible without keywords.

Data sources:
- 8-class Set B AUROC: experiments/18_v2_summary/outputs/summary_table.csv
- Binary Set B AUROC: experiments/19_zeroshot_control/outputs/probe_comparison.csv
  (confirmed for llama1b_inst); Phase 1 results used as proxy for other v2 models.

Output: outputs/fig01_dissociation.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polars as pl
from rich.console import Console

# Allow importing style_config from same directory
sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_ORDER, MODEL_LABELS,
    MODEL_COLORS, COND_COLORS, save_fig
)

console = Console()
apply_style()

EXP_DIR   = Path(__file__).parent.parent.parent / "experiments"
SUMMARY   = EXP_DIR / "18_v2_summary" / "outputs" / "summary_table.csv"
PROBE19   = EXP_DIR / "19_zeroshot_control" / "outputs" / "probe_comparison.csv"

# ============================================================================
# LOAD 8-CLASS SET B AUROC (from v2 summary table)
# ============================================================================

console.print("[bold cyan]Loading summary table...[/bold cyan]")
summary = pl.read_csv(SUMMARY)
console.print(f"  {len(summary)} models loaded")

eightclass_setb = {
    row["model_key"]: row["setb_h_peak_auroc"]
    for row in summary.iter_rows(named=True)
}

# ============================================================================
# LOAD / DERIVE BINARY SET B AUROC
#
# For llama1b_inst: take max(fs_binary_auroc) from exp19 probe_comparison.csv.
# For all others: use Phase 1 results which used the same models on the same
# Set B stimuli (different probe training protocol, same stimulus set).
# Phase 1 results show binary saturates to ≥0.9995 for all models tested.
#
# Source citations:
#   llama1b_inst: experiments/19_zeroshot_control (v2 pipeline, direct)
#   llama1b_base: Phase 1 (same architecture, binary = 0.9995 at L10)
#   llama8b_*:    Phase 1 (L9, binary = 1.0000)
#   gemma9b_*:    Phase 1 (L17, binary = 1.0000)
# ============================================================================

binary_setb = {
    "llama1b_inst": None,   # will be filled from exp19
    "llama1b_base": 0.9995,  # Phase 1 proxy (same architecture)
    "llama8b_inst": 1.0000,  # Phase 1 result (Llama-3-8B-Instruct, L9)
    "llama8b_base": 1.0000,  # Phase 1 result (Llama-3-8B, L9)
    "gemma9b_inst": 1.0000,  # Phase 1 result (Gemma-2-9B, L17)
    "gemma9b_base": 1.0000,  # Phase 1 result (Gemma-2-9B, L17)
}

if PROBE19.exists():
    df19 = pl.read_csv(PROBE19)
    # fs_binary_auroc peaks to 1.0 from layer 4 onward for llama1b_inst
    binary_setb["llama1b_inst"] = float(df19["fs_binary_auroc"].max())
    console.print(f"  llama1b_inst binary peak (exp19): {binary_setb['llama1b_inst']:.4f}")
else:
    console.print("[yellow]  exp19 probe_comparison.csv not found — using Phase 1 proxy for llama1b_inst[/yellow]")
    binary_setb["llama1b_inst"] = 0.9995

console.print("  Binary Set B AUROCs (all models):")
for mk in MODEL_ORDER:
    console.print(f"    {mk}: {binary_setb[mk]:.4f}")

# ============================================================================
# BUILD PLOT
#
# Dumbbell / lollipop chart:
# - One row per model (y-axis)
# - Filled dot = binary AUROC (blue)
# - Open dot = 8-class AUROC (orange)
# - Horizontal line connecting them = the gap
# - Models grouped by family with faint band separators
# ============================================================================

console.print("[bold cyan]Building figure...[/bold cyan]")

fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2))

y_positions = np.arange(len(MODEL_ORDER))
gap_text_x_offset = 0.005   # for gap annotation

for i, mk in enumerate(MODEL_ORDER):
    y       = y_positions[i]
    b_auroc = binary_setb[mk]
    c_auroc = eightclass_setb.get(mk, np.nan)
    gap_pp  = (b_auroc - c_auroc) * 100

    # Connecting line (gap)
    ax.hlines(y, c_auroc, b_auroc, colors="#cccccc", linewidth=2.5, zorder=2)

    # 8-class dot (open)
    ax.scatter(c_auroc, y, color=COND_COLORS["8class"],
               s=60, zorder=4, edgecolors=COND_COLORS["8class"], facecolors="white",
               linewidths=1.5)

    # Binary dot (filled)
    ax.scatter(b_auroc, y, color=COND_COLORS["binary"],
               s=60, zorder=4)

    # Gap annotation (pp label between the two dots)
    mid_x  = (b_auroc + c_auroc) / 2
    ax.text(mid_x, y + 0.22, f"Δ{gap_pp:.1f}pp",
            ha="center", va="bottom", fontsize=7, color="#555555")

# Family separator bands (subtle)
ax.axhspan(1.5, 3.5, color="#f5f5f5", zorder=0)   # Llama-8B band

# Axis labels and style
ax.set_yticks(y_positions)
ax.set_yticklabels([MODEL_LABELS[mk] for mk in MODEL_ORDER], fontsize=8)
ax.set_xlabel("AUROC on Set B (keyword-free clinical vignettes)", fontsize=9)
ax.set_xlim(0.885, 1.015)
ax.set_ylim(-0.8, 5.6)   # extra headroom so top annotation clears the title
ax.set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
ax.set_xticklabels(["0.90", "0.92", "0.94", "0.96", "0.98", "1.00"])

# Chance-level note (0.125 would be far left, off-scale — note in text)
ax.text(0.888, -0.55, "Chance (8-class) = 0.125  [off scale]",
        fontsize=7, color="#888888", style="italic")

# Family labels on right spine
ax.text(1.017, 0.5,  "Llama\n1B",  fontsize=8, va="center", ha="left", color="#666666",
        transform=ax.get_yaxis_transform())
ax.text(1.017, 2.5,  "Llama\n8B",  fontsize=8, va="center", ha="left", color="#666666",
        transform=ax.get_yaxis_transform())
ax.text(1.017, 4.5,  "Gemma\n9B", fontsize=8, va="center", ha="left", color="#666666",
        transform=ax.get_yaxis_transform())

# Legend
legend_handles = [
    mpatches.Patch(facecolor=COND_COLORS["binary"],  label="Binary AUROC (affect reception)"),
    mpatches.Patch(facecolor="white", edgecolor=COND_COLORS["8class"], linewidth=1.5,
                   label="8-class AUROC (emotion categorization)"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          bbox_to_anchor=(0.01, 0.99))

ax.set_title(
    "Affect reception vs. emotion categorization on keyword-free stimuli (Set B)",
    fontsize=9, pad=8
)

plt.tight_layout()

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig01_dissociation")
console.print("[bold green]✓ Fig 1 complete[/bold green]")
plt.close()
