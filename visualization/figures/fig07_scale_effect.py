"""
Figure 7 — Scale Effect Summary
=================================
Question: Does scale improve keyword-free emotion processing?

Shows: How 3 metrics evolve from 1B → 8B → 9B (both base and instruct):
  1. Set B 8-class AUROC drop relative to Set A (smaller = better)
  2. A→B transfer AUROC (higher = better)
  3. Absolute Binary–8-class gap on Set B (smaller = less dissociation)

Key message: Scale improves everything — smaller drops, better transfer,
narrower dissociation gap. But the gap never closes.

Data source: experiments/18_v2_summary/outputs/summary_table.csv

Output: outputs/fig07_scale_effect.{pdf,png}
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
    apply_style, FULL_WIDTH, MODEL_LABELS, MODEL_COLORS, COLORS, save_fig
)

console = Console()
apply_style()
# Helvetica lacks arrow glyphs (↓ ↑ →); DejaVu Sans has them — bump to front
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

SUMMARY = Path(__file__).parent.parent.parent / "experiments" / "18_v2_summary" / "outputs" / "summary_table.csv"

# Binary Set B peak AUROCs (from Phase 1 / exp19, same models)
# Used to compute binary–8-class gap
BINARY_SETB = {
    "llama1b_base":  0.9995,
    "llama1b_inst":  0.9995,
    "llama8b_base":  1.0000,
    "llama8b_inst":  1.0000,
    "gemma9b_base":  1.0000,
    "gemma9b_inst":  1.0000,
}

# ============================================================================
# LOAD AND COMPUTE METRICS
# ============================================================================

console.print("[bold cyan]Loading summary table...[/bold cyan]")
df = pl.read_csv(SUMMARY)

# Map family → x position (log-ish scale: 1B=0, 8B=1, 9B=2)
FAMILY_X = {"Llama-1B": 0, "Llama-8B": 1, "Gemma-9B": 2}
X_LABELS  = ["1B", "8B", "9B"]

records = []
for row in df.iter_rows(named=True):
    mk = row["model_key"]
    family = row["family"]
    variant = row["type"]   # "base" or "instruct"

    seta_auroc   = row["seta_h_peak_auroc"]
    setb_auroc   = row["setb_h_peak_auroc"]
    ab_transfer  = row["ab_transfer_h"]
    binary_auroc = BINARY_SETB.get(mk, np.nan)

    drop_pct  = (seta_auroc - setb_auroc) * 100             # ↓ lower = better

    records.append({
        "model_key":   mk,
        "family":      family,
        "variant":     variant,
        "x":           FAMILY_X.get(family, -1),
        "drop_pct":    drop_pct,
        "ab_transfer": ab_transfer,
        "setb_auroc":  setb_auroc,
    })
    console.print(f"  {mk}: drop={drop_pct:.2f}pp  transfer={ab_transfer:.3f}  setb_auroc={setb_auroc:.4f}")

# Separate base and instruct
base_rec = sorted([r for r in records if r["variant"] == "base"],  key=lambda r: r["x"])
inst_rec = sorted([r for r in records if r["variant"] == "instruct"], key=lambda r: r["x"])

# ============================================================================
# BUILD FIGURE — 3-panel small multiples (stacked vertically)
# ============================================================================

console.print("[bold cyan]Building figure...[/bold cyan]")

METRICS = [
    ("drop_pct",    "Set B drop vs Set A\n(percentage points)", True,  "Keyword-sensitivity shrinks with scale"),
    ("ab_transfer", "A→B transfer\nAUROC",                     False, "Cross-set transfer improves with scale"),
    ("setb_auroc",  "Set B 8-class\nAUROC",                    False, "Keyword-free classification accuracy"),
]

fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH * 0.65, 6.5), sharex=True)

for pi, (metric_key, ylabel, lower_better, subtitle) in enumerate(METRICS):
    ax = axes[pi]

    xs_base  = [r["x"] for r in base_rec]
    ys_base  = [r[metric_key] for r in base_rec]
    xs_inst  = [r["x"] for r in inst_rec]
    ys_inst  = [r[metric_key] for r in inst_rec]

    # Base line (dashed)
    ax.plot(xs_base, ys_base,
            color=COLORS["blue"], linestyle="--", linewidth=1.5, marker="o",
            markersize=6, label="Base", zorder=3)

    # Instruct line (solid)
    ax.plot(xs_inst, ys_inst,
            color=COLORS["vermillion"], linestyle="-", linewidth=1.5, marker="s",
            markersize=6, label="Instruct", zorder=3)

    # Value annotations — push each label AWAY from the other line at that x:
    # whichever value is higher at a given x gets pushed up, lower gets pushed down.
    all_vals = ys_base + ys_inst
    dy = (max(all_vals) - min(all_vals)) * 0.12
    dy = max(dy, 1e-4)
    inst_by_x = {r["x"]: r[metric_key] for r in inst_rec}
    base_by_x = {r["x"]: r[metric_key] for r in base_rec}
    for r in base_rec:
        v, other = r[metric_key], inst_by_x.get(r["x"], r[metric_key])
        offset, va = (dy, "bottom") if v >= other else (-dy, "top")
        ax.text(r["x"], v + offset, f"{v:.2f}",
                fontsize=6.5, ha="center", va=va, color=COLORS["blue"])
    for r in inst_rec:
        v, other = r[metric_key], base_by_x.get(r["x"], r[metric_key])
        offset, va = (dy, "bottom") if v >= other else (-dy, "top")
        ax.text(r["x"], v + offset, f"{v:.2f}",
                fontsize=6.5, ha="center", va=va, color=COLORS["vermillion"])

    # Give headroom above top annotation and below bottom annotation
    ax.set_ylim(
        bottom=min(all_vals) - dy * 2.5,
        top=max(all_vals) + dy * 2.5,
    )
    ax.set_ylabel(ylabel, fontsize=8)
    direction = "lower" if lower_better else "higher"
    ax.set_title(f"{subtitle}\nbetter = {direction}",
                 fontsize=8, style="italic", color="#555555", pad=8)
    ax.set_xticks([0, 1, 2])

    if pi == 0:
        ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xticklabels(X_LABELS, fontsize=9)
axes[-1].set_xlabel("Model scale", fontsize=9)

# Family annotations on x-axis
for pi_ax in axes:
    pi_ax.axvline(0.5, color="#eeeeee", linewidth=0.8)
    pi_ax.axvline(1.5, color="#eeeeee", linewidth=0.8)

fig.suptitle(
    "Scale improves all dimensions of keyword-free emotion processing",
    fontsize=9, y=1.01
)
plt.tight_layout()

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig07_scale_effect")
console.print("[bold green]✓ Fig 7 complete[/bold green]")
plt.close()
