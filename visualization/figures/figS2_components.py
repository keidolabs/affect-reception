"""
Figure S2 — Component Decomposition: MHSA vs FFN (Supplementary)
=================================================================
Section: 4

Shows: Layer-wise AUROC for residual stream (h), attention output (a),
and FFN/MLP output (m) across all 6 models.

Key message: MHSA peaks before FFN, suggesting a two-stage integration:
attention integrates context, FFN specialises emotion categories.

Data source:
- experiments/13_v2_probes/outputs/probe_results_{model}_{set}_{act}.csv
  act ∈ {h, a, m}, set ∈ {seta, setb_clinical}

Output: outputs/figS2_components.{pdf,png}
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
    apply_style, FULL_WIDTH, MODEL_ORDER, MODEL_LABELS, COLORS, save_fig
)

console = Console()
apply_style()

PROBES_DIR = Path(__file__).parent.parent.parent / "experiments" / "13_v2_probes" / "outputs"

# Component styles
ACT_STYLES = {
    "h": {"color": COLORS["blue"],     "linestyle": "-",   "linewidth": 1.6, "label": "Residual stream (h)"},
    "a": {"color": COLORS["orange"],   "linestyle": "--",  "linewidth": 1.2, "label": "MHSA output (a)",   "dashes": (5,3)},
    "m": {"color": COLORS["vermillion"],"linestyle": ":",  "linewidth": 1.2, "label": "FFN output (m)"},
}
ACT_TYPES = ["h", "a", "m"]

# Panel map: 3 rows × 2 cols
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

# Show Set A only (cleaner; supplementary can show both if desired)
SET_KEY   = "seta"
SET_LABEL = "Set A"

# ============================================================================
# BUILD FIGURE
# ============================================================================

console.print("[bold cyan]Building component decomposition figure...[/bold cyan]")

fig, axes = plt.subplots(3, 2, figsize=(FULL_WIDTH, 8.5), sharex=False, sharey=False)

for (row, col), mk in PANEL_MAP.items():
    ax = axes[row, col]

    for act in ACT_TYPES:
        path = PROBES_DIR / f"probe_results_{mk}_{SET_KEY}_{act}.csv"
        if not path.exists():
            continue
        df     = pl.read_csv(path)
        x      = df["norm_depth"].to_numpy()
        y      = df["auroc_mean"].to_numpy()
        style  = ACT_STYLES[act]

        plot_kwargs = {
            "color":     style["color"],
            "linestyle": style["linestyle"],
            "linewidth": style["linewidth"],
            "label":     style["label"],
            "zorder":    3,
        }
        if "dashes" in style:
            plot_kwargs["dashes"] = style["dashes"]

        ax.plot(x, y, **plot_kwargs)

        # Peak dot
        peak_idx = np.argmax(y)
        ax.scatter(x[peak_idx], y[peak_idx],
                   color=style["color"], s=30, zorder=5,
                   edgecolors="white", linewidths=0.7)

    # Panel styling
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.set_yticks([0.5, 0.7, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.7", "0.9", "1.0"])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"] if row == 2 else [])

    if col == 0:
        ax.set_ylabel("AUROC", fontsize=8)
        ax.text(-0.28, 0.5, ROW_LABELS[row],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="center", ha="right", rotation=90)

    if row == 0:
        ax.set_title(COL_LABELS[col], fontsize=9, fontweight="bold", pad=6)

    if row == 2:
        ax.set_xlabel("Normalised layer depth", fontsize=8)

    # Chance line
    ax.axhline(0.125, color="#dddddd", linewidth=0.7, linestyle=":", zorder=0)

    console.print(f"  Panel ({row},{col}) {mk} — done")

# Shared legend
legend_handles = [
    mlines.Line2D([], [], **{k: v for k, v in ACT_STYLES[a].items()
                              if k in ("color", "linestyle", "linewidth")},
                  label=ACT_STYLES[a]["label"])
    for a in ACT_TYPES
]
fig.legend(
    handles=legend_handles,
    loc="upper center", bbox_to_anchor=(0.5, 1.01),
    ncol=3, fontsize=8, frameon=False,
)

fig.suptitle(
    f"S2 — Component decomposition (MHSA / FFN / Residual) on {SET_LABEL}",
    fontsize=9, y=1.035
)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "figS2_components")
console.print("[bold green]✓ Fig S2 complete[/bold green]")
plt.close()
