"""
Figure 3 — Cross-Set Transfer Asymmetry
========================================
Question: Does a probe trained on clinical text (Set B) generalize to keyword-rich
text (Set A), or vice versa? And which direction is stronger?

Shows: Peak A→B and B→A transfer AUROC per model — the asymmetry reveals that
Set B probes subsume Set A representations (clinical ⊂ keyword-rich).

Key message: B→A consistently exceeds A→B. Clinical representations are more
general. The gap narrows with scale (better representations).

Data source:
- experiments/13_v2_probes/outputs/transfer_results_{model}_h.csv
  Columns: layer, norm_depth, model_key, act_type, condition, auroc_mean

Output: outputs/fig03_transfer_asymmetry.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polars as pl
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_ORDER, MODEL_LABELS,
    COLORS, COND_COLORS, save_fig
)

console = Console()
apply_style()
# Helvetica lacks U+2192 (→); DejaVu Sans has it — bump to front for this figure only
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

PROBES_DIR = Path(__file__).parent.parent.parent / "experiments" / "13_v2_probes" / "outputs"

# ============================================================================
# LOAD TRANSFER RESULTS FOR ALL MODELS
# ============================================================================

console.print("[bold cyan]Loading transfer results...[/bold cyan]")

records = []

for mk in MODEL_ORDER:
    path = PROBES_DIR / f"transfer_results_{mk}_h.csv"
    if not path.exists():
        console.print(f"[yellow]  Missing: {path.name}[/yellow]")
        continue
    df = pl.read_csv(path)

    ab = df.filter(pl.col("condition") == "A_to_B")["auroc_mean"]
    ba = df.filter(pl.col("condition") == "B_to_A")["auroc_mean"]

    peak_ab = float(ab.max())
    peak_ba = float(ba.max())
    asymmetry = peak_ba - peak_ab

    records.append({
        "model_key": mk,
        "peak_ab":   peak_ab,
        "peak_ba":   peak_ba,
        "asymmetry": asymmetry,
    })
    console.print(f"  {mk}: A→B={peak_ab:.4f}  B→A={peak_ba:.4f}  Δ={asymmetry:+.4f}")

# ============================================================================
# BUILD FIGURE — Grouped bar chart
# ============================================================================

console.print("[bold cyan]Building figure...[/bold cyan]")

n_models   = len(records)
x          = np.arange(n_models)
bar_width  = 0.35

fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))

# A→B bars (lighter fill, same color family)
bars_ab = ax.bar(
    x - bar_width / 2,
    [r["peak_ab"] for r in records],
    width=bar_width,
    color=[COLORS["sky_blue"]] * n_models,
    edgecolor="white", linewidth=0.5,
    label="A → B (keyword-trained probe on clinical text)",
    zorder=3,
)

# B→A bars (darker fill)
bars_ba = ax.bar(
    x + bar_width / 2,
    [r["peak_ba"] for r in records],
    width=bar_width,
    color=[COLORS["blue"]] * n_models,
    edgecolor="white", linewidth=0.5,
    label="B → A (clinical-trained probe on keyword-rich text)",
    zorder=3,
)

# Asymmetry annotation (Δ pp above each pair)
for i, r in enumerate(records):
    xi     = x[i]
    high   = max(r["peak_ab"], r["peak_ba"])
    delta  = r["asymmetry"] * 100
    sign   = "+" if delta >= 0 else ""
    ax.text(xi, high + 0.006,
            f"{sign}{delta:.1f}pp",
            ha="center", va="bottom", fontsize=7.5, color="#444444")

# Chance line
ax.axhline(1.0, color="#dddddd", linewidth=0.8, linestyle=":", zorder=0)

# Family separator bands
ax.axvspan(-0.5,  1.5, alpha=0.06, color=COLORS["orange"], zorder=0)   # Llama-1B
ax.axvspan(1.5,   3.5, alpha=0.06, color=COLORS["blue"],   zorder=0)   # Llama-8B
ax.axvspan(3.5,   5.5, alpha=0.06, color=COLORS["green"],  zorder=0)   # Gemma-9B

# Family labels (x-axis top)
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([0.5, 2.5, 4.5])
ax_top.set_xticklabels(["Llama-3.2-1B", "Llama-3-8B", "Gemma-2-9B"], fontsize=8)
ax_top.tick_params(length=0)
ax_top.spines["top"].set_visible(False)

# Axis labels — family is already shown on top axis, so just Base/Instruct below
short_labels = ["Base" if "base" in r["model_key"] else "Instruct" for r in records]
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=8)
ax.set_ylabel("Transfer AUROC (peak across layers)", fontsize=9)
ax.set_ylim(0.6, 1.06)
ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=8)
ax.set_title(
    "Cross-set probe transfer: clinical-trained probes generalise better (B→A > A→B)",
    fontsize=9, pad=8
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig03_transfer_asymmetry")
console.print("[bold green]✓ Fig 3 complete[/bold green]")
plt.close()
