"""
Figure 4 — Activation Patching Results
=======================================
Question: Can we causally transfer emotion representations between stimuli?
Does patching cross-set (A→B or B→A) work, and what does it transfer?

Shows: Success rate under 4 conditions across 3 models:
  1. Within-set Set A patching (same-emotion pairs)
  2. Within-set Set B patching (same-emotion pairs)
  3. Cross-set, same-emotion (A→B or B→A, source/target share emotion)
  4. Cross-set, diff-emotion (A→B or B→A, source/target have different emotion)

Key message: Cross-set patching works at 100% for diff-emotion condition —
the model transfers affective salience, not emotion identity.

Data sources:
- Within-set: experiments/14_v2_patching/outputs/patching_success_by_layer_{model}_{set}_h.csv
- Cross-set: experiments/14_v2_patching/outputs/patching_crossset_results_{model}_h.csv

Models shown: llama1b_inst, llama8b_inst, gemma9b_base (as per paper spec)

Output: outputs/fig04_patching.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polars as pl
from scipy.stats import norm as scipy_norm
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_LABELS, COLORS, save_fig
)

console = Console()
apply_style()
# Helvetica lacks many Unicode symbols; DejaVu Sans is more complete
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

PATCH_DIR = Path(__file__).parent.parent.parent / "experiments" / "14_v2_patching" / "outputs"

# Models to show (per paper spec: §6.1)
MODELS_SHOWN = ["llama1b_inst", "llama8b_inst", "gemma9b_base"]

# ============================================================================
# WILSON CI HELPER
# ============================================================================

def wilson_ci(successes, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return float(centre - margin), float(centre + margin)

# ============================================================================
# LOAD DATA FOR EACH MODEL AND CONDITION
# ============================================================================

console.print("[bold cyan]Loading patching data...[/bold cyan]")

results = {}  # mk -> {condition_label: {"mean": float, "ci_lo": float, "ci_hi": float}}

CONDITIONS = [
    "Set A\n(within)",
    "Set B\n(within)",
    "Cross-set\nsame-emotion",
    "Cross-set\ndiff-emotion",
]

for mk in MODELS_SHOWN:
    results[mk] = {}

    # ── Within-set Set A ─────────────────────────────────────────────────
    seta_path = PATCH_DIR / f"patching_success_by_layer_{mk}_seta_h.csv"
    if seta_path.exists():
        df = pl.read_csv(seta_path)
        best_row = df.sort("success_rate", descending=True).row(0, named=True)
        peak_sr  = best_row["success_rate"]
        n_pairs  = int(best_row["n_pairs"])
        successes = round(peak_sr * n_pairs)
        ci_lo, ci_hi = wilson_ci(successes, n_pairs)
        results[mk]["Set A\n(within)"] = {"mean": peak_sr, "ci_lo": ci_lo, "ci_hi": ci_hi}
        console.print(f"  {mk} Set A within: {peak_sr:.3f} (n={n_pairs})")
    else:
        console.print(f"[yellow]  Missing: {seta_path.name}[/yellow]")
        results[mk]["Set A\n(within)"] = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    # ── Within-set Set B ─────────────────────────────────────────────────
    setb_path = PATCH_DIR / f"patching_success_by_layer_{mk}_setb_h.csv"
    if setb_path.exists():
        df = pl.read_csv(setb_path)
        best_row = df.sort("success_rate", descending=True).row(0, named=True)
        peak_sr  = best_row["success_rate"]
        n_pairs  = int(best_row["n_pairs"])
        successes = round(peak_sr * n_pairs)
        ci_lo, ci_hi = wilson_ci(successes, n_pairs)
        results[mk]["Set B\n(within)"] = {"mean": peak_sr, "ci_lo": ci_lo, "ci_hi": ci_hi}
        console.print(f"  {mk} Set B within: {peak_sr:.3f} (n={n_pairs})")
    else:
        console.print(f"[yellow]  Missing: {setb_path.name}[/yellow]")
        results[mk]["Set B\n(within)"] = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    # ── Cross-set conditions ──────────────────────────────────────────────
    cross_path = PATCH_DIR / f"patching_crossset_results_{mk}_h.csv"
    if cross_path.exists():
        df = pl.read_csv(cross_path)

        # Find best layer for cross-set overall, then compute rates per condition at that layer
        # Strategy: for each condition, find peak layer independently
        for cond_key, cond_label in [
            ("crossset_same_emotion", "Cross-set\nsame-emotion"),
            ("crossset_diff_emotion", "Cross-set\ndiff-emotion"),
        ]:
            sub = df.filter(pl.col("condition") == cond_key)
            if len(sub) == 0:
                console.print(f"[yellow]  No {cond_key} rows for {mk}[/yellow]")
                results[mk][cond_label] = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
                continue
            # Aggregate success by layer, pick best layer
            by_layer = (
                sub.group_by("layer")
                .agg([
                    pl.col("success").sum().alias("n_success"),
                    pl.col("success").count().alias("n_total"),
                ])
                .with_columns((pl.col("n_success") / pl.col("n_total")).alias("sr"))
                .sort("sr", descending=True)
            )
            best = by_layer.row(0, named=True)
            peak_sr   = best["sr"]
            n_success = int(best["n_success"])
            n_total   = int(best["n_total"])
            ci_lo, ci_hi = wilson_ci(n_success, n_total)
            results[mk][cond_label] = {"mean": peak_sr, "ci_lo": ci_lo, "ci_hi": ci_hi}
            console.print(f"  {mk} {cond_key}: {peak_sr:.3f} (n={n_total})")
    else:
        console.print(f"[yellow]  Missing: {cross_path.name}[/yellow]")
        for lbl in ["Cross-set\nsame-emotion", "Cross-set\ndiff-emotion"]:
            results[mk][lbl] = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

# ============================================================================
# BUILD FIGURE
# ============================================================================

console.print("[bold cyan]Building figure...[/bold cyan]")

n_conds   = len(CONDITIONS)
n_models  = len(MODELS_SHOWN)
bar_width = 0.22
group_gap = 0.1

# X positions: groups of conditions, bars within each group for models
x_centers = np.arange(n_conds) * (n_models * bar_width + group_gap + 0.15)

COND_COLORS_MAP = [
    COLORS["blue"],
    COLORS["sky_blue"],
    COLORS["orange"],
    COLORS["vermillion"],
]

fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2))

for mi, mk in enumerate(MODELS_SHOWN):
    x_offsets = x_centers + (mi - (n_models - 1) / 2) * bar_width

    means  = [results[mk][c]["mean"]  for c in CONDITIONS]
    ci_los = [results[mk][c]["ci_lo"] for c in CONDITIONS]
    ci_his = [results[mk][c]["ci_hi"] for c in CONDITIONS]

    # Use model-specific shading rather than condition color
    alpha_vals = [0.7, 0.85, 1.0]
    bar_color = {
        "llama1b_inst": COLORS["sky_blue"],
        "llama8b_inst": COLORS["vermillion"],
        "gemma9b_base": COLORS["green"],
    }[mk]

    for ci, (cond_lbl, x, m, lo, hi) in enumerate(zip(CONDITIONS, x_offsets, means, ci_los, ci_his)):
        if np.isnan(m):
            continue
        # Bar
        ax.bar(x, m * 100, width=bar_width * 0.9,
               color=bar_color, alpha=alpha_vals[mi],
               edgecolor="white", linewidth=0.4, zorder=3)
        # Wilson CI error bar
        err_lo = (m - lo) * 100
        err_hi = (hi - m) * 100
        ax.errorbar(x, m * 100, yerr=[[err_lo], [err_hi]],
                    fmt="none", color="#555555", linewidth=0.9, capsize=2, zorder=4)

        # Star annotation for 100% cross-set diff-emotion
        if cond_lbl == "Cross-set\ndiff-emotion" and m >= 0.999:
            ax.plot(x, m * 100 + err_hi + 3.0,
                    marker='*', markersize=9, color=bar_color,
                    linestyle='none', zorder=5)

# Chance line (12.5% for 8-class, but success rate is binary = 50% random,
# actually patching "success" = correctly changing to target emotion → chance ~12.5%)
ax.axhline(12.5, color="#888888", linewidth=1.1, linestyle="--", zorder=5)
ax.text(x_centers[-1] + bar_width, 13.5, "Chance (12.5%)",
        fontsize=7.5, color="#555555", va="bottom")

# X axis: condition group labels
ax.set_xticks(x_centers)
ax.set_xticklabels(CONDITIONS, fontsize=8)
ax.set_ylabel("Patching success rate (%)", fontsize=9)
ax.set_ylim(0, 115)
ax.set_yticks([0, 25, 50, 75, 100])

# Model legend
legend_handles = [
    mpatches.Patch(color=COLORS["sky_blue"],   alpha=0.7,  label=MODEL_LABELS["llama1b_inst"].replace("\n", " ")),
    mpatches.Patch(color=COLORS["vermillion"], alpha=0.85, label=MODEL_LABELS["llama8b_inst"].replace("\n", " ")),
    mpatches.Patch(color=COLORS["green"],      alpha=1.0,  label=MODEL_LABELS["gemma9b_base"].replace("\n", " ")),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
ax.set_title(
    "Activation patching success: cross-set patching transfers affective salience, not emotion identity",
    fontsize=9, pad=8
)

plt.tight_layout()

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig04_patching")
console.print("[bold green]✓ Fig 4 complete[/bold green]")
plt.close()
