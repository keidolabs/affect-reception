"""
Figure 8 — Set C Validation (Binary Probe)
===========================================
Question: Does the binary probe merely detect narrative complexity (rich text),
or does it detect computed emotional significance?

Shows two panels:
  Panel A: Distribution of P(emotional) scores for Set B emotional, Set B neutral,
           and Set C complex-neutral. The probe correctly rejects Set C.
  Panel B: Layer-by-layer P(emotional) for Set C (median + IQR band).
           Set B shown as flat reference lines only (probe trained on Set B —
           applying it back to Set B at all layers is a train-set tautology).
           Set C stays near zero throughout all layers → correctly rejected.

Key message: The probe reads computed emotional meaning, not narrative richness.
Set C (complex, emotionally-neutral) is correctly rejected (P ≈ 0.005 median).

Data sources:
  Panel A: experiments/20_binary_confound_control/outputs/confound_scores.csv
  Panel B: experiments/20_binary_confound_control/outputs/layer_profiles.csv

Output: outputs/fig08_setc_validation.{pdf,png}
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
    apply_style, FULL_WIDTH, COLORS, COND_COLORS, save_fig
)

console = Console()
apply_style()

EXP20_DIR = Path(__file__).parent.parent.parent / "experiments" / "20_binary_confound_control" / "outputs"
SCORES_CSV  = EXP20_DIR / "confound_scores.csv"
PROFILES_CSV = EXP20_DIR / "layer_profiles.csv"

# ============================================================================
# LOAD DATA
# ============================================================================

console.print("[bold cyan]Loading Set C data...[/bold cyan]")

# ── Panel A data ─────────────────────────────────────────────────────────────
# Set C scores from exp20
setc_scores = None
if SCORES_CSV.exists():
    df_scores = pl.read_csv(SCORES_CSV)
    setc_scores = df_scores["p_emotional"].to_numpy()
    console.print(f"  Set C scores: n={len(setc_scores)}  mean={setc_scores.mean():.4f}  "
                  f"max={setc_scores.max():.4f}")
else:
    console.print("[yellow]  confound_scores.csv not found[/yellow]")

# ── Panel A + B data from layer_profiles.csv (all real) ──────────────────────
PEAK_LAYER = 10   # matches exp20/run.py — L10 (norm_depth 0.625) for llama1b_inst

if not PROFILES_CSV.exists():
    console.print("[bold red]layer_profiles.csv not found — run exp20 first.[/bold red]")
    raise SystemExit(1)

df_profiles = pl.read_csv(PROFILES_CSV)

def peak_scores(cond):
    return (df_profiles
            .filter((pl.col("condition") == cond) & (pl.col("layer") == PEAK_LAYER))
            ["p_emotional"].to_numpy())

setb_emo_scores  = peak_scores("B_emotional")
setb_neut_scores = peak_scores("B_neutral")
console.print(f"  Set B emotional (L{PEAK_LAYER}): n={len(setb_emo_scores)}  mean={setb_emo_scores.mean():.4f}")
console.print(f"  Set B neutral   (L{PEAK_LAYER}): n={len(setb_neut_scores)}  mean={setb_neut_scores.mean():.4f}")

if setc_scores is None:
    setc_scores = peak_scores("C_complex_neutral")
    console.print(f"[yellow]  Set C scores from layer_profiles at L{PEAK_LAYER}[/yellow]")

def layer_curve(cond):
    agg = (df_profiles
           .filter(pl.col("condition") == cond)
           .group_by(["layer", "norm_depth"])
           .agg(pl.col("p_emotional").mean().alias("mean_p"))
           .sort("layer"))
    return agg["norm_depth"].to_numpy(), agg["mean_p"].to_numpy()

def layer_curve_median_iqr(cond):
    """Per-layer median and IQR — more robust than mean for small-n with outliers."""
    agg = (df_profiles
           .filter(pl.col("condition") == cond)
           .group_by(["layer", "norm_depth"])
           .agg([
               pl.col("p_emotional").median().alias("median_p"),
               pl.col("p_emotional").quantile(0.25).alias("q25"),
               pl.col("p_emotional").quantile(0.75).alias("q75"),
           ])
           .sort("layer"))
    return (agg["norm_depth"].to_numpy(),
            agg["median_p"].to_numpy(),
            agg["q25"].to_numpy(),
            agg["q75"].to_numpy())

# Set B reference values: scalar medians at each layer
# (not shown as trajectories — train-set tautology; shown as flat reference lines)
depths, setb_emo_line,  _, _ = layer_curve_median_iqr("B_emotional")
_,      setb_neut_line, _, _ = layer_curve_median_iqr("B_neutral")

# Set C: median + IQR band across all layers
depths_c, setc_median_p, setc_q25, setc_q75 = layer_curve_median_iqr("C_complex_neutral")

# Reference scalars: overall median at peak layer (for flat reference lines in Panel B)
setb_emo_ref  = float(np.median(setb_emo_scores))
setb_neut_ref = float(np.median(setb_neut_scores))
setc_ref      = float(np.median(setc_scores))

console.print(f"  Set B emotional reference P: {setb_emo_ref:.4f}")
console.print(f"  Set B neutral   reference P: {setb_neut_ref:.4f}")
console.print(f"  Set C median trajectory: L0={setc_median_p[0]:.3f}  L{PEAK_LAYER}={setc_median_p[PEAK_LAYER]:.3f}  final={setc_median_p[-1]:.3f}")

# ============================================================================
# BUILD FIGURE — two panels side by side
# ============================================================================

console.print("[bold cyan]Building figure...[/bold cyan]")

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 4.5),
                                  gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.28})

# ── Panel A: Score distributions ──────────────────────────────────────────

def jitter_strip(ax, x, data, color, alpha=0.6, width=0.15, size=18):
    """Horizontal jitter strip plot."""
    np.random.seed(99)
    jitter = np.random.uniform(-width, width, size=len(data))
    ax.scatter(data, np.full(len(data), x) + jitter,
               color=color, alpha=alpha, s=size, edgecolors="none", zorder=3)

CONDITIONS_A = [
    (setb_emo_scores,  "Set B\nEmotional",     COLORS["vermillion"]),
    (setb_neut_scores, "Set B\nNeutral",        COLORS["sky_blue"]),
    (setc_scores,      "Set C\nComplex-Neutral", COLORS["green"]),
]

for xi, (scores, label, color) in enumerate(CONDITIONS_A):
    jitter_strip(ax_a, xi, scores, color=color)


# Decision boundary
ax_a.axvline(0.5, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
ax_a.text(0.52, 1.0, "Decision\nboundary", rotation=90,
          ha="left", va="center", fontsize=6.5, color="#888888")

ax_a.set_yticks([0, 1, 2])
ax_a.set_yticklabels([c[1] for c in CONDITIONS_A], fontsize=8)
ax_a.set_xlabel("Binary probe score  P(emotional)", fontsize=8.5)
ax_a.set_xlim(-0.05, 1.1)
ax_a.set_title("A — Probe score distributions\n(0 = neutral, 1 = emotional)",
               fontsize=9, fontweight="bold")


# ── Panel B: Set C layer trajectory (median + IQR) ────────────────────────
# Set B NOT shown as trajectories: applying a probe trained on Set B back to
# Set B at all layers is a train-set tautology — the flat lines are artefacts.
# Instead, show Set B as thin horizontal reference dashes at their peak-layer
# median values, so viewers can see where Set C sits relative to each reference.

# Set B reference dashes (flat horizontal lines)
ax_b.axhline(setb_emo_ref, color=COLORS["vermillion"], linewidth=1.2,
             linestyle="--", alpha=0.6, zorder=2,
             label=f"Set B Emotional (ref. P={setb_emo_ref:.3f})")
ax_b.axhline(setb_neut_ref, color=COLORS["sky_blue"], linewidth=1.2,
             linestyle="--", alpha=0.6, zorder=2,
             label=f"Set B Neutral (ref. P={setb_neut_ref:.3f})")

# Annotate reference lines at right margin
ax_b.text(1.04, setb_emo_ref,  "Set B\nEmotional", fontsize=6.5,
          color=COLORS["vermillion"], va="center", ha="left", alpha=0.8)
ax_b.text(1.04, setb_neut_ref, "Set B\nNeutral",   fontsize=6.5,
          color=COLORS["sky_blue"],   va="center", ha="left", alpha=0.8)

# Set C: median line
ax_b.plot(depths_c, setc_median_p, color=COLORS["green"], linewidth=2.0,
          linestyle="-", zorder=3,
          label="Set C Complex-Neutral (median)")

# Decision boundary
ax_b.axhline(0.5, color="#cccccc", linewidth=0.8, linestyle=":", zorder=0)
ax_b.text(0.01, 0.52, "decision boundary", fontsize=6.5, color="#aaaaaa")

ax_b.set_xlabel("Normalised layer depth", fontsize=8.5)
ax_b.set_ylabel("P(emotional)", fontsize=8.5)
ax_b.set_xlim(-0.02, 1.10)   # slight extra room for right-margin labels
ax_b.set_ylim(-0.05, 1.10)
ax_b.set_title("B — Set C probe trajectory across layers\n"
               "(probe correctly rejects complex-neutral text throughout)",
               fontsize=8.5, fontweight="bold")
ax_b.legend(
    loc="upper center", bbox_to_anchor=(-0.15, -0.18),
    fontsize=8, ncol=3, frameon=True, framealpha=0.9, edgecolor="#dddddd",
)

plt.suptitle(
    "Set C validation: the binary probe detects emotional meaning, not narrative richness",
    fontsize=9, y=1.02
)
plt.tight_layout()

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig08_setc_validation")
console.print("[bold green]✓ Fig 8 complete[/bold green]")
plt.close()
