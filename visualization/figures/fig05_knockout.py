"""
Figure 5 — Knockout Fragility Asymmetry (Heatmap)
==================================================
Question: How many layers are "critical" for Set A vs Set B processing?
Does keyword-free processing require more layers?

Shows: Heatmap of accuracy drop when each layer's MHSA or FFN is zeroed out,
split by Set A vs Set B. White = no drop, red = catastrophic.

Key message: Keyword-free processing is distributed and fragile (many critical
layers). Keywords create shortcuts through narrow bottlenecks (few critical layers).

Data source:
- experiments/16_v2_knockout/outputs/knockout_summary_{model}.csv
  Columns: layer, norm_depth, act_type, knockout_type, stimulus_set,
           accuracy_drop, n_changed, n_total, is_critical

Primary model: gemma9b_inst (most complete data available in v2 pipeline)

Output: outputs/fig05_knockout.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import polars as pl
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import apply_style, FULL_WIDTH, SINGLE_COL, save_fig, COLORS

console = Console()
apply_style()

KO_DIR = Path(__file__).parent.parent.parent / "experiments" / "16_v2_knockout" / "outputs"

# Use zero knockout (most interpretable: silence a layer completely)
KNOCKOUT_TYPE = "zero"

# Primary model — gemma9b_inst is the model with full v2 knockout data
PRIMARY_MODEL = "gemma9b_inst"

# Also try these if available
FALLBACK_MODELS = ["llama8b_inst", "llama1b_inst"]

# ============================================================================
# LOAD KNOCKOUT SUMMARY
# ============================================================================

def load_model(mk):
    path = KO_DIR / f"knockout_summary_{mk}.csv"
    if not path.exists():
        return None
    return pl.read_csv(path)

console.print("[bold cyan]Loading knockout data...[/bold cyan]")

df = load_model(PRIMARY_MODEL)
model_key = PRIMARY_MODEL
if df is None:
    for mk in FALLBACK_MODELS:
        df = load_model(mk)
        if df is not None:
            model_key = mk
            break

if df is None:
    console.print("[bold red]No knockout data found! Check exp16 outputs.[/bold red]")
    raise SystemExit(1)

console.print(f"  Using: {model_key} ({len(df)} rows)")

# Filter to zero-knockout only
df = df.filter(pl.col("knockout_type") == KNOCKOUT_TYPE)

# ============================================================================
# BUILD HEATMAP DATA
#
# 4 columns: (Set A, MHSA), (Set A, FFN), (Set B, MHSA), (Set B, FFN)
# Rows: layers (sorted)
# Values: accuracy_drop (0–1)
# ============================================================================

# Get layer range
layers = sorted(df["layer"].unique().to_list())
n_layers = len(layers)
layer_arr = np.array(layers)

SETS    = ["seta", "setb"]
ACTS    = ["a", "m"]   # a = MHSA, m = FFN
ACT_LABELS = {"a": "MHSA", "m": "FFN"}
SET_LABELS = {"seta": "Set A", "setb": "Set B"}
COLUMNS = [(s, a) for s in SETS for a in ACTS]
COL_LABELS = [f"{SET_LABELS[s]}\n{ACT_LABELS[a]}" for s, a in COLUMNS]

# Build matrix: rows = layers, cols = 4 conditions
heatmap_data = np.full((n_layers, len(COLUMNS)), np.nan)

for col_i, (s, a) in enumerate(COLUMNS):
    sub = df.filter(
        (pl.col("stimulus_set") == s) & (pl.col("act_type") == a)
    ).sort("layer")
    for li, layer in enumerate(layers):
        row = sub.filter(pl.col("layer") == layer)
        if len(row) > 0:
            heatmap_data[li, col_i] = float(row["accuracy_drop"][0])

console.print(f"  Heatmap shape: {heatmap_data.shape} (layers × conditions)")
console.print(f"  Max drop: {np.nanmax(heatmap_data):.3f}  Mean drop: {np.nanmean(heatmap_data):.3f}")

# Count critical layers per condition (is_critical is where drop > threshold)
# Threshold used in exp16 run.py was typically 0.2 (20% accuracy drop)
CRITICAL_THRESHOLD = 0.20

n_critical = {}
for col_i, (s, a) in enumerate(COLUMNS):
    col = heatmap_data[:, col_i]
    n_crit = int(np.sum(col > CRITICAL_THRESHOLD))
    n_critical[f"{s}_{a}"] = n_crit
    console.print(f"  {SET_LABELS[s]} {ACT_LABELS[a]}: {n_crit}/{n_layers} critical layers")

# ============================================================================
# BUILD FIGURE
# ============================================================================

console.print("[bold cyan]Building knockout heatmap...[/bold cyan]")

# Three-zone colormap (vmax=0.4):
#   Zone 1  0.00–0.10  → blue   (data pos 0.000–0.250)  clearly sub-threshold
#   Zone 2  0.10–0.19  → white  (data pos 0.250–0.475)  near-threshold buffer
#   Zone 3  0.20–0.40  → red    (data pos 0.500–1.000)  at or above threshold
# Sharp boundaries make the three zones immediately readable.
VMAX = 0.4
cmap = mcolors.LinearSegmentedColormap.from_list(
    "ko_drop",
    [(0.000, "#0072B2"),   # deep blue   (data 0.00)  — clearly sub-threshold
     (0.499, "#aad4ef"),   # pale blue   (data ~0.20) — approaching threshold
     (0.500, "#ffcc88"),   # pale orange (data  0.20) — just above threshold
     (1.000, "#881100")],  # dark red    (data 0.40)  — clearly critical
)

# Horizontal layout: conditions on y-axis (4 rows), layers on x-axis
# Transpose: (n_layers, n_conditions) → (n_conditions, n_layers)
fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))

im = ax.imshow(
    heatmap_data.T,
    aspect="auto",
    cmap=cmap,
    vmin=0.0, vmax=VMAX,
    origin="upper",
    interpolation="nearest",
)

# Critical layer markers: vertical lines where any condition > threshold
for li, layer in enumerate(layers):
    row_data = heatmap_data[li, :]
    if np.any(row_data > CRITICAL_THRESHOLD):
        ax.axvline(li - 0.5, color="#aaaaaa", linewidth=0.3, zorder=2)

# Row separator between Set A and Set B (between condition rows 1 and 2)
ax.axhline(1.5, color="#333333", linewidth=1.5, zorder=3)

# Y axis: condition labels with crit counts
col_labels_with_crit = [
    f"{SET_LABELS[s]} {ACT_LABELS[a]}  ({n_critical[f'{s}_{a}']} crit.)"
    for s, a in COLUMNS
]
ax.set_yticks(range(len(COLUMNS)))
ax.set_yticklabels(col_labels_with_crit, fontsize=8)

# X axis: layer numbers (show every Nth to avoid crowding)
step = max(1, n_layers // 20)
x_ticks = list(range(0, n_layers, step))
ax.set_xticks(x_ticks)
ax.set_xticklabels([str(layers[i]) for i in x_ticks], fontsize=7)
ax.set_xlabel("Layer", fontsize=9)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label(f"Accuracy drop on knockout\n(crit. threshold = {CRITICAL_THRESHOLD})", fontsize=8)
cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
cbar.ax.tick_params(labelsize=7)

# Critical threshold line on colorbar — at data=0.2, midpoint of the scale
cbar.ax.axhline(CRITICAL_THRESHOLD, color="black", linewidth=1, linestyle="--")

ax.set_title(
    f"Knockout fragility: {model_key.replace('_', ' ').title()}"
    f"  (zero-ablation; Set A = keyword-rich, Set B = clinical)",
    fontsize=8.5, pad=8
)

plt.tight_layout(pad=0.5)

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "fig05_knockout")
console.print("[bold green]✓ Fig 5 complete[/bold green]")
plt.close()
