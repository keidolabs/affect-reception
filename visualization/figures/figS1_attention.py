"""
Figure S1 — Attention Pattern Visualization (Supplementary)
============================================================
Section: 6.3

Shows: For each model, heatmap of emo_sensitivity (= proportion of attention
on emotion keywords in Set A minus proportion in Set B) per (layer, head).
High sensitivity = head preferentially attends to emotional keywords.

Key message: The model attends to structural anchors (colons, BOS, label space),
not to target emotional content — consistent with a probing paradigm, not a
lexical lookup.

Data source:
- experiments/15_v2_attention/outputs/head_sensitivity_{model}.csv
  Columns: layer, head, norm_depth, seta_emo_kw_prop, setb_emo_kw_prop, emo_sensitivity

Output: outputs/figS1_attention.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import polars as pl
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from style_config import (
    apply_style, FULL_WIDTH, MODEL_LABELS, MODEL_ORDER, save_fig, COLORS
)

console = Console()
apply_style()

ATTN_DIR = Path(__file__).parent.parent.parent / "experiments" / "15_v2_attention" / "outputs"

# Show these models (pick representative set to avoid crowding)
MODELS_SHOWN = ["llama1b_base", "llama8b_base", "gemma9b_base"]

# ============================================================================
# LOAD HEAD SENSITIVITY DATA
# ============================================================================

console.print("[bold cyan]Loading head sensitivity data...[/bold cyan]")

data = {}
for mk in MODELS_SHOWN:
    path = ATTN_DIR / f"head_sensitivity_{mk}.csv"
    if not path.exists():
        console.print(f"[yellow]  Missing: {path.name}[/yellow]")
        continue
    df = pl.read_csv(path)
    layers = sorted(df["layer"].unique().to_list())
    heads  = sorted(df["head"].unique().to_list())
    n_layers = len(layers)
    n_heads  = len(heads)
    # Build (n_layers, n_heads) matrix
    mat = np.full((n_layers, n_heads), np.nan)
    layer_idx = {l: i for i, l in enumerate(layers)}
    head_idx  = {h: i for i, h in enumerate(heads)}
    for row in df.iter_rows(named=True):
        li = layer_idx[row["layer"]]
        hi = head_idx[row["head"]]
        mat[li, hi] = row["emo_sensitivity"]
    data[mk] = {"mat": mat, "layers": layers, "heads": heads}
    console.print(f"  {mk}: {n_layers} layers × {n_heads} heads  "
                  f"max sensitivity={np.nanmax(mat):.3f}  mean={np.nanmean(mat):.3f}")

if not data:
    console.print("[bold red]No attention data found![/bold red]")
    raise SystemExit(1)

# ============================================================================
# BUILD FIGURE — one panel per model (side by side)
# ============================================================================

console.print("[bold cyan]Building attention sensitivity heatmap...[/bold cyan]")

n_panels = len(data)
fig, axes = plt.subplots(n_panels, 1,
                          figsize=(FULL_WIDTH, 6.5),
                          sharex=False)

if n_panels == 1:
    axes = [axes]

# Shared colormap: diverging, centered at 0
vmax = max(np.nanmax(d["mat"]) for d in data.values())
vabs = max(abs(vmax), 0.1)
cmap = plt.cm.RdBu_r

for pi, mk in enumerate(MODELS_SHOWN):
    if mk not in data:
        axes[pi].text(0.5, 0.5, "Data\nnot found",
                      ha="center", va="center", transform=axes[pi].transAxes,
                      fontsize=9, color="#999999")
        continue

    ax  = axes[pi]
    d   = data[mk]
    mat = d["mat"]

    # Transpose: rows=heads, cols=layers
    im = ax.imshow(
        mat.T,
        aspect="auto",
        cmap=cmap,
        vmin=-vabs, vmax=vabs,
        origin="upper",
        interpolation="nearest",
    )

    # x-axis: layers
    n_layers = len(d["layers"])
    x_step = max(1, n_layers // 10)
    x_ticks = list(range(0, n_layers, x_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(d["layers"][i]) for i in x_ticks], fontsize=7)
    ax.set_xlabel("Layer" if pi == n_panels - 1 else "", fontsize=8)

    # y-axis: attention heads
    n_heads = len(d["heads"])
    y_step = max(1, n_heads // 8)
    y_ticks = list(range(0, n_heads, y_step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(d["heads"][i]) for i in y_ticks], fontsize=7)
    ax.set_ylabel("Head", fontsize=8)

    short = MODEL_LABELS[mk].replace("\n", " ")
    ax.set_title(short, fontsize=8.5, pad=4)

fig.suptitle(
    "S1 — Attention head sensitivity to emotional keywords across layers",
    fontsize=9, y=1.01
)
plt.tight_layout(rect=[0, 0.10, 1, 1])

# Colorbar — placed manually below panels after tight_layout reserves space
cax = fig.add_axes([0.15, 0.03, 0.70, 0.022])
cbar = fig.colorbar(
    plt.cm.ScalarMappable(
        norm=mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs),
        cmap=cmap
    ),
    cax=cax, orientation="horizontal",
)
cbar.set_label("Keyword sensitivity  (Set A − Set B attention proportion)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# ============================================================================
# SAVE
# ============================================================================

save_fig(fig, "figS1_attention")
console.print("[bold green]✓ Fig S1 complete[/bold green]")
plt.close()
