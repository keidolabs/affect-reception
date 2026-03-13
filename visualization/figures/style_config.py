"""
style_config.py — Shared visualization style for "Affect Without Keywords"
===========================================================================
Import this in every figure script:

    from style_config import (
        apply_style, COLORS, MODEL_COLORS, EMOTION_COLORS,
        FULL_WIDTH, SINGLE_COL, save_fig
    )
    apply_style()  # call once before any plt.figure()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================================
# LAYOUT CONSTANTS (inches, arXiv single-column = 6.5 in max)
# ============================================================================

FULL_WIDTH  = 6.5   # arXiv max width (inches)
SINGLE_COL  = 3.5   # single-panel figures (half-width)
OUTPUT_DIR  = Path(__file__).parent / "outputs"

# ============================================================================
# OKABE-ITO COLORBLIND-SAFE PALETTE
# ============================================================================

COLORS = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
}

# ============================================================================
# MODEL COLORS — consistent across all figures
# Llama-1B family: orange / sky_blue
# Llama-8B family: blue / vermillion
# Gemma-9B family: green / pink
# ============================================================================

MODEL_COLORS = {
    "llama1b_base":  COLORS["orange"],
    "llama1b_inst":  COLORS["sky_blue"],
    "llama8b_base":  COLORS["blue"],
    "llama8b_inst":  COLORS["vermillion"],
    "gemma9b_base":  COLORS["green"],
    "gemma9b_inst":  COLORS["pink"],
}

MODEL_LABELS = {
    "llama1b_base":  "Llama-3.2-1B\nBase",
    "llama1b_inst":  "Llama-3.2-1B\nInstruct",
    "llama8b_base":  "Llama-3-8B\nBase",
    "llama8b_inst":  "Llama-3-8B\nInstruct",
    "gemma9b_base":  "Gemma-2-9B\nBase",
    "gemma9b_inst":  "Gemma-2-9B\nInstruct",
}

# Short labels for axes with limited space
MODEL_LABELS_SHORT = {
    "llama1b_base":  "1B-B",
    "llama1b_inst":  "1B-I",
    "llama8b_base":  "8B-B",
    "llama8b_inst":  "8B-I",
    "gemma9b_base":  "9B-B",
    "gemma9b_inst":  "9B-I",
}

# Display order (grouped by family)
MODEL_ORDER = [
    "llama1b_base", "llama1b_inst",
    "llama8b_base", "llama8b_inst",
    "gemma9b_base", "gemma9b_inst",
]

# ============================================================================
# EMOTION COLORS — 8 categories
# ============================================================================

EMOTION_COLORS = {
    "anger":    COLORS["vermillion"],
    "disgust":  COLORS["green"],
    "fear":     COLORS["pink"],
    "grief":    COLORS["blue"],
    "happiness":COLORS["yellow"],
    "love":     COLORS["orange"],
    "neutral":  "#999999",
    "surprise": COLORS["sky_blue"],
}

EMOTION_MARKERS = {
    "anger":    "o",
    "disgust":  "s",
    "fear":     "^",
    "grief":    "D",
    "happiness":"P",
    "love":     "*",
    "neutral":  "X",
    "surprise": "v",
}

# ============================================================================
# CONDITION COLORS — binary vs 8-class, Set A vs Set B
# ============================================================================

COND_COLORS = {
    "binary":    COLORS["blue"],
    "8class":    COLORS["orange"],
    "set_a":     COLORS["blue"],
    "set_b":     COLORS["vermillion"],
    "set_c":     COLORS["green"],
    "transfer_ab": COLORS["sky_blue"],
    "transfer_ba": COLORS["vermillion"],
}

# ============================================================================
# MATPLOTLIB rcPARAMS — publication quality
# ============================================================================

RC_PARAMS = {
    # Font
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            9,
    "axes.labelsize":       9,
    "axes.titlesize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.title_fontsize":8,
    # Figure
    "figure.dpi":           150,   # screen; save_fig overrides to 300
    "savefig.dpi":          300,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    # Spines and ticks
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.minor.size":     2,
    "ytick.minor.size":     2,
    # Lines
    "lines.linewidth":      1.5,
    "lines.markersize":     5,
    # Grid
    "axes.grid":            False,
    # Legend
    "legend.frameon":       False,
    "legend.borderpad":     0.3,
    # Patch
    "patch.linewidth":      0.8,
    # PDF output
    "pdf.fonttype":         42,   # embeds fonts in PDF (arXiv requirement)
    "ps.fonttype":          42,
}


def apply_style():
    """Call once before creating any figure. Applies the shared rcParams."""
    mpl.rcParams.update(RC_PARAMS)


def save_fig(fig, name: str, output_dir: Path = OUTPUT_DIR):
    """
    Save figure as both PDF (vector, for arXiv) and PNG (300 DPI, for preview).

    Usage:
        save_fig(fig, "fig01_dissociation")
    Outputs:
        outputs/fig01_dissociation.pdf
        outputs/fig01_dissociation.png
    """
    output_dir.mkdir(exist_ok=True)
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    print(f"  Saved: {png_path.name}")
