"""
Figure 9 — Experimental Design Schematic
=========================================
Renders fig09_schematic.mmd via the mermaid CLI (mmdc) to produce
publication-ready PNG and SVG outputs.

Requires:  mmdc  (install once: npm install -g @mermaid-js/mermaid-cli)

Output: outputs/fig09_schematic.{png,svg}
"""

import subprocess
import sys
import json
from pathlib import Path
from rich.console import Console

console = Console()

FIGURES_DIR = Path(__file__).parent
OUTPUT_DIR  = FIGURES_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MMD_FILE    = FIGURES_DIR / "fig09_schematic.mmd"
PNG_OUT     = OUTPUT_DIR  / "fig09_schematic.png"
SVG_OUT     = OUTPUT_DIR  / "fig09_schematic.svg"
PDF_OUT     = OUTPUT_DIR  / "fig09_schematic.pdf"

# mmdc config: white background, wide canvas for the flowchart
MMDC_CONFIG = {
    "theme": "base"
}
CONFIG_FILE = FIGURES_DIR / "_mmdc_config.json"
CONFIG_FILE.write_text(json.dumps(MMDC_CONFIG))

# ============================================================================
# VERIFY mmdc IS AVAILABLE
# ============================================================================

def find_mmdc() -> str:
    """Return path to mmdc, searching PATH and common npm locations."""
    import shutil, os
    found = shutil.which("mmdc")
    if found:
        return found
    # nvm installs under ~/.nvm
    nvm_bin = Path.home() / ".nvm" / "versions" / "node"
    if nvm_bin.exists():
        for node_ver in sorted(nvm_bin.iterdir(), reverse=True):
            candidate = node_ver / "bin" / "mmdc"
            if candidate.exists():
                return str(candidate)
    raise RuntimeError(
        "mmdc not found. Install with: npm install -g @mermaid-js/mermaid-cli"
    )

mmdc = find_mmdc()
console.print(f"[dim]mmdc:[/dim] {mmdc}")

# ============================================================================
# RENDER PNG
# ============================================================================

console.print("[bold cyan]Rendering PNG...[/bold cyan]")
result = subprocess.run(
    [
        mmdc,
        "--input",           str(MMD_FILE),
        "--output",          str(PNG_OUT),
        "--configFile",      str(CONFIG_FILE),
        "--backgroundColor", "white",
        "--width",           "1400",
        "--scale",           "2",         # retina — ~300 DPI effective
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    console.print(f"[bold red]mmdc PNG error:[/bold red]\n{result.stderr}")
    sys.exit(1)
console.print(f"  Saved: {PNG_OUT.name}")

# ============================================================================
# RENDER SVG
# ============================================================================

console.print("[bold cyan]Rendering SVG...[/bold cyan]")
result = subprocess.run(
    [
        mmdc,
        "--input",           str(MMD_FILE),
        "--output",          str(SVG_OUT),
        "--configFile",      str(CONFIG_FILE),
        "--backgroundColor", "white",
        "--width",           "1400",
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    console.print(f"[bold red]mmdc SVG error:[/bold red]\n{result.stderr}")
    sys.exit(1)
console.print(f"  Saved: {SVG_OUT.name}")

# ============================================================================
# RENDER PDF (from PNG via matplotlib for tight bounding box)
# ============================================================================

console.print("[bold cyan]Rendering PDF (tight crop from PNG)...[/bold cyan]")
try:
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(PNG_OUT).convert("RGB")
    # Auto-crop whitespace: find bounding box of non-white content
    import numpy as np
    arr = np.array(img)
    non_white = np.any(arr < 245, axis=2)  # pixel is non-white if any channel < 245
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = (cmin, rmin, cmax + 1, rmax + 1)
    else:
        bbox = None
    if bbox:
        # Add small margin (20px at render scale)
        margin = 20
        bbox = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(img.width, bbox[2] + margin),
            min(img.height, bbox[3] + margin),
        )
        img = img.crop(bbox)

    # Render into matplotlib figure sized to image aspect ratio
    dpi = 150
    fig_w = img.width / dpi
    fig_h = img.height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(str(PDF_OUT), format="pdf", bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    console.print(f"  Saved: {PDF_OUT.name}")
except Exception as e:
    console.print(f"[yellow]PDF crop failed ({e}), falling back to mmdc direct render[/yellow]")
    result = subprocess.run(
        [mmdc, "--input", str(MMD_FILE), "--output", str(PDF_OUT),
         "--configFile", str(CONFIG_FILE), "--backgroundColor", "white", "--width", "1400"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[bold red]mmdc PDF error:[/bold red]\n{result.stderr}")
        sys.exit(1)
    console.print(f"  Saved: {PDF_OUT.name} (uncropped)")

# Clean up temp config
CONFIG_FILE.unlink(missing_ok=True)

console.print("[bold green]✓ Fig 9 complete[/bold green]")
