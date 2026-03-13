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
# RENDER PDF
# ============================================================================

console.print("[bold cyan]Rendering PDF...[/bold cyan]")
result = subprocess.run(
    [
        mmdc,
        "--input",           str(MMD_FILE),
        "--output",          str(PDF_OUT),
        "--configFile",      str(CONFIG_FILE),
        "--backgroundColor", "white",
        "--width",           "1400",
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    console.print(f"[bold red]mmdc PDF error:[/bold red]\n{result.stderr}")
    sys.exit(1)
console.print(f"  Saved: {PDF_OUT.name}")

# Clean up temp config
CONFIG_FILE.unlink(missing_ok=True)

console.print("[bold green]✓ Fig 9 complete[/bold green]")
