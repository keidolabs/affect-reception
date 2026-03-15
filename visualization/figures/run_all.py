"""
run_all.py — Generate all arXiv figures
========================================
Runs every figure script in order. Failed scripts are reported at the end
but do not stop subsequent figures.

Usage:
    uv run python visualization/figures/run_all.py

Individual figures:
    uv run python visualization/figures/fig01_dissociation.py
    uv run python visualization/figures/fig02_layer_trajectories.py
    ...

Outputs land in: visualization/figures/outputs/
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

FIGURES_DIR = Path(__file__).parent
SCRIPTS = [
    "fig09_schematic.py",         # no data — always works
    "fig01_dissociation.py",
    "fig02_layer_trajectories.py",
    "fig03_transfer_asymmetry.py",
    "fig04_patching.py",
    "fig05_knockout.py",
    "fig06_geometry.py",
    "fig07_scale_effect.py",
    "fig08_setc_validation.py",
    "fig10_confusion.py",
    "figS1_attention.py",
    "figS2_components.py",
    "figS3_geometry_detail.py",
]

results = []

console.rule("[bold cyan]Generating all arXiv figures[/bold cyan]")

for script in SCRIPTS:
    script_path = FIGURES_DIR / script
    if not script_path.exists():
        console.print(f"[yellow]SKIP[/yellow] {script} — file not found")
        results.append((script, "SKIP", ""))
        continue

    console.print(f"\n[bold]Running {script}...[/bold]")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True
    )

    if proc.returncode == 0:
        console.print(f"[green]✓ {script}[/green]")
        results.append((script, "OK", ""))
    else:
        # Show last 5 lines of stderr for quick diagnosis
        err_lines = proc.stderr.strip().split("\n")[-5:]
        err_summary = " | ".join(err_lines)
        console.print(f"[red]✗ {script}[/red]\n  {err_summary}")
        results.append((script, "FAIL", err_summary[:120]))

# ── Summary table ─────────────────────────────────────────────────────────

console.rule("[bold]Summary[/bold]")
table = Table(show_header=True, header_style="bold")
table.add_column("Script", style="dim", width=32)
table.add_column("Status", width=6)
table.add_column("Error", overflow="fold")

for script, status, err in results:
    color = {"OK": "green", "FAIL": "red", "SKIP": "yellow"}.get(status, "white")
    table.add_row(script, f"[{color}]{status}[/{color}]", err)

console.print(table)

n_ok   = sum(1 for _, s, _ in results if s == "OK")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
n_skip = sum(1 for _, s, _ in results if s == "SKIP")

console.print(f"\n[bold green]{n_ok} OK[/bold green]  "
              f"[bold red]{n_fail} FAIL[/bold red]  "
              f"[yellow]{n_skip} SKIP[/yellow]")

output_dir = FIGURES_DIR / "outputs"
pdfs = list(output_dir.glob("*.pdf"))
console.print(f"\nOutputs in: {output_dir}")
console.print(f"  {len(pdfs)} PDF files generated")
