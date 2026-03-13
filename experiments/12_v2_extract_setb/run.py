"""
Experiment 12 — v2 Set B Extraction: Orchestrator
Question: Extract residual stream (h), attention (a), FFN (m), and attention weights
          from all 6 models on 192 Set B stimuli (96 clinical + 96 neutral controls).
Date: 2026-02-25

Runs each model-specific extraction script in sequence. Each script loads
one model, runs 192 forward passes, saves per-stimulus .npz files, then exits
(freeing GPU memory before the next model loads).

Run Exp 11 before this experiment. Downstream analysis (Exp 13) loads
activations from both experiments together.

Usage:
  uv run python experiments/12_v2_extract_setb/run.py
  uv run python experiments/12_v2_extract_setb/run.py --models llama1b_inst llama8b_base

See README.md for full documentation.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

EXP_DIR = Path(__file__).parent

# ============================================================================
# MODEL REGISTRY
# ============================================================================

ALL_MODELS = [
    ("llama1b_base", "extract_llama1b_base.py"),
    ("llama1b_inst", "extract_llama1b_inst.py"),
    ("llama8b_base", "extract_llama8b_base.py"),
    ("llama8b_inst", "extract_llama8b_inst.py"),
    ("gemma9b_base", "extract_gemma9b_base.py"),
    ("gemma9b_inst", "extract_gemma9b_inst.py"),
]

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="Run Set B extraction for all or selected models")
parser.add_argument(
    "--models",
    nargs="+",
    choices=[key for key, _ in ALL_MODELS],
    default=None,
    help="Subset of models to run (default: all). Example: --models llama1b_inst gemma9b_base",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Re-run even if manifest.csv already exists (overwrite existing outputs)",
)
args = parser.parse_args()

selected_keys = set(args.models) if args.models else {key for key, _ in ALL_MODELS}
models_to_run = [(key, script) for key, script in ALL_MODELS if key in selected_keys]

# ============================================================================
# RUN
# ============================================================================

console.print(f"\n[bold cyan]Experiment 12 — Set B Extraction[/bold cyan]")
console.print(f"Stimuli: 96 clinical vignettes + 96 neutral controls = 192 total")
console.print(f"Models selected: {[k for k, _ in models_to_run]}\n")

# Warn if Exp 11 looks incomplete
exp11_act_dir = EXP_DIR.parent / "11_v2_extract_seta" / "outputs" / "activations"
missing_exp11 = [key for key, _ in models_to_run if not (exp11_act_dir / key / "manifest.csv").exists()]
if missing_exp11:
    console.print(f"[yellow]Warning:[/yellow] Exp 11 outputs missing for: {missing_exp11}")
    console.print(f"  Exp 13 (probes) requires activations from both Exp 11 and 12.")
    console.print(f"  Run experiments/11_v2_extract_seta/run.py first if you haven't.\n")

results = []

for model_key, script_name in models_to_run:
    manifest_path = EXP_DIR / "outputs" / "activations" / model_key / "manifest.csv"
    script_path = EXP_DIR / script_name

    # Skip if already done (unless --force)
    if manifest_path.exists() and not args.force:
        console.print(f"[yellow]SKIP[/yellow]  {model_key} — manifest.csv exists (use --force to re-run)")
        results.append((model_key, "skipped", 0.0))
        continue

    console.print(f"[bold]Running {model_key}...[/bold]  ({script_name})")
    t0 = time.time()

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        # Inherit environment (includes HF_TOKEN, EMO_DEVICE, etc.)
        # Inherit stdout/stderr so rich progress bars display correctly
    )
    elapsed = time.time() - t0

    if proc.returncode == 0:
        console.print(f"[green]✓ DONE[/green]  {model_key} in {elapsed/60:.1f} min\n")
        results.append((model_key, "ok", elapsed))
    else:
        console.print(f"[red]✗ FAIL[/red]  {model_key} (exit code {proc.returncode}) — stopping\n")
        results.append((model_key, f"failed (exit {proc.returncode})", elapsed))
        break

# ============================================================================
# SUMMARY
# ============================================================================

console.print()
table = Table(title="Experiment 12 — Extraction Summary")
table.add_column("Model", style="cyan")
table.add_column("Status")
table.add_column("Time")
table.add_column("Output dir")

for model_key, status, elapsed in results:
    act_dir = EXP_DIR / "outputs" / "activations" / model_key
    n_files = len(list(act_dir.glob("*.npz"))) if act_dir.exists() else 0
    status_style = "green" if status == "ok" else ("yellow" if status == "skipped" else "red")
    table.add_row(
        model_key,
        f"[{status_style}]{status}[/{status_style}]",
        f"{elapsed/60:.1f} min" if elapsed > 0 else "—",
        f"{n_files} .npz files" if n_files > 0 else str(act_dir),
    )

console.print(table)

failed = [m for m, s, _ in results if s.startswith("failed")]
if failed:
    console.print(f"\n[red]Failed models: {failed}[/red]")
    console.print("Check output above for error details.")
    sys.exit(1)

done = [m for m, s, _ in results if s == "ok"]
skipped = [m for m, s, _ in results if s == "skipped"]
console.print(f"\n[bold green]✓ Exp 12 complete:[/bold green] {len(done)} extracted, {len(skipped)} skipped")
console.print(f"Outputs → {EXP_DIR / 'outputs' / 'activations'}")
console.print("Next: uv run python experiments/13_v2_probes/run.py")
