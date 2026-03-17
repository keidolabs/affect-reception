"""
Experiment 01 — Phase 0a: TransformerLens Smoke Test
Model: meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
Date: 2026-02-24

Verifies model loads on CUDA via TransformerLens, architecture matches
expectations, and cache-based activation extraction produces correct shapes.

Run on Scrig: uv run python experiments/01_phase0a_llama3-8b/smoke_test.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEVICE   = "cuda"

# ============================================================================
# LOAD HF TOKEN
# ============================================================================

env_path = Path(__file__).parent.parent.parent / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
console.print("[bold cyan]HF token loaded[/bold cyan]")

# ============================================================================
# LOAD MODEL WITH TRANSFORMERLENS
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on {DEVICE} (float16)...[/bold cyan]")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    dtype=torch.float16,
)
model.eval()
console.print("[bold green]✓ Model loaded[/bold green]")

# ============================================================================
# VERIFY ARCHITECTURE
# ============================================================================

n_layers = model.cfg.n_layers
d_model  = model.cfg.d_model
n_heads  = model.cfg.n_heads
d_head   = model.cfg.d_head

checks = [
    ("n_layers", n_layers, 32,   "32 transformer layers"),
    ("d_model",  d_model,  4096, "4096 hidden dim"),
    ("n_heads",  n_heads,  32,   "32 attention heads"),
    ("d_head",   d_head,   128,  "128 dims per head"),
]

table = Table(title=f"Model Configuration: {MODEL_ID}")
table.add_column("Parameter"); table.add_column("Value", justify="right")
table.add_column("Expected", justify="right", style="dim"); table.add_column("Status")

all_pass = True
for name, actual, expected, _ in checks:
    ok = actual == expected
    table.add_row(name, str(actual), str(expected),
                  "[green]✓ PASS[/green]" if ok else "[red]✗ FAIL[/red]")
    if not ok:
        all_pass = False
console.print(table)

# ============================================================================
# FORWARD PASS + CACHE CHECK
# ============================================================================

console.print("\n[bold cyan]Testing cache-based activation extraction...[/bold cyan]")
test_text = "The kitchen counter still had the grocery list in his handwriting."
tokens = model.to_tokens(test_text)
seq_len = tokens.shape[1]
console.print(f"Test prompt: {seq_len} tokens")

with torch.no_grad():
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: "hook_resid_post" in name,
    )

# Check mid-layer activation shape and values
act_l15 = cache["resid_post", 15][:, -1, :].squeeze().cpu().numpy()
hook_shape_ok = act_l15.shape == (d_model,)
hook_finite_ok = np.all(np.isfinite(act_l15))
hook_nonzero_ok = act_l15.std() > 0.001

console.print(f"  Layer 15 cache shape: {act_l15.shape} — {'[green]PASS[/green]' if hook_shape_ok else '[red]FAIL[/red]'}")
console.print(f"  Values finite: {'[green]PASS[/green]' if hook_finite_ok else '[red]FAIL[/red]'}")
console.print(f"  Non-trivial std={act_l15.std():.4f}: {'[green]PASS[/green]' if hook_nonzero_ok else '[red]FAIL[/red]'}")
console.print(f"  mean={act_l15.mean():.4f}, std={act_l15.std():.4f}")

if not (hook_shape_ok and hook_finite_ok and hook_nonzero_ok):
    all_pass = False

del logits, cache

# ============================================================================
# REPORT
# ============================================================================

report = f"""Smoke Test Report — {MODEL_ID}
==========================================
Date: 2026-02-24 | Device: {DEVICE} | Dtype: float16
Framework: TransformerLens (HookedTransformer, run_with_cache)
Status: {"PASS" if all_pass else "FAIL"}

Architecture:
  n_layers: {n_layers}  d_model: {d_model}  n_heads: {n_heads}  d_head: {d_head}

Cache test (layer 15, final token):
  shape: {act_l15.shape}  mean: {act_l15.mean():.4f}  std: {act_l15.std():.4f}

Gate: {"PASS — proceed to extract.py" if all_pass else "FAIL — do not proceed"}
"""
(OUTPUT_DIR / "smoke_test_report.txt").write_text(report)

status_color = "bold green" if all_pass else "bold red"
console.print(f"\n[{status_color}]Smoke Test: {'PASS' if all_pass else 'FAIL'}[/{status_color}]")
if not all_pass:
    sys.exit(1)
