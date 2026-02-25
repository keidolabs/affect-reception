"""
Experiment 01 — Phase 0a: HF Smoke Test
Model: meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
Date: 2026-02-24

Verifies model loads on MPS, architecture matches expectations, and hook-based
activation extraction produces correct shapes. Uses HF transformers (not
TransformerLens — see TL issue #1178 for MPS correctness concerns).

Run: uv run python experiments/01_phase0a_llama3-8b/smoke_test.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEVICE   = "mps"
DTYPE    = torch.float16

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
# LOAD MODEL & TOKENIZER
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on {DEVICE} (float16)...[/bold cyan]")
console.print("First run: ~16GB download. Model is ~16GB in float16.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
model.eval()
console.print("[bold green]✓ Model loaded[/bold green]")

# ============================================================================
# VERIFY ARCHITECTURE
# ============================================================================

cfg = model.config
n_layers = cfg.num_hidden_layers
d_model  = cfg.hidden_size
n_heads  = cfg.num_attention_heads
d_head   = d_model // n_heads

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
# FORWARD PASS + HOOK CHECK
# ============================================================================
# Verify that our hook-based extraction approach works correctly before
# committing to the full 90-stimulus extraction run.

console.print("\n[bold cyan]Testing hook-based activation extraction...[/bold cyan]")
test_text = "The kitchen counter still had the grocery list in his handwriting."
inputs = tokenizer(test_text, return_tensors="pt").to(DEVICE)
seq_len = inputs["input_ids"].shape[1]
console.print(f"Test prompt: {seq_len} tokens")

# Hook a single layer (mid-model) to verify shape and values
hook_capture = {}

def test_hook(module, input, output):
    # Newer transformers returns hidden states as a plain tensor; older returns a tuple.
    h = output[0] if isinstance(output, tuple) else output
    hook_capture["resid_post"] = h[0, -1, :].detach().cpu().to(torch.float32).numpy()

handle = model.model.layers[15].register_forward_hook(test_hook)

with torch.no_grad():
    out = model(**inputs, use_cache=False)

handle.remove()

act_l15 = hook_capture["resid_post"]
hook_shape_ok = act_l15.shape == (d_model,)
hook_finite_ok = np.all(np.isfinite(act_l15))
hook_nonzero_ok = act_l15.std() > 0.001

console.print(f"  Layer 15 hook shape: {act_l15.shape} — {'[green]PASS[/green]' if hook_shape_ok else '[red]FAIL[/red]'}")
console.print(f"  Values finite: {'[green]PASS[/green]' if hook_finite_ok else '[red]FAIL[/red]'}")
console.print(f"  Non-trivial std={act_l15.std():.4f}: {'[green]PASS[/green]' if hook_nonzero_ok else '[red]FAIL[/red]'}")
console.print(f"  mean={act_l15.mean():.4f}, std={act_l15.std():.4f}")

if not (hook_shape_ok and hook_finite_ok and hook_nonzero_ok):
    all_pass = False

# Quick behavioral check
top5_ids = out.logits[0, -1, :].topk(5).indices.tolist()
top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]
console.print(f"  Next-token top-5: {top5_tokens}")

del out, inputs

# ============================================================================
# REPORT
# ============================================================================

report = f"""Smoke Test Report — {MODEL_ID}
==========================================
Date: 2026-02-24 | Device: {DEVICE} | Dtype: float16
Framework: HF transformers (register_forward_hook, NOT TransformerLens)
Status: {"PASS" if all_pass else "FAIL"}

Architecture:
  n_layers: {n_layers}  d_model: {d_model}  n_heads: {n_heads}  d_head: {d_head}

Hook test (layer 15, final token):
  shape: {act_l15.shape}  mean: {act_l15.mean():.4f}  std: {act_l15.std():.4f}

Gate: {"PASS — proceed to extract.py" if all_pass else "FAIL — do not proceed"}
"""
(OUTPUT_DIR / "smoke_test_report.txt").write_text(report)

status_color = "bold green" if all_pass else "bold red"
console.print(f"\n[{status_color}]Smoke Test: {'PASS' if all_pass else 'FAIL'}[/{status_color}]")
if not all_pass:
    sys.exit(1)
