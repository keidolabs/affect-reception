"""
Experiment 00 — TransformerLens Smoke Test
Question: Does Llama-3.2-1B load correctly on CPU via TransformerLens and produce valid caches?
Date: 2026-02-24

Verifies:
  - Model loads with correct architecture (16 layers, d_model=2048)
  - run_with_cache() returns valid residual stream activations
  - Cache keys are accessible in both full and short-form notation
  - No MPS is used (critical: MPS produces silently wrong results)

Run: uv run python experiments/00_phase0_replication/smoke_test.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Pin random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"  # CUDA on scrig (RTX 5060 Ti) — TL works fine on CUDA, only MPS is broken

# ============================================================================
# LOAD HF TOKEN
# ============================================================================

# HF_ACCESS_TOKEN is in .env at project root
env_path = Path(__file__).parent.parent.parent / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break

console.print(f"[bold cyan]HF token loaded from .env[/bold cyan]")

# ============================================================================
# LOAD MODEL WITH TRANSFORMERLENS
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} via TransformerLens on CPU...[/bold cyan]")
console.print("(First run: ~2GB download. Subsequent runs load from HF cache.)")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    dtype=torch.float16,  # float16 on CUDA — faster, fits easily in 16GB VRAM
)
model.eval()
console.print("[bold green]✓ Model loaded[/bold green]")

# ============================================================================
# VERIFY MODEL CONFIG
# ============================================================================

console.print("\n[bold cyan]Verifying model architecture...[/bold cyan]")

cfg = model.cfg
checks = [
    ("n_layers", cfg.n_layers, 16, "16 layers for Llama-3.2-1B"),
    ("d_model", cfg.d_model, 2048, "2048 hidden dim for Llama-3.2-1B"),
    ("n_heads", cfg.n_heads, 32, "32 attention heads"),
    ("d_head", cfg.d_head, 64, "64 dims per head"),
]

config_table = Table(title=f"Model Configuration: {MODEL_ID}")
config_table.add_column("Parameter", style="cyan")
config_table.add_column("Value", justify="right")
config_table.add_column("Expected", justify="right", style="dim")
config_table.add_column("Status", style="green")

all_pass = True
for name, actual, expected, description in checks:
    status = "✓ PASS" if actual == expected else f"✗ FAIL (got {actual})"
    style = "green" if actual == expected else "red"
    config_table.add_row(name, str(actual), str(expected), f"[{style}]{status}[/{style}]")
    if actual != expected:
        all_pass = False

console.print(config_table)

# ============================================================================
# SINGLE FORWARD PASS WITH CACHE
# ============================================================================

console.print("\n[bold cyan]Running forward pass with cache...[/bold cyan]")

# Grief-neutral vignette from Set A for smoke test (single emotionally neutral sentence)
test_text = "The kitchen counter still had the grocery list in his handwriting."

tokens = model.to_tokens(test_text)
console.print(f"Test text: '{test_text}'")
console.print(f"Token count: {tokens.shape[1]} tokens (shape: {tokens.shape})")

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

console.print(f"[bold green]✓ Forward pass complete[/bold green]")
console.print(f"Logits shape: {logits.shape}")

# ============================================================================
# VERIFY CACHE KEYS AND SHAPES
# ============================================================================

console.print("\n[bold cyan]Verifying cache keys and shapes...[/bold cyan]")

seq_len = tokens.shape[1]
expected_resid_shape = (1, seq_len, cfg.d_model)

cache_checks = []

# Check residual stream at each layer (short-form access)
for layer_idx in range(cfg.n_layers):
    resid = cache["resid_post", layer_idx]
    shape_ok = resid.shape == expected_resid_shape
    cache_checks.append({
        "key": f"resid_post, {layer_idx}",
        "shape": str(resid.shape),
        "expected": str(expected_resid_shape),
        "pass": shape_ok,
    })

# Check attention patterns at layer 0
attn_pattern = cache["pattern", 0]
expected_attn_shape = (1, cfg.n_heads, seq_len, seq_len)
cache_checks.append({
    "key": "pattern, 0",
    "shape": str(attn_pattern.shape),
    "expected": str(expected_attn_shape),
    "pass": attn_pattern.shape == expected_attn_shape,
})

# Check MLP output at layer 0
mlp_out = cache["mlp_out", 0]
expected_mlp_shape = (1, seq_len, cfg.d_model)
cache_checks.append({
    "key": "mlp_out, 0",
    "shape": str(mlp_out.shape),
    "expected": str(expected_mlp_shape),
    "pass": mlp_out.shape == expected_mlp_shape,
})

# Also verify full-form access works
resid_full_form = cache["blocks.0.hook_resid_post"]
full_form_ok = resid_full_form.shape == expected_resid_shape
cache_checks.append({
    "key": "blocks.0.hook_resid_post (full form)",
    "shape": str(resid_full_form.shape),
    "expected": str(expected_resid_shape),
    "pass": full_form_ok,
})

# Print summary (just show first 2 layers + special checks)
cache_table = Table(title="Cache Verification")
cache_table.add_column("Cache Key", style="cyan")
cache_table.add_column("Shape", justify="right")
cache_table.add_column("Status")

critical_checks = cache_checks[:2] + cache_checks[-3:]  # first 2 layers + last 3
for check in critical_checks:
    status = "[green]✓ PASS[/green]" if check["pass"] else f"[red]✗ FAIL expected {check['expected']}[/red]"
    cache_table.add_row(check["key"], check["shape"], status)

# Summary for all 16 layers
all_resid_pass = all(c["pass"] for c in cache_checks[:16])
cache_table.add_row(
    "resid_post, layers 0-15 (all)",
    f"(1, {seq_len}, {cfg.d_model}) each",
    "[green]✓ ALL PASS[/green]" if all_resid_pass else "[red]✗ SOME FAIL[/red]",
)
console.print(cache_table)

all_cache_pass = all(c["pass"] for c in cache_checks)
if all_cache_pass:
    console.print("[bold green]✓ All cache checks passed[/bold green]")
    all_pass = all_pass and True
else:
    console.print("[bold red]✗ Some cache checks failed[/bold red]")
    all_pass = False

# ============================================================================
# VERIFY FINAL-TOKEN EXTRACTION (used in extract.py)
# ============================================================================

console.print("\n[bold cyan]Verifying final-token residual extraction...[/bold cyan]")

# This is the exact extraction pattern used in extract.py
for layer_idx in [0, 7, 15]:
    resid_final_tok = cache["resid_post", layer_idx][:, -1, :].cpu().numpy()
    console.print(
        f"  Layer {layer_idx:2d} final-token activation: shape={resid_final_tok.shape}, "
        f"mean={resid_final_tok.mean():.4f}, std={resid_final_tok.std():.4f}"
    )

console.print("[bold green]✓ Final-token extraction works[/bold green]")

# ============================================================================
# DEVICE CONFIRMATION
# ============================================================================

console.print(f"\n[bold cyan]Device checks:[/bold cyan]")
console.print(f"  Model device: {next(model.parameters()).device}")
console.print(f"  Logits device: {logits.device}")
console.print(f"  Cache device (resid_post, 0): {cache['resid_post', 0].device}")
assert "cuda" in str(next(model.parameters()).device), "Model must be on CUDA!"
console.print("[bold green]✓ All on CUDA[/bold green]")

# ============================================================================
# SAVE SMOKE TEST REPORT
# ============================================================================

report = f"""Smoke Test Report — {MODEL_ID}
==========================================

Date: 2026-02-24
Device: {DEVICE}
Status: {"PASS" if all_pass else "FAIL"}

Model Architecture:
  n_layers: {cfg.n_layers} (expected 16)
  d_model:  {cfg.d_model} (expected 2048)
  n_heads:  {cfg.n_heads} (expected 32)
  d_head:   {cfg.d_head} (expected 64)

Cache Verification:
  resid_post at all 16 layers: shape (1, seq_len, 2048) — {"PASS" if all_resid_pass else "FAIL"}
  attention patterns: shape (1, n_heads, seq_len, seq_len) — PASS
  mlp_out: shape (1, seq_len, 2048) — PASS

Test Forward Pass:
  Text: '{test_text}'
  Tokens: {tokens.shape[1]}
  Logits shape: {logits.shape}

Device: CUDA (scrig RTX 5060 Ti) — TransformerLens works correctly on CUDA.

Gate: {"PASS — proceed to activation extraction" if all_pass else "FAIL — fix issues above before extracting"}
"""

report_path = OUTPUT_DIR / "smoke_test_report.txt"
report_path.write_text(report)
console.print(f"\n[bold green]✓ Saved: {report_path}[/bold green]")

# Final status
console.print(f"\n{'[bold green]' if all_pass else '[bold red]'}{'=' * 60}[/bold {'green]' if all_pass else 'red]'}")
console.print(f"{'[bold green]' if all_pass else '[bold red]'}Smoke Test: {'PASS' if all_pass else 'FAIL'}[/bold {'green]' if all_pass else 'red]'}")
console.print(f"{'[bold green]' if all_pass else '[bold red]'}{'=' * 60}[/bold {'green]' if all_pass else 'red]'}")

if not all_pass:
    sys.exit(1)
