"""
Experiment 01 — Phase 0a: Activation Extraction
Model: meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
Date: 2026-02-24

Extracts final-token residual stream at all 32 layers for all 90 Set A stimuli.
Output shape: (90, 32, 4096)

Uses HF transformers directly on MPS — NOT TransformerLens.
TL issue #1178 documents MPS correctness bugs specific to TransformerLens
(torch.triu / torch.where mismatches causing inverted logit differences).
HF transformers has independent MPS testing for Llama-3 and is not affected.

Estimated time: ~3-8 min on MPS (vs 25-40 min on CPU)
Run: uv run python experiments/01_phase0a_llama3-8b/extract.py
"""

import gc
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.progress import track
import polars as pl
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXP_DIR.parent.parent))

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEVICE   = "mps"
DTYPE    = torch.float16   # ~16GB — safe on 32GB M1 unified memory

# ============================================================================
# LOAD HF TOKEN
# ============================================================================

env_path = EXP_DIR.parent.parent / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
console.print("[bold cyan]HF token loaded[/bold cyan]")

# ============================================================================
# LOAD STIMULI
# ============================================================================

from stimuli.loader import load_set_a

console.print("\n[bold cyan]Loading Set A stimuli...[/bold cyan]")
set_a = load_set_a()
set_a_sorted = sorted(set_a, key=lambda s: s.id)
console.print(f"Set A: {len(set_a_sorted)} stimuli")
console.print(f"Emotion distribution: {dict(Counter(s.emotion for s in set_a_sorted))}")

# ============================================================================
# LOAD MODEL & TOKENIZER (HF, not TransformerLens)
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on {DEVICE} (float16)...[/bold cyan]")
console.print("  Note: ~16GB model — allow 2-4 min for initial load to MPS")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load weights in float16, move to MPS.
# device_map="auto" would shard across CPU+MPS on memory pressure — avoid that,
# we want all weights on MPS for consistent activation extraction.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
).to(DEVICE)
model.eval()

n_layers = model.config.num_hidden_layers   # 32
d_model  = model.config.hidden_size         # 4096
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}")

# ============================================================================
# REGISTER HOOKS (once, before the extraction loop)
# ============================================================================
# Hook point: model.model.layers[i] is LlamaDecoderLayer.
# output[0] shape: (batch, seq_len, d_model) = residual stream after full layer
# (attention + MLP residual additions) = equivalent to TL's "resid_post".
#
# We register all hooks once and keep them live for the entire extraction loop.
# layer_acts is cleared after each stimulus.

layer_acts = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # Newer transformers returns hidden states as a plain tensor (batch, seq, d_model);
        # older versions wrap it in a tuple. Handle both.
        h = output[0] if isinstance(output, tuple) else output
        layer_acts[layer_idx] = h[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

hooks = []
for i in range(n_layers):
    h = model.model.layers[i].register_forward_hook(make_hook(i))
    hooks.append(h)

console.print(f"✓ Registered {len(hooks)} forward hooks")

# ============================================================================
# MPS VALIDATION — 2 behavioral sanity checks
# ============================================================================
# If MPS is silently computing wrong values, these should fail.
# We check that top-10 predicted tokens include sensible completions.

console.print("\n[bold yellow]MPS validation (behavioral sanity checks)...[/bold yellow]")

validation_prompts = [
    # (prompt, list of acceptable next tokens)
    ("The Eiffel Tower is located in the city of", ["Paris", " Paris", "paris"]),
    ("The boiling point of water is 100 degrees", ["Celsius", " Celsius", "C", " C", "celsius"]),
]

validation_ok = True
for prompt_text, acceptable in validation_prompts:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    layer_acts.clear()
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    top10_ids = out.logits[0, -1, :].topk(10).indices.tolist()
    top10_tokens = [tokenizer.decode([tid]) for tid in top10_ids]
    found = any(tok in top10_tokens for tok in acceptable)
    status = "[bold green]✓[/bold green]" if found else "[bold red]✗[/bold red]"
    console.print(f"  {status} Prompt: '{prompt_text[:50]}...'")
    console.print(f"       Top-5: {top10_tokens[:5]}  (expected one of {acceptable})")
    if not found:
        validation_ok = False

layer_acts.clear()  # discard hook captures from validation runs

if not validation_ok:
    for h in hooks:
        h.remove()
    raise RuntimeError(
        "MPS validation FAILED — model output is wrong, do not trust activations. "
        "Try DEVICE='cpu' as fallback."
    )

console.print("[bold green]✓ MPS validation passed — proceeding with extraction[/bold green]")

# ============================================================================
# EXTRACT ACTIVATIONS
# ============================================================================

console.print(f"\n[bold cyan]Extracting residual stream for {len(set_a_sorted)} stimuli...[/bold cyan]")
console.print(f"Target shape: ({len(set_a_sorted)}, {n_layers}, {d_model})")
console.print(f"Storage: {len(set_a_sorted) * n_layers * d_model * 4 / 1e6:.1f} MB (float32 on CPU)\n")

# Accumulate as float32 numpy — the (90, 32, 4096) array is only ~42MB
all_activations = np.zeros((len(set_a_sorted), n_layers, d_model), dtype=np.float32)
metadata = []

t_start = time.time()

for stim_idx, stimulus in enumerate(track(set_a_sorted, description="Extracting...")):
    inputs = tokenizer(stimulus.text, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model(**inputs, use_cache=False)

    # Pull from hook captures — all 32 layers filled by the forward pass
    for layer_idx in range(n_layers):
        all_activations[stim_idx, layer_idx, :] = layer_acts[layer_idx]

    metadata.append({
        "index":        stim_idx,
        "id":           stimulus.id,
        "emotion":      stimulus.emotion,
        "stimulus_set": stimulus.stimulus_set,
        "intensity":    stimulus.intensity,
        "word_count":   stimulus.word_count,
        "n_tokens":     n_tokens,
    })

    layer_acts.clear()
    torch.mps.empty_cache()
    gc.collect()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(set_a_sorted):.2f}s/stimulus)")
console.print(f"Shape: {all_activations.shape} | Memory: {all_activations.nbytes / 1e6:.1f} MB")

# Remove hooks — no longer needed
for h in hooks:
    h.remove()

# ============================================================================
# SANITY CHECKS
# ============================================================================

assert np.all(np.isfinite(all_activations)), "Non-finite values in activations!"
console.print("✓ All activations finite")

inter_std = all_activations[:, n_layers // 2, :].std(axis=0).mean()
console.print(f"✓ Inter-stimulus std at mid-layer: {inter_std:.4f}")
assert inter_std > 0.001, "Activations look identical across stimuli — something is wrong!"

for li in [0, n_layers // 4, n_layers // 2, n_layers - 1]:
    la = all_activations[:, li, :]
    console.print(f"  Layer {li:2d}: mean={la.mean():.3f} std={la.std():.3f} range=[{la.min():.3f}, {la.max():.3f}]")

# ============================================================================
# SAVE
# ============================================================================

np.save(ACT_DIR / "set_a_residuals.npy", all_activations)
console.print(f"\n[bold green]✓ Saved: set_a_residuals.npy ({all_activations.shape})[/bold green]")

with open(ACT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

pl.DataFrame(metadata).write_csv(ACT_DIR / "metadata.csv")
console.print(f"[bold green]✓ Saved: metadata.json + metadata.csv[/bold green]")
console.print(f"\n[bold green]✓ Ready for run.py[/bold green]")
