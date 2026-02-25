"""
Experiment 08 — Phase 1: Activation Extraction
Model: google/gemma-2-9b (42 layers, d_model=3584)
Date: 2026-02-24

Extracts final-token residual stream at all 42 layers for all 192 Set B stimuli
(96 clinical vignettes + 96 matched neutral controls).
Output shape: (192, 42, 3584)

Uses HF transformers + MPS — same method as Phase 0c (03_phase0c_gemma2-9b).
Gemma-2 alternating local/global attention has no impact on resid_post extraction.

Estimated time: ~10-20 min on MPS (192 stimuli × 42 layers)
Run: uv run python experiments/08_phase1_gemma2-9b/extract.py
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

MODEL_ID = "google/gemma-2-9b"
DEVICE   = "mps"
DTYPE    = torch.float16

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
# LOAD STIMULI — Set B (clinical + neutral, 192 total)
# ============================================================================

from stimuli.loader import load_set_b_clinical, load_set_b_neutral

console.print("\n[bold cyan]Loading Set B stimuli...[/bold cyan]")
set_b_clinical = load_set_b_clinical()
set_b_neutral  = load_set_b_neutral()
all_stimuli = sorted(set_b_clinical + set_b_neutral, key=lambda s: s.id)

console.print(f"Set B clinical: {len(set_b_clinical)} stimuli")
console.print(f"Set B neutral:  {len(set_b_neutral)} stimuli")
console.print(f"Total:          {len(all_stimuli)}")
console.print(f"Clinical emotion distribution: {dict(Counter(s.emotion for s in set_b_clinical))}")

# ============================================================================
# LOAD MODEL & TOKENIZER
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on {DEVICE} (float16)...[/bold cyan]")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
model.eval()

n_layers = model.config.num_hidden_layers   # 42
d_model  = model.config.hidden_size         # 3584
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}")
console.print(f"  (Alternating local/global attention — no impact on resid_post)")

# ============================================================================
# REGISTER HOOKS
# ============================================================================

layer_acts = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_acts[layer_idx] = h[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

hooks = []
for i in range(n_layers):
    h = model.model.layers[i].register_forward_hook(make_hook(i))
    hooks.append(h)

console.print(f"✓ Registered {len(hooks)} forward hooks")

# ============================================================================
# MPS VALIDATION
# ============================================================================

console.print("\n[bold yellow]MPS validation...[/bold yellow]")
validation_prompts = [
    ("The Eiffel Tower is located in the city of", ["Paris", " Paris", "paris"]),
    ("The boiling point of water is 100 degrees", ["Celsius", " Celsius", "C", " C", "celsius"]),
]
validation_ok = True
for prompt_text, acceptable in validation_prompts:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    layer_acts.clear()
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    top10 = [tokenizer.decode([tid]) for tid in out.logits[0, -1, :].topk(10).indices.tolist()]
    found = any(tok in top10 for tok in acceptable)
    status = "[green]✓[/green]" if found else "[red]✗[/red]"
    console.print(f"  {status} '{prompt_text[:45]}...' → top-5: {top10[:5]}")
    if not found:
        validation_ok = False
layer_acts.clear()
if not validation_ok:
    for h in hooks:
        h.remove()
    raise RuntimeError("MPS validation FAILED")
console.print("[bold green]✓ MPS validation passed[/bold green]")

# ============================================================================
# EXTRACT ACTIVATIONS
# ============================================================================

console.print(f"\n[bold cyan]Extracting residual stream for {len(all_stimuli)} stimuli...[/bold cyan]")
console.print(f"Output shape: ({len(all_stimuli)}, {n_layers}, {d_model})")
console.print(f"Memory: {len(all_stimuli) * n_layers * d_model * 4 / 1e6:.1f} MB\n")

all_activations = np.zeros((len(all_stimuli), n_layers, d_model), dtype=np.float32)
metadata = []

t_start = time.time()

for stim_idx, stimulus in enumerate(track(all_stimuli, description="Extracting...")):
    inputs = tokenizer(stimulus.text, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model(**inputs, use_cache=False)

    for layer_idx in range(n_layers):
        all_activations[stim_idx, layer_idx, :] = layer_acts[layer_idx]

    metadata.append({
        "index":              stim_idx,
        "id":                 stimulus.id,
        "emotion":            stimulus.emotion,
        "stimulus_set":       stimulus.stimulus_set,
        "domain":             stimulus.domain,
        "matched_control_id": stimulus.matched_control_id,
        "word_count":         stimulus.word_count,
        "n_tokens":           n_tokens,
    })

    layer_acts.clear()
    torch.mps.empty_cache()
    gc.collect()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(all_stimuli):.2f}s/stimulus)")

for h in hooks:
    h.remove()

# ============================================================================
# SANITY CHECKS
# ============================================================================

assert np.all(np.isfinite(all_activations)), "Non-finite values!"
inter_std = all_activations[:, n_layers // 2, :].std(axis=0).mean()
console.print(f"✓ Inter-stimulus std at mid-layer: {inter_std:.4f}")
assert inter_std > 0.001

for li in [0, n_layers // 4, n_layers // 2, n_layers - 1]:
    la = all_activations[:, li, :]
    console.print(f"  Layer {li:2d}: mean={la.mean():.3f} std={la.std():.3f} range=[{la.min():.3f}, {la.max():.3f}]")

# ============================================================================
# SAVE
# ============================================================================

np.save(ACT_DIR / "set_b_residuals.npy", all_activations)
console.print(f"\n[bold green]✓ Saved: set_b_residuals.npy ({all_activations.shape})[/bold green]")

with open(ACT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

pl.DataFrame(metadata).write_csv(ACT_DIR / "metadata.csv")
console.print(f"[bold green]✓ Saved: metadata.json + metadata.csv[/bold green]")
console.print(f"\n[bold green]✓ Ready for run.py[/bold green]")
