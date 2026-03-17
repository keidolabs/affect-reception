"""
Experiment 01 — Phase 0a: Activation Extraction
Model: meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
Date: 2026-02-24

Extracts final-token residual stream at all 32 layers for all 90 Set A stimuli.
Output shape: (90, 32, 4096)

Uses TransformerLens on CUDA (Scrig GPU rig).
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

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXP_DIR.parent.parent))

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEVICE   = "cuda"

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
# LOAD MODEL WITH TRANSFORMERLENS
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on CUDA (float16)...[/bold cyan]")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    dtype=torch.float16,
)
model.eval()

n_layers = model.cfg.n_layers   # 32
d_model  = model.cfg.d_model    # 4096
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}")

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
    tokens = model.to_tokens(stimulus.text)
    n_tokens = tokens.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "hook_resid_post" in name,
        )

    # Extract final-token activation at every layer
    for layer_idx in range(n_layers):
        act = cache["resid_post", layer_idx][:, -1, :].squeeze().cpu().numpy()
        all_activations[stim_idx, layer_idx, :] = act

    metadata.append({
        "index":        stim_idx,
        "id":           stimulus.id,
        "emotion":      stimulus.emotion,
        "stimulus_set": stimulus.stimulus_set,
        "intensity":    stimulus.intensity,
        "word_count":   stimulus.word_count,
        "n_tokens":     n_tokens,
    })

    del logits, cache
    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(set_a_sorted):.2f}s/stimulus)")
console.print(f"Shape: {all_activations.shape} | Memory: {all_activations.nbytes / 1e6:.1f} MB")

# ============================================================================
# SANITY CHECKS
# ============================================================================

assert np.all(np.isfinite(all_activations)), "Non-finite values in activations!"
console.print("✓ All activations finite")

inter_std = all_activations[:, n_layers // 2, :].std(axis=0).mean()
console.print(f"✓ Inter-stimulus std at mid-layer: {inter_std:.4f}")
assert inter_std > 0.001, "Activations look identical across stimuli — something is wrong!"

for li in [0, 8, 16, 24, 31]:
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
