"""
Experiment 00 — Activation Extraction
Question: What do residual stream activations look like for Set A emotional vs neutral stimuli?
Date: 2026-02-24

Extracts final-token residual stream activations at all 16 layers of Llama-3.2-1B
for all 90 Set A stimuli (80 emotional + 10 neutral).

Saves:
  - outputs/activations/set_a_residuals.npy — shape (90, 16, 2048)
  - outputs/activations/metadata.json — index → {stimulus_id, emotion, stimulus_set}

Run: uv run python experiments/00_phase0_replication/extract.py
Estimated time: ~10-15 minutes on CPU (1B model, 90 stimuli)
"""

import gc
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.progress import track

console = Console()

# Pin random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"  # CUDA on scrig (RTX 5060 Ti) — TL works fine on CUDA, only MPS is broken

# Add project root to path for stimuli.loader import
sys.path.insert(0, str(EXP_DIR.parent.parent))

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
# LOAD STIMULI — Set A only (90 stimuli for Phase 0)
# ============================================================================

from stimuli.loader import load_set_a, load_all

console.print("\n[bold cyan]Loading Set A stimuli...[/bold cyan]")
set_a = load_set_a()
console.print(f"Set A: {len(set_a)} stimuli")

from collections import Counter
emotion_counts = Counter(s.emotion for s in set_a)
console.print(f"Emotion distribution: {dict(emotion_counts)}")

# Sort by ID for reproducibility
set_a_sorted = sorted(set_a, key=lambda s: s.id)

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

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}")

# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================

console.print(f"\n[bold cyan]Extracting residual stream activations for {len(set_a_sorted)} stimuli...[/bold cyan]")
console.print("Extracting final-token activations at all 16 layers.")
console.print("Each stimulus: cache['resid_post', layer][:, -1, :] → shape (d_model,)\n")

# Accumulate (n_stimuli, n_layers, d_model) — small enough to hold in memory
# 90 stimuli × 16 layers × 2048 dims × 4 bytes = ~11.5 MB — trivial
all_activations = np.zeros((len(set_a_sorted), n_layers, d_model), dtype=np.float32)

metadata = []  # list of dicts with stimulus info per index

for stim_idx, stimulus in enumerate(track(set_a_sorted, description="Extracting...")):
    # Tokenize
    tokens = model.to_tokens(stimulus.text)

    # Forward pass with cache — only cache residual stream (saves memory vs full cache)
    # names_filter accepts a callable: True = cache this hook
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "hook_resid_post" in name,
        )

    # Extract final-token activation at every layer: (d_model,)
    for layer_idx in range(n_layers):
        act = cache["resid_post", layer_idx][:, -1, :].squeeze().cpu().numpy()
        all_activations[stim_idx, layer_idx, :] = act

    # Record metadata
    metadata.append({
        "index": stim_idx,
        "id": stimulus.id,
        "emotion": stimulus.emotion,
        "stimulus_set": stimulus.stimulus_set,
        "intensity": stimulus.intensity,
        "word_count": stimulus.word_count,
        "n_tokens": tokens.shape[1],
    })

    # Free cache and logits to avoid memory accumulation
    del logits, cache
    gc.collect()
    torch.cuda.empty_cache()

console.print(f"\n[bold green]✓ Extraction complete[/bold green]")
console.print(f"Activations shape: {all_activations.shape}")
console.print(f"  → ({len(set_a_sorted)} stimuli, {n_layers} layers, {d_model} d_model)")
console.print(f"Memory footprint: {all_activations.nbytes / 1e6:.1f} MB")

# ============================================================================
# SANITY CHECK — Activations should be non-zero and have reasonable statistics
# ============================================================================

console.print("\n[bold cyan]Sanity checks...[/bold cyan]")

# Check that activations are finite and non-trivial
assert np.all(np.isfinite(all_activations)), "FAIL: Non-finite values in activations!"
console.print("✓ All activations are finite")

# Check that activations differ meaningfully across stimuli
# (if they were all identical, something is very wrong)
inter_stimulus_std = all_activations[:, 7, :].std(axis=0).mean()  # layer 7 (middle)
console.print(f"✓ Inter-stimulus std at layer 7 (mean across dims): {inter_stimulus_std:.4f}")
assert inter_stimulus_std > 0.001, "FAIL: Activations look identical across stimuli!"

# Per-layer activation statistics
console.print("\nPer-layer statistics (final-token activations):")
for layer_idx in [0, 4, 8, 12, 15]:
    layer_acts = all_activations[:, layer_idx, :]
    console.print(
        f"  Layer {layer_idx:2d}: mean={layer_acts.mean():.3f}, "
        f"std={layer_acts.std():.3f}, "
        f"range=[{layer_acts.min():.3f}, {layer_acts.max():.3f}]"
    )

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

console.print("\n[bold cyan]Saving activations...[/bold cyan]")

# Save raw activations: (90, 16, 2048) — gitignored due to *.npy rule
act_path = ACT_DIR / "set_a_residuals.npy"
np.save(act_path, all_activations)
console.print(f"[bold green]✓ Saved: {act_path}[/bold green]")
console.print(f"  Shape: {all_activations.shape}, Size: {act_path.stat().st_size / 1e6:.1f} MB")

# Save metadata JSON
meta_path = ACT_DIR / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
console.print(f"[bold green]✓ Saved: {meta_path}[/bold green]")

# Save a small summary CSV (not gitignored — useful for quick inspection)
import polars as pl
meta_df = pl.DataFrame(metadata)
meta_df.write_csv(ACT_DIR / "metadata.csv")
console.print(f"[bold green]✓ Saved: {ACT_DIR / 'metadata.csv'}[/bold green]")

console.print(f"\n[bold green]✓ Extraction complete! Ready for analysis (run.py)[/bold green]")
