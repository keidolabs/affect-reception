"""
Experiment 10 — Set C Tak Replication: Activation Extraction (Llama-3.2-1B)
Question: Does prediction-task framing shift emotion consolidation to mid-layers?
Date: 2026-02-25

Identical to Exp 00 (Phase 0 Llama-1B) except:
  - Only 80 emotional stimuli (no neutrals — 8-class probe doesn't use them)
  - Suffix "\n\nEmotion:" appended to each stimulus before tokenization
  - Extraction point = final token of SUFFIXED text (the ":" token)
  - This replicates the next-token-prediction task framing of Tak et al.

Model: TransformerLens + CPU (never MPS with TL — GitHub #1178)
Output: outputs/activations/llama1b_set_c_residuals.npy — shape (80, 16, 2048)

Run: uv run python experiments/10_setc_tak_replication/extract_llama1b.py
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

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EXP_DIR / "outputs"

sys.path.insert(0, str(EXP_DIR.parent.parent))

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE   = "cpu"   # CRITICAL: Never use MPS with TransformerLens (TL issue #1178)

# The key experimental manipulation: suffix that forces next-token prediction framing.
# ":" is the final token — model is about to predict an emotion label.
SUFFIX = "\n\nEmotion:"

# Plausible emotion tokens to expect in top-10 for behavioral check (broad set)
EMOTION_VOCAB = {
    "rage":       [" anger", " rage", " fury", " frustration", " resentment", " wrath", "anger", "rage"],
    "grief":      [" sadness", " grief", " sorrow", " loss", " depression", " loneliness", "sadness", "grief"],
    "terror":     [" fear", " terror", " panic", " anxiety", " dread", " horror", "fear", "terror"],
    "ecstasy":    [" joy", " happiness", " euphoria", " ecstasy", " elation", " excitement", "joy", "happiness"],
    "loathing":   [" disgust", " loathing", " contempt", " hatred", " repulsion", "disgust", "loathing"],
    "amazement":  [" surprise", " wonder", " amazement", " shock", " awe", "surprise", "wonder", "amazement"],
    "admiration": [" admiration", " respect", " awe", " pride", " love", "admiration", "respect"],
    "vigilance":  [" anxiety", " vigilance", " alertness", " anticipation", " worry", "anxiety", "vigilance"],
}

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
# LOAD STIMULI — 80 emotional only (no neutrals for 8-class probe)
# ============================================================================

from stimuli.loader import load_set_a

console.print("\n[bold cyan]Loading Set A stimuli (emotional only)...[/bold cyan]")
set_a_all = load_set_a()
# Exclude neutral controls — 8-class probe uses emotional stimuli only
set_a_emotional = [s for s in set_a_all if s.emotion != "neutral"]
set_a_sorted = sorted(set_a_emotional, key=lambda s: s.id)

console.print(f"Emotional stimuli: {len(set_a_sorted)}")
console.print(f"Emotion distribution: {dict(Counter(s.emotion for s in set_a_sorted))}")
assert len(set_a_sorted) == 80, f"Expected 80 emotional stimuli, got {len(set_a_sorted)}"

# ============================================================================
# LOAD MODEL — TransformerLens on CPU
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on CPU (float32, TransformerLens)...[/bold cyan]")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.float32)
model.eval()

n_layers = model.cfg.n_layers   # 16
d_model  = model.cfg.d_model    # 2048
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}")

# ============================================================================
# SUFFIX SANITY CHECK — verify ":" is a single token for this tokenizer
# ============================================================================

console.print(f"\n[bold cyan]Suffix sanity check...[/bold cyan]")
console.print(f"Suffix: {repr(SUFFIX)}")

test_text = "I feel something.\n\nEmotion:"
test_tokens = model.to_tokens(test_text)
final_token_id = test_tokens[0, -1].item()
final_token_str = model.tokenizer.decode([final_token_id])
console.print(f"Final token of suffixed test text: {repr(final_token_str)} (id={final_token_id})")
if ":" not in final_token_str:
    console.print(f"[yellow]WARNING: Final token is {repr(final_token_str)}, not ':' — check suffix tokenization[/yellow]")
else:
    console.print(f"[bold green]✓ Final token contains ':' as expected[/bold green]")

# ============================================================================
# ACTIVATION EXTRACTION — suffixed texts, extract at final ":" token
# ============================================================================

console.print(f"\n[bold cyan]Extracting Set C residuals for {len(set_a_sorted)} stimuli...[/bold cyan]")
console.print(f"Target shape: ({len(set_a_sorted)}, {n_layers}, {d_model})")

all_activations = np.zeros((len(set_a_sorted), n_layers, d_model), dtype=np.float32)
metadata = []

t_start = time.time()

for stim_idx, stimulus in enumerate(track(set_a_sorted, description="Extracting...")):
    suffixed_text = stimulus.text + SUFFIX

    tokens_orig  = model.to_tokens(stimulus.text)
    tokens_suf   = model.to_tokens(suffixed_text)
    n_orig = tokens_orig.shape[1]
    n_suf  = tokens_suf.shape[1]
    final_tok_str = model.tokenizer.decode([tokens_suf[0, -1].item()])

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens_suf,
            names_filter=lambda name: "hook_resid_post" in name,
        )

    # Extract final-token residual at every layer (the ":" token)
    for layer_idx in range(n_layers):
        act = cache["resid_post", layer_idx][:, -1, :].squeeze().cpu().numpy()
        all_activations[stim_idx, layer_idx, :] = act

    metadata.append({
        "index":            stim_idx,
        "id":               stimulus.id,
        "emotion":          stimulus.emotion,
        "stimulus_set":     stimulus.stimulus_set,
        "suffix":           SUFFIX,
        "n_tokens_original": n_orig,
        "n_tokens_suffixed": n_suf,
        "final_token_str":  final_tok_str,
        "intensity":        stimulus.intensity,
        "word_count":       stimulus.word_count,
    })

    del logits, cache
    gc.collect()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(set_a_sorted):.2f}s/stimulus)")
console.print(f"Shape: {all_activations.shape}")

# ============================================================================
# SANITY CHECKS
# ============================================================================

assert np.all(np.isfinite(all_activations)), "Non-finite values in activations!"
console.print("✓ All activations finite")

inter_std = all_activations[:, n_layers // 2, :].std(axis=0).mean()
console.print(f"✓ Inter-stimulus std at mid-layer: {inter_std:.4f}")
assert inter_std > 0.001, "Activations look identical — something is wrong!"

for li in [0, n_layers // 4, n_layers // 2, n_layers - 1]:
    la = all_activations[:, li, :]
    console.print(f"  Layer {li:2d}: mean={la.mean():.3f} std={la.std():.3f} range=[{la.min():.3f}, {la.max():.3f}]")

# Check most stimuli end at ":"
final_tokens = [m["final_token_str"] for m in metadata]
colon_count = sum(1 for t in final_tokens if ":" in t)
console.print(f"✓ Stimuli ending at ':': {colon_count}/{len(final_tokens)}")
if colon_count < len(final_tokens) * 0.9:
    console.print(f"[yellow]WARNING: Only {colon_count}/{len(final_tokens)} stimuli end at ':' — check suffix[/yellow]")

# ============================================================================
# BEHAVIORAL CHECK — top-10 predictions at ":" for one stimulus per emotion
# ============================================================================

console.print("\n[bold cyan]Behavioral check: top-10 predictions at ':' position...[/bold cyan]")

# Pick first stimulus per emotion (sorted list, first occurrence of each)
samples = {}
for s in set_a_sorted:
    if s.emotion not in samples:
        samples[s.emotion] = s

behavioral_lines = ["Set C Behavioral Check — Llama-3.2-1B", "=" * 50, ""]
behavioral_lines.append(f"Suffix: {repr(SUFFIX)}")
behavioral_lines.append(f"Pass criterion: ≥5/8 emotions have plausible emotion token in top-10")
behavioral_lines.append("")

n_pass = 0
for emotion, stimulus in sorted(samples.items()):
    suffixed = stimulus.text + SUFFIX
    tokens = model.to_tokens(suffixed)

    with torch.no_grad():
        logits, _ = model.run_with_cache(
            tokens, names_filter=lambda name: False  # no cache, just logits
        )

    top10_ids = logits[0, -1, :].topk(10).indices.tolist()
    top10_toks = [model.tokenizer.decode([tid]) for tid in top10_ids]
    expected = EMOTION_VOCAB.get(emotion, [])
    found = any(tok in top10_toks for tok in expected)
    if found:
        n_pass += 1

    status = "PASS" if found else "FAIL"
    console.print(f"  [{status}] {emotion:12s} | top-10: {top10_toks[:5]}")
    behavioral_lines.append(f"[{status}] {emotion}")
    behavioral_lines.append(f"  Top-10: {top10_toks}")
    behavioral_lines.append(f"  Expected any of: {expected[:4]}")
    behavioral_lines.append("")

gate = "PASS" if n_pass >= 5 else "FAIL"
behavioral_lines.append(f"Result: {n_pass}/8 emotions passed → Behavioral gate: {gate}")
console.print(f"\nBehavioral gate: {n_pass}/8 passed → [bold {'green' if gate == 'PASS' else 'red'}]{gate}[/bold {'green' if gate == 'PASS' else 'red'}]")
if gate == "FAIL":
    console.print("[yellow]NOTE: Low pass rate may indicate suffix format needs adjustment[/yellow]")

(OUT_DIR / "behavioral_check_llama1b.txt").write_text("\n".join(behavioral_lines))
console.print("✓ behavioral_check_llama1b.txt")

# ============================================================================
# SAVE
# ============================================================================

np.save(ACT_DIR / "llama1b_set_c_residuals.npy", all_activations)
console.print(f"\n[bold green]✓ Saved: llama1b_set_c_residuals.npy ({all_activations.shape})[/bold green]")

with open(ACT_DIR / "llama1b_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

import polars as pl
pl.DataFrame(metadata).write_csv(ACT_DIR / "llama1b_metadata.csv")
console.print(f"[bold green]✓ Saved: metadata.json + metadata.csv[/bold green]")
console.print(f"\n[bold green]✓ Ready for run.py[/bold green]")
