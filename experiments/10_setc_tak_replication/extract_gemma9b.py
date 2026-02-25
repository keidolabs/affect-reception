"""
Experiment 10 — Set C Tak Replication: Activation Extraction (Gemma-2-9B)
Question: Does prediction-task framing shift emotion consolidation to mid-layers?
Date: 2026-02-25

Identical to Exp 01 (Phase 0a Llama-3-8B) except:
  - Only 80 emotional stimuli (no neutrals)
  - Suffix "\n\nEmotion:" appended to each stimulus before tokenization
  - Extraction point = final token of SUFFIXED text (the ":" token)
  - This replicates the next-token-prediction task framing of Tak et al.

Model: HF transformers + MPS (NOT TransformerLens — TL issue #1178)
Output: outputs/activations/gemma9b_set_c_residuals.npy — shape (80, 42, 3584)

Run: uv run python experiments/10_setc_tak_replication/extract_gemma9b.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

torch.manual_seed(42)
np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EXP_DIR / "outputs"

sys.path.insert(0, str(EXP_DIR.parent.parent))

MODEL_ID = "google/gemma-2-9b"
DEVICE   = "mps"
DTYPE    = torch.float16

SUFFIX = "\n\nEmotion:"

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
# LOAD STIMULI — 80 emotional only
# ============================================================================

from stimuli.loader import load_set_a

console.print("\n[bold cyan]Loading Set A stimuli (emotional only)...[/bold cyan]")
set_a_all = load_set_a()
set_a_emotional = [s for s in set_a_all if s.emotion != "neutral"]
set_a_sorted = sorted(set_a_emotional, key=lambda s: s.id)

console.print(f"Emotional stimuli: {len(set_a_sorted)}")
console.print(f"Emotion distribution: {dict(Counter(s.emotion for s in set_a_sorted))}")
assert len(set_a_sorted) == 80

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

# ============================================================================
# SUFFIX SANITY CHECK
# ============================================================================

console.print(f"\n[bold cyan]Suffix sanity check...[/bold cyan]")
test_ids = tokenizer("I feel something." + SUFFIX, return_tensors="pt")["input_ids"]
final_tok = tokenizer.decode([test_ids[0, -1].item()])
console.print(f"Final token of suffixed text: {repr(final_tok)}")
if ":" not in final_tok:
    console.print(f"[yellow]WARNING: Final token is {repr(final_tok)}, not ':' — check suffix[/yellow]")
else:
    console.print(f"[bold green]✓ Final token contains ':' as expected[/bold green]")

# ============================================================================
# REGISTER HOOKS (once, for all 32 layers)
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
# MPS VALIDATION — pipeline correctness check
# ============================================================================

console.print("\n[bold yellow]MPS validation (behavioral sanity checks)...[/bold yellow]")

validation_prompts = [
    ("The Eiffel Tower is located in the city of", ["Paris", " Paris", "paris"]),
    ("The boiling point of water is 100 degrees", ["Celsius", " Celsius", "C", " C"]),
]

validation_ok = True
for prompt_text, acceptable in validation_prompts:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    layer_acts.clear()
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    top10 = [tokenizer.decode([tid]) for tid in out.logits[0, -1, :].topk(10).indices.tolist()]
    found = any(tok in top10 for tok in acceptable)
    status = "[bold green]✓[/bold green]" if found else "[bold red]✗[/bold red]"
    console.print(f"  {status} '{prompt_text[:50]}' → top-5: {top10[:5]}")
    if not found:
        validation_ok = False

layer_acts.clear()

if not validation_ok:
    for h in hooks:
        h.remove()
    raise RuntimeError("MPS validation FAILED — do not trust activations. Try DEVICE='cpu'.")

console.print("[bold green]✓ MPS validation passed[/bold green]")

# ============================================================================
# EXTRACT ACTIVATIONS — suffixed texts, hook at final ":" token
# ============================================================================

console.print(f"\n[bold cyan]Extracting Set C residuals for {len(set_a_sorted)} stimuli...[/bold cyan]")
console.print(f"Target shape: ({len(set_a_sorted)}, {n_layers}, {d_model})")

all_activations = np.zeros((len(set_a_sorted), n_layers, d_model), dtype=np.float32)
metadata = []

t_start = time.time()

for stim_idx, stimulus in enumerate(track(set_a_sorted, description="Extracting...")):
    suffixed_text = stimulus.text + SUFFIX

    inputs_orig = tokenizer(stimulus.text, return_tensors="pt")
    inputs_suf  = tokenizer(suffixed_text, return_tensors="pt").to(DEVICE)
    n_orig = inputs_orig["input_ids"].shape[1]
    n_suf  = inputs_suf["input_ids"].shape[1]
    final_tok = tokenizer.decode([inputs_suf["input_ids"][0, -1].item()])

    with torch.no_grad():
        _ = model(**inputs_suf, use_cache=False)

    for layer_idx in range(n_layers):
        all_activations[stim_idx, layer_idx, :] = layer_acts[layer_idx]

    metadata.append({
        "index":             stim_idx,
        "id":                stimulus.id,
        "emotion":           stimulus.emotion,
        "stimulus_set":      stimulus.stimulus_set,
        "suffix":            SUFFIX,
        "n_tokens_original": n_orig,
        "n_tokens_suffixed": n_suf,
        "final_token_str":   final_tok,
        "intensity":         stimulus.intensity,
        "word_count":        stimulus.word_count,
    })

    layer_acts.clear()
    torch.mps.empty_cache()
    gc.collect()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(set_a_sorted):.2f}s/stimulus)")
console.print(f"Shape: {all_activations.shape}")

for h in hooks:
    h.remove()

# ============================================================================
# SANITY CHECKS
# ============================================================================

assert np.all(np.isfinite(all_activations)), "Non-finite values in activations!"
console.print("✓ All activations finite")

inter_std = all_activations[:, n_layers // 2, :].std(axis=0).mean()
console.print(f"✓ Inter-stimulus std at mid-layer: {inter_std:.4f}")
assert inter_std > 0.001

for li in [0, n_layers // 4, n_layers // 2, n_layers - 1]:
    la = all_activations[:, li, :]
    console.print(f"  Layer {li:2d}: mean={la.mean():.3f} std={la.std():.3f} range=[{la.min():.3f}, {la.max():.3f}]")

colon_count = sum(1 for m in metadata if ":" in m["final_token_str"])
console.print(f"✓ Stimuli ending at ':': {colon_count}/{len(metadata)}")

# ============================================================================
# BEHAVIORAL CHECK — top-10 emotion predictions at ":" for one per emotion
# ============================================================================

console.print("\n[bold cyan]Behavioral check: top-10 emotion predictions at ':' position...[/bold cyan]")

# Pick one sample per emotion
samples = {}
for s in set_a_sorted:
    if s.emotion not in samples:
        samples[s.emotion] = s

behavioral_lines = ["Set C Behavioral Check — Meta-Llama-3-8B", "=" * 50, "",
                    f"Suffix: {repr(SUFFIX)}", "Pass criterion: ≥5/8 emotions have plausible emotion token in top-10", ""]

n_pass = 0
for emotion, stimulus in sorted(samples.items()):
    inputs_suf = tokenizer(stimulus.text + SUFFIX, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs_suf, use_cache=False)
    top10_ids = out.logits[0, -1, :].topk(10).indices.tolist()
    top10_toks = [tokenizer.decode([tid]) for tid in top10_ids]
    expected = EMOTION_VOCAB.get(emotion, [])
    found = any(tok in top10_toks for tok in expected)
    if found:
        n_pass += 1
    status = "PASS" if found else "FAIL"
    console.print(f"  [{status}] {emotion:12s} | top-5: {top10_toks[:5]}")
    behavioral_lines += [f"[{status}] {emotion}", f"  Top-10: {top10_toks}",
                         f"  Expected any of: {expected[:4]}", ""]
    torch.mps.empty_cache()

gate = "PASS" if n_pass >= 5 else "FAIL"
behavioral_lines.append(f"Result: {n_pass}/8 passed → Behavioral gate: {gate}")
console.print(f"\nBehavioral gate: {n_pass}/8 passed → [bold {'green' if gate == 'PASS' else 'red'}]{gate}[/bold {'green' if gate == 'PASS' else 'red'}]")

(OUT_DIR / "behavioral_check_gemma9b.txt").write_text("\n".join(behavioral_lines))
console.print("✓ behavioral_check_gemma9b.txt")

# ============================================================================
# SAVE
# ============================================================================

np.save(ACT_DIR / "gemma9b_set_c_residuals.npy", all_activations)
console.print(f"\n[bold green]✓ Saved: gemma9b_set_c_residuals.npy ({all_activations.shape})[/bold green]")

with open(ACT_DIR / "gemma9b_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

import polars as pl
pl.DataFrame(metadata).write_csv(ACT_DIR / "gemma9b_metadata.csv")
console.print(f"[bold green]✓ Saved: metadata + csv[/bold green]")
console.print(f"\n[bold green]✓ Ready for run.py[/bold green]")
