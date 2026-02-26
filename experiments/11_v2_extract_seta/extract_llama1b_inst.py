"""
Experiment 11 — v2 Set A Extraction: Llama-3.2-1B-Instruct (PRIMARY MODEL)
Question: Do mid-layer circuits encode emotion with the Tak et al. few-shot prompt?
Date: 2026-02-25

PRIMARY model for the full v2 mechanistic study. Extracts h (residual stream),
a (MHSA output), m (FFN output), and attn (attention patterns from ":") for
all 80 Set A emotional stimuli using the Tak et al. few-shot classification format.

Framework: TransformerLens on CPU (NEVER MPS — GitHub issue #1178)
Output: outputs/activations/llama1b_inst/{stimulus_id}.npz (80 files)

GATE CHECK: After running, use experiments/13_v2_probes/run_gate_check.py
to verify h probe peak is at normalized depth 0.50–0.75 (Tak replication).

Run: uv run python experiments/11_v2_extract_seta/extract_llama1b_inst.py
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
ROOT    = EXP_DIR.parent.parent
ACT_DIR = EXP_DIR / "outputs" / "activations" / "llama1b_inst"
OUT_DIR = EXP_DIR / "outputs"
ACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_KEY = "llama1b_inst"
MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE    = "cpu"   # CRITICAL: Never use MPS with TransformerLens (TL issue #1178)

# ============================================================================
# TAK ET AL. FEW-SHOT PROMPT FORMAT
# Fixed across all v2 experiments. Only TARGET_TEXT varies.
# ============================================================================

SHOT_1_TEXT    = "My dog died last week. I miss him every day."
SHOT_1_EMOTION = "sadness"
SHOT_2_TEXT    = "I got promoted and my boss praised my work in front of everyone."
SHOT_2_EMOTION = "joy"

def build_tak_prompt(target_text: str) -> str:
    """Build Tak et al. few-shot classification prompt."""
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\n"
        f"Answer: {SHOT_1_EMOTION}\n\n"
        f"Context: {SHOT_2_TEXT}\n"
        f"Answer: {SHOT_2_EMOTION}\n\n"
        f"Context: {target_text}\n"
        "Answer:"
    )

# Plausible emotion tokens for behavioral check (broad set covering multiple surface forms)
EMOTION_VOCAB = {
    "rage":       [" anger", " rage", " fury", " frustration", " resentment", " wrath",
                   "anger", "rage", "Anger", "Rage"],
    "grief":      [" sadness", " grief", " sorrow", " loss", " depression",
                   "sadness", "grief", "Sadness", "Grief"],
    "terror":     [" fear", " terror", " panic", " anxiety", " dread", " horror",
                   "fear", "terror", "Fear", "Terror"],
    "ecstasy":    [" joy", " happiness", " euphoria", " ecstasy", " elation",
                   "joy", "happiness", "Joy", "Happiness"],
    "loathing":   [" disgust", " loathing", " contempt", " hatred", " repulsion",
                   "disgust", "loathing", "Disgust"],
    "amazement":  [" surprise", " wonder", " amazement", " shock", " awe",
                   "surprise", "amazement", "Surprise"],
    "admiration": [" admiration", " respect", " awe", " pride", " love",
                   "admiration", "Admiration"],
    "vigilance":  [" anxiety", " vigilance", " alertness", " anticipation", " worry",
                   "anxiety", "vigilance", "Anticipation"],
}

# ============================================================================
# LOAD HF TOKEN
# ============================================================================

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
console.print("[bold cyan]HF token loaded[/bold cyan]")

# ============================================================================
# LOAD STIMULI — 80 emotional only (no neutrals — 8-class probe doesn't use them)
# ============================================================================

from stimuli.loader import load_set_a

console.print("\n[bold cyan]Loading Set A stimuli (emotional only)...[/bold cyan]")
set_a_all = load_set_a()
set_a_emotional = [s for s in set_a_all if s.emotion != "neutral"]
set_a_sorted = sorted(set_a_emotional, key=lambda s: s.id)

console.print(f"Emotional stimuli: {len(set_a_sorted)}")
console.print(f"Emotion distribution: {dict(Counter(s.emotion for s in set_a_sorted))}")
assert len(set_a_sorted) == 80, f"Expected 80 emotional stimuli, got {len(set_a_sorted)}"

# ============================================================================
# LOAD MODEL — TransformerLens on CPU (float32)
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on CPU (float32, TransformerLens)...[/bold cyan]")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.float32)
model.eval()

n_layers = model.cfg.n_layers   # 16
d_model  = model.cfg.d_model    # 2048
n_heads  = model.cfg.n_heads    # 32
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}, n_heads={n_heads}")

# ============================================================================
# DISCOVER CACHE KEYS — verify h, a, m, attn keys for this TL version
# ============================================================================

console.print(f"\n[bold cyan]Discovering TransformerLens cache keys...[/bold cyan]")
test_prompt = build_tak_prompt("This is a test.")
test_tokens = model.to_tokens(test_prompt)
_, test_cache = model.run_with_cache(
    test_tokens,
    names_filter=lambda name: any(k in name for k in ["resid_post", "attn_out", "mlp_out", "pattern"]),
)

# Show available keys for layer 0 to confirm naming
layer0_keys = [k for k in test_cache.keys() if "0" in k or ".0." in k][:10]
console.print(f"Sample cache keys (layer 0 context): {layer0_keys[:5]}")

# Try to find the correct key for attn_out (varies by TL version)
attn_keys = [k for k in test_cache.keys() if "attn" in k.lower()][:8]
mlp_keys  = [k for k in test_cache.keys() if "mlp" in k.lower()][:5]
console.print(f"Attention-related keys: {attn_keys}")
console.print(f"MLP-related keys: {mlp_keys}")

# Try short-form access for layer 0
try:
    _ = test_cache["resid_post", 0]
    console.print("[bold green]✓ cache['resid_post', l] works[/bold green]")
except:
    console.print("[bold red]✗ cache['resid_post', l] failed — check TL version[/bold red]")
    raise

# Check attn_out key
ATTN_OUT_KEY = None
for candidate in ["attn_out", "hook_attn_out"]:
    try:
        _ = test_cache[candidate, 0]
        ATTN_OUT_KEY = candidate
        console.print(f"[bold green]✓ cache['{candidate}', l] works — using for MHSA output[/bold green]")
        break
    except:
        pass
if ATTN_OUT_KEY is None:
    # Try full key name fallback
    attn_out_full_keys = [k for k in test_cache.keys() if "attn_out" in k and "0" in k]
    if attn_out_full_keys:
        console.print(f"[yellow]Short-form attn_out failed. Full keys: {attn_out_full_keys}[/yellow]")
    console.print("[yellow]WARNING: Cannot find attn_out key — will save zeros for 'a' component[/yellow]")

# Check mlp_out key
MLP_OUT_KEY = None
for candidate in ["mlp_out", "hook_mlp_out"]:
    try:
        _ = test_cache[candidate, 0]
        MLP_OUT_KEY = candidate
        console.print(f"[bold green]✓ cache['{candidate}', l] works — using for FFN output[/bold green]")
        break
    except:
        pass
if MLP_OUT_KEY is None:
    console.print("[yellow]WARNING: Cannot find mlp_out key — will save zeros for 'm' component[/yellow]")

# Check pattern key
PATTERN_KEY = None
for candidate in ["pattern", "attn.hook_pattern", "hook_pattern"]:
    try:
        pat = test_cache[candidate, 0]
        PATTERN_KEY = candidate
        console.print(f"[bold green]✓ cache['{candidate}', l] works — shape: {pat.shape}[/bold green]")
        break
    except:
        pass
if PATTERN_KEY is None:
    console.print("[yellow]WARNING: Cannot find attention pattern key — will save zeros for 'attn' component[/yellow]")

del test_cache
gc.collect()

# ============================================================================
# PROMPT SANITY CHECK — verify ":" is the final token
# ============================================================================

console.print(f"\n[bold cyan]Prompt sanity check: verifying final token is ':'...[/bold cyan]")
test_prompt = build_tak_prompt("She was devastated to hear the terrible news.")
test_tokens = model.to_tokens(test_prompt)
final_tok_id = test_tokens[0, -1].item()
final_tok_str = model.tokenizer.decode([final_tok_id])
n_test_tokens = test_tokens.shape[1]

console.print(f"Prompt (last 60 chars): {repr(test_prompt[-60:])}")
console.print(f"Total tokens: {n_test_tokens}")
console.print(f"Final token: {repr(final_tok_str)} (id={final_tok_id})")

if ":" in final_tok_str:
    console.print(f"[bold green]✓ Final token is ':' as expected[/bold green]")
else:
    console.print(f"[bold red]✗ Final token is {repr(final_tok_str)}, not ':' — check prompt format![/bold red]")
    raise ValueError(f"Final token mismatch: expected ':', got {repr(final_tok_str)}")

# ============================================================================
# EXTRACTION — per-stimulus .npz with h, a, m, attn
# ============================================================================

console.print(f"\n[bold cyan]Extracting activations for {len(set_a_sorted)} stimuli...[/bold cyan]")
console.print(f"Components: h (residual), a (MHSA out), m (FFN out), attn (attention from ':')")

# names_filter: capture all 4 component types
def names_filter(name):
    return any(k in name for k in ["hook_resid_post", "hook_attn_out", "hook_mlp_out", "hook_pattern"])

manifest_rows = []
t_start = time.time()
n_missing_colon = 0

for stim_idx, stimulus in enumerate(track(set_a_sorted, description="Extracting...")):
    prompt = build_tak_prompt(stimulus.text)
    tokens = model.to_tokens(prompt)
    n_tokens = tokens.shape[1]
    colon_pos = n_tokens - 1  # final token is ":"

    # Verify this stimulus ends at ":"
    final_tok_str = model.tokenizer.decode([tokens[0, -1].item()])
    if ":" not in final_tok_str:
        n_missing_colon += 1
        console.print(f"[yellow]WARNING stim {stimulus.id}: final token={repr(final_tok_str)}[/yellow]")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

    # Extract h: residual stream at ":" for all layers — shape (n_layers, d_model)
    h_all = np.stack([
        cache["resid_post", l][:, colon_pos, :].squeeze().cpu().numpy()
        for l in range(n_layers)
    ])  # (n_layers, d_model)

    # Extract a: MHSA output at ":" for all layers
    if ATTN_OUT_KEY is not None:
        a_all = np.stack([
            cache[ATTN_OUT_KEY, l][:, colon_pos, :].squeeze().cpu().numpy()
            for l in range(n_layers)
        ])
    else:
        a_all = np.zeros((n_layers, d_model), dtype=np.float32)

    # Extract m: FFN output at ":" for all layers
    if MLP_OUT_KEY is not None:
        m_all = np.stack([
            cache[MLP_OUT_KEY, l][:, colon_pos, :].squeeze().cpu().numpy()
            for l in range(n_layers)
        ])
    else:
        m_all = np.zeros((n_layers, d_model), dtype=np.float32)

    # Extract attn: attention from ":" to all tokens, all heads — shape (n_layers, n_heads, seq_len)
    if PATTERN_KEY is not None:
        attn_all = np.stack([
            cache[PATTERN_KEY, l][:, :, colon_pos, :].squeeze(0).cpu().numpy()
            for l in range(n_layers)
        ])  # (n_layers, n_heads, seq_len)
    else:
        attn_all = np.zeros((n_layers, n_heads, n_tokens), dtype=np.float32)

    # Get top-5 predictions at ":" position
    top5_ids = logits[0, -1, :].topk(5).indices.tolist()
    top5_toks = [model.tokenizer.decode([tid]).strip() for tid in top5_ids]

    # Check if any top-5 token matches expected emotion vocab
    expected = EMOTION_VOCAB.get(stimulus.emotion, [])
    correct = any(tok in top5_toks or f" {tok}" in top5_toks
                  for tok in [t.strip() for t in expected])

    # Metadata for this stimulus
    meta = {
        "id": stimulus.id,
        "emotion": stimulus.emotion,
        "stimulus_set": stimulus.stimulus_set,
        "domain": stimulus.domain,
        "n_tokens_prompt": n_tokens,
        "colon_token_idx": colon_pos,
        "model_id": MODEL_KEY,
        "top5_predictions": top5_toks,
        "correct": correct,
        "final_token_str": final_tok_str,
    }

    # Save per-stimulus .npz
    out_path = ACT_DIR / f"{stimulus.id}.npz"
    np.savez_compressed(
        out_path,
        h=h_all.astype(np.float32),
        a=a_all.astype(np.float32),
        m=m_all.astype(np.float32),
        attn=attn_all.astype(np.float32),
        metadata=json.dumps(meta),
    )

    manifest_rows.append(meta)

    del logits, cache
    gc.collect()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(set_a_sorted):.2f}s/stim)")
console.print(f"✓ Saved {len(set_a_sorted)} .npz files to {ACT_DIR}")

# ============================================================================
# SANITY CHECKS
# ============================================================================

console.print("\n[bold cyan]Running sanity checks...[/bold cyan]")

# 1. All files exist
n_missing = sum(1 for s in set_a_sorted if not (ACT_DIR / f"{s.id}.npz").exists())
console.print(f"{'✓' if n_missing == 0 else '✗'} Files created: {len(set_a_sorted) - n_missing}/{len(set_a_sorted)}")

# 2. Load a few and check finite + shapes
n_finite_ok = 0
for s in set_a_sorted[:5]:
    npz = np.load(ACT_DIR / f"{s.id}.npz", allow_pickle=True)
    if np.all(np.isfinite(npz["h"])) and np.all(np.isfinite(npz["a"])):
        n_finite_ok += 1
console.print(f"✓ Finite check on first 5: {n_finite_ok}/5 pass")

# 3. Check colon token coverage
colon_count = sum(1 for r in manifest_rows if ":" in r["final_token_str"])
console.print(f"✓ Stimuli ending at ':': {colon_count}/{len(manifest_rows)}")
if n_missing_colon > 0:
    console.print(f"[yellow]  WARNING: {n_missing_colon} stimuli did not end at ':' — check prompt[/yellow]")

# 4. Inter-stimulus std check: load all h for mid-layer
mid_layer = n_layers // 2
all_h_mid = np.stack([
    np.load(ACT_DIR / f"{s.id}.npz", allow_pickle=True)["h"][mid_layer]
    for s in set_a_sorted
])
inter_std = all_h_mid.std(axis=0).mean()
console.print(f"✓ Inter-stimulus std at mid-layer ({mid_layer}): {inter_std:.4f}")
assert inter_std > 0.001, f"Activations look identical at mid-layer! std={inter_std:.6f}"

# ============================================================================
# BEHAVIORAL CHECK — top-5 predictions for one stimulus per emotion
# ============================================================================

console.print("\n[bold cyan]Behavioral check: top-5 predictions at ':' position...[/bold cyan]")

samples = {}
for s in set_a_sorted:
    if s.emotion not in samples:
        samples[s.emotion] = s

behavioral_lines = [
    f"v2 Set A Behavioral Check — {MODEL_ID}", "=" * 60, "",
    "Prompt format: Tak et al. few-shot (2-shot classification)",
    "Extraction point: ':' (final token of 'Answer:')",
    "Pass criterion: ≥5/8 emotions have plausible token in top-10", "",
]

n_pass = 0
for emotion, stimulus in sorted(samples.items()):
    prompt = build_tak_prompt(stimulus.text)
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits, _ = model.run_with_cache(tokens, names_filter=lambda n: False)
    top10_ids = logits[0, -1, :].topk(10).indices.tolist()
    top10_toks = [model.tokenizer.decode([tid]) for tid in top10_ids]
    expected = EMOTION_VOCAB.get(emotion, [])
    found = any(tok in top10_toks for tok in expected)
    if found:
        n_pass += 1
    status = "PASS" if found else "FAIL"
    console.print(f"  [{status}] {emotion:12s} | top-5: {top10_toks[:5]}")
    behavioral_lines += [
        f"[{status}] {emotion}",
        f"  Top-10: {top10_toks}",
        f"  Expected any of: {expected[:5]}",
        "",
    ]

gate = "PASS" if n_pass >= 5 else "FAIL"
gate_color = "green" if gate == "PASS" else "red"
behavioral_lines.append(f"Result: {n_pass}/8 passed → Behavioral gate: {gate}")
console.print(f"\nBehavioral gate: {n_pass}/8 passed → [bold {gate_color}]{gate}[/bold {gate_color}]")

(OUT_DIR / f"behavioral_check_{MODEL_KEY}.txt").write_text("\n".join(behavioral_lines))
console.print(f"✓ behavioral_check_{MODEL_KEY}.txt saved")

# ============================================================================
# SAVE MANIFEST
# ============================================================================

import polars as pl

manifest_df = pl.DataFrame([
    {k: v if not isinstance(v, list) else str(v)
     for k, v in row.items()}
    for row in manifest_rows
])
manifest_df.write_csv(ACT_DIR / "manifest.csv")
console.print(f"\n[bold green]✓ manifest.csv saved ({len(manifest_rows)} rows)[/bold green]")

# Accuracy summary
n_correct = sum(1 for r in manifest_rows if r["correct"])
console.print(f"  Correct top-5 predictions: {n_correct}/{len(manifest_rows)} ({100*n_correct/len(manifest_rows):.1f}%)")

by_emotion = {}
for r in manifest_rows:
    by_emotion.setdefault(r["emotion"], []).append(r["correct"])
for emo, vals in sorted(by_emotion.items()):
    pct = 100 * sum(vals) / len(vals)
    console.print(f"    {emo:12s}: {sum(vals)}/{len(vals)} ({pct:.0f}%)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

console.print(f"\n[bold green]✓ Experiment 11 Set A — {MODEL_KEY} complete![/bold green]")
console.print(f"  Output: {ACT_DIR}/")
console.print(f"  Files:  {len(set_a_sorted)} × .npz (h, a, m, attn per stimulus)")
console.print(f"  Shapes: h=(16, 2048), a=(16, 2048), m=(16, 2048), attn=(16, 32, seq_len)")
console.print(f"\n[bold cyan]Next: uv run python experiments/13_v2_probes/run_gate_check.py[/bold cyan]")
