"""
Experiment 12 — v2 Set B Extraction: Llama-3.2-1B-Instruct
Question: Do keyword-free clinical vignettes activate the same circuits as Set A?
Date: 2026-02-25

Extracts h, a, m, attn at ":" token for all 192 Set B stimuli
(96 clinical + 96 neutral controls) using the Tak et al. few-shot prompt.

Framework: TransformerLens on CPU (NEVER MPS — GitHub issue #1178)
Output: outputs/activations/llama1b_inst/{stimulus_id}.npz (192 files)

Run: uv run python experiments/12_v2_extract_setb/extract_llama1b_inst.py
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
DEVICE    = "cpu"

# ============================================================================
# TAK ET AL. FEW-SHOT PROMPT (same as Exp 11 — fixed across all v2 experiments)
# ============================================================================

SHOT_1_TEXT    = "My dog died last week. I miss him every day."
SHOT_1_EMOTION = "sadness"
SHOT_2_TEXT    = "I got promoted and my boss praised my work in front of everyone."
SHOT_2_EMOTION = "joy"

def build_tak_prompt(target_text: str) -> str:
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\n"
        f"Answer: {SHOT_1_EMOTION}\n\n"
        f"Context: {SHOT_2_TEXT}\n"
        f"Answer: {SHOT_2_EMOTION}\n\n"
        f"Context: {target_text}\n"
        "Answer:"
    )

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
# LOAD STIMULI — 96 clinical + 96 neutral controls = 192 total
# ============================================================================

from stimuli.loader import load_set_b_clinical, load_set_b_neutral

console.print("\n[bold cyan]Loading Set B stimuli (clinical + neutral)...[/bold cyan]")
clinical = load_set_b_clinical()
neutral  = load_set_b_neutral()
all_stimuli = sorted(clinical + neutral, key=lambda s: s.id)

console.print(f"Clinical: {len(clinical)} stimuli")
console.print(f"Neutral: {len(neutral)} stimuli")
console.print(f"Total: {len(all_stimuli)} stimuli")
console.print(f"Emotion distribution (clinical): {dict(Counter(s.emotion for s in clinical))}")

assert len(clinical) == 96, f"Expected 96 clinical, got {len(clinical)}"
assert len(neutral) == 96,  f"Expected 96 neutral, got {len(neutral)}"

# ============================================================================
# LOAD MODEL — TransformerLens on CPU
# ============================================================================

console.print(f"\n[bold cyan]Loading {MODEL_ID} on CPU (float32, TransformerLens)...[/bold cyan]")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.float32)
model.eval()

n_layers = model.cfg.n_layers   # 16
d_model  = model.cfg.d_model    # 2048
n_heads  = model.cfg.n_heads    # 32
console.print(f"[bold green]✓ Model loaded[/bold green] — {n_layers} layers, d_model={d_model}")

# ============================================================================
# DISCOVER CACHE KEYS
# ============================================================================

console.print(f"\n[bold cyan]Discovering cache keys...[/bold cyan]")
test_tokens = model.to_tokens(build_tak_prompt("Test text here."))
_, test_cache = model.run_with_cache(
    test_tokens,
    names_filter=lambda n: any(k in n for k in ["resid_post", "attn_out", "mlp_out", "pattern"]),
)

ATTN_OUT_KEY, MLP_OUT_KEY, PATTERN_KEY = None, None, None
for candidate in ["attn_out", "hook_attn_out"]:
    try:
        _ = test_cache[candidate, 0]; ATTN_OUT_KEY = candidate; break
    except: pass
for candidate in ["mlp_out", "hook_mlp_out"]:
    try:
        _ = test_cache[candidate, 0]; MLP_OUT_KEY = candidate; break
    except: pass
for candidate in ["pattern", "attn.hook_pattern"]:
    try:
        _ = test_cache[candidate, 0]; PATTERN_KEY = candidate; break
    except: pass

console.print(f"Cache keys: resid_post=✓, attn_out={ATTN_OUT_KEY}, mlp_out={MLP_OUT_KEY}, pattern={PATTERN_KEY}")
del test_cache; gc.collect()

# ============================================================================
# PROMPT SANITY CHECK
# ============================================================================

test_prompt  = build_tak_prompt("The patient described persistent chest pain and shortness of breath.")
test_tokens  = model.to_tokens(test_prompt)
final_tok_str = model.tokenizer.decode([test_tokens[0, -1].item()])
console.print(f"\nFinal token check: {repr(final_tok_str)} — {'✓ OK' if ':' in final_tok_str else '✗ FAIL'}")
if ":" not in final_tok_str:
    raise ValueError(f"Final token mismatch: {repr(final_tok_str)}")

# ============================================================================
# EXTRACTION — per-stimulus .npz
# ============================================================================

console.print(f"\n[bold cyan]Extracting for {len(all_stimuli)} Set B stimuli...[/bold cyan]")

def names_filter(name):
    return any(k in name for k in ["hook_resid_post", "hook_attn_out", "hook_mlp_out", "hook_pattern"])

manifest_rows = []
t_start = time.time()

for stimulus in track(all_stimuli, description="Extracting Set B..."):
    prompt   = build_tak_prompt(stimulus.text)
    tokens   = model.to_tokens(prompt)
    n_tokens = tokens.shape[1]
    colon_pos = n_tokens - 1
    final_tok  = model.tokenizer.decode([tokens[0, -1].item()])

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

    h_all = np.stack([cache["resid_post", l][:, colon_pos, :].squeeze().cpu().numpy()
                      for l in range(n_layers)])

    a_all = (np.stack([cache[ATTN_OUT_KEY, l][:, colon_pos, :].squeeze().cpu().numpy()
                       for l in range(n_layers)])
             if ATTN_OUT_KEY else np.zeros((n_layers, d_model), dtype=np.float32))

    m_all = (np.stack([cache[MLP_OUT_KEY, l][:, colon_pos, :].squeeze().cpu().numpy()
                       for l in range(n_layers)])
             if MLP_OUT_KEY else np.zeros((n_layers, d_model), dtype=np.float32))

    attn_all = (np.stack([cache[PATTERN_KEY, l][:, :, colon_pos, :].squeeze(0).cpu().numpy()
                           for l in range(n_layers)])
                if PATTERN_KEY else np.zeros((n_layers, n_heads, n_tokens), dtype=np.float32))

    top5_ids  = logits[0, -1, :].topk(5).indices.tolist()
    top5_toks = [model.tokenizer.decode([tid]).strip() for tid in top5_ids]
    expected  = EMOTION_VOCAB.get(stimulus.emotion, [])
    correct   = any(tok in top5_toks for tok in [t.strip() for t in expected])

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
        "final_token_str": final_tok,
        "matched_control_id": stimulus.matched_control_id,
    }

    np.savez_compressed(
        ACT_DIR / f"{stimulus.id}.npz",
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
console.print(f"\n[bold green]✓ Extraction complete[/bold green] — {elapsed:.1f}s ({elapsed/len(all_stimuli):.2f}s/stim)")

# ============================================================================
# SANITY CHECKS
# ============================================================================

console.print("\n[bold cyan]Sanity checks...[/bold cyan]")

n_missing = sum(1 for s in all_stimuli if not (ACT_DIR / f"{s.id}.npz").exists())
console.print(f"{'✓' if n_missing == 0 else '✗'} Files: {len(all_stimuli) - n_missing}/{len(all_stimuli)}")

mid_layer = n_layers // 2
all_h_mid = np.stack([
    np.load(ACT_DIR / f"{s.id}.npz", allow_pickle=True)["h"][mid_layer]
    for s in all_stimuli[:20]  # spot-check 20 to save time
])
inter_std = all_h_mid.std(axis=0).mean()
console.print(f"✓ Inter-stimulus std (spot-check 20): {inter_std:.4f}")
assert inter_std > 0.001, f"Activations look identical! std={inter_std:.6f}"

colon_count = sum(1 for r in manifest_rows if ":" in r["final_token_str"])
console.print(f"✓ Stimuli ending at ':': {colon_count}/{len(manifest_rows)}")

# ============================================================================
# BEHAVIORAL CHECK — clinical stimuli
# ============================================================================

console.print("\n[bold cyan]Behavioral check (clinical stimuli)...[/bold cyan]")

clinical_samples = {}
for s in clinical:
    if s.emotion not in clinical_samples:
        clinical_samples[s.emotion] = s

behavioral_lines = [
    f"v2 Set B Clinical Behavioral Check — {MODEL_ID}", "=" * 60, "",
    "Note: For Set B, model may predict diverse emotion-adjacent terms.",
    "Low accuracy is expected — Set B tests generalization, not in-distribution perf.",
    "",
]

n_pass = 0
for emotion, stimulus in sorted(clinical_samples.items()):
    prompt = build_tak_prompt(stimulus.text)
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits, _ = model.run_with_cache(tokens, names_filter=lambda n: False)
    top5_toks = [model.tokenizer.decode([tid]) for tid in logits[0, -1, :].topk(5).indices.tolist()]
    expected  = EMOTION_VOCAB.get(emotion, [])
    found     = any(tok in top5_toks for tok in expected)
    if found: n_pass += 1
    status = "PASS" if found else "FAIL"
    console.print(f"  [{status}] {emotion:12s} | top-5: {top5_toks}")
    behavioral_lines += [f"[{status}] {emotion}", f"  Top-5: {top5_toks}", ""]

behavioral_lines.append(f"Result: {n_pass}/8 clinical emotions triggered correct top-5")
(OUT_DIR / f"behavioral_check_setb_{MODEL_KEY}.txt").write_text("\n".join(behavioral_lines))
console.print(f"  {n_pass}/8 correct (lower expected for keyword-free Set B)")

# ============================================================================
# SAVE MANIFEST
# ============================================================================

import polars as pl

manifest_df = pl.DataFrame([
    {k: v if not isinstance(v, list) else str(v) for k, v in row.items()}
    for row in manifest_rows
])
manifest_df.write_csv(ACT_DIR / "manifest.csv")

n_correct_clinical = sum(1 for r in manifest_rows if r["correct"] and r["stimulus_set"] == "B_clinical")
n_correct_neutral  = sum(1 for r in manifest_rows if r["correct"] and r["stimulus_set"] == "neutral")
console.print(f"\n[bold green]✓ manifest.csv saved ({len(manifest_rows)} rows)[/bold green]")
console.print(f"  Clinical correct: {n_correct_clinical}/96 ({100*n_correct_clinical/96:.1f}%)")
console.print(f"  Neutral correct:  {n_correct_neutral}/96 — (neutral has no gold label)")

console.print(f"\n[bold green]✓ Experiment 12 Set B — {MODEL_KEY} complete![/bold green]")
console.print(f"  Output: {ACT_DIR}/")
console.print(f"\n[bold cyan]Next: uv run python experiments/13_v2_probes/run.py[/bold cyan]")
