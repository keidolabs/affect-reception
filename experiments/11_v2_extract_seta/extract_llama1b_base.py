"""
Experiment 11 — v2 Set A Extraction: Llama-3.2-1B (base)
Question: How does base model compare to instruct for emotion encoding with Tak prompt?
Date: 2026-02-25

Base counterpart to extract_llama1b_inst.py. Same stimuli, same Tak prompt,
same extraction point. Comparison reveals RLHF impact on emotion circuits.

Framework: TransformerLens on CPU (NEVER MPS — GitHub issue #1178)
Run: uv run python experiments/11_v2_extract_seta/extract_llama1b_base.py
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
torch.manual_seed(42); np.random.seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
ACT_DIR = EXP_DIR / "outputs" / "activations" / "llama1b_base"
OUT_DIR = EXP_DIR / "outputs"
ACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_KEY = "llama1b_base"
MODEL_ID  = "meta-llama/Llama-3.2-1B"
DEVICE    = "cpu"

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
    "rage":       [" anger", " rage", " fury", " frustration", "anger", "rage", "Anger"],
    "grief":      [" sadness", " grief", " sorrow", " loss", "sadness", "grief", "Sadness"],
    "terror":     [" fear", " terror", " panic", " anxiety", "fear", "terror", "Fear"],
    "ecstasy":    [" joy", " happiness", " euphoria", "joy", "happiness", "Joy"],
    "loathing":   [" disgust", " loathing", " contempt", "disgust", "loathing", "Disgust"],
    "amazement":  [" surprise", " wonder", " amazement", "surprise", "amazement", "Surprise"],
    "admiration": [" admiration", " respect", " awe", "admiration", "Admiration"],
    "vigilance":  [" anxiety", " vigilance", " alertness", "anxiety", "vigilance"],
}

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
console.print("[bold cyan]HF token loaded[/bold cyan]")

from stimuli.loader import load_set_a
set_a_emotional = sorted([s for s in load_set_a() if s.emotion != "neutral"], key=lambda s: s.id)
console.print(f"Stimuli: {len(set_a_emotional)} emotional Set A")
assert len(set_a_emotional) == 80

console.print(f"\n[bold cyan]Loading {MODEL_ID} (TransformerLens, CPU)...[/bold cyan]")
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.float32)
model.eval()
n_layers = model.cfg.n_layers; d_model = model.cfg.d_model; n_heads = model.cfg.n_heads
console.print(f"[bold green]✓ {n_layers} layers, d_model={d_model}[/bold green]")

def _try_cache(cache, key):
    try: _ = cache[key, 0]; return True
    except: return False

# Discover cache keys
test_tokens = model.to_tokens(build_tak_prompt("Test."))
_, tc = model.run_with_cache(test_tokens, names_filter=lambda n: any(k in n for k in ["resid_post", "attn_out", "mlp_out", "pattern"]))
ATTN_OUT_KEY = None
for k in ["attn_out", "hook_attn_out"]:
    try: _ = tc[k, 0]; ATTN_OUT_KEY = k; break
    except: pass
MLP_OUT_KEY = None
for k in ["mlp_out", "hook_mlp_out"]:
    try: _ = tc[k, 0]; MLP_OUT_KEY = k; break
    except: pass
PATTERN_KEY = None
for k in ["pattern", "attn.hook_pattern"]:
    try: _ = tc[k, 0]; PATTERN_KEY = k; break
    except: pass
del tc; gc.collect()
console.print(f"Cache keys: attn_out={ATTN_OUT_KEY}, mlp_out={MLP_OUT_KEY}, pattern={PATTERN_KEY}")

# Verify final token
final_tok = model.tokenizer.decode([model.to_tokens(build_tak_prompt("Test."))[ 0, -1].item()])
console.print(f"Final token: {repr(final_tok)} — {'✓' if ':' in final_tok else '✗'}")
if ":" not in final_tok:
    raise ValueError(f"Final token mismatch: {repr(final_tok)}")

def names_filter(name):
    return any(k in name for k in ["hook_resid_post", "hook_attn_out", "hook_mlp_out", "hook_pattern"])

manifest_rows = []
t_start = time.time()

for stimulus in track(set_a_emotional, description="Extracting..."):
    prompt = build_tak_prompt(stimulus.text)
    tokens = model.to_tokens(prompt)
    n_tokens = tokens.shape[1]; colon_pos = n_tokens - 1
    final_tok = model.tokenizer.decode([tokens[0, -1].item()])

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

    h_all = np.stack([cache["resid_post", l][:, colon_pos, :].squeeze().cpu().numpy() for l in range(n_layers)])
    a_all = (np.stack([cache[ATTN_OUT_KEY, l][:, colon_pos, :].squeeze().cpu().numpy() for l in range(n_layers)])
             if ATTN_OUT_KEY else np.zeros((n_layers, d_model), dtype=np.float32))
    m_all = (np.stack([cache[MLP_OUT_KEY, l][:, colon_pos, :].squeeze().cpu().numpy() for l in range(n_layers)])
             if MLP_OUT_KEY else np.zeros((n_layers, d_model), dtype=np.float32))
    attn_all = (np.stack([cache[PATTERN_KEY, l][:, :, colon_pos, :].squeeze(0).cpu().numpy() for l in range(n_layers)])
                if PATTERN_KEY else np.zeros((n_layers, n_heads, n_tokens), dtype=np.float32))

    top5_toks = [model.tokenizer.decode([tid]).strip() for tid in logits[0, -1, :].topk(5).indices.tolist()]
    expected  = EMOTION_VOCAB.get(stimulus.emotion, [])
    correct   = any(tok in top5_toks for tok in [t.strip() for t in expected])

    meta = {"id": stimulus.id, "emotion": stimulus.emotion, "stimulus_set": stimulus.stimulus_set,
            "domain": stimulus.domain, "n_tokens_prompt": n_tokens, "colon_token_idx": colon_pos,
            "model_id": MODEL_KEY, "top5_predictions": top5_toks, "correct": correct, "final_token_str": final_tok}

    np.savez_compressed(ACT_DIR / f"{stimulus.id}.npz",
                        h=h_all.astype(np.float32), a=a_all.astype(np.float32),
                        m=m_all.astype(np.float32), attn=attn_all.astype(np.float32),
                        metadata=json.dumps(meta))
    manifest_rows.append(meta)
    del logits, cache; gc.collect()

elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green]")

# Sanity: inter-stimulus std
mid_l = n_layers // 2
all_h = np.stack([np.load(ACT_DIR / f"{s.id}.npz", allow_pickle=True)["h"][mid_l] for s in set_a_emotional])
std = all_h.std(axis=0).mean()
console.print(f"✓ Inter-stimulus std mid-layer: {std:.4f}")
assert std > 0.001

import polars as pl
pl.DataFrame([{k: v if not isinstance(v, list) else str(v) for k, v in r.items()} for r in manifest_rows]).write_csv(ACT_DIR / "manifest.csv")
console.print(f"[bold green]✓ manifest.csv saved — {MODEL_KEY} Set A complete![/bold green]")
