"""
Experiment 11 — v2 Set A Extraction: Meta-Llama-3-8B (base)
Question: How does the 8B base model encode emotions with the Tak prompt?
Date: 2026-02-25

Framework: HF transformers + register_forward_hook on MPS
Run: uv run python experiments/11_v2_extract_seta/extract_llama8b_base.py
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
torch.manual_seed(42); np.random.seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
ACT_DIR = EXP_DIR / "outputs" / "activations" / "llama8b_base"
OUT_DIR = EXP_DIR / "outputs"
ACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_KEY = "llama8b_base"
MODEL_ID  = "meta-llama/Meta-Llama-3-8B"
DEVICE    = os.getenv("EMO_DEVICE", "cuda" if torch.cuda.is_available() else "mps")
DTYPE     = torch.float16

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
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip(); break
console.print("[bold cyan]HF token loaded[/bold cyan]")

from stimuli.loader import load_set_a
set_a_emotional = sorted([s for s in load_set_a() if s.emotion != "neutral"], key=lambda s: s.id)
console.print(f"Stimuli: {len(set_a_emotional)} emotional Set A")
assert len(set_a_emotional) == 80

console.print(f"\n[bold cyan]Loading {MODEL_ID} on MPS (float16)...[/bold cyan]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, attn_implementation="eager").to(DEVICE)
model.eval()
n_layers = model.config.num_hidden_layers   # 32
d_model  = model.config.hidden_size         # 4096
n_heads  = model.config.num_attention_heads  # 32
console.print(f"[bold green]✓ {n_layers} layers, d_model={d_model}[/bold green]")

# Final token check
test_ids = tokenizer(build_tak_prompt("Test."), return_tensors="pt")["input_ids"]
final_tok = tokenizer.decode([test_ids[0, -1].item()])
console.print(f"Final token: {repr(final_tok)} — {'✓' if ':' in final_tok else '✗'}")
if ":" not in final_tok:
    raise ValueError(f"Final token mismatch: {repr(final_tok)}")

# ============================================================================
# MPS VALIDATION
# ============================================================================

console.print("\n[bold yellow]Device validation...[/bold yellow]")
validation_prompts = [
    ("The Eiffel Tower is located in the city of", ["Paris", " Paris"]),
    ("The boiling point of water is 100 degrees", ["Celsius", " Celsius", "C", " C"]),
]
for prompt_text, acceptable in validation_prompts:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    top5 = [tokenizer.decode([tid]) for tid in out.logits[0, -1, :].topk(5).indices.tolist()]
    found = any(tok in top5 for tok in acceptable)
    status = "✓" if found else "✗"
    console.print(f"  {status} '{prompt_text[:50]}' → top-3: {top5[:3]}")
    if not found:
        raise RuntimeError(f"device validation FAILED — do not trust activations")
    if DEVICE == "mps": torch.mps.empty_cache()
    elif DEVICE == "cuda": torch.cuda.empty_cache()
console.print("[bold green]✓ device validation passed[/bold green]")

# ============================================================================
# HOOK SETUP — h, a, m (register once, extract per-stimulus)
# ============================================================================

layer_h, layer_a, layer_m = {}, {}, {}

def make_h_hook(l):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_h[l] = h[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

def make_a_hook(l):
    def hook(module, input, output):
        a = output[0] if isinstance(output, tuple) else output
        layer_a[l] = a[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

def make_m_hook(l):
    def hook(module, input, output):
        m = output[0] if isinstance(output, tuple) else output
        layer_m[l] = m[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

handles = []
for l in range(n_layers):
    handles.append(model.model.layers[l].register_forward_hook(make_h_hook(l)))
    handles.append(model.model.layers[l].self_attn.register_forward_hook(make_a_hook(l)))
    handles.append(model.model.layers[l].mlp.register_forward_hook(make_m_hook(l)))
console.print(f"✓ Registered {len(handles)} hooks")

# ============================================================================
# EXTRACTION
# ============================================================================

console.print(f"\n[bold cyan]Extracting for {len(set_a_emotional)} stimuli...[/bold cyan]")

manifest_rows = []
t_start = time.time()

for stimulus in track(set_a_emotional, description="Extracting..."):
    prompt = build_tak_prompt(stimulus.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]
    colon_pos = n_tokens - 1
    final_tok = tokenizer.decode([inputs["input_ids"][0, -1].item()])

    layer_h.clear(); layer_a.clear(); layer_m.clear()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False, output_attentions=True)

    h_all = np.stack([layer_h[l] for l in range(n_layers)])
    a_all = np.stack([layer_a[l] for l in range(n_layers)])
    m_all = np.stack([layer_m[l] for l in range(n_layers)])

    # Attention patterns from outputs.attentions (tuple of n_layers × (batch, heads, seq, seq))
    # Extract row at colon_pos: (n_heads, seq_len)
    attn_all = np.stack([
        outputs.attentions[l][0, :, colon_pos, :].cpu().to(torch.float32).numpy()
        for l in range(n_layers)
    ])

    top5_toks = [tokenizer.decode([tid]).strip() for tid in outputs.logits[0, -1, :].topk(5).indices.tolist()]
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
    del outputs
    if DEVICE == "mps": torch.mps.empty_cache()
    elif DEVICE == "cuda": torch.cuda.empty_cache()
    gc.collect()

for h in handles:
    h.remove()
elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Done in {elapsed:.1f}s ({elapsed/len(set_a_emotional):.2f}s/stim)[/bold green]")

# Sanity
mid_l = n_layers // 2
all_h = np.stack([np.load(ACT_DIR / f"{s.id}.npz", allow_pickle=True)["h"][mid_l] for s in set_a_emotional])
std = all_h.std(axis=0).mean()
console.print(f"✓ Inter-stimulus std mid-layer: {std:.4f}")
assert std > 0.001

import polars as pl
pl.DataFrame([{k: v if not isinstance(v, list) else str(v) for k, v in r.items()} for r in manifest_rows]).write_csv(ACT_DIR / "manifest.csv")
console.print(f"[bold green]✓ manifest.csv saved — {MODEL_KEY} Set A complete![/bold green]")
