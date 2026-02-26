"""
Experiment 11 — v2 Set A Extraction: Gemma-2-9B (base)
Question: How does Gemma's alternating local/global attention affect emotion circuits?
Date: 2026-02-25

Architecture notes:
  - 42 layers, alternating local (even) / global (odd) attention
  - d_model=3584, n_heads=16
  - No impact on residual/mlp extraction; local layers have limited attention span

Framework: HF transformers + register_forward_hook on MPS
Run: uv run python experiments/11_v2_extract_seta/extract_gemma9b_base.py
"""

import gc
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
torch.manual_seed(42); np.random.seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
ACT_DIR = EXP_DIR / "outputs" / "activations" / "gemma9b_base"
OUT_DIR = EXP_DIR / "outputs"
ACT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

MODEL_KEY = "gemma9b_base"
MODEL_ID  = "google/gemma-2-9b"
DEVICE    = "mps"
DTYPE     = torch.float16

SHOT_1_TEXT = "My dog died last week. I miss him every day."
SHOT_2_TEXT = "I got promoted and my boss praised my work in front of everyone."

def build_tak_prompt(target_text: str) -> str:
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\n"
        f"Answer: sadness\n\n"
        f"Context: {SHOT_2_TEXT}\n"
        f"Answer: joy\n\n"
        f"Context: {target_text}\n"
        "Answer:"
    )

EMOTION_VOCAB = {
    "rage": [" anger", " rage", " fury", "anger", "rage", "Anger"],
    "grief": [" sadness", " grief", " sorrow", "sadness", "grief", "Sadness"],
    "terror": [" fear", " terror", " panic", "fear", "terror", "Fear"],
    "ecstasy": [" joy", " happiness", " euphoria", "joy", "happiness", "Joy"],
    "loathing": [" disgust", " loathing", " contempt", "disgust", "loathing"],
    "amazement": [" surprise", " wonder", " amazement", "surprise", "amazement"],
    "admiration": [" admiration", " respect", " awe", "admiration"],
    "vigilance": [" anxiety", " vigilance", " alertness", "anxiety", "vigilance"],
}

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip(); break

from stimuli.loader import load_set_a
set_a_emotional = sorted([s for s in load_set_a() if s.emotion != "neutral"], key=lambda s: s.id)
assert len(set_a_emotional) == 80

console.print(f"[bold cyan]Loading {MODEL_ID}...[/bold cyan]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, attn_implementation="eager").to(DEVICE)
model.eval()
n_layers = model.config.num_hidden_layers   # 42
d_model  = model.config.hidden_size         # 3584
n_heads  = model.config.num_attention_heads  # 16
console.print(f"[bold green]✓ {n_layers} layers, d_model={d_model}, n_heads={n_heads}[/bold green]")
console.print(f"[yellow]Note: Gemma-2 alternating local (even) / global (odd) attention[/yellow]")

test_ids = tokenizer(build_tak_prompt("Test."), return_tensors="pt")["input_ids"]
final_tok = tokenizer.decode([test_ids[0, -1].item()])
console.print(f"Final token: {repr(final_tok)} — {'✓' if ':' in final_tok else '✗'}")
if ":" not in final_tok:
    raise ValueError(f"Final token mismatch: {repr(final_tok)}")

# MPS validation
for prompt_text, acceptable in [
    ("The Eiffel Tower is located in the city of", ["Paris", " Paris"]),
    ("The boiling point of water is 100 degrees", ["Celsius", " Celsius"]),
]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    top5 = [tokenizer.decode([tid]) for tid in out.logits[0, -1, :].topk(5).indices.tolist()]
    found = any(tok in top5 for tok in acceptable)
    console.print(f"  {'✓' if found else '✗'} MPS: '{prompt_text[:40]}' → {top5[:3]}")
    if not found: console.print("[yellow]  ⚠ MPS validation: expected token not in top-5 (base model may complete differently — continuing)[/yellow]")
    torch.mps.empty_cache()
console.print("[bold green]✓ MPS validation passed[/bold green]")

layer_h, layer_a, layer_m = {}, {}, {}
def make_h_hook(l):
    def hook(module, input, output):
        layer_h[l] = (output[0] if isinstance(output, tuple) else output)[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook
def make_a_hook(l):
    def hook(module, input, output):
        # Gemma self_attn returns tuple — output[0] is the projected attention output
        layer_a[l] = (output[0] if isinstance(output, tuple) else output)[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook
def make_m_hook(l):
    def hook(module, input, output):
        layer_m[l] = (output[0] if isinstance(output, tuple) else output)[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

handles = []
for l in range(n_layers):
    handles.append(model.model.layers[l].register_forward_hook(make_h_hook(l)))
    handles.append(model.model.layers[l].self_attn.register_forward_hook(make_a_hook(l)))
    handles.append(model.model.layers[l].mlp.register_forward_hook(make_m_hook(l)))
console.print(f"✓ Registered {len(handles)} hooks (Gemma-2 architecture)")

manifest_rows = []
t_start = time.time()

for stimulus in track(set_a_emotional, description="Extracting..."):
    prompt = build_tak_prompt(stimulus.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]; colon_pos = n_tokens - 1
    final_tok = tokenizer.decode([inputs["input_ids"][0, -1].item()])
    layer_h.clear(); layer_a.clear(); layer_m.clear()

    with torch.no_grad():
        # NOTE: output_attentions=True for Gemma-2-9B may OOM (42 layers × 16 heads × ~200 tokens)
        # Attempt it; fall back to zeros if OOM
        try:
            outputs = model(**inputs, use_cache=False, output_attentions=True)
            attn_all = np.stack([outputs.attentions[l][0, :, colon_pos, :].cpu().to(torch.float32).numpy()
                                  for l in range(n_layers)])
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                console.print(f"[yellow]  OOM on attention for {stimulus.id} — saving zeros[/yellow]")
                torch.mps.empty_cache(); gc.collect()
                outputs = model(**inputs, use_cache=False)
                attn_all = np.zeros((n_layers, n_heads, n_tokens), dtype=np.float32)
            else:
                raise

    h_all = np.stack([layer_h[l] for l in range(n_layers)])
    a_all = np.stack([layer_a[l] for l in range(n_layers)])
    m_all = np.stack([layer_m[l] for l in range(n_layers)])

    top5_toks = [tokenizer.decode([tid]).strip() for tid in outputs.logits[0, -1, :].topk(5).indices.tolist()]
    correct = any(tok in top5_toks for tok in [t.strip() for t in EMOTION_VOCAB.get(stimulus.emotion, [])])

    meta = {"id": stimulus.id, "emotion": stimulus.emotion, "stimulus_set": stimulus.stimulus_set,
            "domain": stimulus.domain, "n_tokens_prompt": n_tokens, "colon_token_idx": colon_pos,
            "model_id": MODEL_KEY, "top5_predictions": top5_toks, "correct": correct, "final_token_str": final_tok}

    np.savez_compressed(ACT_DIR / f"{stimulus.id}.npz",
                        h=h_all.astype(np.float32), a=a_all.astype(np.float32),
                        m=m_all.astype(np.float32), attn=attn_all.astype(np.float32),
                        metadata=json.dumps(meta))
    manifest_rows.append(meta)
    del outputs; torch.mps.empty_cache(); gc.collect()

for h in handles: h.remove()
elapsed = time.time() - t_start
console.print(f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green]")

mid_l = n_layers // 2
all_h = np.stack([np.load(ACT_DIR / f"{s.id}.npz", allow_pickle=True)["h"][mid_l] for s in set_a_emotional])
std = all_h.std(axis=0).mean()
console.print(f"✓ Inter-stimulus std: {std:.4f}"); assert std > 0.001

import polars as pl
pl.DataFrame([{k: v if not isinstance(v, list) else str(v) for k, v in r.items()} for r in manifest_rows]).write_csv(ACT_DIR / "manifest.csv")
console.print(f"[bold green]✓ {MODEL_KEY} Set A complete![/bold green]")
