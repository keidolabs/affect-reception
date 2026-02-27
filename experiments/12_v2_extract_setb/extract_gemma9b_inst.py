"""
Experiment 12 — v2 Set B Extraction: Gemma-2-9B-IT (instruct)
Date: 2026-02-25
Framework: HF transformers + register_forward_hook on MPS
Run: uv run python experiments/12_v2_extract_setb/extract_gemma9b_inst.py
"""

import gc, json, os, sys, time, torch, numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
torch.manual_seed(42); np.random.seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
ACT_DIR = EXP_DIR / "outputs" / "activations" / "gemma9b_inst"
ACT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

MODEL_KEY = "gemma9b_inst"; MODEL_ID = "google/gemma-2-9b-it"
DEVICE    = os.getenv("EMO_DEVICE", "cuda" if torch.cuda.is_available() else "mps")
DTYPE     = torch.float16

SHOT_1_TEXT = "My dog died last week. I miss him every day."
SHOT_2_TEXT = "I got promoted and my boss praised my work in front of everyone."
def build_tak_prompt(t): return (
    "What are the inferred emotions in the following contexts?\n\n"
    f"Context: {SHOT_1_TEXT}\nAnswer: sadness\n\n"
    f"Context: {SHOT_2_TEXT}\nAnswer: joy\n\n"
    f"Context: {t}\nAnswer:"
)

EMOTION_VOCAB = {
    "rage": [" anger", " rage", "anger", "rage"], "grief": [" sadness", " grief", "sadness"],
    "terror": [" fear", " terror", "fear"], "ecstasy": [" joy", " happiness", "joy"],
    "loathing": [" disgust", "disgust"], "amazement": [" surprise", "surprise"],
    "admiration": [" admiration", " respect"], "vigilance": [" anxiety", "anxiety"],
}

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip(); break

from stimuli.loader import load_set_b_clinical, load_set_b_neutral
all_stimuli = sorted(load_set_b_clinical() + load_set_b_neutral(), key=lambda s: s.id)
assert len(all_stimuli) == 192; console.print(f"Set B: {len(all_stimuli)} stimuli")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, attn_implementation="eager").to(DEVICE)
model.eval()
n_layers = model.config.num_hidden_layers; d_model = model.config.hidden_size; n_heads = model.config.num_attention_heads
console.print(f"[bold green]✓ {MODEL_ID} — {n_layers}L[/bold green]")

for p, acc in [("The capital of France is", ["Paris", " Paris"])]:
    inp = tokenizer(p, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): out = model(**inp, use_cache=False)
    top3 = [tokenizer.decode([tid]) for tid in out.logits[0, -1, :].topk(3).indices.tolist()]
    if not any(t in top3 for t in acc): raise RuntimeError("device validation FAILED")
    if DEVICE == "mps": torch.mps.empty_cache()
    elif DEVICE == "cuda": torch.cuda.empty_cache()
console.print("[bold green]✓ device OK[/bold green]")

layer_h, layer_a, layer_m = {}, {}, {}
def make_h_hook(l):
    def hook(mod, inp, out):
        layer_h[l] = (out[0] if isinstance(out, tuple) else out)[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook
def make_a_hook(l):
    def hook(mod, inp, out):
        layer_a[l] = (out[0] if isinstance(out, tuple) else out)[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook
def make_m_hook(l):
    def hook(mod, inp, out):
        layer_m[l] = (out[0] if isinstance(out, tuple) else out)[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook

handles = []
for l in range(n_layers):
    handles += [model.model.layers[l].register_forward_hook(make_h_hook(l)),
                model.model.layers[l].self_attn.register_forward_hook(make_a_hook(l)),
                model.model.layers[l].mlp.register_forward_hook(make_m_hook(l))]

manifest_rows = []; t_start = time.time()

for stimulus in track(all_stimuli, description="Extracting Set B..."):
    prompt = build_tak_prompt(stimulus.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]; colon_pos = n_tokens - 1
    final_tok = tokenizer.decode([inputs["input_ids"][0, -1].item()])
    layer_h.clear(); layer_a.clear(); layer_m.clear()

    with torch.no_grad():
        try:
            outputs = model(**inputs, use_cache=False, output_attentions=True)
            attn_all = np.stack([outputs.attentions[l][0, :, colon_pos, :].cpu().to(torch.float32).numpy() for l in range(n_layers)])
        except RuntimeError:
            if DEVICE == "mps": torch.mps.empty_cache()
            elif DEVICE == "cuda": torch.cuda.empty_cache()
            gc.collect()
            outputs = model(**inputs, use_cache=False)
            attn_all = np.zeros((n_layers, n_heads, n_tokens), dtype=np.float32)

    h_all = np.stack([layer_h[l] for l in range(n_layers)])
    a_all = np.stack([layer_a[l] for l in range(n_layers)])
    m_all = np.stack([layer_m[l] for l in range(n_layers)])

    top5_toks = [tokenizer.decode([tid]).strip() for tid in outputs.logits[0, -1, :].topk(5).indices.tolist()]
    correct = any(t in top5_toks for t in [t2.strip() for t2 in EMOTION_VOCAB.get(stimulus.emotion, [])])

    meta = {"id": stimulus.id, "emotion": stimulus.emotion, "stimulus_set": stimulus.stimulus_set,
            "domain": stimulus.domain, "n_tokens_prompt": n_tokens, "colon_token_idx": colon_pos,
            "model_id": MODEL_KEY, "top5_predictions": top5_toks, "correct": correct,
            "final_token_str": final_tok, "matched_control_id": stimulus.matched_control_id}

    np.savez_compressed(ACT_DIR / f"{stimulus.id}.npz",
                        h=h_all.astype(np.float32), a=a_all.astype(np.float32),
                        m=m_all.astype(np.float32), attn=attn_all.astype(np.float32),
                        metadata=json.dumps(meta))
    manifest_rows.append(meta)
    del outputs
    if DEVICE == "mps": torch.mps.empty_cache()
    elif DEVICE == "cuda": torch.cuda.empty_cache()
    gc.collect()

for h in handles: h.remove()
console.print(f"\n[bold green]✓ Done in {time.time()-t_start:.1f}s[/bold green]")

import polars as pl
pl.DataFrame([{k: v if not isinstance(v, list) else str(v) for k, v in r.items()} for r in manifest_rows]).write_csv(ACT_DIR / "manifest.csv")
console.print(f"[bold green]✓ {MODEL_KEY} Set B complete![/bold green]")
