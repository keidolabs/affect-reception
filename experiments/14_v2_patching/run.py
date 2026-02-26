"""
Experiment 14 — v2 Activation Patching
Question: Do the same circuits causally mediate emotion in Set A and Set B?
Date: 2026-02-25

Optimizations vs. naive approach:
  - Source activations loaded from Exp 11/12 .npz files (no source forward pass)
  - Target baselines precomputed once per unique stimulus (not per pair)
  - PROBE_LAYERS: 8 key layers around probe peak (not rolling windows)
  - PAIRS_PER_EMOTION: 1 pair per (src_emo, tgt_emo) combination
  - HF AutoModelForCausalLM on MPS (same as Exp 11/12 extraction)
  - Expected runtime: ~10-15 min on MPS vs. ~20h CPU rolling-window approach

Two conditions:
  A. Within-set patching: source and target from same stimulus set
     - Success = top-1 shifts to source emotion after patching
  B. Cross-set patching: patch Set A activation into Set B target (same vs diff emotion)
     - If same-emotion cross-set matches within-set rates → shared circuits

Run: uv run python experiments/14_v2_patching/run.py
"""

import gc
import json
import os
import sys
import time
import torch
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from rich.console import Console
from rich.progress import track

console = Console()
torch.manual_seed(42); np.random.seed(42)

EXP_DIR   = Path(__file__).parent
ROOT      = EXP_DIR.parent.parent
OUT_DIR   = EXP_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_CONFIGS = {
    "llama1b_inst": {"id": "meta-llama/Llama-3.2-1B-Instruct",          "n_layers": 16},
    "llama1b_base": {"id": "meta-llama/Llama-3.2-1B",                    "n_layers": 16},
    "llama8b_base": {"id": "meta-llama/Meta-Llama-3-8B",                 "n_layers": 32},
    "llama8b_inst": {"id": "meta-llama/Meta-Llama-3-8B-Instruct",        "n_layers": 32},
    "gemma9b_base": {"id": "google/gemma-2-9b",                          "n_layers": 42},
    "gemma9b_inst": {"id": "google/gemma-2-9b-it",                       "n_layers": 42},
}

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "llama1b_inst"
assert MODEL_KEY in MODEL_CONFIGS, f"Unknown MODEL_KEY: {MODEL_KEY}. Options: {list(MODEL_CONFIGS)}"
MODEL_ID  = MODEL_CONFIGS[MODEL_KEY]["id"]
N_LAYERS  = MODEL_CONFIGS[MODEL_KEY]["n_layers"]
DEVICE    = "mps"

SETA_DIR  = ROOT / "experiments" / "11_v2_extract_seta" / "outputs" / "activations" / MODEL_KEY
SETB_DIR  = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations" / MODEL_KEY

# 8 probe layers at proportional depths — adapts to any model size
# Emphasises the 55–95% depth range where emotion circuits consistently peak
_probe_depths = [0.20, 0.40, 0.55, 0.65, 0.72, 0.80, 0.87, 0.95]
PROBE_LAYERS  = sorted(set([min(N_LAYERS - 1, max(0, int(d * N_LAYERS) - 1)) for d in _probe_depths]))

PAIRS_PER_EMOTION = 1   # pairs per (src_emo, tgt_emo) combination

# ============================================================================
# EMOTION VOCABULARY (for success check)
# ============================================================================

EMOTION_VOCAB = {
    "rage":       [" anger", " rage", " fury", " frustration", "anger", "rage", "Anger", "Rage"],
    "grief":      [" sadness", " grief", " sorrow", " loss", "sadness", "grief", "Sadness", "Grief"],
    "terror":     [" fear", " terror", " panic", " anxiety", "fear", "terror", "Fear", "Terror"],
    "ecstasy":    [" joy", " happiness", " euphoria", " ecstasy", "joy", "happiness", "Joy", "Happiness"],
    "loathing":   [" disgust", " loathing", " contempt", " hatred", "disgust", "loathing", "Disgust"],
    "amazement":  [" surprise", " wonder", " amazement", " shock", "surprise", "amazement", "Surprise"],
    "admiration": [" admiration", " respect", " awe", " pride", "admiration", "Admiration"],
    "vigilance":  [" anxiety", " vigilance", " alertness", " anticipation", "anxiety", "vigilance"],
}

EMOTIONS = list(EMOTION_VOCAB.keys())

SHOT_1_TEXT = "My dog died last week. I miss him every day."
SHOT_2_TEXT = "I got promoted and my boss praised my work in front of everyone."

def build_tak_prompt(target_text: str) -> str:
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\nAnswer: sadness\n\n"
        f"Context: {SHOT_2_TEXT}\nAnswer: joy\n\n"
        f"Context: {target_text}\nAnswer:"
    )

# ============================================================================
# CHECK PREREQUISITES
# ============================================================================

if not SETA_DIR.exists() or not (SETA_DIR / "manifest.csv").exists():
    console.print(f"[bold red]✗ Set A activations not found: {SETA_DIR}[/bold red]")
    console.print("[bold red]  Run: uv run python experiments/11_v2_extract_seta/extract_llama1b_inst.py[/bold red]")
    sys.exit(1)

if not SETB_DIR.exists() or not (SETB_DIR / "manifest.csv").exists():
    console.print(f"[bold red]✗ Set B activations not found: {SETB_DIR}[/bold red]")
    console.print("[bold red]  Run: uv run python experiments/12_v2_extract_setb/extract_llama1b_inst.py[/bold red]")
    sys.exit(1)

# ============================================================================
# LOAD MODEL — HF on MPS (same as extraction scripts)
# ============================================================================

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip(); break

console.print(f"\n[bold cyan]Loading {MODEL_ID} on {DEVICE}...[/bold cyan]")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
model.eval()

console.print(f"[bold green]✓ Model loaded — {N_LAYERS} layers, {DEVICE}[/bold green]")

# ============================================================================
# LOAD STIMULI
# ============================================================================

from stimuli.loader import load_set_a, load_set_b_clinical

seta_stims     = {s.id: s for s in load_set_a() if s.emotion != "neutral"}
setb_clin_stims = {s.id: s for s in load_set_b_clinical()}

seta_by_emotion = {}
for s in seta_stims.values():
    seta_by_emotion.setdefault(s.emotion, []).append(s)

setb_by_emotion = {}
for s in setb_clin_stims.values():
    setb_by_emotion.setdefault(s.emotion, []).append(s)

console.print(f"Set A by emotion: {dict((k, len(v)) for k, v in seta_by_emotion.items())}")
console.print(f"Set B by emotion: {dict((k, len(v)) for k, v in setb_by_emotion.items())}")

# ============================================================================
# PRECOMPUTE BASELINES — 1 forward pass per unique target stimulus
# ============================================================================

console.print(f"\n[bold cyan]Precomputing baselines ({len(seta_stims) + len(setb_clin_stims)} stimuli)...[/bold cyan]")

all_stims_list = list(seta_stims.values()) + list(setb_clin_stims.values())
baselines = {}

for stim in track(all_stims_list, description="Baselines"):
    prompt = build_tak_prompt(stim.text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    pred = tokenizer.decode([out.logits[0, -1, :].argmax().item()]).strip()
    baselines[stim.id] = pred

console.print(f"  {len(baselines)} baselines computed")

# ============================================================================
# PATCHING HELPERS
# ============================================================================

def get_module(model, act_type, layer_idx):
    """Return the module to hook for a given act_type."""
    if act_type == "h":
        return model.model.layers[layer_idx]        # full layer → resid_post
    elif act_type == "a":
        return model.model.layers[layer_idx].self_attn   # attention output only
    elif act_type == "m":
        return model.model.layers[layer_idx].mlp         # MLP output only
    raise ValueError(f"Unknown act_type: {act_type}")


def make_patch_hook(colon_pos: int, source_act: torch.Tensor):
    """
    Returns a forward hook that injects source_act at colon_pos.
    source_act: shape (d_model,) on device — the saved activation from .npz.
    Works for full layer (tuple output) and sublayer (tensor output).
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            out_list = list(output)
            modified = out_list[0].clone()
            modified[:, colon_pos, :] = source_act
            out_list[0] = modified
            return tuple(out_list)
        else:
            modified = output.clone()
            modified[:, colon_pos, :] = source_act
            return modified
    return hook


def load_source_act(stim_id: str, act_type: str, layer_idx: int, act_dir: Path) -> torch.Tensor:
    """Load saved activation from .npz, return shape (d_model,) on DEVICE."""
    npz = np.load(act_dir / f"{stim_id}.npz", allow_pickle=True)
    arr = npz[act_type][layer_idx].copy()   # (d_model,)
    return torch.from_numpy(arr).to(torch.float16).to(DEVICE)


def run_patch(model, tokenizer, target_stim, source_act: torch.Tensor,
              layer_idx: int, act_type: str) -> str:
    """Single patched forward pass. Returns top-1 prediction string."""
    prompt = build_tak_prompt(target_stim.text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    colon_pos = input_ids.shape[1] - 1

    module = get_module(model, act_type, layer_idx)
    handle = module.register_forward_hook(make_patch_hook(colon_pos, source_act))
    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        pred = tokenizer.decode([out.logits[0, -1, :].argmax().item()]).strip()
    finally:
        handle.remove()
    return pred


def is_success(patched_pred: str, source_emotion: str) -> bool:
    toks = EMOTION_VOCAB.get(source_emotion, [])
    return any(t.strip() == patched_pred or t == patched_pred for t in toks)

# ============================================================================
# MAIN PATCHING LOOP — within-set
# ============================================================================

console.print(f"\n[bold cyan]Running activation patching...[/bold cyan]")
console.print(f"  Probe layers: {PROBE_LAYERS}  |  Pairs/emotion: {PAIRS_PER_EMOTION}")

all_rows = []

for act_type in ["h", "a", "m"]:
    console.print(f"\n  [bold white]--- {act_type.upper()} component ---[/bold white]")

    for set_name, stims_by_emo, src_dir in [
        ("seta", seta_by_emotion, SETA_DIR),
        ("setb", setb_by_emotion, SETB_DIR),
    ]:
        console.print(f"  {set_name} within-set...")
        set_rows = []

        for layer_idx in track(PROBE_LAYERS, description=f"  {set_name} {act_type}"):
            norm_depth = (layer_idx + 1) / N_LAYERS

            for src_emo, tgt_emo in product(EMOTIONS, EMOTIONS):
                if src_emo == tgt_emo:
                    continue
                if src_emo not in stims_by_emo or tgt_emo not in stims_by_emo:
                    continue

                src_pool = stims_by_emo[src_emo]
                tgt_pool = stims_by_emo[tgt_emo]
                rng = np.random.default_rng(42 + layer_idx)
                n_pairs = min(PAIRS_PER_EMOTION, len(src_pool), len(tgt_pool))
                src_idx = rng.choice(len(src_pool), n_pairs, replace=False)
                tgt_idx = rng.choice(len(tgt_pool), n_pairs, replace=False)

                for si, ti in zip(src_idx, tgt_idx):
                    src_stim = src_pool[si]
                    tgt_stim = tgt_pool[ti]
                    try:
                        src_act   = load_source_act(src_stim.id, act_type, layer_idx, src_dir)
                        patched   = run_patch(model, tokenizer, tgt_stim, src_act, layer_idx, act_type)
                        orig_pred = baselines[tgt_stim.id]
                        success   = is_success(patched, src_emo)
                        set_rows.append({
                            "source_id":      src_stim.id,
                            "target_id":      tgt_stim.id,
                            "source_emotion": src_emo,
                            "target_emotion": tgt_emo,
                            "original_pred":  orig_pred,
                            "patched_pred":   patched,
                            "success":        success,
                            "layer":          layer_idx,
                            "norm_depth":     round(norm_depth, 4),
                            "act_type":       act_type,
                            "stimulus_set":   set_name,
                            "condition":      "within_set",
                        })
                    except Exception as e:
                        console.print(f"[yellow]    skip ({src_emo}→{tgt_emo} L{layer_idx}): {e}[/yellow]")

        if set_rows:
            df = pl.DataFrame(set_rows)
            df.write_csv(OUT_DIR / f"patching_results_{MODEL_KEY}_{set_name}_{act_type}.csv")

            by_layer = (
                df.group_by("layer")
                  .agg(pl.col("success").mean().alias("success_rate"),
                       pl.len().alias("n_pairs"),
                       pl.col("norm_depth").first())
                  .sort("layer")
            )
            by_layer.write_csv(
                OUT_DIR / f"patching_success_by_layer_{MODEL_KEY}_{set_name}_{act_type}.csv"
            )
            peak = by_layer.filter(pl.col("success_rate") == by_layer["success_rate"].max())
            console.print(
                f"    {set_name} {act_type}: peak L{peak['layer'][0]} "
                f"(depth {peak['norm_depth'][0]:.3f}), success={peak['success_rate'][0]:.3f}, "
                f"n={peak['n_pairs'][0]}"
            )
        all_rows.extend(set_rows)

# ============================================================================
# CROSS-SET PATCHING — Set A → Set B
# ============================================================================

console.print(f"\n[bold cyan]Cross-set patching (Set A → Set B)...[/bold cyan]")

for act_type in ["h", "a", "m"]:
    crossset_rows = []

    for layer_idx in track(PROBE_LAYERS, description=f"  crossset {act_type}"):
        norm_depth = (layer_idx + 1) / N_LAYERS

        # Condition 1: same emotion (A:rage → B:rage)
        for emotion in EMOTIONS:
            if emotion not in seta_by_emotion or emotion not in setb_by_emotion:
                continue
            src_pool = seta_by_emotion[emotion]
            tgt_pool = setb_by_emotion[emotion]
            rng = np.random.default_rng(99 + layer_idx)
            n_pairs = min(PAIRS_PER_EMOTION, len(src_pool), len(tgt_pool))
            for si, ti in zip(rng.choice(len(src_pool), n_pairs, replace=False),
                               rng.choice(len(tgt_pool), n_pairs, replace=False)):
                try:
                    src_act = load_source_act(src_pool[si].id, act_type, layer_idx, SETA_DIR)
                    patched = run_patch(model, tokenizer, tgt_pool[ti], src_act, layer_idx, act_type)
                    success = is_success(patched, emotion)
                    crossset_rows.append({
                        "source_id": src_pool[si].id, "target_id": tgt_pool[ti].id,
                        "source_emotion": emotion, "target_emotion": emotion,
                        "patched_pred": patched, "success": success,
                        "layer": layer_idx, "norm_depth": round(norm_depth, 4),
                        "act_type": act_type, "condition": "crossset_same_emotion",
                    })
                except Exception:
                    pass

        # Condition 2: different emotion (A:X → B:Y, sample 20 pairs)
        for src_emo, tgt_emo in list(product(EMOTIONS, EMOTIONS))[:20]:
            if src_emo == tgt_emo:
                continue
            if src_emo not in seta_by_emotion or tgt_emo not in setb_by_emotion:
                continue
            src_pool = seta_by_emotion[src_emo]
            tgt_pool = setb_by_emotion[tgt_emo]
            rng = np.random.default_rng(199 + layer_idx)
            n_pairs = min(1, len(src_pool), len(tgt_pool))
            for si, ti in zip(rng.choice(len(src_pool), n_pairs, replace=False),
                               rng.choice(len(tgt_pool), n_pairs, replace=False)):
                try:
                    src_act = load_source_act(src_pool[si].id, act_type, layer_idx, SETA_DIR)
                    patched = run_patch(model, tokenizer, tgt_pool[ti], src_act, layer_idx, act_type)
                    success = is_success(patched, src_emo)
                    crossset_rows.append({
                        "source_id": src_pool[si].id, "target_id": tgt_pool[ti].id,
                        "source_emotion": src_emo, "target_emotion": tgt_emo,
                        "patched_pred": patched, "success": success,
                        "layer": layer_idx, "norm_depth": round(norm_depth, 4),
                        "act_type": act_type, "condition": "crossset_diff_emotion",
                    })
                except Exception:
                    pass

    if crossset_rows:
        df = pl.DataFrame(crossset_rows)
        df.write_csv(OUT_DIR / f"patching_crossset_results_{MODEL_KEY}_{act_type}.csv")
        for cond in ["crossset_same_emotion", "crossset_diff_emotion"]:
            sub = df.filter(pl.col("condition") == cond)
            if len(sub) > 0:
                console.print(
                    f"  {act_type} {cond}: success={sub['success'].mean():.3f} (n={len(sub)})"
                )

# ============================================================================
# VISUALIZATION
# ============================================================================

console.print("\n[bold cyan]Generating figures...[/bold cyan]")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Activation Patching Success — {MODEL_KEY}", fontsize=13)

for col_idx, act_type in enumerate(["h", "a", "m"]):
    for row_idx, set_name in enumerate(["seta", "setb"]):
        ax = axes[row_idx, col_idx]
        csv_path = OUT_DIR / f"patching_success_by_layer_{MODEL_KEY}_{set_name}_{act_type}.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            nd  = df["norm_depth"].to_numpy()
            sr  = df["success_rate"].to_numpy()
            color = "#e74c3c" if set_name == "seta" else "#2980b9"
            ax.plot(nd, sr, linewidth=2.5, marker="o", markersize=5, color=color)
            ax.fill_between(nd, 0, sr, alpha=0.15, color=color)
            peak_i = int(np.argmax(sr))
            ax.axvline(nd[peak_i], linestyle="--", alpha=0.5, color=color)
            ax.text(nd[peak_i] + 0.01, sr[peak_i], f"L{df['layer'][peak_i]}\n{sr[peak_i]:.2f}",
                    fontsize=7, color=color)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{act_type.upper()} — {'Set A' if set_name == 'seta' else 'Set B'}", fontsize=10)
        ax.set_xlabel("Normalized depth"); ax.set_ylabel("Success rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
        ax.axhline(0, linestyle=":", color="gray", alpha=0.4); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / f"patching_heatmap_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"patching_heatmap_{MODEL_KEY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ patching_heatmap_{MODEL_KEY}.png/.svg saved[/bold green]")

# Cross-set figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Cross-Set Patching (Set A → Set B) — {MODEL_KEY}", fontsize=12)

for col_idx, act_type in enumerate(["h", "a", "m"]):
    ax = axes[col_idx]
    csv_path = OUT_DIR / f"patching_crossset_results_{MODEL_KEY}_{act_type}.csv"
    if csv_path.exists():
        df = pl.read_csv(csv_path)
        for cond, color, label in [
            ("crossset_same_emotion", "#27ae60", "Same emotion (A:X→B:X)"),
            ("crossset_diff_emotion", "#e74c3c", "Diff emotion (A:X→B:Y)"),
        ]:
            sub = df.filter(pl.col("condition") == cond)
            if len(sub) > 0:
                by_l = (sub.group_by("layer")
                           .agg(pl.col("success").mean().alias("sr"),
                                pl.col("norm_depth").first())
                           .sort("layer"))
                ax.plot(by_l["norm_depth"].to_list(), by_l["sr"].to_list(),
                        color=color, linewidth=2.5, marker="o", markersize=4, label=label)
    ax.set_title(f"{act_type.upper()} component", fontsize=10)
    ax.set_xlabel("Normalized depth"); ax.set_ylabel("Success rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / f"patching_crossset_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"patching_crossset_{MODEL_KEY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ patching_crossset_{MODEL_KEY}.png/.svg saved[/bold green]")

# ============================================================================
# README
# ============================================================================

readme = f"""# Experiment 14 — v2 Activation Patching

**Date:** 2026-02-25
**Model:** {MODEL_KEY} (Llama-3.2-1B-Instruct, HF on MPS)
**Probe layers:** {PROBE_LAYERS}
**Pairs per emotion combo:** {PAIRS_PER_EMOTION}

## Research Question

Do the same circuits causally mediate emotion inference in keyword-rich (Set A)
and keyword-free clinical (Set B) stimuli?

## Method

Activation patching: replace target stimulus's layer-{act_type} activation at ":"
with the saved activation from a source stimulus of a different emotion.
Success = top-1 prediction shifts to source emotion.

- Source activations loaded from Exp 11/12 .npz files (no re-run)
- Baselines precomputed once per stimulus
- Within-set: source and target from same set
- Cross-set: source from Set A, target from Set B (same vs different emotion)

## Interpretation

If cross-set SAME emotion patching succeeds at similar rates to within-set,
the emotion representations are causally shared across stimulus types (Set A ≈ Set B circuits).

## Outputs

- `outputs/patching_results_{MODEL_KEY}_{{seta,setb}}_{{h,a,m}}.csv`
- `outputs/patching_success_by_layer_*.csv`
- `outputs/patching_crossset_results_{MODEL_KEY}_*.csv`
- `outputs/patching_heatmap_{MODEL_KEY}.png/.svg`
- `outputs/patching_crossset_{MODEL_KEY}.png/.svg`
"""

(EXP_DIR / "README.md").write_text(readme)
console.print(f"[bold green]✓ README.md written[/bold green]")
console.print(f"\n[bold green]✓ Experiment 14 complete![/bold green]")
