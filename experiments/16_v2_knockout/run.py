"""
Experiment 16 — v2 Knockout Experiments
Question: Which layers are causally necessary for emotion inference?
Date: 2026-02-25

Two knockout types at ":" position:
  1. Zero knockout: set activation to 0 (most disruptive)
  2. Random knockout: replace with magnitude-preserving random vector

Two activation types:
  a. MHSA output (a): zeroing removes attention's contribution to residual
  m. FFN output (m): zeroing removes MLP's contribution to residual

For each (layer, knockout_type, activation_type):
  - Run all Set A and Set B stimuli through the model WITH the knockout hook
  - Measure accuracy drop (fraction where top-1 prediction changes)
  - "Critical layer" = accuracy drop > 20%

Shared circuits test: critical layers in Set A should overlap with Set B.

Uses HF AutoModelForCausalLM on MPS (same as extraction scripts).
No saved activations needed — runs live with register_forward_hook.

Run: uv run python experiments/16_v2_knockout/run.py
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
from rich.console import Console
from rich.progress import track

console = Console()
torch.manual_seed(42); np.random.seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
OUT_DIR = EXP_DIR / "outputs"
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
DEVICE    = os.getenv("EMO_DEVICE", "cuda" if torch.cuda.is_available() else "mps")

EMOTION_VOCAB = {
    "rage":       [" anger", " rage", " fury", "anger", "rage", "Anger", "Rage"],
    "grief":      [" sadness", " grief", " sorrow", "sadness", "grief", "Sadness"],
    "terror":     [" fear", " terror", " panic", "fear", "terror", "Fear"],
    "ecstasy":    [" joy", " happiness", " euphoria", "joy", "happiness", "Joy"],
    "loathing":   [" disgust", " loathing", " contempt", "disgust", "loathing"],
    "amazement":  [" surprise", " wonder", " amazement", "surprise", "amazement"],
    "admiration": [" admiration", " respect", " awe", "admiration"],
    "vigilance":  [" anxiety", " vigilance", " alertness", "anxiety", "vigilance"],
}

SHOT_1_TEXT = "My dog died last week. I miss him every day."
SHOT_2_TEXT = "I got promoted and my boss praised my work in front of everyone."

def build_tak_prompt(t):
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\nAnswer: sadness\n\n"
        f"Context: {SHOT_2_TEXT}\nAnswer: joy\n\n"
        f"Context: {t}\nAnswer:"
    )

# ============================================================================
# LOAD HF TOKEN + MODEL
# ============================================================================

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip(); break

console.print(f"[bold cyan]Loading {MODEL_ID} on {DEVICE}...[/bold cyan]")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
model.eval()
console.print(f"[bold green]✓ Model loaded — {N_LAYERS} layers, {DEVICE}[/bold green]")

# ============================================================================
# LOAD STIMULI
# ============================================================================

from stimuli.loader import load_set_a, load_set_b_clinical

seta_emotional = sorted([s for s in load_set_a() if s.emotion != "neutral"], key=lambda s: s.id)
setb_clinical  = sorted(load_set_b_clinical(), key=lambda s: s.id)

console.print(f"Set A: {len(seta_emotional)} stimuli")
console.print(f"Set B: {len(setb_clinical)} stimuli")

# ============================================================================
# BASELINE PREDICTIONS
# ============================================================================

def get_top1(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    return tokenizer.decode([out.logits[0, -1, :].argmax().item()]).strip()


console.print(f"\n[bold cyan]Getting baseline predictions...[/bold cyan]")

baselines = {}
for stim_list, set_name in [(seta_emotional, "seta"), (setb_clinical, "setb")]:
    baselines[set_name] = {}
    for stimulus in track(stim_list, description=f"Baseline {set_name}"):
        prompt = build_tak_prompt(stimulus.text)
        baselines[set_name][stimulus.id] = get_top1(model, tokenizer, prompt)
    console.print(f"  {set_name}: {len(baselines[set_name])} baselines done")

# ============================================================================
# KNOCKOUT HELPERS
# ============================================================================

def get_module(model, act_type: str, layer_idx: int):
    """Return the submodule to hook for a given act_type."""
    if act_type == "a":
        return model.model.layers[layer_idx].self_attn
    elif act_type == "m":
        return model.model.layers[layer_idx].mlp
    elif act_type == "h":
        return model.model.layers[layer_idx]
    raise ValueError(f"Unknown act_type: {act_type}")


def make_zero_hook(colon_pos: int):
    """Zero out the activation at colon_pos."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = list(output)
            modified = out[0].clone()
            modified[:, colon_pos, :] = 0.0
            out[0] = modified
            return tuple(out)
        else:
            modified = output.clone()
            modified[:, colon_pos, :] = 0.0
            return modified
    return hook


def make_random_hook(colon_pos: int):
    """Replace activation at colon_pos with magnitude-preserving random vector."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = list(output)
            act = out[0].clone()
            orig = act[:, colon_pos, :]
            r = torch.randn_like(orig)
            orig_norm = orig.norm(dim=-1, keepdim=True)
            r_norm    = r.norm(dim=-1, keepdim=True)
            act[:, colon_pos, :] = r * (orig_norm / (r_norm + 1e-8))
            out[0] = act
            return tuple(out)
        else:
            out = output.clone()
            orig = out[:, colon_pos, :]
            r = torch.randn_like(orig)
            orig_norm = orig.norm(dim=-1, keepdim=True)
            r_norm    = r.norm(dim=-1, keepdim=True)
            out[:, colon_pos, :] = r * (orig_norm / (r_norm + 1e-8))
            return out
    return hook


def run_knockout_layer(model, tokenizer, stim_list, baselines_dict,
                       layer_idx, act_type, knockout_type):
    """
    Run knockout at one layer for all stimuli.
    Returns list of dicts {id, emotion, original_pred, ko_pred, changed, ...}.
    """
    results = []
    module = get_module(model, act_type, layer_idx)
    hook_fn = make_zero_hook if knockout_type == "zero" else make_random_hook

    for stimulus in stim_list:
        prompt   = build_tak_prompt(stimulus.text)
        inputs   = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        colon_pos = input_ids.shape[1] - 1
        orig_pred = baselines_dict.get(stimulus.id, "")

        handle = module.register_forward_hook(hook_fn(colon_pos))
        try:
            with torch.no_grad():
                out = model(input_ids, use_cache=False)
            ko_pred = tokenizer.decode([out.logits[0, -1, :].argmax().item()]).strip()
        finally:
            handle.remove()

        results.append({
            "id":            stimulus.id,
            "emotion":       stimulus.emotion,
            "original_pred": orig_pred,
            "ko_pred":       ko_pred,
            "changed":       ko_pred != orig_pred,
            "layer":         layer_idx,
            "act_type":      act_type,
            "knockout_type": knockout_type,
        })

    return results

# ============================================================================
# RUN KNOCKOUT ACROSS ALL LAYERS
# ============================================================================

console.print(f"\n[bold cyan]Running knockout experiments...[/bold cyan]")
console.print(f"  {N_LAYERS} layers × 2 act_types × 2 ko_types × 2 sets = {N_LAYERS * 2 * 2 * 2} configurations")

all_rows    = []
summary_rows = []

for act_type in ["a", "m"]:
    for knockout_type in ["zero", "random"]:
        for set_name, stim_list in [("seta", seta_emotional), ("setb", setb_clinical)]:
            console.print(f"\n  {act_type} × {knockout_type} × {set_name}...")

            by_layer_results = []
            for l in track(range(N_LAYERS), description=f"  L0→L{N_LAYERS-1}"):
                layer_rows = run_knockout_layer(
                    model, tokenizer, stim_list, baselines[set_name],
                    l, act_type, knockout_type
                )
                all_rows.extend(layer_rows)
                by_layer_results.append(layer_rows)

            # Accuracy drop per layer
            for l, layer_rows in enumerate(by_layer_results):
                n_changed = sum(r["changed"] for r in layer_rows)
                n_total   = len(layer_rows)
                drop_rate = n_changed / n_total if n_total > 0 else 0
                summary_rows.append({
                    "layer":         l,
                    "norm_depth":    round((l + 1) / N_LAYERS, 4),
                    "act_type":      act_type,
                    "knockout_type": knockout_type,
                    "stimulus_set":  set_name,
                    "accuracy_drop": drop_rate,
                    "n_changed":     n_changed,
                    "n_total":       n_total,
                    "is_critical":   drop_rate >= 0.20,
                })

            # Save per-config CSV
            cfg_df = pl.DataFrame([r for ll in by_layer_results for r in ll])
            cfg_df.write_csv(
                OUT_DIR / f"knockout_results_{MODEL_KEY}_{set_name}_{knockout_type}_{act_type}.csv"
            )

# ============================================================================
# SAVE + ANALYZE
# ============================================================================

summary_df = pl.DataFrame(summary_rows)
summary_df.write_csv(OUT_DIR / f"knockout_summary_{MODEL_KEY}.csv")
console.print(f"\n[bold green]✓ knockout_summary_{MODEL_KEY}.csv saved[/bold green]")

critical_layers = (
    summary_df.filter(pl.col("is_critical") == True)
              .select("layer").unique().sort("layer")
)
console.print(f"\nCritical layers (≥20% drop): {critical_layers['layer'].to_list()}")

# Shared critical layers between Set A and Set B
for act_type in ["a", "m"]:
    for ko_type in ["zero", "random"]:
        seta_crit = set(
            summary_df.filter(
                (pl.col("act_type") == act_type) & (pl.col("knockout_type") == ko_type) &
                (pl.col("stimulus_set") == "seta") & (pl.col("is_critical") == True)
            )["layer"].to_list()
        )
        setb_crit = set(
            summary_df.filter(
                (pl.col("act_type") == act_type) & (pl.col("knockout_type") == ko_type) &
                (pl.col("stimulus_set") == "setb") & (pl.col("is_critical") == True)
            )["layer"].to_list()
        )
        overlap = seta_crit & setb_crit
        console.print(
            f"  {act_type} {ko_type}: A={sorted(seta_crit)}, B={sorted(setb_crit)}, overlap={sorted(overlap)}"
        )

# Save critical layers CSV
critical_rows = []
for act_type in ["a", "m"]:
    for ko_type in ["zero", "random"]:
        for set_name in ["seta", "setb"]:
            crit = summary_df.filter(
                (pl.col("act_type") == act_type) & (pl.col("knockout_type") == ko_type) &
                (pl.col("stimulus_set") == set_name) & (pl.col("is_critical") == True)
            )
            for row in crit.iter_rows(named=True):
                critical_rows.append({**row, "model_key": MODEL_KEY})

if critical_rows:
    pl.DataFrame(critical_rows).write_csv(OUT_DIR / f"critical_layers_{MODEL_KEY}.csv")
    console.print(f"[bold green]✓ critical_layers_{MODEL_KEY}.csv saved[/bold green]")

# ============================================================================
# VISUALIZATION
# ============================================================================

console.print("\n[bold cyan]Generating knockout figures...[/bold cyan]")

norm_depths = np.array([(l + 1) / N_LAYERS for l in range(N_LAYERS)])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Knockout Accuracy Drop — {MODEL_KEY}", fontsize=13)

for row_idx, act_type in enumerate(["a", "m"]):
    for col_idx, ko_type in enumerate(["zero", "random"]):
        ax = axes[row_idx, col_idx]
        for set_name, color, ls in [("seta", "#e74c3c", "-"), ("setb", "#2980b9", "--")]:
            sub = summary_df.filter(
                (pl.col("act_type") == act_type) & (pl.col("knockout_type") == ko_type) &
                (pl.col("stimulus_set") == set_name)
            ).sort("layer")
            if len(sub) > 0:
                drops = sub["accuracy_drop"].to_numpy()
                ax.plot(norm_depths[:len(drops)], drops, color=color, linestyle=ls,
                        linewidth=2.5, marker="o", markersize=4,
                        label=f"{'Set A' if set_name == 'seta' else 'Set B'}")
        ax.axhline(0.20, linestyle=":", color="black", alpha=0.5, label="Critical threshold (20%)")
        ax.set_title(f"{act_type.upper()} × {ko_type} knockout", fontsize=10)
        ax.set_xlabel("Normalized depth"); ax.set_ylabel("Accuracy drop (fraction changed)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / f"knockout_curves_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"knockout_curves_{MODEL_KEY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ knockout_curves_{MODEL_KEY}.png/.svg saved[/bold green]")

# ============================================================================
# README
# ============================================================================

zero_a_seta = sorted(summary_df.filter(
    (pl.col("act_type") == "a") & (pl.col("knockout_type") == "zero") &
    (pl.col("stimulus_set") == "seta") & (pl.col("is_critical") == True))["layer"].to_list())
zero_m_seta = sorted(summary_df.filter(
    (pl.col("act_type") == "m") & (pl.col("knockout_type") == "zero") &
    (pl.col("stimulus_set") == "seta") & (pl.col("is_critical") == True))["layer"].to_list())
zero_a_setb = sorted(summary_df.filter(
    (pl.col("act_type") == "a") & (pl.col("knockout_type") == "zero") &
    (pl.col("stimulus_set") == "setb") & (pl.col("is_critical") == True))["layer"].to_list())

readme = f"""# Experiment 16 — v2 Knockout Experiments

**Date:** 2026-02-25
**Model:** {MODEL_KEY} (Llama-3.2-1B-Instruct, HF on MPS)
**Threshold for "critical layer":** ≥20% accuracy drop

## Research Question

Which layers are causally necessary for emotion inference in Set A and Set B?

## Key Findings

| Configuration | Critical Layers (Set A) | Critical Layers (Set B) |
|---------------|------------------------|------------------------|
| MHSA zero knockout | {zero_a_seta} | {zero_a_setb} |
| FFN zero knockout  | {zero_m_seta} | see CSV |

**Overlap (shared critical layers):** Layers appearing in both Set A and Set B
indicate circuits that are causally shared across stimulus types.

## Methodology

- Zero knockout: set activation to 0 at ":" position (most disruptive)
- Random knockout: magnitude-preserving random vector at ":" position
- Accuracy drop = fraction of stimuli where top-1 prediction changes post-knockout
- Critical layer = accuracy drop ≥ 20%

## Outputs

- `outputs/knockout_summary_{MODEL_KEY}.csv` — per-layer accuracy drops
- `outputs/critical_layers_{MODEL_KEY}.csv` — critical layers per configuration
- `outputs/knockout_curves_{MODEL_KEY}.png/.svg` — visualization
"""

(EXP_DIR / "README.md").write_text(readme)
console.print(f"[bold green]✓ README.md written[/bold green]")
console.print(f"\n[bold green]✓ Experiment 16 complete![/bold green]")
