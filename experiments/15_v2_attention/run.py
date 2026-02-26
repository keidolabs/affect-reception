"""
Experiment 15 — v2 Attention Pattern Analysis
Question: What does the ":" token attend to in Set A vs Set B?
Date: 2026-02-25

Key hypothesis:
  - Set A (keyword-rich): mid-layer attention at ":" should hit explicit emotion words
  - Set B (clinical): mid-layer attention should hit contextual/behavioral cues, NOT emotion words
  - If Set B attention hits non-keyword cues yet still encodes emotion (from probes),
    the model extracts emotion through semantic context, not lexical shortcuts

Method:
  - Load attention patterns from .npz files (extracted at ":" position)
  - For each layer × head: find top-attended tokens in each stimulus
  - Classify tokens as: emotion_keyword, context_word, function_word, punctuation
  - Compare Set A vs Set B distributions
  - Head sensitivity: which heads have highest emotion-keyword attention in Set A?

Prerequisite:
  - Run experiments/11_v2_extract_seta/extract_llama1b_inst.py (must have attn arrays)
  - Run experiments/12_v2_extract_setb/extract_llama1b_inst.py (must have attn arrays)

Run: uv run python experiments/15_v2_attention/run.py
"""

import gc
import json
import os
import sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter, defaultdict
from rich.console import Console
from rich.table import Table

console = Console()
np.random.seed(42)

EXP_DIR  = Path(__file__).parent
ROOT     = EXP_DIR.parent.parent
OUT_DIR  = EXP_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_CONFIGS = {
    "llama1b_inst": {"id": "meta-llama/Llama-3.2-1B-Instruct",          "n_layers": 16, "n_heads": 32},
    "llama1b_base": {"id": "meta-llama/Llama-3.2-1B",                    "n_layers": 16, "n_heads": 32},
    "llama8b_base": {"id": "meta-llama/Meta-Llama-3-8B",                 "n_layers": 32, "n_heads": 32},
    "llama8b_inst": {"id": "meta-llama/Meta-Llama-3-8B-Instruct",        "n_layers": 32, "n_heads": 32},
    "gemma9b_base": {"id": "google/gemma-2-9b",                          "n_layers": 42, "n_heads": 16},
    "gemma9b_inst": {"id": "google/gemma-2-9b-it",                       "n_layers": 42, "n_heads": 16},
}

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "llama1b_inst"
assert MODEL_KEY in MODEL_CONFIGS, f"Unknown MODEL_KEY: {MODEL_KEY}. Options: {list(MODEL_CONFIGS)}"
MODEL_ID  = MODEL_CONFIGS[MODEL_KEY]["id"]
N_LAYERS  = MODEL_CONFIGS[MODEL_KEY]["n_layers"]
N_HEADS   = MODEL_CONFIGS[MODEL_KEY]["n_heads"]

SETA_DIR = ROOT / "experiments" / "11_v2_extract_seta" / "outputs" / "activations" / MODEL_KEY
SETB_DIR = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations" / MODEL_KEY

# ============================================================================
# TOKEN CLASSIFICATION
# ============================================================================

# Emotion keywords (broad — for Set A detection)
EMOTION_KEYWORDS = {
    "rage", "anger", "angry", "furious", "fury", "mad", "enraged", "wrath",
    "grief", "sad", "sadness", "sorrow", "cry", "weep", "mourn", "loss",
    "fear", "terror", "afraid", "scared", "panic", "dread", "horror",
    "joy", "happy", "happiness", "excited", "elated", "euphoria", "ecstasy",
    "disgust", "loathing", "disgusted", "contempt", "repulsion",
    "surprise", "shocked", "amazed", "amazement", "astonished", "wonder",
    "admiration", "admired", "respect", "proud", "pride",
    "anxiety", "anxious", "vigilance", "worry", "worried", "dread", "anticipation",
}

FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "i", "he", "she", "they", "we", "it", "me", "him", "her", "them", "us",
    "my", "his", "her", "their", "our", "its", "your",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "and", "or", "but", "not", "no", "yes",
    "that", "this", "these", "those", "what", "which", "who",
    ".", ",", "!", "?", ";", ":", "'", '"',
}

def classify_token(token_str: str) -> str:
    """Classify a token into one of 4 categories."""
    t = token_str.strip().lower().rstrip(".,!?;:'\"")
    if t in EMOTION_KEYWORDS or any(ek in t for ek in EMOTION_KEYWORDS):
        return "emotion_keyword"
    if t in FUNCTION_WORDS or len(t) <= 1:
        return "function_word"
    if not t.isalpha():
        return "punctuation"
    return "context_word"

# ============================================================================
# CHECK PREREQUISITES
# ============================================================================

for act_dir, name in [(SETA_DIR, "Set A"), (SETB_DIR, "Set B")]:
    if not act_dir.exists() or not (act_dir / "manifest.csv").exists():
        console.print(f"[bold red]✗ {name} activations not found: {act_dir}[/bold red]")
        sys.exit(1)

# ============================================================================
# LOAD MODEL TOKENIZER (for token decoding)
# ============================================================================

console.print(f"\n[bold cyan]Loading tokenizer for {MODEL_KEY}...[/bold cyan]")

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip(); break

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
console.print("[bold green]✓ Tokenizer loaded[/bold green]")

# Also load stimuli to map token IDs back to text
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
# LOAD STIMULI
# ============================================================================

from stimuli.loader import load_set_a, load_set_b_clinical
seta_stims = {s.id: s for s in load_set_a() if s.emotion != "neutral"}
setb_stims = {s.id: s for s in load_set_b_clinical()}

# ============================================================================
# ATTENTION ANALYSIS — per stimulus set
# ============================================================================

def analyze_attention_set(act_dir: Path, stims_dict: dict, set_name: str):
    """
    For each stimulus in set, decode attention weights from ":" position.
    Returns: per-layer per-head token type distribution, top token lists.
    """
    manifest = pl.read_csv(act_dir / "manifest.csv")

    # Aggregate: token_type distribution per (layer, head) across stimuli
    # shape: (n_layers, n_heads, 4_token_types) → accumulated
    type_map = {"emotion_keyword": 0, "context_word": 1, "function_word": 2, "punctuation": 3}
    type_counts = np.zeros((N_LAYERS, N_HEADS, 4), dtype=np.float32)
    n_stimuli_processed = 0

    # For top-token analysis at each layer
    top_tokens_by_layer = defaultdict(list)  # layer -> list of (token_str, emotion) tuples

    for row in manifest.iter_rows(named=True):
        stim_id = row["id"]
        npz_path = act_dir / f"{stim_id}.npz"
        if not npz_path.exists():
            continue

        stim = stims_dict.get(stim_id)
        if stim is None:
            continue

        npz = np.load(npz_path, allow_pickle=True)
        attn = npz["attn"]  # (n_layers, n_heads, seq_len)

        if attn.shape[0] != N_LAYERS or attn.max() == 0:
            continue  # zeros = attention not extracted

        # Decode prompt tokens
        prompt = build_tak_prompt(stim.text)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_ids]
        seq_len = min(attn.shape[2], len(prompt_tokens))

        # Classify each token
        token_types = [classify_token(t) for t in prompt_tokens[:seq_len]]

        # Accumulate: for each (layer, head), find top-attended token and classify
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                attn_row = attn[l, h, :seq_len]
                if attn_row.sum() < 0.01:
                    continue
                top_idx = int(np.argmax(attn_row))
                if top_idx < len(token_types):
                    tok_type = token_types[top_idx]
                    type_counts[l, h, type_map[tok_type]] += 1

                    # Record for top-token analysis (mid-layers)
                    if int(0.25 * N_LAYERS) <= l <= int(0.75 * N_LAYERS):
                        tok_str = prompt_tokens[top_idx].strip()
                        top_tokens_by_layer[l].append((tok_str, stim.emotion))

        n_stimuli_processed += 1

    console.print(f"  {set_name}: processed {n_stimuli_processed} stimuli with attention data")

    # Normalize type counts to proportions
    totals = type_counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    type_props = type_counts / totals  # (n_layers, n_heads, 4)

    return type_props, type_counts, top_tokens_by_layer


console.print(f"\n[bold cyan]Analyzing attention patterns...[/bold cyan]")
console.print("  Set A:")
seta_props, seta_counts, seta_top_tokens = analyze_attention_set(SETA_DIR, seta_stims, "Set A")
console.print("  Set B:")
setb_props, setb_counts, setb_top_tokens = analyze_attention_set(SETB_DIR, setb_stims, "Set B")

# ============================================================================
# EMOTION KEYWORD ATTENTION — per layer average
# ============================================================================

# emotion_keyword proportion averaged across all heads, per layer
seta_emo_kwprop = seta_props[:, :, 0].mean(axis=1)  # (n_layers,)
setb_emo_kwprop = setb_props[:, :, 0].mean(axis=1)

console.print(f"\n[bold cyan]Emotion keyword attention proportion:[/bold cyan]")
for l in range(N_LAYERS):
    console.print(f"  L{l:02d}: Set A={seta_emo_kwprop[l]:.3f}  Set B={setb_emo_kwprop[l]:.3f}")

# Head sensitivity: which heads show highest emotion_keyword attention in Set A?
seta_head_emo = seta_props[:, :, 0]  # (n_layers, n_heads)
head_sensitivity = []
for l in range(N_LAYERS):
    for h in range(N_HEADS):
        head_sensitivity.append({
            "layer": l, "head": h,
            "norm_depth": round((l+1)/N_LAYERS, 4),
            "seta_emo_kw_prop": float(seta_props[l, h, 0]),
            "setb_emo_kw_prop": float(setb_props[l, h, 0]),
            "emo_sensitivity": float(seta_props[l, h, 0] - setb_props[l, h, 0]),
        })

head_df = pl.DataFrame(head_sensitivity)
head_df.write_csv(OUT_DIR / f"head_sensitivity_{MODEL_KEY}.csv")
console.print(f"\n[bold green]✓ head_sensitivity_{MODEL_KEY}.csv saved[/bold green]")

top_sensitive_heads = head_df.sort("emo_sensitivity", descending=True).head(10)
table = Table(title="Top 10 Most Emotion-Sensitive Heads (Set A − Set B)")
table.add_column("Layer"); table.add_column("Head"); table.add_column("Set A emo_kw %");
table.add_column("Set B emo_kw %"); table.add_column("Sensitivity")
for row in top_sensitive_heads.iter_rows(named=True):
    table.add_row(str(row["layer"]), str(row["head"]),
                  f"{row['seta_emo_kw_prop']:.3f}", f"{row['setb_emo_kw_prop']:.3f}",
                  f"{row['emo_sensitivity']:.3f}")
console.print(table)

# ============================================================================
# TOP TOKENS PER LAYER — for mid-layers
# ============================================================================

top_token_rows = []
for l in range(4, 13):
    for set_name, top_tokens in [("seta", seta_top_tokens), ("setb", setb_top_tokens)]:
        token_counter = Counter(tok for tok, emo in top_tokens.get(l, []))
        for tok, count in token_counter.most_common(5):
            top_token_rows.append({
                "layer": l, "norm_depth": round((l+1)/N_LAYERS, 4),
                "stimulus_set": set_name,
                "token": tok, "count": count,
                "token_type": classify_token(tok),
            })

pl.DataFrame(top_token_rows).write_csv(OUT_DIR / f"attention_top_tokens_{MODEL_KEY}.csv")
console.print(f"[bold green]✓ attention_top_tokens_{MODEL_KEY}.csv saved[/bold green]")

# ============================================================================
# VISUALIZATION
# ============================================================================

console.print("\n[bold cyan]Generating attention figures...[/bold cyan]")

norm_depths = np.array([(l+1)/N_LAYERS for l in range(N_LAYERS)])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"Attention Pattern Analysis — {MODEL_KEY}", fontsize=13)

type_names = ["emotion_keyword", "context_word", "function_word", "punctuation"]
type_colors = ["#e74c3c", "#2980b9", "#bdc3c7", "#7f8c8d"]

# Plot 1: Token type distribution per layer (Set A)
ax = axes[0, 0]
bottoms = np.zeros(N_LAYERS)
for t_idx, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
    vals = seta_props[:, :, t_idx].mean(axis=1)
    ax.bar(norm_depths, vals, bottom=bottoms, color=tcolor, label=tname, width=0.04)
    bottoms += vals
ax.set_title("Set A — Token type distribution at ':'", fontsize=10)
ax.set_xlabel("Normalized depth"); ax.set_ylabel("Proportion")
ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(0, 1)

# Plot 2: Token type distribution per layer (Set B)
ax = axes[0, 1]
bottoms = np.zeros(N_LAYERS)
for t_idx, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
    vals = setb_props[:, :, t_idx].mean(axis=1)
    ax.bar(norm_depths, vals, bottom=bottoms, color=tcolor, label=tname, width=0.04)
    bottoms += vals
ax.set_title("Set B — Token type distribution at ':'", fontsize=10)
ax.set_xlabel("Normalized depth"); ax.set_ylabel("Proportion")
ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(0, 1)

# Plot 3: Emotion keyword attention — Set A vs Set B
ax = axes[1, 0]
ax.plot(norm_depths, seta_emo_kwprop, color="#e74c3c", linewidth=2.5,
        marker="o", markersize=5, label="Set A (keyword-rich)")
ax.plot(norm_depths, setb_emo_kwprop, color="#2980b9", linewidth=2.5,
        marker="s", markersize=5, linestyle="--", label="Set B (clinical)")
ax.set_title("Emotion keyword attention proportion", fontsize=10)
ax.set_xlabel("Normalized depth"); ax.set_ylabel("Prop of heads attending to emo keyword")
ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

# Plot 4: Head sensitivity scatter (which heads are most emotion-sensitive)
ax = axes[1, 1]
peak_l = int(np.argmax(seta_emo_kwprop))
ax.scatter(seta_head_emo[peak_l, :], setb_props[peak_l, :, 0],
           c=np.arange(N_HEADS), cmap="viridis", s=60)
ax.axline((0, 0), slope=1, linestyle=":", color="gray", alpha=0.5)
ax.set_title(f"Head sensitivity at peak layer L{peak_l}", fontsize=10)
ax.set_xlabel("Set A emo_kw attention"); ax.set_ylabel("Set B emo_kw attention")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
for h in range(N_HEADS):
    if seta_head_emo[peak_l, h] > 0.3:
        ax.annotate(f"H{h}", (seta_head_emo[peak_l, h], setb_props[peak_l, h, 0]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")

plt.tight_layout()
plt.savefig(OUT_DIR / f"attention_heatmap_{MODEL_KEY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"attention_heatmap_{MODEL_KEY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ attention_heatmap_{MODEL_KEY}.png/.svg saved[/bold green]")

# ============================================================================
# README
# ============================================================================

peak_seta_l = int(np.argmax(seta_emo_kwprop))
peak_setb_l = int(np.argmax(setb_emo_kwprop))
peak_diff = float(seta_emo_kwprop[peak_seta_l] - setb_emo_kwprop[peak_seta_l])

readme = f"""# Experiment 15 — v2 Attention Pattern Analysis

**Date:** 2026-02-25
**Model:** {MODEL_KEY} (Llama-3.2-1B-Instruct)

## Research Question

What does the ":" token attend to in Set A (keyword-rich) vs Set B (clinical)?

## Key Findings

| Metric | Set A | Set B | Difference |
|--------|-------|-------|------------|
| Peak emotion keyword attention layer | L{peak_seta_l} ({seta_emo_kwprop[peak_seta_l]:.3f}) | L{peak_setb_l} ({setb_emo_kwprop[peak_setb_l]:.3f}) | {peak_diff:.3f} |

### Top sensitivity heads (L{peak_seta_l}):
See `outputs/head_sensitivity_{MODEL_KEY}.csv` for full head-by-head breakdown.

## Interpretation

Set A shows {'higher' if peak_diff > 0 else 'similar'} emotion keyword attention than Set B at peak layer.
This {'confirms' if peak_diff > 0.1 else 'does not confirm'} that Set A attention is driven partly by lexical emotion words,
while Set B relies {'more on contextual cues' if peak_diff > 0.1 else 'on similar mechanisms'}.

## Outputs

- `outputs/head_sensitivity_{MODEL_KEY}.csv` — per-head emotion sensitivity
- `outputs/attention_top_tokens_{MODEL_KEY}.csv` — top-attended tokens per layer
- `outputs/attention_heatmap_{MODEL_KEY}.png/.svg` — visualizations

## Methodology

1. Load attention patterns from .npz files (attn array: n_layers × n_heads × seq_len)
2. For each (layer, head): identify top-attended token, classify it
3. Token types: emotion_keyword, context_word, function_word, punctuation
4. Head sensitivity = Set A emo_kw proportion − Set B emo_kw proportion
"""

(EXP_DIR / "README.md").write_text(readme)
console.print(f"[bold green]✓ README.md written[/bold green]")
console.print(f"\n[bold green]✓ Experiment 15 complete![/bold green]")
