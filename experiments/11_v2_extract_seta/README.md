# Experiment 11 — v2 Activation Extraction: Set A (Keyword-Rich Emotional Text)

**Date:** 2026-02-25
**Stimulus set:** Set A — 80 emotional stimuli (8 Plutchik emotions × 10 stimuli each, neutral excluded)
**Models:** 6 (Llama-3.2-1B base/inst, Llama-3-8B base/inst, Gemma-2-9B base/inst)
**Output:** Per-stimulus `.npz` files containing residual stream, attention, and FFN activations at every layer

---

## Research Question

What activations should we capture to characterize how each model encodes emotion?

This experiment is pure data collection — no analysis. It extracts the raw activation arrays that all downstream experiments (Exp 13–17) consume.

---

## Extraction Design

### Prompt Format (TAK — Task-Anchored Knowledge)

Each stimulus is wrapped in a two-shot prompt:

```
What are the inferred emotions in the following contexts?

Context: My dog died last week. I miss him every day.
Answer: sadness

Context: I got promoted and my boss praised my work in front of everyone.
Answer: joy

Context: {stimulus.text}
Answer:
```

The model is queried for the next token after `"Answer:"`. We extract activations at this final token position (the colon token, index `n_tokens - 1`). This captures the model's internal state at the moment of emotion classification — after reading the full stimulus and before generating the label.

**Why this position?** The final token position integrates the full context. At this position, the residual stream represents the model's "readout" of the input, making it the most informative position for probing what emotion has been encoded.

**Why two-shot?** Zero-shot probing (Exp 19) provides the ablation. The two-shot format gives the model a clear task framing and produces more stable, interpretable activations.

### What Is Extracted

For each stimulus, at each layer `l`, we capture three activation types via `register_forward_hook`:

| Array | Hook target | Shape | Meaning |
|-------|------------|-------|---------|
| `h` | `model.model.layers[l]` (output) | `(n_layers, d_model)` | Residual stream after full layer (h = a + m + residual) |
| `a` | `model.model.layers[l].self_attn` | `(n_layers, d_model)` | Multi-head self-attention output only |
| `m` | `model.model.layers[l].mlp` | `(n_layers, d_model)` | Feed-forward network output only |
| `attn` | `output_attentions=True` | `(n_layers, n_heads, n_tokens)` | Attention weights at the final token position |

`h`, `a`, `m` are float16→float32 at final token. `attn` is per-head attention over the input sequence at the classification position.

The h/a/m decomposition lets Exp 13 ask: *which component carries the emotion signal?* (Answer: primarily `h`, meaning the residual stream integrates information across both components.)

### Output Format

Each stimulus produces one `.npz` file:

```
outputs/activations/{model_key}/{stimulus_id}.npz
  h     : float32, shape (n_layers, d_model)
  a     : float32, shape (n_layers, d_model)
  m     : float32, shape (n_layers, d_model)
  attn  : float32, shape (n_layers, n_heads, n_prompt_tokens)
  metadata : JSON string with id, emotion, top5_predictions, correct, n_tokens_prompt, ...
```

Plus a `manifest.csv` in the same directory with one row per stimulus.

---

## Model Specifications

| Model key | HuggingFace ID | Layers | d_model | Heads |
|-----------|---------------|--------|---------|-------|
| `llama1b_base` | `meta-llama/Llama-3.2-1B` | 16 | 2048 | 32 |
| `llama1b_inst` | `meta-llama/Llama-3.2-1B-Instruct` | 16 | 2048 | 32 |
| `llama8b_base` | `meta-llama/Meta-Llama-3-8B` | 32 | 4096 | 32 |
| `llama8b_inst` | `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 4096 | 32 |
| `gemma9b_base` | `google/gemma-2-9b` | 42 | 3584 | 16 |
| `gemma9b_inst` | `google/gemma-2-9b-it` | 42 | 3584 | 16 |

---

## How to Run

### Prerequisites

```bash
# HuggingFace token with access to gated models
cp .env.sample .env  # then set HF_ACCESS_TOKEN=hf_...

# Optional: force device (default: cuda if available, else mps)
export EMO_DEVICE=cuda   # or mps, or cpu
```

### Run all 6 models (recommended)

```bash
uv run python experiments/11_v2_extract_seta/run.py
```

The orchestrator runs models sequentially, skipping any already completed (checks for `manifest.csv`). To re-run a specific model:

```bash
uv run python experiments/11_v2_extract_seta/run.py --models llama8b_inst gemma9b_base
```

### Run individual models

```bash
uv run python experiments/11_v2_extract_seta/extract_llama1b_base.py
uv run python experiments/11_v2_extract_seta/extract_llama1b_inst.py
uv run python experiments/11_v2_extract_seta/extract_llama8b_base.py
uv run python experiments/11_v2_extract_seta/extract_llama8b_inst.py
uv run python experiments/11_v2_extract_seta/extract_gemma9b_base.py
uv run python experiments/11_v2_extract_seta/extract_gemma9b_inst.py
```

Each script loads one model, runs 80 forward passes, saves activations, then exits (releasing GPU memory before the next script starts).

---

## Time and Disk Estimates

| Model | GPU time (A6000) | GPU time (M1 MPS) | Disk per model |
|-------|-----------------|------------------|----------------|
| llama1b_base/inst | ~5 min | ~15 min | ~50 MB |
| llama8b_base/inst | ~15 min | ~45 min | ~180 MB |
| gemma9b_base/inst | ~20 min | ~60 min | ~230 MB |
| **Total (6 models)** | **~80 min** | **~4 hrs** | **~920 MB** |

*Estimates for 80 stimuli on Set A. Add ~2.4× for Set B (Exp 12, 192 stimuli).*

---

## Validation Built Into Each Script

Each extraction script runs two sanity checks before processing stimuli:

1. **Final token check** — verifies the prompt ends with a colon token (the position we extract from)
2. **Device validation** — runs 2 factual prompts (`"The Eiffel Tower is in..."`, `"Boiling point of water..."`) and checks the top-5 predictions include the correct answer. Raises `RuntimeError` if the model is producing garbage output.

After extraction, inter-stimulus activation std is checked (`> 0.001`) to confirm activations vary meaningfully across stimuli.

---

## Outputs

```
outputs/activations/
  llama1b_base/
    {stimulus_id}.npz   × 80
    manifest.csv
  llama1b_inst/
    ...
  llama8b_base/
  llama8b_inst/
  gemma9b_base/
  gemma9b_inst/
```

Activation `.npz` files are gitignored (too large). The `manifest.csv` files are tracked and record per-stimulus metadata including model top-5 predictions and whether the correct emotion was in top-5.

---

## Notes

- `attn_implementation="eager"` is set explicitly to ensure `output_attentions=True` works on MPS (the default SDPA implementation does not return attention weights on MPS)
- Each script processes one model then exits — this is intentional. Loading two 8B models simultaneously would OOM on 32GB M1
- `use_cache=False` on all forward passes — required for correct hook behavior
- Gemma-2-9B uses alternating local/global attention (even=local, odd=global); this does not affect residual stream extraction but is relevant for interpreting `attn` arrays in Exp 15
