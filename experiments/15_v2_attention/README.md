# Experiment 15 — v2 Attention Pattern Analysis

**Date:** 2026-02-25
**Models:** All 6 (llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst)
**Framework:** Analysis-only (loads attention weights from Exp 11/12 .npz files, tokenizer only — no model forward passes)
**Note:** Exploratory analysis. Results characterize attention patterns but do not establish causal claims.

## Research Question

What does the ":" token attend to in Set A (keyword-rich) vs Set B (clinical)?

## Methodology

1. Load attention patterns from .npz files (attn array: n_layers × n_heads × seq_len)
2. For each (layer, head): identify top-attended token, classify it
3. Token types: emotion_keyword, context_word, function_word, punctuation
4. Head sensitivity = Set A emo_kw proportion − Set B emo_kw proportion

## Outputs

Per model (`{MODEL_KEY}` = llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst):

- `outputs/head_sensitivity_{MODEL_KEY}.csv` — per-head emotion sensitivity
- `outputs/attention_top_tokens_{MODEL_KEY}.csv` — top-attended tokens per layer
- `outputs/attention_heatmap_{MODEL_KEY}.png/.svg` — visualizations
