# Experiment 15 — v2 Attention Pattern Analysis

**Date:** 2026-02-25
**Model:** llama1b_inst (Llama-3.2-1B-Instruct)

## Research Question

What does the ":" token attend to in Set A (keyword-rich) vs Set B (clinical)?

## Key Findings

| Metric | Set A | Set B | Difference |
|--------|-------|-------|------------|
| Peak emotion keyword attention layer | L9 (0.192) | L9 (0.227) | -0.035 |

### Top sensitivity heads (L9):
See `outputs/head_sensitivity_llama1b_inst.csv` for full head-by-head breakdown.

## Interpretation

Set A shows similar emotion keyword attention than Set B at peak layer.
This does not confirm that Set A attention is driven partly by lexical emotion words,
while Set B relies on similar mechanisms.

## Outputs

- `outputs/head_sensitivity_llama1b_inst.csv` — per-head emotion sensitivity
- `outputs/attention_top_tokens_llama1b_inst.csv` — top-attended tokens per layer
- `outputs/attention_heatmap_llama1b_inst.png/.svg` — visualizations

## Methodology

1. Load attention patterns from .npz files (attn array: n_layers × n_heads × seq_len)
2. For each (layer, head): identify top-attended token, classify it
3. Token types: emotion_keyword, context_word, function_word, punctuation
4. Head sensitivity = Set A emo_kw proportion − Set B emo_kw proportion
