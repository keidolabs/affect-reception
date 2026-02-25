# Phase 0 — Cross-Model Comparison

**Date:** 2026-02-24
**Stimuli:** Set A standard (n=90: 80 emotional + 10 neutral)
**Method:** Layer-wise linear probes, 8-class OvR, 5-fold CV, macro AUROC

## Summary

| Model | Layers | d_model | Best Layer | Rel. Depth | Best AUROC | Gate |
|-------|--------|---------|-----------|-----------|-----------|------|
| Llama-3.2-1B | 16 | 2048 | 14 | 0.93 | 0.9875 | ✓ PASS |
| Llama-3-8B (base) | 32 | 4096 | 25 | 0.81 | 0.9938 | ✓ PASS |
| Llama-3-8B (instruct) | 32 | 4096 | 30 | 0.97 | 0.9804 | ✓ PASS |
| Gemma-2-9B | 42 | 3584 | 39 | 0.95 | 0.9884 | ✓ PASS |

## Key Observations

- All models trained on keyword-rich stimuli (Set A) — probes have the easiest possible task
- Best layer depth (normalized) varies by model — compare across architectures in the AUROC plot
- Base vs Instruct delta shows where RLHF fine-tuning reshapes emotion representations
- Per-class comparison shows which emotions decode most reliably across all model families

## Outputs

- `outputs/auroc_comparison.png/.svg` — overlaid AUROC curves (absolute + normalized)
- `outputs/auroc_comparison.html` — interactive version with hover
- `outputs/base_vs_instruct_delta.png/.svg` — base → instruct AUROC delta
- `outputs/base_vs_instruct_delta.html` — interactive version
- `outputs/per_class_auroc_comparison.html` — per-class grouped bar (interactive)