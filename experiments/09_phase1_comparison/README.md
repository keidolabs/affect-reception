# Phase 1 Cross-Model Comparison

**Date:** 2026-02-24
**Question:** Does emotion category information survive in the residual stream
when keyword-rich stimuli (Set A) are replaced by clinical vignettes (Set B)?

## Summary

| Model | P0 AUROC | P0 peak (norm) | P1 AUROC | P1 peak (norm) | Gap | Gap % |
|-------|---------|---------------|---------|---------------|-----|-------|
| Llama-3.2-1B | 0.9875 | 93.3% | 0.9075 | 100.0% | 0.0800 | 8.1% |
| Llama-3-8B (base) | 0.9938 | 80.7% | 0.9642 | 83.9% | 0.0295 | 3.0% |
| Llama-3-8B (instruct) | 0.9804 | 96.8% | 0.9479 | 90.3% | 0.0325 | 3.3% |
| Gemma-2-9B | 0.9884 | 95.1% | 0.9458 | 70.7% | 0.0425 | 4.3% |


## Outputs

- `outputs/phase0_vs_phase1_panel.png/.svg` — 2×2 per-model overlay
- `outputs/decodability_gap.png/.svg` — gap by normalized depth, all models
- `outputs/summary.csv` — machine-readable summary table