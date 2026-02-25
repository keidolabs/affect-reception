# Phase 0c — google/gemma-2-9b

**Date:** 2026-02-24
**Model:** google/gemma-2-9b (42 layers, d_model=3584)
**Note:** Gemma-2-9b uses alternating global/local attention (even layers = local sliding window).

## Gate Decision: PASS ✓

**Gate: PASS** — Emotion information is linearly decodable from the Gemma-2 residual stream.

**Best layer: 39** (global attention) — AUROC = **0.9884** ± 0.0092
- Chance: 0.125 | Improvement: 0.8634 (690.7%)
- Silhouette at best layer (PCA-10): 0.0850

## Layer-wise AUROC

| Layer | Mean AUROC | Std | Attn type |
|-------|-----------|-----|----------|
|  0 | 0.6286 | 0.0581 | local |
|  1 | 0.7875 | 0.0676 | global |
|  2 | 0.7616 | 0.0572 | local |
|  3 | 0.8054 | 0.0634 | global |
|  4 | 0.8464 | 0.0364 | local |
|  5 | 0.8634 | 0.0273 | global |
|  6 | 0.8866 | 0.0513 | local |
|  7 | 0.8732 | 0.0305 | global |
|  8 | 0.9045 | 0.0445 | local |
|  9 | 0.9268 | 0.0416 | global |
| 10 | 0.9527 | 0.0336 | local |
| 11 | 0.9500 | 0.0312 | global |
| 12 | 0.9491 | 0.0275 | local |
| 13 | 0.9464 | 0.0293 | global |
| 14 | 0.9330 | 0.0235 | local |
| 15 | 0.9304 | 0.0356 | global |
| 16 | 0.9241 | 0.0311 | local |
| 17 | 0.9357 | 0.0334 | global |
| 18 | 0.9402 | 0.0344 | local |
| 19 | 0.9366 | 0.0426 | global |
| 20 | 0.9411 | 0.0421 | local |
| 21 | 0.9473 | 0.0319 | global |
| 22 | 0.9545 | 0.0295 | local |
| 23 | 0.9634 | 0.0208 | global |
| 24 | 0.9616 | 0.0201 | local |
| 25 | 0.9768 | 0.0139 | global |
| 26 | 0.9768 | 0.0206 | local |
| 27 | 0.9777 | 0.0246 | global |
| 28 | 0.9795 | 0.0199 | local |
| 29 | 0.9821 | 0.0157 | global |
| 30 | 0.9768 | 0.0177 | local |
| 31 | 0.9795 | 0.0151 | global |
| 32 | 0.9804 | 0.0146 | local |
| 33 | 0.9812 | 0.0142 | global |
| 34 | 0.9839 | 0.0122 | local |
| 35 | 0.9875 | 0.0087 | global |
| 36 | 0.9875 | 0.0091 | local |
| 37 | 0.9866 | 0.0102 | global |
| 38 | 0.9839 | 0.0143 | local |
| 39 | 0.9884 | 0.0092 | global | **← best**
| 40 | 0.9848 | 0.0108 | local |
| 41 | 0.9812 | 0.0156 | global |


## Outputs

- `outputs/activations/set_a_residuals.npy` — shape (90, 42, 3584)
- `outputs/results.csv` — per-layer AUROC
- `outputs/auroc_curve.png/.svg`
- `outputs/pca_scatter.png/.svg`
- `outputs/tsne_scatter.png/.svg`