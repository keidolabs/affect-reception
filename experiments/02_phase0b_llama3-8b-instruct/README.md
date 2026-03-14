# Phase 0b — meta-llama/Meta-Llama-3-8B-Instruct

**Date:** 2026-02-24
**Model:** meta-llama/Meta-Llama-3-8B-Instruct (32 layers, d_model=4096)
**Stimuli:** Set A standard (n=90: 80 emotional + 10 neutral)

## Gate Decision: PASS ✓

**Gate: PASS** — Emotion information is linearly decodable from the residual stream.

**Best layer: 30** — AUROC = **0.9804** ± 0.0128
- Chance: 0.125 | Improvement: 0.8554 (684.3%)
- Silhouette at best layer (PCA-10): 0.0717

## Layer-wise AUROC

| Layer | Mean AUROC | Std |
|-------|-----------|-----|
|  0 | 0.6937 | 0.0394 |
|  1 | 0.6446 | 0.0527 |
|  2 | 0.6902 | 0.0351 |
|  3 | 0.8946 | 0.0323 |
|  4 | 0.9223 | 0.0269 |
|  5 | 0.9250 | 0.0290 |
|  6 | 0.9411 | 0.0332 |
|  7 | 0.9384 | 0.0306 |
|  8 | 0.9339 | 0.0426 |
|  9 | 0.9375 | 0.0409 |
| 10 | 0.9286 | 0.0399 |
| 11 | 0.8991 | 0.0483 |
| 12 | 0.9152 | 0.0426 |
| 13 | 0.9366 | 0.0276 |
| 14 | 0.9402 | 0.0161 |
| 15 | 0.9482 | 0.0197 |
| 16 | 0.9580 | 0.0146 |
| 17 | 0.9616 | 0.0061 |
| 18 | 0.9705 | 0.0092 |
| 19 | 0.9670 | 0.0092 |
| 20 | 0.9643 | 0.0113 |
| 21 | 0.9661 | 0.0115 |
| 22 | 0.9723 | 0.0111 |
| 23 | 0.9679 | 0.0131 |
| 24 | 0.9741 | 0.0111 |
| 25 | 0.9759 | 0.0092 |
| 26 | 0.9768 | 0.0107 |
| 27 | 0.9795 | 0.0115 |
| 28 | 0.9768 | 0.0121 |
| 29 | 0.9777 | 0.0138 |
| 30 | 0.9804 | 0.0128 | **← best**
| 31 | 0.9688 | 0.0160 |


## Methodology

- **Framework:** HF `AutoModelForCausalLM` + MPS (float16) — NOT TransformerLens
- **Extraction:** Final-token residual stream (`register_forward_hook` on `model.model.layers[l]`) at all 32 layers
- **Probe:** 5-fold stratified CV, `LogisticRegression` (OvR, C=1.0, max_iter=1000), macro AUROC
- **Probe normalization:** None (raw activations passed to classifier)
- **Seeds:** 42 everywhere

## Outputs

- `outputs/activations/set_a_residuals.npy` — shape (90, 32, 4096)
- `outputs/results.csv` — per-layer AUROC
- `outputs/auroc_curve.png/.svg`
- `outputs/pca_scatter.png/.svg`
- `outputs/tsne_scatter.png/.svg`