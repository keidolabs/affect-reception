# Phase 0a — meta-llama/Meta-Llama-3-8B

**Date:** 2026-02-24
**Model:** meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
**Stimuli:** Set A standard (n=90: 80 emotional + 10 neutral)

## Gate Decision: PASS ✓

**Gate: PASS** — Emotion information is linearly decodable from the residual stream.

**Best layer: 25** — AUROC = **0.9938** ± 0.0036
- Chance: 0.125 | Improvement: 0.8688 (695.0%)
- Silhouette at best layer (PCA-10): 0.0885

## Layer-wise AUROC

| Layer | Mean AUROC | Std |
|-------|-----------|-----|
|  0 | 0.7152 | 0.0605 |
|  1 | 0.6804 | 0.0686 |
|  2 | 0.7098 | 0.0389 |
|  3 | 0.9045 | 0.0348 |
|  4 | 0.9250 | 0.0298 |
|  5 | 0.9232 | 0.0303 |
|  6 | 0.9375 | 0.0321 |
|  7 | 0.9455 | 0.0317 |
|  8 | 0.9304 | 0.0425 |
|  9 | 0.9357 | 0.0346 |
| 10 | 0.9277 | 0.0307 |
| 11 | 0.9045 | 0.0480 |
| 12 | 0.9214 | 0.0267 |
| 13 | 0.9455 | 0.0250 |
| 14 | 0.9455 | 0.0177 |
| 15 | 0.9563 | 0.0131 |
| 16 | 0.9705 | 0.0100 |
| 17 | 0.9714 | 0.0122 |
| 18 | 0.9768 | 0.0066 |
| 19 | 0.9750 | 0.0036 |
| 20 | 0.9777 | 0.0056 |
| 21 | 0.9830 | 0.0059 |
| 22 | 0.9812 | 0.0059 |
| 23 | 0.9821 | 0.0063 |
| 24 | 0.9866 | 0.0028 |
| 25 | 0.9938 | 0.0036 | **← best**
| 26 | 0.9920 | 0.0059 |
| 27 | 0.9902 | 0.0033 |
| 28 | 0.9902 | 0.0033 |
| 29 | 0.9857 | 0.0107 |
| 30 | 0.9812 | 0.0177 |
| 31 | 0.9750 | 0.0216 |


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