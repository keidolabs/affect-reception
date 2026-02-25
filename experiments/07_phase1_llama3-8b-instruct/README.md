# Phase 1 — meta-llama/Meta-Llama-3-8B-Instruct

**Date:** 2026-02-24
**Model:** meta-llama/Meta-Llama-3-8B-Instruct (32 layers, d_model=4096)
**Stimuli:** Set B (96 clinical vignettes + 96 matched neutral controls)

## Gate Decision: PASS ✓

**Best layer: 28** (norm: 90.3%) — AUROC = **0.9479** ± 0.0258
- **Phase 0 (Set A):** best AUROC = 0.9804 (layer 30/31)
- **Decodability drop:** 0.0325 (3.3%)
- Chance: 0.125 | Gate threshold: 0.40

## Binary Probe (clinical vs neutral)
- Best layer: 9 — AUROC = **1.0000**

## Silhouette at Best Layer
- 0.0539 (PCA-10, emotional stimuli only)

## Token Length Note
Clinical vignettes are longer than neutral controls on average
(clinical: 135 ± 21 tokens vs neutral: 119 ± 15).
37/96 pairs are within ±10% token length threshold.
- Sensitivity analysis skipped (only 37 matched pairs)
See `outputs/token_lengths.csv` for per-pair breakdown.

## Layer-wise AUROC

| Layer | AUROC (8-class) | Std | Binary AUROC | Std |
|-------|----------------|-----|-------------|-----|
|  0 | 0.6994 | 0.0429 | 0.8959 | 0.0497 |
|  1 | 0.6785 | 0.0491 | 0.9237 | 0.0234 |
|  2 | 0.7383 | 0.0862 | 0.9640 | 0.0234 |
|  3 | 0.8290 | 0.0606 | 0.9952 | 0.0052 |
|  4 | 0.8343 | 0.0502 | 0.9973 | 0.0041 |
|  5 | 0.8240 | 0.0573 | 0.9973 | 0.0041 |
|  6 | 0.8080 | 0.0660 | 0.9984 | 0.0021 |
|  7 | 0.8114 | 0.0533 | 0.9978 | 0.0021 |
|  8 | 0.8373 | 0.0454 | 0.9989 | 0.0013 |
|  9 | 0.8524 | 0.0435 | 1.0000 | 0.0000 |
| 10 | 0.8698 | 0.0540 | 1.0000 | 0.0000 |
| 11 | 0.8727 | 0.0549 | 1.0000 | 0.0000 |
| 12 | 0.8822 | 0.0550 | 1.0000 | 0.0000 |
| 13 | 0.9008 | 0.0477 | 1.0000 | 0.0000 |
| 14 | 0.9114 | 0.0383 | 1.0000 | 0.0000 |
| 15 | 0.9128 | 0.0360 | 1.0000 | 0.0000 |
| 16 | 0.9155 | 0.0320 | 1.0000 | 0.0000 |
| 17 | 0.9217 | 0.0281 | 1.0000 | 0.0000 |
| 18 | 0.9213 | 0.0353 | 1.0000 | 0.0000 |
| 19 | 0.9270 | 0.0346 | 1.0000 | 0.0000 |
| 20 | 0.9304 | 0.0280 | 1.0000 | 0.0000 |
| 21 | 0.9261 | 0.0308 | 1.0000 | 0.0000 |
| 22 | 0.9274 | 0.0288 | 1.0000 | 0.0000 |
| 23 | 0.9396 | 0.0274 | 1.0000 | 0.0000 |
| 24 | 0.9383 | 0.0227 | 1.0000 | 0.0000 |
| 25 | 0.9425 | 0.0288 | 1.0000 | 0.0000 |
| 26 | 0.9450 | 0.0253 | 1.0000 | 0.0000 |
| 27 | 0.9435 | 0.0267 | 1.0000 | 0.0000 |
| 28 | 0.9479 | 0.0258 | 1.0000 | 0.0000 | **← best**
| 29 | 0.9467 | 0.0307 | 0.9994 | 0.0011 |
| 30 | 0.9445 | 0.0337 | 0.9984 | 0.0021 |
| 31 | 0.9447 | 0.0345 | 0.9984 | 0.0021 |


## Outputs
- `outputs/activations/set_b_residuals.npy` — shape (192, 32, 4096)
- `outputs/results_8class.csv` — 8-class AUROC by layer
- `outputs/results_binary.csv` — binary AUROC by layer
- `outputs/token_lengths.csv` — per-pair token length comparison
- `outputs/auroc_setA_vs_setB.png/.svg` — Phase 0 vs Phase 1 comparison
- `outputs/auroc_binary.png/.svg` — binary probe
- `outputs/pca_scatter.png/.svg` + `outputs/tsne_scatter.png/.svg`