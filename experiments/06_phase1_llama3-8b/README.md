# Phase 1 — meta-llama/Meta-Llama-3-8B

**Date:** 2026-02-24
**Model:** meta-llama/Meta-Llama-3-8B (32 layers, d_model=4096)
**Stimuli:** Set B (96 clinical vignettes + 96 matched neutral controls)

## Gate Decision: PASS ✓

**Best layer: 26** (norm: 83.9%) — AUROC = **0.9642** ± 0.0233
- **Phase 0 (Set A):** best AUROC = 0.9938 (layer 25/31)
- **Decodability drop:** 0.0295 (3.0%)
- Chance: 0.125 | Gate threshold: 0.40

## Binary Probe (clinical vs neutral)
- Best layer: 9 — AUROC = **1.0000**

## Silhouette at Best Layer
- 0.0417 (PCA-10, emotional stimuli only)

## Token Length Note
Clinical vignettes are longer than neutral controls on average
(clinical: 135 ± 21 tokens vs neutral: 119 ± 15).
37/96 pairs are within ±10% token length threshold.
- Sensitivity analysis skipped (only 37 matched pairs)
See `outputs/token_lengths.csv` for per-pair breakdown.

## Layer-wise AUROC

| Layer | AUROC (8-class) | Std | Binary AUROC | Std |
|-------|----------------|-----|-------------|-----|
|  0 | 0.7296 | 0.0230 | 0.9087 | 0.0515 |
|  1 | 0.7194 | 0.0513 | 0.9356 | 0.0273 |
|  2 | 0.7641 | 0.0703 | 0.9612 | 0.0273 |
|  3 | 0.8379 | 0.0623 | 0.9941 | 0.0076 |
|  4 | 0.8491 | 0.0541 | 0.9968 | 0.0039 |
|  5 | 0.8245 | 0.0675 | 0.9968 | 0.0039 |
|  6 | 0.8160 | 0.0728 | 0.9978 | 0.0020 |
|  7 | 0.8309 | 0.0644 | 0.9978 | 0.0021 |
|  8 | 0.8511 | 0.0516 | 0.9984 | 0.0013 |
|  9 | 0.8698 | 0.0510 | 1.0000 | 0.0000 |
| 10 | 0.8875 | 0.0541 | 1.0000 | 0.0000 |
| 11 | 0.8781 | 0.0556 | 1.0000 | 0.0000 |
| 12 | 0.8882 | 0.0538 | 1.0000 | 0.0000 |
| 13 | 0.9020 | 0.0526 | 1.0000 | 0.0000 |
| 14 | 0.9116 | 0.0433 | 1.0000 | 0.0000 |
| 15 | 0.9202 | 0.0380 | 1.0000 | 0.0000 |
| 16 | 0.9258 | 0.0321 | 1.0000 | 0.0000 |
| 17 | 0.9379 | 0.0290 | 1.0000 | 0.0000 |
| 18 | 0.9456 | 0.0312 | 1.0000 | 0.0000 |
| 19 | 0.9475 | 0.0302 | 1.0000 | 0.0000 |
| 20 | 0.9508 | 0.0319 | 1.0000 | 0.0000 |
| 21 | 0.9562 | 0.0299 | 1.0000 | 0.0000 |
| 22 | 0.9574 | 0.0307 | 1.0000 | 0.0000 |
| 23 | 0.9530 | 0.0326 | 1.0000 | 0.0000 |
| 24 | 0.9544 | 0.0291 | 1.0000 | 0.0000 |
| 25 | 0.9632 | 0.0245 | 1.0000 | 0.0000 |
| 26 | 0.9642 | 0.0233 | 1.0000 | 0.0000 | **← best**
| 27 | 0.9611 | 0.0265 | 1.0000 | 0.0000 |
| 28 | 0.9613 | 0.0259 | 1.0000 | 0.0000 |
| 29 | 0.9620 | 0.0279 | 1.0000 | 0.0000 |
| 30 | 0.9612 | 0.0335 | 1.0000 | 0.0000 |
| 31 | 0.9541 | 0.0405 | 0.9994 | 0.0011 |


## Outputs
- `outputs/activations/set_b_residuals.npy` — shape (192, 32, 4096)
- `outputs/results_8class.csv` — 8-class AUROC by layer
- `outputs/results_binary.csv` — binary AUROC by layer
- `outputs/token_lengths.csv` — per-pair token length comparison
- `outputs/auroc_setA_vs_setB.png/.svg` — Phase 0 vs Phase 1 comparison
- `outputs/auroc_binary.png/.svg` — binary probe
- `outputs/pca_scatter.png/.svg` + `outputs/tsne_scatter.png/.svg`