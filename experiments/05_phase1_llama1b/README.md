# Phase 1 — meta-llama/Llama-3.2-1B

**Date:** 2026-02-24
**Model:** meta-llama/Llama-3.2-1B (16 layers, d_model=2048)
**Stimuli:** Set B (96 clinical vignettes + 96 matched neutral controls)

## Gate Decision: PASS ✓

**Best layer: 15** (norm: 100.0%) — AUROC = **0.9075** ± 0.0569
- **Phase 0 (Set A):** best AUROC = 0.9875 (layer 14/15)
- **Decodability drop:** 0.0800 (8.1%)
- Chance: 0.125 | Gate threshold: 0.40

## Binary Probe (clinical vs neutral)
- Best layer: 10 — AUROC = **0.9995**

## Silhouette at Best Layer
- -0.0014 (PCA-10, emotional stimuli only)

## Token Length Note
Clinical vignettes are longer than neutral controls on average
(clinical: 135 ± 21 tokens vs neutral: 119 ± 15).
37/96 pairs are within ±10% token length threshold.
- Sensitivity analysis skipped (only 37 matched pairs)
See `outputs/token_lengths.csv` for per-pair breakdown.

## Layer-wise AUROC

| Layer | AUROC (8-class) | Std | Binary AUROC | Std |
|-------|----------------|-----|-------------|-----|
|  0 | 0.7089 | 0.0296 | 0.9112 | 0.0340 |
|  1 | 0.7530 | 0.0463 | 0.9558 | 0.0309 |
|  2 | 0.8020 | 0.0658 | 0.9822 | 0.0140 |
|  3 | 0.8341 | 0.0804 | 0.9936 | 0.0048 |
|  4 | 0.8258 | 0.0777 | 0.9962 | 0.0021 |
|  5 | 0.8105 | 0.0857 | 0.9962 | 0.0022 |
|  6 | 0.8272 | 0.0579 | 0.9961 | 0.0038 |
|  7 | 0.8485 | 0.0549 | 0.9983 | 0.0022 |
|  8 | 0.8488 | 0.0614 | 0.9978 | 0.0021 |
|  9 | 0.8739 | 0.0518 | 0.9989 | 0.0013 |
| 10 | 0.8863 | 0.0512 | 0.9995 | 0.0011 |
| 11 | 0.8956 | 0.0551 | 0.9995 | 0.0011 |
| 12 | 0.9050 | 0.0409 | 0.9995 | 0.0011 |
| 13 | 0.9063 | 0.0410 | 0.9989 | 0.0021 |
| 14 | 0.9022 | 0.0483 | 0.9989 | 0.0021 |
| 15 | 0.9075 | 0.0569 | 0.9984 | 0.0022 | **← best**


## Outputs
- `outputs/activations/set_b_residuals.npy` — shape (192, 16, 2048)
- `outputs/results_8class.csv` — 8-class AUROC by layer
- `outputs/results_binary.csv` — binary AUROC by layer
- `outputs/token_lengths.csv` — per-pair token length comparison
- `outputs/auroc_setA_vs_setB.png/.svg` — Phase 0 vs Phase 1 comparison
- `outputs/auroc_binary.png/.svg` — binary probe
- `outputs/pca_scatter.png/.svg` + `outputs/tsne_scatter.png/.svg`