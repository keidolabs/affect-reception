# Phase 1 — google/gemma-2-9b

**Date:** 2026-02-24
**Model:** google/gemma-2-9b (42 layers, d_model=3584)
**Stimuli:** Set B (96 clinical vignettes + 96 matched neutral controls)

## Gate Decision: PASS ✓

**Best layer: 29** (norm: 70.7%) — AUROC = **0.9458** ± 0.0288
- **Phase 0 (Set A):** best AUROC = 0.9884 (layer 39/41)
- **Decodability drop:** 0.0425 (4.3%)
- Chance: 0.125 | Gate threshold: 0.40

## Binary Probe (clinical vs neutral)
- Best layer: 17 — AUROC = **1.0000**

## Silhouette at Best Layer
- 0.0400 (PCA-10, emotional stimuli only)

## Token Length Note
Clinical vignettes are longer than neutral controls on average
(clinical: 136 ± 21 tokens vs neutral: 120 ± 15).
37/96 pairs are within ±10% token length threshold.
- Sensitivity analysis skipped (only 37 matched pairs)
See `outputs/token_lengths.csv` for per-pair breakdown.

## Layer-wise AUROC

| Layer | AUROC (8-class) | Std | Binary AUROC | Std |
|-------|----------------|-----|-------------|-----|
|  0 | 0.7091 | 0.0551 | 0.9308 | 0.0171 |
|  1 | 0.8433 | 0.0773 | 0.9780 | 0.0110 |
|  2 | 0.8143 | 0.0765 | 0.9729 | 0.0088 |
|  3 | 0.8217 | 0.0630 | 0.9839 | 0.0104 |
|  4 | 0.8525 | 0.0593 | 0.9920 | 0.0088 |
|  5 | 0.8195 | 0.0463 | 0.9920 | 0.0086 |
|  6 | 0.7949 | 0.0447 | 0.9952 | 0.0056 |
|  7 | 0.7749 | 0.0616 | 0.9930 | 0.0035 |
|  8 | 0.7876 | 0.0408 | 0.9973 | 0.0018 |
|  9 | 0.7967 | 0.0596 | 0.9973 | 0.0017 |
| 10 | 0.8211 | 0.0573 | 0.9978 | 0.0021 |
| 11 | 0.8127 | 0.0543 | 0.9978 | 0.0021 |
| 12 | 0.8177 | 0.0562 | 0.9978 | 0.0021 |
| 13 | 0.8196 | 0.0502 | 0.9973 | 0.0017 |
| 14 | 0.8378 | 0.0495 | 0.9973 | 0.0024 |
| 15 | 0.8381 | 0.0551 | 0.9978 | 0.0021 |
| 16 | 0.8277 | 0.0513 | 0.9989 | 0.0013 |
| 17 | 0.8662 | 0.0545 | 1.0000 | 0.0000 |
| 18 | 0.8842 | 0.0561 | 1.0000 | 0.0000 |
| 19 | 0.9050 | 0.0404 | 1.0000 | 0.0000 |
| 20 | 0.9049 | 0.0460 | 1.0000 | 0.0000 |
| 21 | 0.9169 | 0.0354 | 1.0000 | 0.0000 |
| 22 | 0.9164 | 0.0442 | 1.0000 | 0.0000 |
| 23 | 0.9087 | 0.0492 | 1.0000 | 0.0000 |
| 24 | 0.9199 | 0.0398 | 1.0000 | 0.0000 |
| 25 | 0.9329 | 0.0306 | 1.0000 | 0.0000 |
| 26 | 0.9392 | 0.0283 | 1.0000 | 0.0000 |
| 27 | 0.9365 | 0.0332 | 1.0000 | 0.0000 |
| 28 | 0.9309 | 0.0379 | 1.0000 | 0.0000 |
| 29 | 0.9458 | 0.0288 | 1.0000 | 0.0000 | **← best**
| 30 | 0.9401 | 0.0278 | 1.0000 | 0.0000 |
| 31 | 0.9327 | 0.0317 | 1.0000 | 0.0000 |
| 32 | 0.9289 | 0.0330 | 1.0000 | 0.0000 |
| 33 | 0.9242 | 0.0293 | 1.0000 | 0.0000 |
| 34 | 0.9268 | 0.0273 | 1.0000 | 0.0000 |
| 35 | 0.9270 | 0.0270 | 1.0000 | 0.0000 |
| 36 | 0.9246 | 0.0324 | 1.0000 | 0.0000 |
| 37 | 0.9267 | 0.0360 | 1.0000 | 0.0000 |
| 38 | 0.9313 | 0.0311 | 1.0000 | 0.0000 |
| 39 | 0.9324 | 0.0330 | 1.0000 | 0.0000 |
| 40 | 0.9276 | 0.0408 | 0.9995 | 0.0011 |
| 41 | 0.9260 | 0.0338 | 0.9995 | 0.0011 |


## Outputs
- `outputs/activations/set_b_residuals.npy` — shape (192, 42, 3584)
- `outputs/results_8class.csv` — 8-class AUROC by layer
- `outputs/results_binary.csv` — binary AUROC by layer
- `outputs/token_lengths.csv` — per-pair token length comparison
- `outputs/auroc_setA_vs_setB.png/.svg` — Phase 0 vs Phase 1 comparison
- `outputs/auroc_binary.png/.svg` — binary probe
- `outputs/pca_scatter.png/.svg` + `outputs/tsne_scatter.png/.svg`