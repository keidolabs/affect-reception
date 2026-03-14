# Experiment 17 — v2 Representational Geometry

**Date:** 2026-02-25
**Models:** All 6 (llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst)
**Framework:** Analysis-only (loads Exp 11/12 activations at peak layer from Exp 13)

## Research Questions

1. Do Set A and Set B emotions cluster together in representation space?
2. Is the structure organized by emotion (shared across sets) or by stimulus set?
3. In Set B, does emotion dominate over topic (domain)?

## Key Finding

Set-level structure dominates over emotion-level structure in the joint geometry
(silhouette by set > silhouette by emotion). This is a genuine negative result:
while probes can linearly decode emotions within each set (Exp 13 AUROC ~1.0),
the representations are not geometrically organized primarily by emotion when
both sets are projected together. The linear probe finds directions that PCA
and silhouette miss — consistent with the AUROC/silhouette dissociation observed
throughout Phase 0/1.

## Methodology

1. **Joint PCA**: Normalize Set A + Set B peak-layer activations, PCA to 2D
2. **Silhouette**: emotion vs set grouping silhouette scores
3. **Cosine similarity**: centroids per (emotion × set), compare same-emotion cross-set vs cross-emotion
4. **Cross-topic clustering**: within-emotion cross-domain vs cross-emotion similarity in Set B
5. **Permutation test**: 500 random label shuffles → empirical p-value

## Outputs

Per model (`{MODEL_KEY}` = llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst):

- `outputs/pca_joint_{MODEL_KEY}.png/.svg` — joint PCA visualization
- `outputs/cosine_similarity_matrix_{MODEL_KEY}.png/.svg` — centroid similarity heatmap
- `outputs/cross_topic_clustering_{MODEL_KEY}.png/.svg` — emotion vs topic comparison
- `outputs/silhouette_scores_{MODEL_KEY}.csv` — silhouette and cosine metrics
- `outputs/permutation_test_{MODEL_KEY}.csv` — permutation test results
