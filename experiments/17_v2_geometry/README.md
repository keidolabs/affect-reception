# Experiment 17 — v2 Representational Geometry

**Date:** 2026-02-25
**Model:** llama1b_inst (Llama-3.2-1B-Instruct)
**Peak layer:** L11 (from Exp 13 probe results)

## Research Questions

1. Do Set A and Set B emotions cluster together in representation space?
2. Is the structure organized by emotion (shared across sets) or by stimulus set?
3. In Set B, does emotion dominate over topic (domain)?

## Key Findings

| Metric | Value |
|--------|-------|
| Silhouette by emotion | 0.0397 |
| Silhouette by set | 0.1290 |
| Emotion > set structure | ✗ |
| Within-emo cross-set cosine similarity | 0.9402 |
| Cross-emotion cosine similarity | 0.9378 |
| Emotions dominating topic (Set B) | 5/8 |

## Methodology

1. **Joint PCA**: Normalize Set A + Set B peak-layer activations, PCA to 2D
2. **Silhouette**: emotion vs set grouping silhouette scores
3. **Cosine similarity**: centroids per (emotion × set), compare same-emotion cross-set vs cross-emotion
4. **Cross-topic clustering**: within-emotion cross-domain vs cross-emotion similarity in Set B
5. **Permutation test**: 500 random label shuffles → empirical p-value

## Outputs

- `outputs/pca_joint_llama1b_inst.png/.svg` — joint PCA visualization
- `outputs/cosine_similarity_matrix_llama1b_inst.png/.svg` — centroid similarity heatmap
- `outputs/cross_topic_clustering_llama1b_inst.png/.svg` — emotion vs topic comparison
- `outputs/silhouette_scores_llama1b_inst.csv` — silhouette and cosine metrics
- `outputs/permutation_test_llama1b_inst.csv` — permutation test results
