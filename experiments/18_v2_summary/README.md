# Experiment 18 — v2 Cross-Model Summary

**Date:** 2026-02-25
**Models:** 6 (3 base/instruct pairs: Llama-1B, Llama-8B, Gemma-9B)

## Summary Table

| Model | h peak depth (A) | h AUROC (A) | h AUROC (B) | A→B transfer | Sil: emo | Sil: set |
|-------|-----------------|------------|------------|-------------|---------|---------|
| Llama-1B Instruct | 0.750 | 0.9991 | 0.9335 | 0.8049 | 0.0397 | 0.1290 |
| Llama-1B Base | 0.750 | 1.0000 | 0.9541 | 0.7919 | 0.0418 | 0.1082 |
| Llama-8B Instruct | 1.000 | 1.0000 | 0.9813 | 0.9043 | 0.0908 | 0.0663 |
| Llama-8B Base | 0.562 | 1.0000 | 0.9875 | 0.9174 | 0.0443 | 0.1172 |
| Gemma-9B Instruct | 0.524 | 1.0000 | 0.9866 | 0.9066 | 0.0383 | 0.1330 |
| Gemma-9B Base | 0.786 | 1.0000 | 0.9893 | 0.9237 | 0.0894 | 0.0715 |

## Research Questions Answered

### 1. Do the same circuits process Set A and Set B?

*Based on patching (Exp 14) and geometry (Exp 17) results.*

### 2. Which components carry the emotion signal?

*Based on h vs a vs m probe peaks (Exp 13).*

### 3. What does instruction tuning do?

*Based on base vs instruct probe AUROC comparison.*

## Figures

- `outputs/figure_probe_comparison.png/.svg` — Base vs instruct probe curves
- `outputs/figure_setA_vs_setB.png/.svg` — Set A vs Set B within-primary-model
- `outputs/figure_cross_model_auroc.png/.svg` — Summary bar chart
- `outputs/summary_table.csv` — Machine-readable summary

## Outputs

All experiment outputs:
- Exp 11/12: Per-stimulus .npz activations (h, a, m, attn)
- Exp 13: Probe curves + transfer results per model
- Exp 14: Activation patching success heatmaps
- Exp 15: Attention token-type analysis
- Exp 16: Knockout critical layer identification
- Exp 17: Representational geometry (PCA, cosine, cross-topic)