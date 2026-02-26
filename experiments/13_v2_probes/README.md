# Experiment 13 — v2 Full Probing Analysis

**Date:** 2026-02-25
**Primary model:** llama1b_inst (Llama-3.2-1B-Instruct)
**Stimuli:** 80 Set A (keyword-rich) + 96 Set B clinical (keyword-free)
**Activation types:** h (residual stream), a (MHSA output), m (FFN output)

## Research Questions

1. At which layers do h, a, m components encode emotion?
2. Does the emotion encoding transfer from keyword-rich (Set A) to keyword-free (Set B)?
3. How does base vs instruct affect probe performance?

## Key Findings — Primary Model (llama1b_inst)

| Metric | Value |
|--------|-------|
| h probe peak | L11 (0.750) |
| h probe peak AUROC | 0.9991 |
| Gate check | See outputs/gate_check_report_llama1b_inst.md |

## Summary Table (h component across models)

| Llama-1B Instruct | L11 (0.750) | 0.9991 AUROC | A→B: 0.8049355158730158 |
| Llama-1B Base | L11 (0.750) | 1.0000 AUROC | A→B: 0.7919146825396826 |
| Llama-8B Instruct | L31 (1.000) | 1.0000 AUROC | A→B: 0.904265873015873 |
| Llama-8B Base | L17 (0.562) | 1.0000 AUROC | A→B: 0.9174107142857142 |
| Gemma-9B Instruct | L21 (0.524) | 1.0000 AUROC | A→B: 0.9066220238095238 |
| Gemma-9B Base | L32 (0.786) | 1.0000 AUROC | A→B: 0.9237351190476191 |

## Methodology

- 8-class logistic regression probes with 5-fold stratified cross-validation
- Macro AUROC (one-vs-rest) as primary metric
- Stimuli normalized per fold before fitting
- Transfer: train on all Set A emotional, test on all Set B clinical (same emotion classes)

## Outputs

- `outputs/probe_results_{model}_{set}_{act_type}.csv` — per-layer AUROC
- `outputs/transfer_results_{model}_{act_type}.csv` — A→B and B→A transfer
- `outputs/probe_summary_all_models.csv` — cross-model comparison
- `outputs/probe_curves_panel_{PRIMARY}.png/.svg` — visualization
