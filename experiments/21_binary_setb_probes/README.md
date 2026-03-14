# Experiment 21 — Binary Set B Probes (Affect Reception)

**Date:** 2026-03-14
**Models:** All 6 (Llama-1B, Llama-8B, Gemma-9B; base + instruct)
**Stimuli:** Set B (96 clinical + 96 neutral, from Exp 12)

## Research Question

At which layer does binary affect detection (emotional vs neutral) saturate
for each model on keyword-free clinical vignettes?

## Motivation

Tables 4 and 6 in the manuscript had gaps — binary Set B probes had only been
run for 4 models during Phase 1 (Exp 05-08), and those used a different
methodology (no z-normalization, passive reading prompt). This experiment fills
the gap using v2 activations (Exp 12) with methodology matching Exp 13
(per-fold z-normalization, 5-fold stratified CV, Tak prompt format).

## Method

- Load v2 Set B activations from Exp 12 (per-stimulus `.npz`, `h` component)
- Binary labels: `B-*` = emotional (1), `N-*` = neutral (0)
- Per-layer logistic regression, 5-fold stratified CV, per-fold z-normalization
- AUROC (binary) as primary metric
- Seeds: 42 everywhere

## Results

| Model | Binary AUROC | Saturates at | Norm. Depth |
|-------|-------------|-------------|-------------|
| Llama-1B Instruct | 1.000 | L4/16 | 0.25 |
| Llama-1B Base | 1.000 | L6/16 | 0.38 |
| Llama-8B Instruct | 1.000 | L3/32 | 0.09 |
| Llama-8B Base | 1.000 | L4/32 | 0.13 |
| Gemma-9B Instruct | 1.000 | L4/42 | 0.10 |
| Gemma-9B Base | 1.000 | L5/42 | 0.12 |

*Saturation defined as first layer with AUROC >= 0.999.*

## Key Findings

1. **Perfect binary detection across all 6 models** — AUROC 1.000 on keyword-free
   clinical vignettes, confirming affect reception is universal
2. **Very early saturation** — 9-38% of network depth, much earlier than 8-class
   categorization peaks (52-100%). The dissociation is stark
3. **Instruct saturates earlier than base** at 1B scale (L4 vs L6), consistent
   with alignment restructuring access to existing representations
4. **Larger models saturate earlier in absolute terms** (L3-5) despite having
   more layers, suggesting affect reception is a low-level computation

## Dependencies

- Exp 12 (Set B activations for all 6 models)

## Outputs

- `outputs/results_binary_{model_key}.csv` — per-layer AUROC for each model
- `outputs/summary_binary_setb.csv` — peak and saturation summary
