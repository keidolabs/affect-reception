# Gate Check Report — Exp 13 v2 Probes (Tak Replication)

**Date:** 2026-02-25
**Model:** llama1b_inst (meta-llama/Llama-3.2-1B-Instruct)
**Stimuli:** 80 emotional Set A
**Prompt:** Tak et al. few-shot (2-shot classification)

## Gate Result: ✅ PASS

| Metric | Value |
|--------|-------|
| h probe peak layer | L11 |
| h probe peak norm depth | 0.7500 |
| h probe peak AUROC | 0.9991 |
| Gate zone | [0.5, 0.75] |
| Gate: PASS | Peak within gate zone |

## All Component Peaks

| Component | Peak Layer | Norm Depth | AUROC |
|-----------|-----------|-----------|-------|
| h (residual) | L11 | 0.7500 | 0.9991 |
| a (MHSA out) | L11 | 0.7500 | 1.0000 |
| m (FFN out)  | L14 | 0.9375 | 1.0000 |

## Interpretation

The h probe peak at normalized depth 0.750 falls within the expected gate zone [0.50, 0.75].
This replicates Tak et al.'s finding that few-shot classification framing consolidates
emotion representations at mid-layer depth (~60%), earlier than passive reading (~87%).

**Conclusion:** Tak replication confirmed. Proceed to Stage 2.

## Outputs

- `outputs/probe_results_llama1b_inst_seta_{h,a,m}.csv` — per-layer AUROC
- `outputs/probe_curves_llama1b_inst_seta_gate.png/.svg` — visualization