# Experiment 10 — Set C: Tak Replication Sanity Check

**Date:** 2026-02-25
**Question:** Does prediction-task framing shift emotion consolidation from late to mid layers?
**Gate:** Set C peak normalized depth in [40%, 75%] (mid-layer zone)
**Status:** FAIL ✗ (0/4 models in mid-zone)

## What This Tests

Internal validation experiment. One thing changes vs Phase 0:
- **Suffix appended:** `"\n\nEmotion:"` — forces next-token prediction framing
- **Extraction point:** final token `":"` of suffixed text (not end-of-narrative)
- Everything else identical (same stimuli, same model, same probe)

This replicates the task framing in Tak et al., where emotion consolidation was found
in mid-layers (~40–75% depth) under prediction-task conditions. Phase 0 used passive
reading (no suffix), which produced late-layer consolidation (~80–95% depth).

## Results

| Model | P0 peak (norm) | Set C peak (norm) | Set C AUROC | Shift + Gate |
|-------|---------------|------------------|-------------|--------------|
| Llama-3.2-1B | 87.5% | 93.3% (L14/15) | 0.9973 | +5.8pp ✗ |
| Llama-3-8B (base) | 78.1% | 96.8% (L30/31) | 0.9929 | +18.7pp ✗ |
| Llama-3-8B (instruct) | 93.8% | 80.6% (L25/31) | 0.9991 | -13.2pp ✗ |
| Gemma-2-9B | 92.9% | 80.5% (L33/41) | 1.0000 | -12.4pp ✗ |

**Mid-zone criterion:** normalized depth 40–75%

## Behavioral Check

Top-10 next-token predictions at ":" verified for one sample per emotion.
See `outputs/behavioral_check_*.txt` for full details.

## Outputs

- `outputs/auroc_setc_vs_phase0_panel.png/.svg` — 4-panel comparison per model
- `outputs/results_*.csv` — per-layer AUROC for each model
- `outputs/summary.csv` — summary table
- `outputs/behavioral_check_*.txt` — top-10 prediction checks

## Interpretation

If Set C peaks in the mid-zone [40–75%]:
The late-layer consolidation in Phase 0/1 is explained by task framing (passive reading vs
prediction). Both paradigms are valid but measure different things. Proceed with Phase 2 analysis
using Phase 0/1 data, noting this methodological distinction.

If Set C still peaks late (>75%):
Task framing alone does not shift the peak in these models. Possible explanations:
(a) The Tak et al. result doesn't generalize to Llama/Gemma architectures;
(b) Our suffix format differs from their stimuli in a critical way;
(c) The late-layer localization in Phase 0/1 is a genuine property of these models,
not a task-framing artifact. Document and proceed.