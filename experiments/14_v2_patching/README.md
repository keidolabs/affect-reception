# Experiment 14 — v2 Activation Patching

**Date:** 2026-02-25
**Model:** llama1b_inst (Llama-3.2-1B-Instruct, HF on MPS)
**Probe layers:** [3, 7, 9, 10, 11, 12, 13, 15]
**Pairs per emotion combo:** 1

## Research Question

Do the same circuits causally mediate emotion inference in keyword-rich (Set A)
and keyword-free clinical (Set B) stimuli?

## Method

Activation patching: replace target stimulus's layer-m activation at ":"
with the saved activation from a source stimulus of a different emotion.
Success = top-1 prediction shifts to source emotion.

- Source activations loaded from Exp 11/12 .npz files (no re-run)
- Baselines precomputed once per stimulus
- Within-set: source and target from same set
- Cross-set: source from Set A, target from Set B (same vs different emotion)

## Interpretation

If cross-set SAME emotion patching succeeds at similar rates to within-set,
the emotion representations are causally shared across stimulus types (Set A ≈ Set B circuits).

## Outputs

- `outputs/patching_results_llama1b_inst_{seta,setb}_{h,a,m}.csv`
- `outputs/patching_success_by_layer_*.csv`
- `outputs/patching_crossset_results_llama1b_inst_*.csv`
- `outputs/patching_heatmap_llama1b_inst.png/.svg`
- `outputs/patching_crossset_llama1b_inst.png/.svg`
