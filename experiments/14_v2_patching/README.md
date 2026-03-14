# Experiment 14 — v2 Activation Patching

**Date:** 2026-02-25
**Models:** All 6 (llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst)
**Framework:** HF AutoModelForCausalLM + MPS (float16)
**Probe layers:** Proportionally scaled per model at normalized depths [0.20, 0.40, 0.55, 0.65, 0.72, 0.80, 0.87, 0.95]
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
- Run on all 6 models

## Interpretation

If cross-set SAME emotion patching succeeds at similar rates to within-set,
the emotion representations are causally shared across stimulus types (Set A ≈ Set B circuits).

## Outputs

Per model (`{MODEL_KEY}` = llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst):

- `outputs/patching_results_{MODEL_KEY}_{seta,setb}_{h,a,m}.csv` — within-set results
- `outputs/patching_success_by_layer_{MODEL_KEY}_{seta,setb}_{h,a,m}.csv` — per-layer success rates
- `outputs/patching_crossset_results_{MODEL_KEY}_{h,a,m}.csv` — cross-set results
- `outputs/patching_crossset_{MODEL_KEY}.png/.svg` — cross-set visualization
