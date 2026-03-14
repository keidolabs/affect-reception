# Experiment 16 — v2 Knockout Experiments

**Date:** 2026-02-25
**Models:** All 6 (llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst)
**Framework:** HF AutoModelForCausalLM + MPS (float16)
**Threshold for "critical layer":** ≥20% accuracy drop

## Research Question

Which layers are causally necessary for emotion inference in Set A and Set B?

## Methodology

- Zero knockout: set activation to 0 at ":" position (most disruptive)
- Random knockout: magnitude-preserving random vector at ":" position
- Accuracy drop = fraction of stimuli where top-1 prediction changes post-knockout
- Critical layer = accuracy drop ≥ 20%
- Run on all 6 models; see per-model CSVs for full results

## Outputs

Per model (`{MODEL_KEY}` = llama1b_base, llama1b_inst, llama8b_base, llama8b_inst, gemma9b_base, gemma9b_inst):

- `outputs/knockout_summary_{MODEL_KEY}.csv` — per-layer accuracy drops
- `outputs/critical_layers_{MODEL_KEY}.csv` — critical layers per configuration
- `outputs/knockout_curves_{MODEL_KEY}.png/.svg` — visualization
