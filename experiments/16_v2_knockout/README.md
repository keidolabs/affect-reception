# Experiment 16 — v2 Knockout Experiments

**Date:** 2026-02-25
**Model:** gemma9b_inst (Llama-3.2-1B-Instruct, HF on MPS)
**Threshold for "critical layer":** ≥20% accuracy drop

## Research Question

Which layers are causally necessary for emotion inference in Set A and Set B?

## Key Findings

| Configuration | Critical Layers (Set A) | Critical Layers (Set B) |
|---------------|------------------------|------------------------|
| MHSA zero knockout | [] | [] |
| FFN zero knockout  | [] | see CSV |

**Overlap (shared critical layers):** Layers appearing in both Set A and Set B
indicate circuits that are causally shared across stimulus types.

## Methodology

- Zero knockout: set activation to 0 at ":" position (most disruptive)
- Random knockout: magnitude-preserving random vector at ":" position
- Accuracy drop = fraction of stimuli where top-1 prediction changes post-knockout
- Critical layer = accuracy drop ≥ 20%

## Outputs

- `outputs/knockout_summary_gemma9b_inst.csv` — per-layer accuracy drops
- `outputs/critical_layers_gemma9b_inst.csv` — critical layers per configuration
- `outputs/knockout_curves_gemma9b_inst.png/.svg` — visualization
