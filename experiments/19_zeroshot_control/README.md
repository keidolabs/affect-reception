# Experiment 19 — Zero-Shot Control

**Date:** 2026-02-28
**Model:** Llama-3.2-1B-Instruct only
**Stimuli:** Set B (96 clinical + 96 neutral) — identical to Exp 12

## Research Question

Do the 2 fixed few-shot exemplars in the Tak et al. prompt drive the emotion signal
we observe in Set B representations? Or does the signal survive without them?

## Hypothesis

The pre-training signal should be robust enough that removing 2 shots causes ≤5%
drop in binary AUROC and ≤10% drop in 8-class AUROC. If both collapse toward
chance, the few-shot context was doing the work, not the stimulus itself.

## Prompt Conditions

| Condition | Structure |
|-----------|-----------|
| Few-shot (Exp 12) | `[Shot 1 → sadness] [Shot 2 → joy] [Set B target] → Answer:` |
| Zero-shot (Exp 19) | `[Set B target] → Answer:` |

Same extraction token (`:` at end of "Answer:"), same model, same stimuli.

## Method

1. Re-extract Set B activations with zero-shot prompt → `outputs/activations/llama1b_inst/`
2. Run binary probe (clinical vs neutral) at each layer, 5-fold CV
3. Run 8-class probe (8 emotions, clinical only) at each layer, 5-fold CV
4. Load Exp 12 few-shot results for direct comparison
5. Report delta at peak layer for each probe type

## Possible Outcomes

1. **Binary ~1.0, 8-class ~0.91** — shots do nothing. Confound dead. Frame as validation.
2. **Both collapse** — shots were load-bearing. Core claim weakened.
3. **Binary stays high, 8-class drops further** — shots help categorization but not binary affect. Strengthens h/m dissociation story.

## Results

## Results

**Date:** 2026-02-28
**Model:** Llama-3.2-1B-Instruct
**Activation type:** h (residual stream)
**Probe CV:** 5-fold stratified

### Peak AUROC Comparison

| Probe | Few-shot (Exp 12) | Zero-shot (Exp 19) | Delta |
|-------|-------------------|-------------------|-------|
| Binary (emotional vs neutral) | 1.0000 at L4 | 1.0000 at L4 | +0.0000 |
| 8-class (clinical emotions)   | 0.9335 at L9 | 0.9377 at L9 | +0.0042 |

### Interpretation

**Outcome 1** — Both probes robust to shot removal. Shots do not drive the signal. Confound eliminated.

### Outputs

- `outputs/probe_comparison.csv` — AUROC by layer, both conditions
- `outputs/probe_comparison.png/.svg` — comparison plot

## Outputs

- `outputs/activations/llama1b_inst/*.npz` — zero-shot activations (192 files)
- `outputs/probe_comparison.csv` — binary + 8-class AUROC by layer, both conditions
- `outputs/probe_comparison.png/.svg` — comparison plot
- `outputs/summary.md` — numeric summary


## Results

**Date:** 2026-02-28
**Model:** Llama-3.2-1B-Instruct
**Activation type:** h (residual stream)
**Probe CV:** 5-fold stratified

### Peak AUROC Comparison

| Probe | Few-shot (Exp 12) | Zero-shot (Exp 19) | Delta |
|-------|-------------------|-------------------|-------|
| Binary (emotional vs neutral) | 1.0000 at L4 | 1.0000 at L4 | +0.0000 |
| 8-class (clinical emotions)   | 0.9335 at L9 | 0.9377 at L9 | +0.0042 |

### Interpretation

**Outcome 1** — Both probes robust to shot removal. Shots do not drive the signal. Confound eliminated.

### Outputs

- `outputs/probe_comparison.csv` — AUROC by layer, both conditions
- `outputs/probe_comparison.png/.svg` — comparison plot
