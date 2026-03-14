# Experiment 10 — Methodology Consistency Check

**Date:** 2026-02-25 (original), 2026-03-13 (rewritten for methodological rigor)
**Question:** Does prompt format (passive reading vs Tak few-shot) change what layer-wise probes measure?
**Gate:** ≥3/4 models show consistent late-layer consolidation across both conditions
**Status:** FAIL (2/4 consistent)

## Why This Experiment Exists

Before trusting 20 experiments of results, we need to verify that:
1. Our core finding (late-layer emotion consolidation) isn't an artifact of prompt format
2. Phase 0 results (passive reading) and v2 results (Tak prompt) agree when probed identically
3. The extraction framework difference (TL+CPU for 1B vs HF+MPS for 8B/9B) isn't hiding bugs

## What Changed in the Rewrite (2026-03-13)

The original Exp 10 had methodological problems:
- Used a **third** prompt format ("\n\nEmotion:" suffix) that matched neither Phase 0 nor v2
- Did NOT z-normalize probes (Phase 0 style), making comparison to v2 (Exp 13) invalid
- Mixed TL+CPU (Llama-1B) with HF+MPS (8B/9B) without documenting the inconsistency
- Called the condition "Set C" which collides with the actual Set C in Exp 20

The rewrite loads existing activations from Phase 0 and v2, runs identical z-normalized
probes on both, and directly compares. No new extraction needed.

## What Is Controlled

| Aspect | Phase 0 (Exp 00–03) | v2 (Exp 11) | Same? |
|--------|---------------------|-------------|-------|
| Llama-1B framework | TL + CPU + float32 | TL + CPU + float32 | ✓ |
| 8B/9B framework | HF + MPS + float16 | HF + MPS + float16 | ✓ |
| Stimuli | Set A, 80 emotional | Set A, 80 emotional | ✓ |
| Probe method | z-norm LogReg 5-fold | z-norm LogReg 5-fold | ✓ (both re-probed here) |
| Random seed | 42 | 42 | ✓ |

## What Varies (the independent variable)

| Aspect | Phase 0 | v2 |
|--------|---------|-----|
| Prompt format | Raw text (no framing) | Tak 2-shot + "Answer:" |
| Extraction position | Final token of narrative | ":" token after "Answer:" |

## Results

| Model | Framework | P0 peak | P0 AUROC | v2 peak | v2 AUROC | Δ depth | Consistent? |
|-------|-----------|---------|----------|---------|----------|---------|-------------|
| Llama-3.2-1B | TL+CPU+f32 | L14 (93.3%) | 0.9884 | L11 (73.3%) | 1.0000 | -20.0pp | ✗ |
| Llama-3-8B (base) | HF+MPS+f16 | L25 (80.7%) | 0.9938 | L17 (54.8%) | 1.0000 | -25.8pp | ✗ |
| Llama-3-8B (instruct) | HF+MPS+f16 | L25 (80.7%) | 0.9812 | L31 (100.0%) | 1.0000 | +19.4pp | ✓ |
| Gemma-2-9B | HF+MPS+f16 | L35 (85.4%) | 0.9875 | L32 (78.0%) | 1.0000 | -7.3pp | ✓ |

**Consistency criterion:** both conditions peak in upper half (>50% depth) AND shift < 20pp.

## Interpretation

If consistent: Late-layer emotion consolidation is a genuine property of these models,
not an artifact of how we prompt them. The Phase 0 findings and v2 findings measure
the same underlying phenomenon. Proceed with confidence.

If inconsistent: The prompt format matters more than expected. This doesn't invalidate
the v2 pipeline (which is internally consistent), but it means Phase 0 results should
be interpreted cautiously and not directly compared to v2 numbers.

## Outputs

- `outputs/consistency_comparison.png/.svg` — 4-panel overlay of Phase 0 vs v2 AUROC curves
- `outputs/results_*.csv` — per-layer AUROC for both conditions, all models
- `outputs/summary.csv` — comparison table

## Confounds Acknowledged

1. **Extraction position confound:** Phase 0 extracts at end-of-narrative, v2 at ":" after
   prompt. These are different sequence positions with different context. A shift in peak
   layer could reflect the model processing different amounts of text, not prompt format per se.
2. **TL vs HF for 1B:** TransformerLens and HF transformers may compute slightly different
   activations due to internal implementation differences. This only affects the Llama-1B
   comparison. The 8B/9B models use HF+MPS in both conditions.
3. **No normalization in original Phase 0 probes:** The original Phase 0 results (Exp 00–03)
   did NOT z-normalize. This experiment re-probes Phase 0 activations WITH z-normalization,
   so the numbers here differ from the original Phase 0 READMEs. This is intentional —
   we're controlling for probe methodology.
