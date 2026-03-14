# Experiment Chain — Study Architecture

This document maps the logical flow of the entire study from Exp 00 to Exp 21.
Each experiment has a specific role, explicit dependencies, and a gate criterion
that determined whether subsequent experiments were justified.

---

## Study Structure

```
                    PHASE 0: Can we decode emotions?
                    ════════════════════════════════
                    Set A (keyword-rich), 4 models
                    Passive reading, final-token extraction
                    No z-norm probes

    ┌──────────┬──────────┬──────────┐
   00          01         02         03         04
 Llama-1B   Llama-8B   Llama-8B   Gemma-9B   Comparison
  (base)    (base)    (instruct)  (base)     (meta)
  TL+CPU    HF+MPS    HF+MPS     HF+MPS
    │          │         │          │
    │  Gate: AUROC > 0.40 on keyword-rich stimuli?
    │  Result: ALL PASS (0.98–0.99)
    │
    ▼
                    PHASE 1: Is it just keywords?
                    ════════════════════════════
                    Set B (keyword-free clinical), same 4 models
                    Same extraction as Phase 0

    ┌──────────┬──────────┬──────────┐
   05          06         07         08         09
 Llama-1B   Llama-8B   Llama-8B   Gemma-9B   Comparison
    │          │         │          │
    │  Gate: AUROC > 0.40 on keyword-free stimuli?
    │  Result: ALL PASS (0.91–0.96), drops only 3–8%
    │  Finding: keyword-spotting hypothesis falsified
    │
    ▼
                    METHODOLOGY CHECK
                    ═════════════════
   10
 Consistency     Does prompt format (passive vs Tak) change findings?
 Check           Loads Phase 0 + v2 activations, runs identical z-norm probes
                 Result: curve shapes agree (r=0.85–0.92), v2 saturates earlier
                 Phase 0 and v2 are qualitatively consistent
    │
    ▼
                    v2: CAUSAL MECHANISTIC STUDY
                    ════════════════════════════
                    Tak few-shot prompt, ":" extraction token
                    6 models (+ base/instruct pairs)
                    z-normalized probes

   11 ──────────► 12
 Extract          Extract        Both use HF+MPS (8B/9B) or TL+CPU (1B)
 Set A            Set B          Per-stimulus .npz with h, a, m, attn
 (80 emo)         (96+96)        Tak 2-shot prompt, extract at ":"
    │                │
    └──────┬─────────┘
           ▼
          13                     8-class probes (h, a, m) + A→B transfer
        Probes ─────────────►    z-normalized, 5-fold CV
           │                     All 6 models
           │
     ┌─────┼─────┬───────┐
     ▼     ▼     ▼       ▼
    14    15     16      17
 Patching Attn  Knockout Geometry
 (causal) (what  (which  (how are
          does   layers  emotions
          ":"    are     organized
          attend needed?) in repr
          to?)            space?)
     │     │      │       │       All 6 models
     └─────┴──────┴───────┘
               │
               ▼
              18
           Summary              Cross-model synthesis of Exp 11–17
               │
               ▼
                    CONTROLS
                    ════════

          19                     Do few-shot exemplars drive the signal?
       Zero-shot ───────────►    Result: No (Δ = +0.004 AUROC)
       Control                   Llama-1B-Instruct only

          20                     Does binary probe detect affect or richness?
       Binary    ───────────►    Result: CLEAN (Set C scores 0.04)
       Confound                  Llama-1B-Instruct only

          21                     Binary affect detection — all 6 models
       Binary    ───────────►    Result: AUROC 1.000 all models, saturates L3–L6
       Set B                     Fills manuscript Tables 4 & 6
       Probes                    Uses Exp 12 activations + Exp 13 methodology
```

---

## Experiment Reference

### Phase 0 — Representational Discovery (Exp 00–04)

| Exp | Purpose | Models | Framework | Stimuli | Probe | Gate |
|-----|---------|--------|-----------|---------|-------|------|
| 00 | Baseline: can emotions be decoded? | Llama-1B base | TL+CPU f32 | Set A (90) | 5-fold CV, no z-norm | AUROC > 0.40 → **PASS** (0.988) |
| 01 | Scale: 8B base | Llama-8B base | HF+MPS f16 | Set A (90) | 5-fold CV, no z-norm | **PASS** (0.994) |
| 02 | RLHF: 8B instruct | Llama-8B instruct | HF+MPS f16 | Set A (90) | 5-fold CV, no z-norm | **PASS** (0.980) |
| 03 | Architecture: Gemma | Gemma-9B base | HF+MPS f16 | Set A (90) | 5-fold CV, no z-norm | **PASS** (0.988) |
| 04 | Cross-model comparison | All 4 above | Analysis only | — | — | All 4 pass |

**Key findings:** Late-layer consolidation (peaks 78–95% depth). L2→L3 jump universal in Llama. AUROC/silhouette dissociation (0.99 AUROC but 0.07 silhouette).

### Phase 1 — Keyword Independence (Exp 05–09)

| Exp | Purpose | Models | Framework | Stimuli | Probe | Gate |
|-----|---------|--------|-----------|---------|-------|------|
| 05 | Keyword-free test: 1B | Llama-1B base | TL+CPU f32 | Set B (192) | 5-fold CV, no z-norm | AUROC > 0.40 → **PASS** (0.908) |
| 06 | Keyword-free test: 8B base | Llama-8B base | HF+MPS f16 | Set B (192) | 5-fold CV, no z-norm | **PASS** (0.964) |
| 07 | Keyword-free test: 8B instruct | Llama-8B instruct | HF+MPS f16 | Set B (192) | 5-fold CV, no z-norm | **PASS** (0.948) |
| 08 | Keyword-free test: Gemma | Gemma-9B base | HF+MPS f16 | Set B (192) | 5-fold CV, no z-norm | **PASS** (0.946) |
| 09 | Cross-model comparison | All 4 above | Analysis only | — | — | All 4 pass |

**Key findings:** Drops 3–8% (not to chance) → keyword-spotting falsified. L2→L3 jump preserved on Set B. Two-stage: binary saturates early, 8-class peaks late. Gemma late layers were keyword-sensitive (24pp peak shift).

### Methodology Bridge (Exp 10)

| Exp | Purpose | Method | Result |
|-----|---------|--------|--------|
| 10 | Do Phase 0 and v2 agree? | Load both activation sets, run identical z-norm probes | Curve shapes agree (r=0.85–0.92), v2 ceiling masks peak differences |

**Note:** Phase 0 → v2 is a methodological boundary. Phase 0 used passive reading + no z-norm. v2 uses Tak prompt + z-norm. Numbers should not be mixed across the boundary without noting this. The qualitative story (late-layer consolidation) holds in both.

### v2 — Causal Mechanistic Study (Exp 11–18)

| Exp | Purpose | Models | Framework | Key outputs |
|-----|---------|--------|-----------|-------------|
| 11 | Extract Set A activations | All 6 | TL+CPU (1B), HF+MPS (8B/9B) | Per-stimulus .npz (h, a, m, attn) |
| 12 | Extract Set B activations | All 6 | Same as 11 | Per-stimulus .npz (h, a, m, attn) |
| 13 | 8-class probes + transfer | All 6 | Analysis (on 11/12 data) | AUROC curves, A→B transfer rates |
| 14 | Activation patching | All 6 | HF+MPS live | Within-set and cross-set patching success |
| 15 | Attention analysis | All 6 | Analysis (tokenizer only) | Head sensitivity, token-type classification |
| 16 | Layer knockout | All 6 | HF+MPS live | Critical layers (≥20% accuracy drop) |
| 17 | Representational geometry | All 6 | Analysis (on 11/12 data) | PCA, silhouette, cosine similarity, permutation tests |
| 18 | Cross-model summary | All 6 | Analysis | Synthesis table of Exp 11–17 |

**Prompt format:** Tak et al. 2-shot (`Context: ... Answer:`) for all v2 experiments.
**Extraction point:** Final ":" token.
**Probe normalization:** Per-fold z-normalization throughout.

### Controls (Exp 19–21)

| Exp | Purpose | Model | Result |
|-----|---------|-------|--------|
| 19 | Do few-shot exemplars drive signal? | Llama-1B instruct | No — zero-shot AUROC matches few-shot (Δ = +0.004) |
| 20 | Does binary probe detect richness? | Llama-1B instruct | No — Set C (vivid neutrals) scores 0.04 at peak layer |
| 21 | Binary affect detection on Set B | All 6 | AUROC 1.000 all models, saturates at L3–L6 (9–38% depth) |

---

## Dependency Graph

```
Exp 00 ──┐
Exp 01 ──┤
Exp 02 ──┼──► Exp 04 (comparison)
Exp 03 ──┘         │
                    │ Gate PASS → proceed to Phase 1
                    ▼
Exp 05 ──┐
Exp 06 ──┤
Exp 07 ──┼──► Exp 09 (comparison)
Exp 08 ──┘         │
                    │ Gate PASS → proceed to v2
                    ▼
Exp 11 ──┬──► Exp 13 (probes) ──┬──► Exp 14 (patching)
Exp 12 ──┘                      ├──► Exp 15 (attention)
                                ├──► Exp 16 (knockout)
                                ├──► Exp 17 (geometry)
                                └──► Exp 18 (summary)

Exp 00–03 ──┬──► Exp 10 (methodology check)
Exp 11    ──┘

Exp 12 ──────────► Exp 19 (zero-shot control)
Exp 12 ──────────► Exp 20 (binary confound control)
Exp 12 ──────────► Exp 21 (binary Set B probes, all 6 models)
```

---

## Framework Summary

| Models | Phase 0/1 (Exp 00–09) | v2 extraction (Exp 11/12) | v2 live (Exp 14/16) |
|--------|----------------------|--------------------------|---------------------|
| Llama-1B (base/inst) | TL + CPU + float32 | TL + CPU + float32 | HF + MPS + float16 |
| Llama-8B (base/inst) | HF + MPS + float16 | HF + MPS + float16 | HF + MPS + float16 |
| Gemma-9B (base/inst) | HF + MPS + float16 | HF + MPS + float16 | HF + MPS + float16 |

**Why TL for 1B?** TransformerLens supports Llama-3.2-1B natively and was used from
the start. TL on MPS gives silently wrong results (GitHub #1178), so CPU is mandatory.
8B/9B models are too slow on TL+CPU, so HF+MPS is used instead. Within each model,
the framework is consistent across all experiments.

---

## Stimulus Sets

| Set | n | Content | Used in |
|-----|---|---------|---------|
| **Set A** | 80 emo + 10 neutral | Keyword-rich first-person narratives (8 Plutchik primaries × 10) | Exp 00–04, 11, 13–18 |
| **Set B** | 96 clinical + 96 neutral | Keyword-free clinical vignettes (8 × 12) + matched controls | Exp 05–09, 12, 13–20 |
| **Set C** | 24 | High-complexity neutral narratives (vivid but no affect) | Exp 20 only |

---

## Methodological Boundaries

There is one methodological boundary in this study that should not be crossed when
comparing numbers:

| | Phase 0/1 (Exp 00–09) | v2 (Exp 11–18) |
|--|----------------------|----------------|
| Prompt | Raw text | Tak 2-shot + "Answer:" |
| Extract position | Final token of narrative | ":" token |
| Probe z-norm | No | Yes |

Exp 10 demonstrates that this boundary does not change the qualitative story
(late-layer consolidation holds in both), but the absolute AUROC values differ.
Phase 0/1 results and v2 results should not be cited interchangeably.

---

## Replication Order

To replicate the full study from scratch:

1. **Stimuli:** Verify `stimuli/stimuli.jsonl` (282 entries) and `stimuli/exp20/set_c_complex_neutral.md` (24 entries)
2. **Phase 0:** Run Exp 00–03 extraction + `run.py`, then Exp 04
3. **Phase 1:** Run Exp 05–08 extraction + `run.py`, then Exp 09
4. **v2 extraction:** Run Exp 11 (Set A, all 6 models), then Exp 12 (Set B, all 6 models)
5. **v2 probes:** Run Exp 13
6. **v2 causal:** Run Exp 14, 15, 16, 17 (can be parallelized across models)
7. **v2 summary:** Run Exp 18
8. **Controls:** Run Exp 19, 20, 21 (independent of each other, all need Exp 12)
9. **Methodology check:** Run Exp 10 (requires Exp 00–03 and Exp 11 outputs)

Each experiment's `run.py` is self-contained. See individual `README.md` files for
exact commands and prerequisites.
