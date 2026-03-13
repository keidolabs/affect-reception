# Experiment 12 — v2 Activation Extraction: Set B (Keyword-Free Clinical Vignettes)

**Date:** 2026-02-25
**Stimulus set:** Set B — 192 stimuli (96 clinical vignettes + 96 matched neutral controls)
**Models:** 6 (Llama-3.2-1B base/inst, Llama-3-8B base/inst, Gemma-2-9B base/inst)
**Output:** Per-stimulus `.npz` files — identical format to Exp 11

---

## Research Question

Does emotion encoding in the residual stream persist when emotional content is conveyed *implicitly* — with no explicit emotion keywords — as in clinical descriptions of distress, grief, or fear?

This experiment is the Set B counterpart to Exp 11. Together, Exp 11 and 12 provide the activation data for all cross-set comparisons: Exp 13 tests whether a probe trained on Set A transfers to Set B; Exp 14 tests whether activations patch causally across sets; Exp 17 tests whether the geometry of emotion representations is preserved.

---

## What Makes Set B Different

Set B stimuli were designed to defeat keyword-spotting as an explanation for Exp 11 results:

- **96 clinical vignettes** across 8 emotion-adjacent domains (rage→anger disorder, grief→bereavement, terror→PTSD, etc.) — written to describe emotional states without naming them
- **96 matched neutral controls** — domain-matched (e.g., same medical domain, similar length) but emotionally neutral in content
- Confirmed keyword-free: `validation/lexical_screening.py` found mean LIWC polarity −0.07 for Set B vs −0.24 for Set A (p < 0.001)
- Token length confound documented: clinical vignettes are systematically ~13% longer than neutral controls (see `validation/surface_features.py`); Exp 20 controls for this

**The test:** If a probe trained on Set A (explicit emotion words) transfers to Set B (no emotion words), keyword-spotting cannot explain the encoding — the model must be representing the *content* of the emotion, not the lexical surface.

---

## Extraction Design

Identical to Exp 11. See `experiments/11_v2_extract_seta/README.md` for full details on:
- TAK prompt format
- Final token extraction position
- Hook targets (h, a, m, attn)
- Output `.npz` format

**Stimulus count difference:** 192 stimuli (vs. 80 in Exp 11) — all Set B clinical + all neutral controls. The neutral controls are included because Exp 13 probes require them to define the binary emotion/neutral contrast.

---

## Model Specifications

Same 6 models as Exp 11:

| Model key | HuggingFace ID | Layers | d_model |
|-----------|---------------|--------|---------|
| `llama1b_base` | `meta-llama/Llama-3.2-1B` | 16 | 2048 |
| `llama1b_inst` | `meta-llama/Llama-3.2-1B-Instruct` | 16 | 2048 |
| `llama8b_base` | `meta-llama/Meta-Llama-3-8B` | 32 | 4096 |
| `llama8b_inst` | `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 4096 |
| `gemma9b_base` | `google/gemma-2-9b` | 42 | 3584 |
| `gemma9b_inst` | `google/gemma-2-9b-it` | 42 | 3584 |

---

## How to Run

### Prerequisites

```bash
cp .env.sample .env  # set HF_ACCESS_TOKEN=hf_...
export EMO_DEVICE=cuda  # or mps
```

**Run Exp 11 first.** Exp 13 (probes) loads activations from both Exp 11 and 12 together. While Exp 12 scripts are independent, the downstream analysis requires both sets.

### Run all 6 models (recommended)

```bash
uv run python experiments/12_v2_extract_setb/run.py
```

Skips models where `manifest.csv` already exists.

```bash
uv run python experiments/12_v2_extract_setb/run.py --models llama8b_inst gemma9b_base
```

### Run individual models

```bash
uv run python experiments/12_v2_extract_setb/extract_llama1b_base.py
uv run python experiments/12_v2_extract_setb/extract_llama1b_inst.py
uv run python experiments/12_v2_extract_setb/extract_llama8b_base.py
uv run python experiments/12_v2_extract_setb/extract_llama8b_inst.py
uv run python experiments/12_v2_extract_setb/extract_gemma9b_base.py
uv run python experiments/12_v2_extract_setb/extract_gemma9b_inst.py
```

---

## Time and Disk Estimates

| Model | GPU time (A6000) | GPU time (M1 MPS) | Disk per model |
|-------|-----------------|------------------|----------------|
| llama1b_base/inst | ~12 min | ~35 min | ~120 MB |
| llama8b_base/inst | ~35 min | ~105 min | ~430 MB |
| gemma9b_base/inst | ~45 min | ~140 min | ~540 MB |
| **Total (6 models)** | **~3 hrs** | **~9 hrs** | **~2.2 GB** |

*2.4× longer than Exp 11 due to 192 vs 80 stimuli.*

---

## Outputs

```
outputs/activations/
  llama1b_base/
    {stimulus_id}.npz   × 192   (96 clinical + 96 neutral)
    manifest.csv
  llama1b_inst/
    ...
  llama8b_base/
  llama8b_inst/
  gemma9b_base/
  gemma9b_inst/
```

Activation `.npz` files are gitignored. The `manifest.csv` per model is tracked and includes `matched_control_id` linking each clinical vignette to its neutral control.

---

## Notes

- Same `attn_implementation="eager"` setting as Exp 11 (required for MPS attention output)
- `use_cache=False` on all forward passes
- Neutral controls are extracted with the same TAK prompt format — the model is asked to infer emotion from a neutral text, which produces a baseline activation pattern against which clinical activations are compared
- Stimulus IDs in Set B follow the pattern `setb_clin_{emotion}_{N}` (clinical) and `setb_neu_{domain}_{N}` (neutral), matching the `matched_control_id` links in `manifest.csv`
