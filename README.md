# Emotional Representations in Large Language Models: A Mechanistic Interpretability Study

**Author:** Michael Keeman
**Affiliation:** Keido Labs, Liverpool, UK
**Contact:** michael@keidolabs.com

[![arXiv](https://img.shields.io/badge/arXiv-forthcoming-b31b1b.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Research area:**
[![Mechanistic Interpretability](https://img.shields.io/badge/research-Mechanistic_Interpretability-purple.svg)]()
[![Emotion in LLMs](https://img.shields.io/badge/research-Emotion_in_LLMs-purple.svg)]()
[![Representational Geometry](https://img.shields.io/badge/research-Representational_Geometry-purple.svg)]()

**Tech stack:**
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/models-HuggingFace-FFD21E.svg?logo=huggingface)](https://huggingface.co)
[![Polars](https://img.shields.io/badge/data-Polars-CD792C.svg?logo=polars)](https://pola.rs)
[![SciPy](https://img.shields.io/badge/stats-SciPy-8CAAE6.svg?logo=scipy)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/figures-Matplotlib-11557C.svg)](https://matplotlib.org)

---

## Overview

This repository contains all code, stimuli, and results for a mechanistic interpretability study investigating how large language models internally represent emotional content. We probe the residual stream and attention/FFN components across six LLMs using two stimulus sets: expressive emotional text (Set A) and keyword-free clinical vignettes (Set B).

> **Note:** This is a research reproducibility repository, not an actively maintained software project. The code is provided as-is to support transparency and reproduction of our published findings.

**Core questions:**
1. Do LLMs encode a structured representation of emotion categories in their residual stream?
2. Does this representation persist when emotional content is conveyed *implicitly* (no explicit emotion words)?
3. Which architectural components carry the signal — attention, FFN, or the full residual stream?
4. Is the representation causally active, or merely correlated?

**Short answers:** Yes, yes, primarily the residual stream, and yes — causal patching and layer knockout confirm functional necessity.

**Key Findings:**
- Emotion categories are linearly decodable from the residual stream at AUROC 0.93–1.00 across all 6 models on keyword-free clinical text
- A probe trained on explicit emotional text (Set A) transfers to keyword-free clinical vignettes (Set B) at AUROC 0.79–0.92, falsifying the keyword-spotting hypothesis
- Activation patching causally transfers emotion representations across stimuli at 46–67% success rate at peak layers
- Two-stage processing: binary emotion/neutral separation saturates at ~50–75% network depth; 8-class discrimination peaks at 75–100% depth
- RLHF instruction tuning slightly degrades cross-domain encoding but does not eliminate it

---

## Key Results

| Model | Set A AUROC | Set B AUROC | A→B Transfer | Peak depth |
|-------|------------|------------|-------------|-----------|
| Llama-3.2-1B Base | 1.000 | 0.954 | 0.792 | 75% |
| Llama-3.2-1B Instruct | 0.999 | 0.934 | 0.805 | 75% |
| Llama-3-8B Base | 1.000 | 0.988 | 0.917 | 56% |
| Llama-3-8B Instruct | 1.000 | 0.981 | 0.904 | 100% |
| Gemma-2-9B Base | 1.000 | 0.989 | 0.924 | 79% |
| Gemma-2-9B Instruct | 1.000 | 0.987 | 0.907 | 52% |

*8-class macro AUROC, logistic regression probes, 5-fold CV. Set B uses keyword-free clinical vignettes (mean polarity −0.07, no explicit emotion words).*

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), HuggingFace access to gated model repos.

```bash
git clone https://github.com/drkeeman/emotion-circuits
cd emotion-circuits
uv sync
cp .env.sample .env
# Edit .env: set HF_ACCESS_TOKEN=hf_...
```

**Models required** (request access on HuggingFace):
- `meta-llama/Llama-3.2-1B` · `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Meta-Llama-3-8B` · `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b` · `google/gemma-2-9b-it`

**Hardware:** Activation extraction requires GPU. Tested on Apple M1 Pro 32GB (MPS) and Vast.ai A6000. Analysis and visualization scripts run on CPU.

> **Note on raw activations:** `.npy`/`.npz` activation arrays are gitignored (too large). Re-generate with the extraction scripts (Exp 11–12), or contact the authors.

See `docs/vast-ai-guide.md` for cloud GPU setup and `Makefile` for cloud orchestration targets.

---

## Experiments

### Phase 0 — Baseline Probing (Exp 00–04)

| # | Directory | Description |
|---|-----------|-------------|
| 00 | `experiments/00_phase0_replication/` | Initial baseline: Llama-3.2-1B-Instruct, Set A, residual stream probes |
| 01 | `experiments/01_phase0a_llama3-8b/` | Replication on Llama-3-8B base |
| 02 | `experiments/02_phase0b_llama3-8b-instruct/` | Replication on Llama-3-8B Instruct |
| 03 | `experiments/03_phase0c_gemma2-9b/` | Replication on Gemma-2-9B |
| 04 | `experiments/04_phase0_comparison/` | Cross-model comparison, AUROC curves, silhouette analysis |

### Phase 1 — Keyword-Free Validation (Exp 05–10)

| # | Directory | Description |
|---|-----------|-------------|
| 05 | `experiments/05_phase1_llama1b/` | Set B probing: Llama-3.2-1B |
| 06 | `experiments/06_phase1_llama8b/` | Set B probing: Llama-3-8B base + instruct |
| 07 | `experiments/07_phase1_gemma/` | Set B probing: Gemma-2-9B base + instruct |
| 08–09 | `experiments/08–09_*/` | Phase 1 comparison and summary |
| 10 | `experiments/10_setc_tak_replication/` | Set C validation (alternate stimulus set) |

### v2 Mechanistic Pipeline (Exp 11–18)

The core contribution. Uses PyTorch forward hooks to extract residual stream (`h`), attention output (`a`), and FFN output (`m`) activations for all 6 models × both stimulus sets.

| # | Directory | Description | Key outputs |
|---|-----------|-------------|-------------|
| 11 | `experiments/11_v2_extract_seta/` | Extract h/a/m/attention — Set A, all 6 models | `.npz` per stimulus |
| 12 | `experiments/12_v2_extract_setb/` | Extract h/a/m/attention — Set B, all 6 models | `.npz` per stimulus |
| 13 | `experiments/13_v2_probes/` | Logistic regression probes: per-layer AUROC, cross-set transfer | `probe_results_*.csv` |
| 14 | `experiments/14_v2_patching/` | Activation patching: causal transfer of emotion representations | `patching_heatmap_*.png/svg` |
| 15 | `experiments/15_v2_attention/` | Attention analysis: which token types drive emotion-sensitive heads | `attention_heatmap_*.png/svg` |
| 16 | `experiments/16_v2_knockout/` | Layer knockout: identify causally necessary layers | `knockout_curves_*.png/svg` |
| 17 | `experiments/17_v2_geometry/` | Representational geometry: PCA, cosine similarity, cluster tests | `pca_joint_*.png/svg` |
| 18 | `experiments/18_v2_summary/` | Cross-model summary and synthesis | `summary_table.csv` |

### Controls (Exp 19–20)

| # | Directory | Description |
|---|-----------|-------------|
| 19 | `experiments/19_zeroshot_control/` | Zero-shot probing (no training) as baseline |
| 20 | `experiments/20_binary_confound_control/` | Binary classification controlling for token-length confound |

**Recommended run order:**
```bash
# 1. Extract activations (GPU required — see Exp 11/12 READMEs for time estimates)
uv run python experiments/11_v2_extract_seta/run.py
uv run python experiments/12_v2_extract_setb/run.py

# 2. Probes, patching, knockout, geometry (CPU OK)
uv run python experiments/13_v2_probes/run.py
uv run python experiments/14_v2_patching/run.py
uv run python experiments/16_v2_knockout/run.py
uv run python experiments/17_v2_geometry/run.py

# 3. Publication figures
uv run python visualization/figures/run_all.py
```

---

## Stimuli

All stimuli are in `stimuli/`. Load programmatically via `stimuli/loader.py`.

| Set | File/Directory | N | Description |
|-----|---------------|---|-------------|
| Set A | `stimuli/set-a-standard.jsonl` | 90 | Expressive emotional text; 8 Plutchik basic emotions × ~11 stimuli |
| Set B Clinical | `stimuli/set-b-clinical/` | 96 | Keyword-free clinical vignettes; 8 emotion-adjacent domains |
| Set B Neutral | `stimuli/neutral-controls/` | 96 | Domain-matched neutral controls for Set B |
| Set C | `stimuli/exp20/` | — | Alternate validation set |

Set B excludes explicit emotion vocabulary, confirmed by `validation/lexical_screening.py`: mean LIWC-based polarity −0.07 vs −0.24 for Set A (p < 0.001, Mann-Whitney U).

---

## Visualization

Publication figures (13 main + 3 supplemental) in `visualization/figures/`. All figures export as `.png` and `.svg`.

```bash
uv run python visualization/figures/run_all.py
```

Figures use the Okabe-Ito colorblind-safe palette. See `visualization/figures/style_config.py` for shared style config.

---

## Repository Structure

```
emotion-circuits/
├── stimuli/                    # All stimulus sets and unified loader
├── experiments/                # 20 numbered experiments
│   └── */
│       ├── README.md           # Research question, methods, key findings
│       ├── run.py              # Executable script (uv run python ...)
│       └── outputs/            # CSVs, PNGs, SVGs (activation .npz gitignored)
├── visualization/
│   └── figures/                # 13 arXiv-ready figures + style config
├── analysis/                   # Cross-experiment statistical utilities
├── validation/                 # Stimulus validation scripts
├── lab-journal/                # Chronological research notes (open science)
├── docs/                       # Cloud GPU setup guide
├── scripts/                    # Cloud sync helpers (Vast.ai)
├── Makefile                    # Cloud GPU orchestration targets
├── pyproject.toml              # All Python dependencies (managed by uv)
└── CLAUDE.md                   # AI-assisted research workflow documentation
```

---

## Reproducibility Notes

- All random seeds pinned at 42 (`torch.manual_seed(42)`, `np.random.seed(42)`)
- All forward-pass experiments use HuggingFace `AutoModelForCausalLM` with MPS (Apple Silicon) or CUDA (cloud GPU)
- TransformerLens is **not** used for forward passes (known MPS compatibility issue); present in `pyproject.toml` as an optional exploratory dependency only
- Model dtype: `float16` throughout; hooks on `model.model.layers[i]` (residual post), `.self_attn`, `.mlp`
- `use_cache=False` on all forward passes

Code in this repository was written with AI assistance. `CLAUDE.md` documents the technical conventions used throughout the codebase.

---

## Citation

```bibtex
@misc{keeman2026emotional,
  title         = {[Paper title — forthcoming]},
  author        = {Keeman, Michael},
  year          = {2026},
  eprint        = {[arXiv ID]},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/[arXiv ID]}
}
```

---

## License

Code: MIT — see [LICENSE](LICENSE).

Model weights are subject to their respective licenses:
- Llama 3: [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)
- Gemma 2: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

---

## Contact

Michael Keeman · michael@keidolabs.com · [@drkeeman](https://github.com/drkeeman) · [Keido Labs](https://keidolabs.com)

For issues with this repository, open a GitHub issue.
