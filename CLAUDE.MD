# CLAUDE.md — Technical Reference

This file documents the technical conventions, tooling, and code patterns used throughout this codebase. It is included in the public repository as a reference for contributors and replicators.

---

## Project

Mechanistic interpretability research into emotional representations in LLMs. This is a research lab in code, not production software — clarity of findings and reproducibility matter more than architectural elegance.

---

## Environment

- **Python:** 3.12+ managed by `uv`
- **Hardware:** Mac M1 Pro 32GB (MPS backend) · Vast.ai A6000 (CUDA)
- **Run scripts:** `uv run python experiments/[name]/run.py`
- **All deps in root** `pyproject.toml`

---

## Tech Stack

### Always use

- **PyTorch** — model loading, hooks, activation extraction
- **transformers (HuggingFace)** — `AutoModelForCausalLM`, forward passes with hooks
- **numpy** — numerical operations on activations
- **scipy** — statistical tests (Mann-Whitney, t-tests, effect sizes)
- **matplotlib / seaborn** — static publication-quality plots
- **polars** — tabular data (not pandas)
- **rich** — terminal output, progress bars

### Use when needed

- **scikit-learn** — logistic regression probes, PCA, clustering
- **plotly** — interactive exploration (`.html`), not default for papers

### Never use

- Jupyter notebooks — plain `.py` scripts only
- pandas — use polars
- Databases — local files only (csv, npy, pt)
- Complex training loops — we probe, we don't train

---

## Code Conventions

**Do:**
- One experiment = one script, top to bottom, no magic
- Section headers: `# ============================================================`
- Use `rich` for all terminal output (progress, shapes, p-values)
- Pin seeds everywhere: `torch.manual_seed(42)`, `np.random.seed(42)`
- Save plots as both `.png` (150 dpi) and `.svg`
- Save raw activations as `.npy` / `.npz`; results as `.csv`
- Comment the *research logic*, not Python syntax
- Write findings to each experiment's `README.md`

**Don't:**
- Build abstractions — copy-paste is fine until it happens 3+ times
- Create classes — flat functions and scripts
- Optimise for speed — if it runs in <10 min on M1 it's fast enough
- Use `.ipynb` — ever

---

## Experiment Structure

Every experiment directory contains:

```
experiments/[name]/
├── README.md      # Research question, hypothesis, method, findings
├── run.py         # Single executable script
└── outputs/
    ├── results.csv
    ├── figure.png
    └── figure.svg
```

### Script template

```python
"""
Experiment: [Short title]
Question: [One-line research question]
Date: YYYY-MM-DD
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import track
import polars as pl

console = Console()
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps"  # or cuda
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# ============================================================
# LOAD MODEL
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
model.eval()

# ============================================================
# REGISTER HOOKS
# ============================================================
activations = {}
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = (output[0] if isinstance(output, tuple) else output)[0, -1, :].detach().cpu().numpy()
    return hook

handles = []
for l in range(model.config.num_hidden_layers):
    handles.append(model.model.layers[l].register_forward_hook(hook_fn(f"layer_{l}")))

# ============================================================
# FORWARD PASSES
# ============================================================
# ... run stimuli, collect activations ...

for h in handles:
    h.remove()

# ============================================================
# ANALYSIS & OUTPUTS
# ============================================================
# ... stats, plots, write README.md ...
```

---

## Common Patterns

### Activation extraction (final token)

```python
def hook_fn(name):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out[0, -1, :].detach().cpu().to(torch.float32).numpy()
    return hook
```

### Mann-Whitney + Cohen's d

```python
from scipy.stats import mannwhitneyu

for i in range(hidden_dim):
    stat, p = mannwhitneyu(emo[:, i], neu[:, i], alternative="two-sided")
    d = (emo[:, i].mean() - neu[:, i].mean()) / np.sqrt((emo[:, i].var() + neu[:, i].var()) / 2)
```

### Logistic regression probe

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in cv.split(X, y):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X[train_idx], y[train_idx])
    auroc = roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx]), multi_class="ovr")
```

### Causal patching

```python
def patch_hook(source_act, pos):
    def hook(module, input, output):
        out = list(output)
        out[0] = out[0].clone()
        out[0][:, pos, :] = torch.from_numpy(source_act).to(out[0].device, dtype=out[0].dtype)
        return tuple(out)
    return hook
```

### Attention extraction

```python
with torch.no_grad():
    outputs = model(**inputs, use_cache=False, output_attentions=True)
# outputs.attentions: tuple of (batch, n_heads, seq_len, seq_len)
attn_at_final = outputs.attentions[layer_idx][0, :, -1, :]  # (n_heads, seq_len)
```

---

## M1 / MPS Gotchas

- Use `torch.float16` — MPS handles it well, saves RAM
- `use_cache=False` on all forward passes
- `attn_implementation="eager"` required for `output_attentions=True` on MPS (SDPA does not return attention weights)
- `torch.compile()` does not work on MPS — don't use it
- If an op raises `NotImplementedError`, fall back: `tensor.cpu().op().to("mps")`
- OOM strategy: save per-stimulus to disk, don't accumulate:
  ```python
  np.save(OUTPUT_DIR / f"act_{i:04d}.npy", act)
  del act
  torch.mps.empty_cache()
  ```

---

## HuggingFace Auth

Token in `.env` at project root:
```
HF_ACCESS_TOKEN=hf_...
```

Load pattern used in all extraction scripts:
```python
env_path = Path(__file__).parent.parent.parent / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
```
