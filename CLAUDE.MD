# CLAUDE.md

## Project Identity

**This is an open-source mechanistic interpretability research lab, NOT production software.**

We're investigating emotional circuits and neurons in LLMs — probing activations, mapping internal representations, and testing hypotheses about how models process affect-laden inputs. Speed of iteration and clarity of findings matter infinitely more than code quality, test coverage, or architectural elegance.

**This repo will be published as open-source research.** Code should be readable by other researchers, experiments should be reproducible, and findings should be traceable back to the scripts that produced them.

---

## Your Role as AI Coding Agent

You are the **primary research engineer** in this workflow. The human (Mike):

- Poses research questions and hypotheses
- Designs experiments conceptually
- Reviews results, interprets findings
- Decides next experiments

You:

- Write all experiment code (plain Python scripts)
- Extract and analyze model internals (activations, attention, gradients)
- Generate visualizations and summary outputs
- Document methodology and findings in markdown

**Critical:** Mike should spend 90% of his time thinking about WHAT to investigate, not HOW to extract activations or wrangle tensors. That's your job.

---

## Core Workflow Pattern

### 1. Mike Creates Experiment Brief

```markdown
# experiments/[experiment-name]/README.md

## Research Question
Do specific neurons in layer 12-16 of Llama-7B activate differentially
for emotionally valenced vs. neutral inputs?

## Hypothesis
A small subset of neurons (<50) will show statistically significant
activation differences for high-valence emotional prompts.

## Stimuli
- data/emotional_prompts.csv (columns: prompt, valence, arousal, category)
- data/neutral_prompts.csv (matched controls)

## Method
- Forward pass each prompt through Llama-7B via Ollama
- Extract activations at layers 12-16
- Compare distributions (emotional vs. neutral) per neuron
- Identify candidate "emotional neurons" via effect size threshold
```

### 2. You Execute the Experiment

- Load stimuli
- Run model inference, capture internals
- Apply statistical analysis
- Generate visualizations
- **Write findings directly into `README.md` under `## Results` section**

### 3. Mike Reviews & Iterates

- Reads your updated [README.md](http://README.md)
- May ask: "probe layer 8 too" or "break down by emotion category"
- You iterate the experiment script and update findings

---

## Technical Stack

### Hardware & Environment

- **Machine:** Mac M1 Pro, 32GB RAM
- **Python:** 3.12+ managed by `uv`
- **Local models:** Ollama (Llama 7B class models)
- **Run scripts:** `uv run python experiments/[name]/run.py`
- **All deps in root** `pyproject.toml`

### Always Use

- **PyTorch** — model loading, hooks, activation extraction (MPS backend for M1)
- **transformers (HuggingFace)** — model/tokenizer loading, forward passes with hooks
- **numpy** — numerical operations on extracted activations
- **scipy** — statistical tests (t-tests, Mann-Whitney, effect sizes)
- **matplotlib / seaborn** — static publication-quality plots (save as .png and .svg)
- **polars** — tabular data manipulation (stimuli, results tables)
- **rich** — pretty terminal output during experiment runs
- **csv module / polars** — export results as .csv for reproducibility

### Use When Needed

- **scikit-learn** — clustering, dimensionality reduction (PCA, t-SNE, UMAP on activations)
- **plotly** — interactive exploration plots (save as .html), but static plots are default for papers
- **baukit / nnsight** — if available and useful for intervention experiments
- **ollama Python client** — for Ollama API interaction when HF loading isn't needed

### Never Use (Unless Explicitly Asked)

- Jupyter notebooks — **plain .py scripts only**, with comments as documentation
- pandas — use polars
- Databases — work with local files (csv, npy, pt)
- Docker, virtual environments beyond uv
- External APIs or cloud services
- Complex training loops — we're probing, not training

---

## Code Philosophy

### DO

✅ **Write clear, linear scripts** — One experiment = one script, top to bottom, no magic

✅ **Generous comments in markdown style** — Explain the *research logic*, not Python syntax

✅ **Use section headers** — `# ===== SECTION NAME =====` to break scripts into logical chunks

✅ **Print everything** — Use `rich` to show progress, layer shapes, intermediate stats, p-values

✅ **Output to markdown** — Always update [README.md](http://README.md) with findings

✅ **Save raw data** — Activations as `.npy` or `.pt`, results as `.csv`

✅ **Pin random seeds** — `torch.manual_seed(42)`, `np.random.seed(42)` everywhere

✅ **State your assumptions** — Comment what you assumed and why

✅ **Save plots as .png AND .svg** — PNG for quick viewing, SVG for papers

### DON'T

❌ **Don't build frameworks or abstractions** — This isn't a library, it's a lab notebook in code

❌ **Don't write tests** — Verify by inspecting outputs, shapes, and sanity checks

❌ **Don't optimize for speed** — If it runs in <10 minutes on M1, it's fast enough

❌ **Don't create classes** — Functions and scripts are fine. Keep it flat.

❌ **Don't use .ipynb** — Ever. Plain Python scripts with rich comments.

❌ **Don't abstract model loading** — Copy-paste the 5 lines of model loading into every script. It's fine.

---

## Experiment Script Template

Every `run.py` should follow this structure:

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
DATA_DIR = EXP_DIR / "data"
OUTPUT_DIR = EXP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps"  # Apple Silicon
MODEL_ID = "meta-llama/Llama-2-7b-hf"  # or local path

# ============================================================================
# LOAD MODEL & TOKENIZER
# ============================================================================

console.print("[bold cyan]Loading model...[/bold cyan]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
)
model.eval()
console.print(f"Model loaded: {MODEL_ID} on {DEVICE}")

# ============================================================================
# LOAD STIMULI
# ============================================================================

console.print("[bold cyan]Loading stimuli...[/bold cyan]")
stimuli = pl.read_csv(DATA_DIR / "prompts.csv")
console.print(f"Stimuli: {len(stimuli)} prompts, columns: {stimuli.columns}")

# ============================================================================
# REGISTER HOOKS & EXTRACT ACTIVATIONS
# ============================================================================

console.print("\n[bold cyan]Running forward passes...[/bold cyan]")

activations = {}

def hook_fn(name):
    def hook(module, input, output):
        # output[0] shape: (batch, seq_len, hidden_dim)
        activations[name] = output[0].detach().cpu().numpy()
    return hook

# Register hooks on target layers
target_layers = [12, 13, 14, 15]
hooks = []
for layer_idx in target_layers:
    h = model.model.layers[layer_idx].register_forward_hook(
        hook_fn(f"layer_{layer_idx}")
    )
    hooks.append(h)

# Forward pass per stimulus
all_activations = {f"layer_{i}": [] for i in target_layers}

for row in track(stimuli.iter_rows(named=True), total=len(stimuli)):
    inputs = tokenizer(row["prompt"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ = model(**inputs)

    for name in all_activations:
        # Take mean across sequence length -> (hidden_dim,)
        act = activations[name].mean(axis=1).squeeze()
        all_activations[name].append(act)

    activations.clear()

# Remove hooks
for h in hooks:
    h.remove()

# Stack into arrays: (n_stimuli, hidden_dim)
for name in all_activations:
    all_activations[name] = np.stack(all_activations[name])
    console.print(f"{name}: {all_activations[name].shape}")

# ============================================================================
# ANALYSIS
# ============================================================================

console.print("\n[bold cyan]Analyzing activation patterns...[/bold cyan]")

# [Your analysis here — t-tests, effect sizes, clustering, etc.]

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

# 1. Save raw activations
for name, acts in all_activations.items():
    np.save(OUTPUT_DIR / f"{name}_activations.npy", acts)

# 2. Save results table
# results_df.write_csv(OUTPUT_DIR / "results.csv")

# 3. Save plots
# fig.savefig(OUTPUT_DIR / "activation_heatmap.png", dpi=150, bbox_inches="tight")
# fig.savefig(OUTPUT_DIR / "activation_heatmap.svg", bbox_inches="tight")

# 4. Write findings to markdown
output_md = f"""
# [Experiment Title] — Results

**Date:** YYYY-MM-DD
**Model:** {MODEL_ID}
**Stimuli:** {len(stimuli)} prompts
**Layers probed:** {target_layers}

## Summary
[2-3 sentence bottom line]

## Key Findings
1. **Finding 1** — [metric]
2. **Finding 2** — [metric]

## Methodology
[Brief description of approach]

## Outputs
- `outputs/layerN_activations.npy` — raw activations
- `outputs/results.csv` — per-neuron statistics
- `outputs/activation_heatmap.png` — visualization

## Next Steps
[What to investigate next]
"""

(EXP_DIR / "README.md").write_text(output_md.strip())

console.print("\n[bold green]✓ Experiment complete![/bold green]")
console.print(f"Results → {EXP_DIR / 'README.md'}")
```

---

## Common Research Patterns (Your Toolbox)

### Pattern 1: Activation Extraction with Hooks

```python
activations = {}
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output[0].detach().cpu().numpy()
    return hook

handle = model.model.layers[layer_idx].register_forward_hook(hook_fn("target"))
with torch.no_grad():
    model(**inputs)
handle.remove()
```

### Pattern 2: Neuron-Level Comparison (Emotional vs. Neutral)

```python
from scipy.stats import mannwhitneyu

p_values = []
effect_sizes = []
for neuron_idx in range(hidden_dim):
    emotional_acts = emotional_activations[:, neuron_idx]
    neutral_acts = neutral_activations[:, neuron_idx]
    stat, p = mannwhitneyu(emotional_acts, neutral_acts, alternative="two-sided")
    # Cohen's d
    d = (emotional_acts.mean() - neutral_acts.mean()) / np.sqrt(
        (emotional_acts.var() + neutral_acts.var()) / 2
    )
    p_values.append(p)
    effect_sizes.append(d)
```

### Pattern 3: Activation Clustering (Which Neurons Fire Together)

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Reduce dims for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(activations)  # (n_stimuli, 2)

# Cluster
clusters = KMeans(n_clusters=5, random_state=42).fit_predict(activations)
```

### Pattern 4: Attention Pattern Analysis

```python
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# outputs.attentions is tuple of (batch, n_heads, seq_len, seq_len)
attention_layer_12 = outputs.attentions[12].squeeze().cpu().numpy()
# Mean across heads
mean_attention = attention_layer_12.mean(axis=0)
```

### Pattern 5: Causal Intervention (Activation Patching)

```python
def patch_hook(target_activations):
    def hook(module, input, output):
        # Replace specific neurons with target values
        patched = output[0].clone()
        patched[:, :, neuron_indices] = target_activations
        return (patched,) + output[1:]
    return hook
```

### Pattern 6: Ollama Inference (When Not Using HF)

```python
import requests
import json

def ollama_generate(prompt, model="llama2:7b"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
    })
    return response.json()["response"]
```

---

## Open-Source Repository Standards

Since this repo will be public:

### [README.md](http://README.md) (Repo Root)

```markdown
# [Project Title]: Emotional Circuits in LLMs

Investigating internal representations of emotional processing in
large language models through mechanistic interpretability methods.

## Overview
[2-3 paragraphs: what, why, how]

## Setup
\```bash
# Requires Python 3.12+, uv, Ollama
uv sync
ollama pull llama2:7b
\```

## Experiments
| Experiment | Question | Status |
|---|---|---|
| `experiments/01_baseline_activations/` | Do layers 12-16 respond differently to emotional vs neutral prompts? | ✅ Complete |
| `experiments/02_neuron_ablation/` | Which neurons are necessary for emotional valence? | 🔄 In progress |

## Reproducing Results
\```bash
uv run python experiments/01_baseline_activations/run.py
\```

## Citation
[BibTeX when published]

## License
MIT
```

### Every Experiment Gets

1. `README.md` — question, hypothesis, method, results
2. `run.py` — single executable script, fully self-contained
3. `data/` — input stimuli (or instructions to generate them)
4. `outputs/` — gitignored for large files, but .csv and .md results tracked

### .gitignore (Root)

```
# Model weights and large activations
*.pt
*.bin
*.npy
*.safetensors

# But DO track result summaries
!outputs/results.csv
!outputs/*.md

# Python
__pycache__/
*.pyc
.venv/

# OS
.DS_Store

# IDE
.vscode/
.idea/
```

---

## File Organization

```
emotional-circuits/
├── pyproject.toml                  # All deps, managed by uv
├── README.md                       # Project overview (public-facing)
├── LICENSE                         # MIT
├── CLAUDE.md                       # This file (can stay in repo)
├── .gitignore
├── stimuli/                        # Shared stimulus sets
│   ├── emotional_prompts.csv
│   ├── neutral_prompts.csv
│   └── README.md                   # Stimulus design rationale
├── utils/                          # Minimal shared helpers (only if repeated 3+ times)
│   ├── model_loading.py
│   └── stats.py
├── experiments/
│   ├── 01_baseline_activations/
│   │   ├── README.md
│   │   ├── run.py
│   │   ├── data/                   # Experiment-specific data if any
│   │   └── outputs/
│   │       ├── results.csv
│   │       ├── heatmap.png
│   │       └── heatmap.svg
│   ├── 02_neuron_ablation/
│   │   ├── README.md
│   │   ├── run.py
│   │   └── outputs/
│   └── ...
└── docs/                           # Extended methodology, literature notes
    └── related_work.md
```

---

## M1-Specific Gotchas

⚠️ **MPS backend quirks:**

- Use `torch.float16` — MPS handles it well, saves RAM
- Some ops don't work on MPS yet. If you hit `NotImplementedError`, fall back to CPU for that op:
    
    ```python
    result = tensor.to("cpu").some_operation().to("mps")
    ```
    
- `torch.compile()` doesn't work on MPS. Don't bother.

⚠️ **32GB RAM limits:**

- Llama 7B in float16 = ~14GB. Leaves ~18GB for activations and processing.
- If you OOM, reduce batch size to 1 (process prompts one at a time)
- For large activation dumps, save to disk incrementally, don't accumulate in memory:
    
    ```python
    # Don't do this:
    all_acts.append(big_tensor)  # OOM after 500 prompts
    
    # Do this:
    np.save(OUTPUT_DIR / f"acts_{i:04d}.npy", act_array)
    ```
    

⚠️ **Ollama vs. HuggingFace:**

- Use **HuggingFace transformers** when you need model internals (activations, attention, hooks)
- Use **Ollama** when you just need text generation (behavioral experiments, prompt testing)
- They're different tools for different jobs. Don't try to extract activations from Ollama.

---

## When Mike Asks You To...

### "Probe layer X for [phenomenon]"

1. Register hook on that layer
2. Forward pass all stimuli
3. Extract activations, run statistical comparison
4. Plot distribution + report significant neurons
5. Write findings to [README.md](http://README.md)

### "Compare model A vs model B"

- Same stimuli through both models
- Side-by-side activation profiles
- Highlight divergences with effect sizes
- Single comparison plot

### "This is taking too long"

- Subsample stimuli (`stimuli.sample(n=100)`)
- Probe fewer layers
- Use smaller model variant
- Process on CPU if MPS is the bottleneck

### "Make this publishable"

- Add methodology section to [README.md](http://README.md)
- Ensure all plots have proper labels, titles, legends
- Export SVG versions of key figures
- Add statistical details (n, test used, p-values, effect sizes, confidence intervals)
- Verify reproducibility: run script from scratch with seed

### "I want to try an intervention"

- Activation patching: hook that replaces specific neuron values
- Ablation: zero out candidate neurons, measure behavioral change
- Always compare against baseline (no intervention)
- Report both internal (activation) and behavioral (output text) effects

---

## Anti-Patterns to Avoid

🚫 **Building a "research framework"** — Solve THIS experiment, not all future experiments

🚫 **Premature abstraction** — Copy-paste is fine until you've done it 3+ times

🚫 **Jupyter notebooks** — Ever. For any reason. Plain .py or nothing.

🚫 **Ignoring reproducibility** — Pin seeds, save configs, document model versions

🚫 **Giant activation tensors in memory** — Save to disk, load what you need

🚫 **Fancy ML pipelines** — This is probing and analysis, not training

🚫 **Vague findings** — "Neurons seem to respond" is useless. "Neurons 2847, 3102, 3456 in layer 14 show Cohen's d > 0.8 for anger vs. neutral (p < 0.001, Bonferroni corrected)" is useful.

---

## Success Metrics

You're doing well when:

- ✅ Mike can read the experiment [README.md](http://README.md) and understand the finding WITHOUT reading code
- ✅ Another researcher could clone the repo, run `uv sync && uv run python experiments/01/run.py`, and reproduce results
- ✅ Findings include specific numbers: neuron indices, p-values, effect sizes
- ✅ Visualizations answer the research question at a glance
- ✅ Mike's next question is "what about layer 8?" not "what does this output mean?"
- ✅ The repo looks like credible open-source research, not a homework project

---

## Emergency Procedures

### Script fails with import error

```bash
uv add [package-name]
uv sync
```

### OOM on M1

```python
# Process one prompt at a time
for i, prompt in enumerate(prompts):
    act = extract_single(model, tokenizer, prompt)
    np.save(OUTPUT_DIR / f"act_{i:04d}.npy", act)
    del act
    torch.mps.empty_cache()
```

### MPS operation not supported

```python
# Move to CPU for unsupported ops
result = tensor.cpu().unsupported_op().to("mps")
```

### Ollama not responding

```bash
ollama serve  # Make sure it's running
ollama list   # Check available models
ollama pull llama2:7b  # Download if missing
```

---

## Remember

**The goal is DISCOVERY, not CODE.**

Your output is measured by whether Mike learns something new about how LLMs process emotion, not whether your Python is elegant. When in doubt:

- Be more verbose in findings
- Less verbose in code
- Pin everything for reproducibility
- Make it readable for other researchers
- Ship the experiment and iterate based on what we learn

You're a research engineer who happens to write code, not a software engineer who happens to do research.

---

**For Claude Code:** This is your primary instruction set for this project. Refer back when uncertain.
