"""
Experiment 20 — Binary Confound Control
Question: Does AUROC ~1.0 reflect genuine affect detection, or just narrative richness?
Date: 2026-02-28

Trains a frozen binary probe on Set B (clinical vs neutral), then scores
24 high-complexity neutral narratives (Set C). Set C is matched to Set B
emotional vignettes on narrative richness, length, and sentence structure —
but contains zero emotional content (no human stakes, no interpersonal dynamics,
no moral weight).

If the probe fires on Set C → it's detecting narrative complexity, not affect.
If the probe stays low on Set C → binary AUROC reflects genuine emotion signal.

Run: uv run python experiments/20_binary_confound_control/run.py
"""

import gc
import json
import os
import re
import sys
import time
import torch
import numpy as np
from pathlib import Path

import joblib
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

console = Console()
torch.manual_seed(42)
np.random.seed(42)

EXP_DIR      = Path(__file__).parent
ROOT         = EXP_DIR.parent.parent
ACT_DIR      = EXP_DIR / "outputs" / "activations"
FIG_DIR      = EXP_DIR / "outputs" / "figures"
OUT_DIR      = EXP_DIR / "outputs"
STIMULI_DIR  = ROOT / "stimuli" / "exp20"
# Set B activations from Exp 12 — same model, same few-shot prompt
SETB_ACT_DIR = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations" / "llama1b_inst"

for d in [ACT_DIR, FIG_DIR, STIMULI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_KEY = "llama1b_inst"
MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE    = "cpu"   # TransformerLens on MPS gives silently wrong results (TL issue #1178)

# ============================================================================
# SAME FEW-SHOT PROMPT AS EXP 12
# Critical: Set C activations must be in the same "prompt space" as Set B
# to make the comparison valid. The spec shows a different template —
# using the actual Exp 12 template here.
# ============================================================================

SHOT_1_TEXT    = "My dog died last week. I miss him every day."
SHOT_1_EMOTION = "sadness"
SHOT_2_TEXT    = "I got promoted and my boss praised my work in front of everyone."
SHOT_2_EMOTION = "joy"

def build_tak_prompt(target_text: str) -> str:
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\n"
        f"Answer: {SHOT_1_EMOTION}\n\n"
        f"Context: {SHOT_2_TEXT}\n"
        f"Answer: {SHOT_2_EMOTION}\n\n"
        f"Context: {target_text}\n"
        "Answer:"
    )

# ============================================================================
# STEP 1: PARSE SET C MARKDOWN → JSONL
# ============================================================================

console.print("[bold cyan]Step 1: Parsing Set C stimuli from markdown...[/bold cyan]")

SETC_MD_PATH   = ROOT / "stimuli" / "exp20" / "set_c_complex_neutral.md"
SETC_JSONL_PATH = STIMULI_DIR / "set_c_complex_neutral.jsonl"

# Domain mapping by stimulus number
def get_domain(stim_id: str) -> str:
    n = int(stim_id.split("-")[1])
    if   n <= 6:  return "technical_mechanical"
    elif n <= 12: return "natural_environment"
    elif n <= 18: return "commercial_urban"
    else:         return "routine_process"

md_text = SETC_MD_PATH.read_text()

# Split on "### C-NN" section headers to extract each stimulus block
stimulus_blocks = re.split(r"\n### (C-\d+)\n", md_text)
# After split: [preamble, "C-01", text_block_1, "C-02", text_block_2, ...]

set_c_stimuli = []
i = 1
while i < len(stimulus_blocks) - 1:
    stim_id_raw  = stimulus_blocks[i].strip()          # e.g. "C-01"
    stim_text_raw = stimulus_blocks[i + 1]

    # Truncate at any subsequent section break (--- or ##)
    stim_text = stim_text_raw.split("\n---")[0].split("\n##")[0].strip()
    # Collapse internal whitespace to single spaces
    stim_text = " ".join(stim_text.split())

    if stim_text and stim_id_raw.startswith("C-"):
        stim_id = stim_id_raw.lower()   # "c-01"
        set_c_stimuli.append({
            "id":             stim_id,
            "text":           stim_text,
            "emotion":        "neutral",
            "stimulus_set":   "C_complex_neutral",
            "expected_class": "neutral",
            "domain":         get_domain(stim_id),
        })
    i += 2

assert len(set_c_stimuli) == 24, f"Expected 24 Set C stimuli, got {len(set_c_stimuli)}"
console.print(f"  Parsed {len(set_c_stimuli)} stimuli")

# Save JSONL for reproducibility
with open(SETC_JSONL_PATH, "w") as f:
    for s in set_c_stimuli:
        f.write(json.dumps(s) + "\n")
console.print(f"  Saved: {SETC_JSONL_PATH}")

domain_counts = {}
for s in set_c_stimuli:
    domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1
console.print(f"  Domain distribution: {domain_counts}")

# ============================================================================
# STEP 2: LOAD SET B ACTIVATIONS (from Exp 12)
# ============================================================================

console.print("\n[bold cyan]Step 2: Loading Set B activations from Exp 12...[/bold cyan]")

manifest = pl.read_csv(SETB_ACT_DIR / "manifest.csv")

setb_emo_acts = []   # clinical (emotional) — label = 1
setb_neu_acts = []   # neutral controls — label = 0

for row in manifest.iter_rows(named=True):
    npz_path = SETB_ACT_DIR / f"{row['id']}.npz"
    if not npz_path.exists():
        continue

    npz = np.load(npz_path, allow_pickle=True)
    act = npz["h"].copy()   # (n_layers, d_model) — key is "h" in Exp 12 files

    if not np.all(np.isfinite(act)):
        act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

    if row["stimulus_set"] == "B_clinical":
        setb_emo_acts.append(act)
    elif row["stimulus_set"] == "neutral":
        setb_neu_acts.append(act)

setb_emo = np.stack(setb_emo_acts)   # (96, n_layers, d_model)
setb_neu = np.stack(setb_neu_acts)   # (96, n_layers, d_model)

assert setb_emo.shape[0] == 96, f"Expected 96 emotional, got {setb_emo.shape[0]}"
assert setb_neu.shape[0] == 96, f"Expected 96 neutral, got {setb_neu.shape[0]}"

N_LAYERS = setb_emo.shape[1]   # 16 for Llama-1B
D_MODEL  = setb_emo.shape[2]   # 2048

console.print(f"  Set B emotional: {setb_emo.shape}")
console.print(f"  Set B neutral:   {setb_neu.shape}")

# ============================================================================
# STEP 3: LOAD MODEL
# ============================================================================

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
console.print("\n[bold cyan]HF token loaded[/bold cyan]")

console.print(f"\n[bold cyan]Step 3: Loading {MODEL_ID} on CPU (TransformerLens, float32)...[/bold cyan]")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.float32)
model.eval()

console.print(f"[bold green]✓ Model loaded[/bold green] — {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# Verify the final token is ':' — same extraction point as Exp 12
test_tokens   = model.to_tokens(build_tak_prompt("Test text."))
final_tok_str = model.tokenizer.decode([test_tokens[0, -1].item()])
assert ":" in final_tok_str, f"Final token mismatch: {repr(final_tok_str)}"
console.print(f"  ✓ Final token check: {repr(final_tok_str)}")

# ============================================================================
# STEP 4: EXTRACT SET C ACTIVATIONS (24 stimuli)
# ============================================================================

console.print(f"\n[bold cyan]Step 4: Extracting Set C activations (24 stimuli)...[/bold cyan]")

def names_filter(name):
    return "hook_resid_post" in name

setc_acts = np.zeros((len(set_c_stimuli), N_LAYERS, D_MODEL), dtype=np.float32)
t_start = time.time()

for i, stim in enumerate(track(set_c_stimuli, description="Extracting Set C...")):
    npz_path = ACT_DIR / f"{stim['id']}.npz"

    # Use cache if already extracted (resumability)
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        setc_acts[i] = npz["h"]
        continue

    prompt    = build_tak_prompt(stim["text"])
    tokens    = model.to_tokens(prompt)
    colon_pos = tokens.shape[1] - 1   # ':' is the last token

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)

    h_all = np.stack([
        cache["resid_post", l][:, colon_pos, :].squeeze().cpu().numpy()
        for l in range(N_LAYERS)
    ]).astype(np.float32)   # (16, 2048)

    setc_acts[i] = h_all
    np.savez_compressed(npz_path, h=h_all)
    del cache
    gc.collect()

elapsed = time.time() - t_start
console.print(f"  ✓ Set C extracted in {elapsed:.1f}s — shape: {setc_acts.shape}")

# Unload model — no longer needed
del model
gc.collect()

# ============================================================================
# STEP 5: TRAIN BINARY PROBES ON SET B (all 16 layers, FULL training set)
# ============================================================================
#
# Train on the FULL Set B (no CV). We want the strongest possible probe for
# diagnostic purposes — CV AUROC measures generalization; here we want to
# know if the representation can encode the distinction at all before testing
# on Set C.
#
# IMPORTANT — peak layer selection:
# With 192 samples in 2048 dimensions, logistic regression perfectly separates
# the training set at EVERY layer (train AUROC = 1.0000 everywhere). Using
# argmax(train_auroc) would pick L0, which is just token embeddings — a
# vocabulary-level bag-of-words classifier, not what the model computed.
#
# The correct peak is L10, established by 5-fold CV AUROC in Phase 1 / Exp 12.
# That's where the model's computed representations (not raw vocab) are most
# discriminative. We hardcode it here and verify it matches expectations.
# ============================================================================

PEAK_LAYER = 10   # from Phase 1 / Exp 12 5-fold CV AUROC (0.9995) for llama1b_inst Set B binary

console.print("\n[bold cyan]Step 5: Training binary probes on full Set B...[/bold cyan]")
console.print(f"  Peak layer fixed at L{PEAK_LAYER} (from Phase 1 / Exp 12 CV results)")

# Stack Set B: (192, n_layers, d_model), labels: 1=emotional, 0=neutral
X_setb = np.concatenate([setb_emo, setb_neu], axis=0)   # (192, 16, 2048)
y_setb = np.array([1] * 96 + [0] * 96)

layer_probes  = {}    # layer → fitted LogisticRegression
layer_scalers = {}    # layer → (mu, std) for normalisation
layer_aurocs  = []    # train AUROC at each layer (diagnostic only, NOT used for peak selection)

for l in track(range(N_LAYERS), description="Training layer probes..."):
    X = X_setb[:, l, :]   # (192, 2048)

    mu  = X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True) + 1e-8
    X_n = (X - mu) / std

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_n, y_setb)

    # Train AUROC — expected to be ~1.0 everywhere due to p>>n (2048 dims, 192 samples)
    # This confirms the probe fitted correctly, not that the layer is informative.
    train_auroc = roc_auc_score(y_setb, clf.predict_proba(X_n)[:, 1])

    layer_probes[l]  = clf
    layer_scalers[l] = (mu, std)
    layer_aurocs.append(train_auroc)

peak_layer = PEAK_LAYER
console.print(f"\n  Train AUROC range: {min(layer_aurocs):.4f} – {max(layer_aurocs):.4f} "
              f"(all ~1.0 expected — p>>n overfitting)")
console.print(f"  Using L{peak_layer} as peak (CV-derived, not train-AUROC-derived)")

# Save frozen probes for reproducibility
joblib.dump(
    {"probes": layer_probes, "scalers": layer_scalers,
     "layer_aurocs": layer_aurocs, "peak_layer": peak_layer},
    OUT_DIR / "layer_probes.joblib"
)
console.print(f"  [bold green]✓ layer_probes.joblib saved[/bold green]")

# ============================================================================
# STEP 6: SCORE SET C + SET B REFERENCE AT ALL LAYERS
# ============================================================================

console.print("\n[bold cyan]Step 6: Scoring Set C and Set B reference across all layers...[/bold cyan]")

score_rows = []

# Set C
for i, stim in enumerate(set_c_stimuli):
    for l in range(N_LAYERS):
        mu, std = layer_scalers[l]
        x_n = (setc_acts[i, l, :].reshape(1, -1) - mu) / std
        p_emo = float(layer_probes[l].predict_proba(x_n)[0, 1])
        score_rows.append({
            "id":          stim["id"],
            "domain":      stim["domain"],
            "condition":   "C_complex_neutral",
            "layer":       l,
            "norm_depth":  round((l + 1) / N_LAYERS, 4),
            "p_emotional": round(p_emo, 6),
        })

# Set B emotional reference
for j in range(setb_emo.shape[0]):
    for l in range(N_LAYERS):
        mu, std = layer_scalers[l]
        x_n = (setb_emo[j, l, :].reshape(1, -1) - mu) / std
        p_emo = float(layer_probes[l].predict_proba(x_n)[0, 1])
        score_rows.append({
            "id":          f"setb_emo_{j:03d}",
            "domain":      "B_clinical",
            "condition":   "B_emotional",
            "layer":       l,
            "norm_depth":  round((l + 1) / N_LAYERS, 4),
            "p_emotional": round(p_emo, 6),
        })

# Set B neutral reference
for j in range(setb_neu.shape[0]):
    for l in range(N_LAYERS):
        mu, std = layer_scalers[l]
        x_n = (setb_neu[j, l, :].reshape(1, -1) - mu) / std
        p_emo = float(layer_probes[l].predict_proba(x_n)[0, 1])
        score_rows.append({
            "id":          f"setb_neu_{j:03d}",
            "domain":      "neutral_control",
            "condition":   "B_neutral",
            "layer":       l,
            "norm_depth":  round((l + 1) / N_LAYERS, 4),
            "p_emotional": round(p_emo, 6),
        })

df_all = pl.DataFrame(score_rows)
df_all.write_csv(OUT_DIR / "layer_profiles.csv")
console.print(f"  [bold green]✓ layer_profiles.csv saved ({len(score_rows)} rows)[/bold green]")

# Peak-layer Set C scores
setc_peak = df_all.filter(
    (pl.col("condition") == "C_complex_neutral") & (pl.col("layer") == peak_layer)
)
p_emo_vals = setc_peak["p_emotional"].to_numpy()
mean_c     = float(p_emo_vals.mean())
median_c   = float(np.median(p_emo_vals))
n_above_50 = int((p_emo_vals > 0.5).sum())
n_above_80 = int((p_emo_vals > 0.8).sum())

# Save per-stimulus confound scores
confound_df = setc_peak.with_columns(
    (pl.col("p_emotional") > 0.5).alias("predicted_emotional")
).select(["id", "domain", "p_emotional", "predicted_emotional"])
confound_df.write_csv(OUT_DIR / "confound_scores.csv")

# Print per-stimulus table
table = Table(title=f"Set C — Binary Probe Scores at L{peak_layer} (peak layer)")
table.add_column("ID", style="cyan")
table.add_column("Domain")
table.add_column("P(emotional)", justify="right")
table.add_column("Predicted", justify="right")

for row in confound_df.sort("p_emotional", descending=True).iter_rows(named=True):
    pred_style = "red" if row["predicted_emotional"] else "green"
    pred_label = "emotional" if row["predicted_emotional"] else "neutral"
    table.add_row(
        row["id"], row["domain"],
        f"{row['p_emotional']:.4f}",
        f"[{pred_style}]{pred_label}[/{pred_style}]"
    )

console.print(table)
console.print(f"\n  Set C mean P(emotional) at L{peak_layer}: {mean_c:.4f}")
console.print(f"  Set C median: {median_c:.4f}")
console.print(f"  Above 0.5: {n_above_50}/24  |  Above 0.8: {n_above_80}/24")

# ============================================================================
# STEP 7: FIGURES
# ============================================================================

console.print("\n[bold cyan]Step 7: Generating figures...[/bold cyan]")

# Helper: compute per-layer mean ± SD for one condition
def layer_stats(condition: str):
    subset = df_all.filter(pl.col("condition") == condition)
    means, stds = [], []
    for l in range(N_LAYERS):
        vals = subset.filter(pl.col("layer") == l)["p_emotional"].to_numpy()
        means.append(vals.mean())
        stds.append(vals.std())
    return np.array(means), np.array(stds)

norm_depths   = np.array([(l + 1) / N_LAYERS for l in range(N_LAYERS)])
c_mean,  c_std  = layer_stats("C_complex_neutral")
be_mean, be_std = layer_stats("B_emotional")
bn_mean, bn_std = layer_stats("B_neutral")

# --- Figure 1: P(emotional) histogram at peak layer ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(p_emo_vals, bins=20, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(0.5,    color="black", linestyle="--", linewidth=1.5, alpha=0.7,
           label="Decision boundary (0.5)")
ax.axvline(mean_c, color="red",   linestyle="-",  linewidth=2.0, alpha=0.85,
           label=f"Mean = {mean_c:.3f}")
ax.set_xlabel("P(emotional)")
ax.set_ylabel(f"Count (n={len(p_emo_vals)})")
ax.set_title(f"Set C (High-Complexity Neutral) — Binary Probe at L{peak_layer}\n"
             f"Llama-3.2-1B-Instruct · h (residual stream)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(FIG_DIR / "set_c_histogram.png", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "set_c_histogram.svg", bbox_inches="tight")
plt.close()
console.print("  ✓ set_c_histogram.png/.svg")

# --- Figure 2: Layer profile overlay ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(norm_depths, be_mean, color="black", linewidth=2.0,
        label="Set B emotional (reference)", marker="o", markersize=3)
ax.fill_between(norm_depths, be_mean - be_std, be_mean + be_std,
                color="black", alpha=0.10)

ax.plot(norm_depths, bn_mean, color="gray", linewidth=2.0,
        label="Set B neutral (reference)", marker="o", markersize=3)
ax.fill_between(norm_depths, bn_mean - bn_std, bn_mean + bn_std,
                color="gray", alpha=0.10)

ax.plot(norm_depths, c_mean, color="steelblue", linewidth=2.5, linestyle="--",
        label="Set C (complex neutral — test)", marker="s", markersize=4)
ax.fill_between(norm_depths, c_mean - c_std, c_mean + c_std,
                color="steelblue", alpha=0.15)

# Mark peak layer
ax.axvline((peak_layer + 1) / N_LAYERS, color="steelblue", linestyle=":",
           alpha=0.5, label=f"Peak layer L{peak_layer}")
ax.axhline(0.5, color="black", linestyle="-", alpha=0.12)

ax.set_xlabel("Normalized depth")
ax.set_ylabel("Mean P(emotional)")
ax.set_title("Binary Probe Layer Profiles: Set C vs Set B Reference\n"
             "Llama-3.2-1B-Instruct · h (residual stream) · probe trained on full Set B")
ax.legend(loc="center right", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "layer_profiles.png", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "layer_profiles.svg", bbox_inches="tight")
plt.close()
console.print("  ✓ layer_profiles.png/.svg")

# ============================================================================
# STEP 8: VERDICT
# ============================================================================

if mean_c < 0.3:
    verdict = "CLEAN"
    verdict_detail = (
        f"Probe distinguishes emotional from vivid-neutral content. "
        f"Set C mean P(emotional) = {mean_c:.4f} — well below decision boundary. "
        "Binary AUROC reflects genuine affect detection, not narrative complexity. "
        "Suitable for Appendix validation in paper."
    )
elif mean_c < 0.7:
    verdict = "AMBIGUOUS"
    verdict_detail = (
        f"Probe partially triggered by narrative complexity. "
        f"Set C mean P(emotional) = {mean_c:.4f} — in ambiguous zone (0.3–0.7). "
        "Run full diagnostic (Sets D–G) before drawing strong conclusions. "
        "8-class dissociation and patching results unaffected."
    )
else:
    verdict = "CONFOUNDED"
    verdict_detail = (
        f"Probe detects narrative richness, not affect. "
        f"Set C mean P(emotional) = {mean_c:.4f} — fires on vivid-neutral content. "
        "Binary AUROC claims require reframing. "
        "8-class dissociation and patching results unaffected."
    )

color_map = {"CLEAN": "bold green", "AMBIGUOUS": "bold yellow", "CONFOUNDED": "bold red"}
vc = color_map[verdict]
console.print(f"\n[{vc}]{'='*60}[/{vc}]")
console.print(f"[{vc}]VERDICT: {verdict}[/{vc}]")
console.print(f"  {verdict_detail}")
console.print(f"[{vc}]{'='*60}[/{vc}]")

# ============================================================================
# STEP 9: UPDATE README
# ============================================================================

# Per-domain summary table
domain_stats = {}
for row in confound_df.iter_rows(named=True):
    d = row["domain"]
    if d not in domain_stats:
        domain_stats[d] = []
    domain_stats[d].append(row["p_emotional"])

domain_table = ["| Domain | n | Mean P(emo) | Above 0.5 |",
                "|--------|---|-------------|-----------|"]
for domain, vals in sorted(domain_stats.items()):
    domain_table.append(
        f"| {domain} | {len(vals)} | {np.mean(vals):.4f} | {sum(v > 0.5 for v in vals)}/{len(vals)} |"
    )

layer_auroc_table = ["| Layer | Norm depth | Train AUROC |",
                     "|-------|-----------|-------------|"]
for l, auroc in enumerate(layer_aurocs):
    marker = " ← peak" if l == peak_layer else ""
    layer_auroc_table.append(f"| L{l} | {(l+1)/N_LAYERS:.3f} | {auroc:.4f}{marker} |")

results_section = f"""
## Results

**Date:** 2026-02-28
**Model:** Llama-3.2-1B-Instruct
**Activation type:** h (residual stream)
**Set B probe:** trained on full 192 stimuli (no CV — diagnostic mode)

### Set B Binary Probe — Train AUROC by Layer

{chr(10).join(layer_auroc_table)}

### Set C Scores at Peak Layer L{peak_layer}

| Metric | Value |
|--------|-------|
| Mean P(emotional) | {mean_c:.4f} |
| Median P(emotional) | {median_c:.4f} |
| Above 0.5 (predicted emotional) | {n_above_50}/24 |
| Above 0.8 | {n_above_80}/24 |

### By Domain

{chr(10).join(domain_table)}

### Verdict: {verdict}

{verdict_detail}
"""

readme_path = EXP_DIR / "README.md"
existing    = readme_path.read_text()
if "*To be filled after run.*" in existing:
    updated = existing.replace("*To be filled after run.*", results_section.strip())
else:
    updated = existing + "\n" + results_section
readme_path.write_text(updated)
console.print("\n[bold green]✓ README.md updated[/bold green]")

console.print(f"\n[bold green]✓ Experiment 20 complete![/bold green]")
console.print(f"  Verdict: {verdict}  |  Set C mean P(emo) = {mean_c:.4f}")
console.print(f"  Figures: {FIG_DIR}/")
console.print(f"  Scores:  {OUT_DIR}/confound_scores.csv")
