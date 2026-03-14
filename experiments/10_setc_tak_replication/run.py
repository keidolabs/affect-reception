"""
Experiment 10 — Methodology Consistency Check
Question: Does the v2 pipeline (Tak prompt + z-norm probes) agree with Phase 0 (passive reading)?
Date: 2026-02-25 (original), 2026-03-13 (rewritten)

This experiment validates that our core finding — late-layer emotion consolidation — is
NOT an artifact of prompt format, extraction point, or probe normalization.

WHAT IT DOES:
  1. Loads Phase 0 activations (Exp 00–03): raw text, final token, no prompt
  2. Loads v2 activations (Exp 11): Tak few-shot prompt, "Answer:" colon token
  3. Runs IDENTICAL z-normalized 8-class probes on both
  4. Compares layer-wise AUROC profiles

WHAT IS CONTROLLED:
  - Extraction framework: same per model (TL+CPU for 1B, HF+MPS for 8B/9B)
  - Stimuli: same 80 Set A emotional stimuli
  - Probe: identical z-normalized LogisticRegression, same folds, same seed
  - Models: same weights

WHAT VARIES (the thing being tested):
  - Prompt format: raw text (Phase 0) vs Tak few-shot + "Answer:" (v2)
  - Extraction position: end-of-text token (Phase 0) vs ":" token (v2)

EXPECTED RESULT:
  If late-layer consolidation is real → both conditions peak late, similar profile shape.
  If it was a prompt artifact → profiles diverge significantly.

Run: uv run python experiments/10_setc_tak_replication/run.py
"""

import json
import sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table

console = Console()
np.random.seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
OUT_DIR = EXP_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL REGISTRY — 4 models with both Phase 0 and v2 activations
# ============================================================================

MODELS = [
    {
        "key":       "llama1b",
        "v2_key":    "llama1b_base",
        "label":     "Llama-3.2-1B",
        "color":     "#e74c3c",
        "n_layers":  16,
        "framework": "TL+CPU+f32",
        "phase0_npy":  ROOT / "experiments/00_phase0_replication/outputs/activations/set_a_residuals.npy",
        "phase0_meta": ROOT / "experiments/00_phase0_replication/outputs/activations/metadata.json",
        "v2_dir":      ROOT / "experiments/11_v2_extract_seta/outputs/activations/llama1b_base",
    },
    {
        "key":       "llama8b",
        "v2_key":    "llama8b_base",
        "label":     "Llama-3-8B (base)",
        "color":     "#2980b9",
        "n_layers":  32,
        "framework": "HF+MPS+f16",
        "phase0_npy":  ROOT / "experiments/01_phase0a_llama3-8b/outputs/activations/set_a_residuals.npy",
        "phase0_meta": ROOT / "experiments/01_phase0a_llama3-8b/outputs/activations/metadata.json",
        "v2_dir":      ROOT / "experiments/11_v2_extract_seta/outputs/activations/llama8b_base",
    },
    {
        "key":       "llama8b_inst",
        "v2_key":    "llama8b_inst",
        "label":     "Llama-3-8B (instruct)",
        "color":     "#8e44ad",
        "n_layers":  32,
        "framework": "HF+MPS+f16",
        "phase0_npy":  ROOT / "experiments/02_phase0b_llama3-8b-instruct/outputs/activations/set_a_residuals.npy",
        "phase0_meta": ROOT / "experiments/02_phase0b_llama3-8b-instruct/outputs/activations/metadata.json",
        "v2_dir":      ROOT / "experiments/11_v2_extract_seta/outputs/activations/llama8b_inst",
    },
    {
        "key":       "gemma9b",
        "v2_key":    "gemma9b_base",
        "label":     "Gemma-2-9B",
        "color":     "#27ae60",
        "n_layers":  42,
        "framework": "HF+MPS+f16",
        "phase0_npy":  ROOT / "experiments/03_phase0c_gemma2-9b/outputs/activations/set_a_residuals.npy",
        "phase0_meta": ROOT / "experiments/03_phase0c_gemma2-9b/outputs/activations/metadata.json",
        "v2_dir":      ROOT / "experiments/11_v2_extract_seta/outputs/activations/gemma9b_base",
    },
]

# ============================================================================
# SECTION 1 — LOAD PHASE 0 ACTIVATIONS (passive reading, final token)
# ============================================================================

console.print("\n[bold cyan]Section 1: Loading Phase 0 activations (passive reading)...[/bold cyan]")

for m in MODELS:
    if not m["phase0_npy"].exists():
        console.print(f"  [red]MISSING: {m['phase0_npy']}[/red]")
        m["p0_acts"] = None
        m["p0_labels"] = None
        continue

    all_acts = np.load(m["phase0_npy"])  # (90, n_layers, d_model)
    with open(m["phase0_meta"]) as f:
        meta = json.load(f)

    # Filter to emotional only (exclude neutrals) — match v2 which has 80 emotional
    emo_mask = np.array([entry["emotion"] != "neutral" for entry in meta])
    acts = all_acts[emo_mask]  # (80, n_layers, d_model)
    labels = np.array([entry["emotion"] for entry in meta if entry["emotion"] != "neutral"])

    m["p0_acts"] = acts
    m["p0_labels"] = labels
    console.print(f"  {m['label']:30s} Phase 0: {acts.shape}, {len(np.unique(labels))} classes")

# ============================================================================
# SECTION 2 — LOAD V2 ACTIVATIONS (Tak prompt, ":" token)
# ============================================================================

console.print("\n[bold cyan]Section 2: Loading v2 activations (Tak prompt)...[/bold cyan]")

for m in MODELS:
    v2_dir = m["v2_dir"]
    manifest_path = v2_dir / "manifest.csv"
    if not manifest_path.exists():
        console.print(f"  [red]MISSING: {manifest_path}[/red]")
        m["v2_acts"] = None
        m["v2_labels"] = None
        continue

    manifest = pl.read_csv(manifest_path)
    # Exclude neutrals if any
    manifest = manifest.filter(pl.col("emotion") != "neutral")

    all_acts = []
    all_labels = []
    for row in manifest.iter_rows(named=True):
        npz_path = v2_dir / f"{row['id']}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        h = npz["h"]  # (n_layers, d_model) — residual stream
        if not np.all(np.isfinite(h)):
            h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        all_acts.append(h)
        all_labels.append(row["emotion"])

    if all_acts:
        m["v2_acts"] = np.stack(all_acts)  # (80, n_layers, d_model)
        m["v2_labels"] = np.array(all_labels)
        console.print(f"  {m['label']:30s} v2:      {m['v2_acts'].shape}, {len(np.unique(m['v2_labels']))} classes")
    else:
        m["v2_acts"] = None
        m["v2_labels"] = None
        console.print(f"  [red]{m['label']:30s} v2: no activations loaded[/red]")

# ============================================================================
# SECTION 3 — Z-NORMALIZED PROBES (identical methodology for both conditions)
# ============================================================================

console.print("\n[bold cyan]Section 3: Running z-normalized 8-class probes on both conditions...[/bold cyan]")


def probe_layers(acts, labels, n_layers):
    """
    8-class logistic regression probe with per-fold z-normalization.
    Matches Exp 13 methodology exactly.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classes = sorted(np.unique(labels))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_map[yi] for yi in labels])

    layer_aurocs = []
    layer_stds = []

    for layer_idx in range(n_layers):
        X = acts[:, layer_idx, :]
        fold_scores = []
        for train_idx, test_idx in skf.split(X, y_int):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_int[train_idx], y_int[test_idx]
            # Z-normalize per fold (matching Exp 13)
            mu = X_tr.mean(0, keepdims=True)
            std = X_tr.std(0, keepdims=True) + 1e-8
            X_tr_n = (X_tr - mu) / std
            X_te_n = (X_te - mu) / std
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X_tr_n, y_tr)
            y_prob = clf.predict_proba(X_te_n)
            if len(np.unique(y_te)) > 1:
                score = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
            else:
                score = float("nan")
            fold_scores.append(score)
        layer_aurocs.append(float(np.nanmean(fold_scores)))
        layer_stds.append(float(np.nanstd(fold_scores)))

    return layer_aurocs, layer_stds


for m in MODELS:
    console.print(f"\n  [cyan]{m['label']}[/cyan] ({m['framework']})")

    for condition, key_acts, key_labels in [
        ("Phase 0 (passive)",  "p0_acts",  "p0_labels"),
        ("v2 (Tak prompt)",    "v2_acts",  "v2_labels"),
    ]:
        acts = m.get(key_acts)
        labels = m.get(key_labels)
        if acts is None:
            console.print(f"    {condition:25s} — skipped (no data)")
            m[f"{key_acts}_aurocs"] = None
            m[f"{key_acts}_stds"] = None
            continue

        aurocs, stds = probe_layers(acts, labels, m["n_layers"])
        m[f"{key_acts}_aurocs"] = aurocs
        m[f"{key_acts}_stds"] = stds

        best_layer = int(np.argmax(aurocs))
        best_auroc = aurocs[best_layer]
        best_norm = best_layer / (m["n_layers"] - 1)

        console.print(
            f"    {condition:25s} peak: L{best_layer}/{m['n_layers']-1} "
            f"({best_norm*100:.1f}%) AUROC={best_auroc:.4f}"
        )

# ============================================================================
# SECTION 4 — COMPARISON TABLE
# ============================================================================

console.print("\n[bold cyan]Section 4: Methodology consistency comparison...[/bold cyan]")

table = Table(title="Phase 0 (passive reading) vs v2 (Tak prompt) — same z-norm probes")
table.add_column("Model", style="cyan")
table.add_column("Framework")
table.add_column("P0 peak layer", justify="right")
table.add_column("P0 peak depth", justify="right")
table.add_column("P0 AUROC", justify="right")
table.add_column("v2 peak layer", justify="right")
table.add_column("v2 peak depth", justify="right")
table.add_column("v2 AUROC", justify="right")
table.add_column("Δ depth (pp)", justify="right")
table.add_column("Consistent?", justify="center")

summary_rows = []
for m in MODELS:
    p0_aurocs = m.get("p0_acts_aurocs")
    v2_aurocs = m.get("v2_acts_aurocs")

    if p0_aurocs is None or v2_aurocs is None:
        table.add_row(m["label"], m["framework"], "—", "—", "—", "—", "—", "—", "—", "—")
        continue

    p0_best = int(np.argmax(p0_aurocs))
    v2_best = int(np.argmax(v2_aurocs))
    n = m["n_layers"] - 1
    p0_norm = p0_best / n
    v2_norm = v2_best / n
    delta = (v2_norm - p0_norm) * 100

    # "Consistent" = both peak in upper half (>50%) AND delta < 20pp
    both_late = p0_norm > 0.50 and v2_norm > 0.50
    small_shift = abs(delta) < 20
    consistent = both_late and small_shift
    cons_str = "[bold green]YES[/bold green]" if consistent else "[bold red]NO[/bold red]"

    table.add_row(
        m["label"],
        m["framework"],
        f"L{p0_best}/{n}",
        f"{p0_norm*100:.1f}%",
        f"{p0_aurocs[p0_best]:.4f}",
        f"L{v2_best}/{n}",
        f"{v2_norm*100:.1f}%",
        f"{v2_aurocs[v2_best]:.4f}",
        f"{delta:+.1f}",
        cons_str,
    )

    summary_rows.append({
        "model":         m["label"],
        "framework":     m["framework"],
        "p0_peak_layer": p0_best,
        "p0_norm_depth": round(p0_norm, 4),
        "p0_best_auroc": round(p0_aurocs[p0_best], 4),
        "v2_peak_layer": v2_best,
        "v2_norm_depth": round(v2_norm, 4),
        "v2_best_auroc": round(v2_aurocs[v2_best], 4),
        "delta_depth_pp": round(delta, 1),
        "consistent":    consistent,
    })

console.print(table)

if summary_rows:
    pl.DataFrame(summary_rows).write_csv(OUT_DIR / "summary.csv")
    console.print("  ✓ summary.csv")

# ============================================================================
# SECTION 5 — SAVE PER-LAYER RESULTS
# ============================================================================

console.print("\n[bold cyan]Section 5: Saving per-layer results...[/bold cyan]")

for m in MODELS:
    p0_aurocs = m.get("p0_acts_aurocs")
    v2_aurocs = m.get("v2_acts_aurocs")
    if p0_aurocs is None and v2_aurocs is None:
        continue

    n = m["n_layers"]
    rows = {
        "layer":     list(range(n)),
        "rel_depth": [round(i / (n - 1), 4) for i in range(n)],
    }
    if p0_aurocs is not None:
        rows["auroc_p0_znorm"] = [round(a, 6) for a in p0_aurocs]
        rows["std_p0_znorm"]   = [round(s, 6) for s in m["p0_acts_stds"]]
    if v2_aurocs is not None:
        rows["auroc_v2_znorm"] = [round(a, 6) for a in v2_aurocs]
        rows["std_v2_znorm"]   = [round(s, 6) for s in m["v2_acts_stds"]]

    pl.DataFrame(rows).write_csv(OUT_DIR / f"results_{m['key']}.csv")
    console.print(f"  ✓ results_{m['key']}.csv")

# ============================================================================
# SECTION 6 — 4-PANEL COMPARISON PLOT
# ============================================================================

console.print("\n[bold cyan]Section 6: Generating comparison plots...[/bold cyan]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for ax_idx, m in enumerate(MODELS):
    ax = axes_flat[ax_idx]
    n = m["n_layers"]
    norm_depths = [i / (n - 1) for i in range(n)]

    # Reference
    ax.axhline(0.125, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance (0.125)")

    # Phase 0 curve
    p0_aurocs = m.get("p0_acts_aurocs")
    if p0_aurocs is not None:
        p0_arr = np.array(p0_aurocs)
        p0_std = np.array(m["p0_acts_stds"])
        ax.fill_between(norm_depths, p0_arr - p0_std, p0_arr + p0_std, alpha=0.10, color=m["color"])
        ax.plot(norm_depths, p0_arr, color=m["color"], linestyle="--", linewidth=2,
                alpha=0.7, label="Phase 0 (passive reading)")
        p0_best = int(np.argmax(p0_aurocs))
        ax.annotate(f"P0: L{p0_best} ({p0_aurocs[p0_best]:.3f})",
                    xy=(norm_depths[p0_best], p0_aurocs[p0_best]),
                    xytext=(norm_depths[p0_best] - 0.15, p0_aurocs[p0_best] - 0.06),
                    fontsize=7, color=m["color"], alpha=0.8,
                    arrowprops=dict(arrowstyle="->", color=m["color"], alpha=0.5, lw=0.8))

    # v2 curve
    v2_aurocs = m.get("v2_acts_aurocs")
    if v2_aurocs is not None:
        v2_arr = np.array(v2_aurocs)
        v2_std = np.array(m["v2_acts_stds"])
        ax.fill_between(norm_depths, v2_arr - v2_std, v2_arr + v2_std, alpha=0.10, color="black")
        ax.plot(norm_depths, v2_arr, color="black", linestyle="-", linewidth=2,
                label="v2 (Tak prompt)", zorder=3)
        v2_best = int(np.argmax(v2_aurocs))
        ax.annotate(f"v2: L{v2_best} ({v2_aurocs[v2_best]:.3f})",
                    xy=(norm_depths[v2_best], v2_aurocs[v2_best]),
                    xytext=(norm_depths[v2_best] + 0.04, v2_aurocs[v2_best] - 0.04),
                    fontsize=7, color="black", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # Correlation annotation
    if p0_aurocs is not None and v2_aurocs is not None:
        r = np.corrcoef(p0_aurocs, v2_aurocs)[0, 1]
        ax.text(0.02, 0.96, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top",
                color="green" if r > 0.95 else "darkorange" if r > 0.85 else "red")

    ax.set_title(f"{m['label']} ({m['framework']})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalized layer depth", fontsize=9)
    ax.set_ylabel("Macro AUROC (8-class, z-norm)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.2)

fig.suptitle("Methodology Consistency: Phase 0 (passive reading) vs v2 (Tak prompt)\n"
             "Same z-normalized probes on both — testing prompt format invariance",
             fontsize=12, fontweight="bold")
plt.tight_layout()

for ext in ["png", "svg"]:
    fig.savefig(OUT_DIR / f"consistency_comparison.{ext}", dpi=150, bbox_inches="tight")
    console.print(f"  ✓ consistency_comparison.{ext}")
plt.close()

# ============================================================================
# SECTION 7 — PROFILE CORRELATION ANALYSIS
# ============================================================================

console.print("\n[bold cyan]Section 7: Profile correlation analysis...[/bold cyan]")

corr_table = Table(title="Layer profile correlations (Phase 0 vs v2)")
corr_table.add_column("Model", style="cyan")
corr_table.add_column("Pearson r", justify="right")
corr_table.add_column("Max AUROC diff", justify="right")
corr_table.add_column("Mean AUROC diff", justify="right")

for m in MODELS:
    p0 = m.get("p0_acts_aurocs")
    v2 = m.get("v2_acts_aurocs")
    if p0 is None or v2 is None:
        corr_table.add_row(m["label"], "—", "—", "—")
        continue

    p0_arr = np.array(p0)
    v2_arr = np.array(v2)
    r = np.corrcoef(p0_arr, v2_arr)[0, 1]
    max_diff = float(np.max(np.abs(p0_arr - v2_arr)))
    mean_diff = float(np.mean(np.abs(p0_arr - v2_arr)))
    corr_table.add_row(m["label"], f"{r:.4f}", f"{max_diff:.4f}", f"{mean_diff:.4f}")

console.print(corr_table)

# ============================================================================
# SECTION 8 — GATE DECISION & README
# ============================================================================

console.print("\n[bold cyan]Section 8: Gate decision...[/bold cyan]")

n_consistent = sum(1 for r in summary_rows if r["consistent"])
n_total = len(summary_rows)
gate = "PASS" if n_consistent >= 3 else "FAIL"

console.print(f"\n  Consistency gate: {n_consistent}/{n_total} models consistent")
console.print(f"  [bold {'green' if gate == 'PASS' else 'red'}]Gate: {gate}[/bold {'green' if gate == 'PASS' else 'red'}]")

# Build README
model_rows = []
for r in summary_rows:
    cons = "✓" if r["consistent"] else "✗"
    model_rows.append(
        f"| {r['model']} | {r['framework']} | "
        f"L{r['p0_peak_layer']} ({r['p0_norm_depth']*100:.1f}%) | {r['p0_best_auroc']:.4f} | "
        f"L{r['v2_peak_layer']} ({r['v2_norm_depth']*100:.1f}%) | {r['v2_best_auroc']:.4f} | "
        f"{r['delta_depth_pp']:+.1f}pp | {cons} |"
    )

readme = f"""# Experiment 10 — Methodology Consistency Check

**Date:** 2026-02-25 (original), 2026-03-13 (rewritten for methodological rigor)
**Question:** Does prompt format (passive reading vs Tak few-shot) change what layer-wise probes measure?
**Gate:** ≥3/4 models show consistent late-layer consolidation across both conditions
**Status:** {gate} ({n_consistent}/{n_total} consistent)

## Why This Experiment Exists

Before trusting 20 experiments of results, we need to verify that:
1. Our core finding (late-layer emotion consolidation) isn't an artifact of prompt format
2. Phase 0 results (passive reading) and v2 results (Tak prompt) agree when probed identically
3. The extraction framework difference (TL+CPU for 1B vs HF+MPS for 8B/9B) isn't hiding bugs

## What Changed in the Rewrite (2026-03-13)

The original Exp 10 had methodological problems:
- Used a **third** prompt format ("\\n\\nEmotion:" suffix) that matched neither Phase 0 nor v2
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
{chr(10).join(model_rows)}

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
"""

(EXP_DIR / "README.md").write_text(readme.strip() + "\n")
console.print("  ✓ README.md")

console.print(f"\n[bold green]✓ Experiment 10 complete — Gate: {gate}[/bold green]")
