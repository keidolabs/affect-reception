"""
Experiment 10 — Set C Tak Replication: Analysis
Question: Does prediction-task framing ("\n\nEmotion:") shift emotion consolidation to mid-layers?
Date: 2026-02-25

Loads Set C activations (suffix="\n\nEmotion:", extracted at ":") from all 4 models,
runs 8-class probes, and overlays against Phase 0 results.

Expected result if replication holds:
  - Set C AUROC peak in normalized depth 0.40–0.75 (mid-layer zone)
  - Phase 0 AUROC peak in normalized depth 0.75–1.0 (late-layer zone)
  - Passive reading → late consolidation; prediction task → mid-layer consolidation

Gate: Set C peak normalized depth in [0.40, 0.75] → REPLICATION CONFIRMED

Run: uv run python experiments/10_setc_tak_replication/run.py
(After running all 4 extract_*.py scripts)
"""

import sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table

console = Console()

np.random.seed(42)

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "outputs" / "activations"
OUT_DIR = EXP_DIR / "outputs"
ROOT    = EXP_DIR.parent.parent

sys.path.insert(0, str(ROOT))

# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODELS = [
    {
        "key":         "llama1b",
        "label":       "Llama-3.2-1B",
        "color":       "#e74c3c",
        "n_layers":    16,
        "npy":         ACT_DIR / "llama1b_set_c_residuals.npy",
        "meta_csv":    ACT_DIR / "llama1b_metadata.csv",
        "beh_check":   OUT_DIR / "behavioral_check_llama1b.txt",
        "phase0_csv":  ROOT / "experiments/00_phase0_replication/outputs/results.csv",
        "phase0_norm_peak": 0.875,  # from Entry 001 (L14/16)
    },
    {
        "key":         "llama8b",
        "label":       "Llama-3-8B (base)",
        "color":       "#2980b9",
        "n_layers":    32,
        "npy":         ACT_DIR / "llama8b_set_c_residuals.npy",
        "meta_csv":    ACT_DIR / "llama8b_metadata.csv",
        "beh_check":   OUT_DIR / "behavioral_check_llama8b.txt",
        "phase0_csv":  ROOT / "experiments/01_phase0a_llama3-8b/outputs/results.csv",
        "phase0_norm_peak": 0.781,  # from Entry 002 (L25/32)
    },
    {
        "key":         "llama8b_instruct",
        "label":       "Llama-3-8B (instruct)",
        "color":       "#8e44ad",
        "n_layers":    32,
        "npy":         ACT_DIR / "llama8b_instruct_set_c_residuals.npy",
        "meta_csv":    ACT_DIR / "llama8b_instruct_metadata.csv",
        "beh_check":   OUT_DIR / "behavioral_check_llama8b_instruct.txt",
        "phase0_csv":  ROOT / "experiments/02_phase0b_llama3-8b-instruct/outputs/results.csv",
        "phase0_norm_peak": 0.938,  # from Entry 002 (L30/32)
    },
    {
        "key":         "gemma9b",
        "label":       "Gemma-2-9B",
        "color":       "#27ae60",
        "n_layers":    42,
        "npy":         ACT_DIR / "gemma9b_set_c_residuals.npy",
        "meta_csv":    ACT_DIR / "gemma9b_metadata.csv",
        "beh_check":   OUT_DIR / "behavioral_check_gemma9b.txt",
        "phase0_csv":  ROOT / "experiments/03_phase0c_gemma2-9b/outputs/results.csv",
        "phase0_norm_peak": 0.929,  # from Entry 002 (L39/42)
    },
]

# Gate zone: mid-layer consolidation expected here if Tak replication holds
MID_ZONE_LO = 0.40
MID_ZONE_HI = 0.75

# ============================================================================
# SECTION 1 — LOAD DATA
# ============================================================================

console.print("\n[bold cyan]Section 1: Loading activations and Phase 0 results...[/bold cyan]")

for m in MODELS:
    if not m["npy"].exists():
        console.print(f"[red]MISSING: {m['npy'].name} — run extract_{m['key']}.py first[/red]")
        m["activations"] = None
        m["labels"]      = None
        m["phase0_aurocs"] = None
        continue

    acts = np.load(m["npy"])           # (80, n_layers, d_model)
    meta = pl.read_csv(m["meta_csv"])
    labels = meta["emotion"].to_numpy()

    m["activations"] = acts
    m["labels"]      = labels

    console.print(f"  {m['label']:30s} activations: {acts.shape}")

    # Load Phase 0 results for overlay
    if m["phase0_csv"].exists():
        p0 = pl.read_csv(m["phase0_csv"])
        m["phase0_aurocs"] = p0["auroc_mean"].to_numpy()
        console.print(f"  {m['label']:30s} Phase 0 AUROC loaded ({len(m['phase0_aurocs'])} layers)")
    else:
        console.print(f"  [yellow]Phase 0 results not found for {m['label']} — overlay skipped[/yellow]")
        m["phase0_aurocs"] = None

# ============================================================================
# SECTION 2 — 8-CLASS PROBES (Set C)
# ============================================================================

console.print("\n[bold cyan]Section 2: Running 8-class probes on Set C...[/bold cyan]")

for m in MODELS:
    if m["activations"] is None:
        m["setc_aurocs"] = None
        continue

    acts   = m["activations"]    # (80, n_layers, d_model)
    labels = m["labels"]
    n_layers = acts.shape[1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    layer_aurocs = []
    layer_stds   = []

    for layer_idx in range(n_layers):
        X = acts[:, layer_idx, :]
        fold_scores = []
        for train_idx, test_idx in cv.split(X, labels):
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X[train_idx], labels[train_idx])
            probs = clf.predict_proba(X[test_idx])
            try:
                score = roc_auc_score(labels[test_idx], probs, multi_class="ovr",
                                      average="macro", labels=clf.classes_)
            except ValueError:
                score = float("nan")
            fold_scores.append(score)
        layer_aurocs.append(float(np.nanmean(fold_scores)))
        layer_stds.append(float(np.nanstd(fold_scores)))

    m["setc_aurocs"] = layer_aurocs
    m["setc_stds"]   = layer_stds
    best_layer = int(np.argmax(layer_aurocs))
    best_auroc = layer_aurocs[best_layer]
    best_norm  = best_layer / (n_layers - 1)
    m["setc_best_layer"] = best_layer
    m["setc_best_auroc"] = best_auroc
    m["setc_best_norm"]  = best_norm
    gate = MID_ZONE_LO <= best_norm <= MID_ZONE_HI

    console.print(f"  {m['label']:30s} best: L{best_layer}/{n_layers-1} ({best_norm*100:.1f}%) AUROC={best_auroc:.4f}  "
                  f"[bold {'green' if gate else 'yellow'}]{'✓ MID-ZONE' if gate else '✗ outside mid-zone'}[/bold {'green' if gate else 'yellow'}]")

# ============================================================================
# SECTION 3 — SAVE SET C RESULTS CSV
# ============================================================================

console.print("\n[bold cyan]Section 3: Saving results CSVs...[/bold cyan]")

for m in MODELS:
    if m["setc_aurocs"] is None:
        continue

    n_layers = len(m["setc_aurocs"])
    rows = {
        "layer":           list(range(n_layers)),
        "rel_depth":       [round(i / (n_layers - 1), 4) for i in range(n_layers)],
        "auroc_setc_mean": [round(a, 6) for a in m["setc_aurocs"]],
        "auroc_setc_std":  [round(s, 6) for s in m["setc_stds"]],
    }

    if m["phase0_aurocs"] is not None:
        # Phase 0 results may have different layer count — align by index
        p0 = m["phase0_aurocs"]
        rows["auroc_phase0_mean"] = [round(p0[i], 6) if i < len(p0) else None for i in range(n_layers)]

    pl.DataFrame(rows).write_csv(OUT_DIR / f"results_{m['key']}.csv")
    console.print(f"  ✓ results_{m['key']}.csv")

# ============================================================================
# SECTION 4 — 4-PANEL COMPARISON PLOT (Set C vs Phase 0)
# ============================================================================

console.print("\n[bold cyan]Section 4: Generating comparison plots...[/bold cyan]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for ax_idx, m in enumerate(MODELS):
    ax = axes_flat[ax_idx]
    n_layers = m["n_layers"]
    norm_depths = [i / (n_layers - 1) for i in range(n_layers)]

    # Mid-layer zone shading
    ax.axvspan(MID_ZONE_LO, MID_ZONE_HI, alpha=0.08, color="green", label="Mid-zone (0.40–0.75)")

    # Reference lines
    ax.axhline(0.125, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance (0.125)")
    ax.axhline(0.40,  color="orange", linewidth=0.8, linestyle=":", alpha=0.7, label="Gate (0.40)")

    # Phase 0 curve (dashed)
    if m.get("phase0_aurocs") is not None:
        p0 = m["phase0_aurocs"]
        p0_depths = [i / (len(p0) - 1) for i in range(len(p0))]
        ax.plot(p0_depths, p0, color=m["color"], linestyle="--", linewidth=1.5,
                alpha=0.5, label="Phase 0 (passive reading)")
        p0_best_layer = int(np.argmax(p0))
        p0_best_norm  = p0_best_layer / (len(p0) - 1)
        ax.axvline(p0_best_norm, color=m["color"], linestyle="--", linewidth=0.8, alpha=0.4)
        ax.annotate(f"P0: {p0[p0_best_layer]:.3f}\n({p0_best_norm*100:.0f}%)",
                    xy=(p0_best_norm, p0[p0_best_layer]),
                    xytext=(p0_best_norm - 0.12, p0[p0_best_layer] - 0.07),
                    fontsize=7, color=m["color"], alpha=0.7,
                    arrowprops=dict(arrowstyle="->", color=m["color"], alpha=0.4, lw=0.8))

    # Set C curve (solid)
    if m.get("setc_aurocs") is not None:
        setc = m["setc_aurocs"]
        stds  = m["setc_stds"]
        setc_arr = np.array(setc)
        std_arr  = np.array(stds)

        ax.fill_between(norm_depths, setc_arr - std_arr, setc_arr + std_arr,
                        alpha=0.12, color=m["color"])
        ax.plot(norm_depths, setc, color=m["color"], linestyle="-", linewidth=2,
                label=f"Set C (prediction task)", zorder=3)

        best_layer = m["setc_best_layer"]
        best_norm  = m["setc_best_norm"]
        best_auroc = m["setc_best_auroc"]

        ax.axvline(best_norm, color=m["color"], linestyle="-", linewidth=1.0, alpha=0.5)
        ax.annotate(f"L{best_layer}: {best_auroc:.3f}\n({best_norm*100:.0f}%)",
                    xy=(best_norm, best_auroc),
                    xytext=(best_norm + 0.04, best_auroc - 0.04),
                    fontsize=8, color=m["color"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=m["color"], lw=0.8))

        # Gate badge
        gate = MID_ZONE_LO <= best_norm <= MID_ZONE_HI
        badge_color = "green" if gate else "darkorange"
        badge_text  = "MID ✓" if gate else "LATE ✗"
        ax.text(0.02, 0.96, badge_text, transform=ax.transAxes, fontsize=9,
                color=badge_color, fontweight="bold", va="top")
    else:
        ax.text(0.5, 0.5, "Not yet run", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=11)

    ax.set_title(m["label"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalized layer depth", fontsize=9)
    ax.set_ylabel("Macro AUROC (8-class OvR)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.2)

fig.suptitle("Set C (Tak Replication) vs Phase 0 — Prediction Task vs Passive Reading\n"
             "Shaded zone = expected mid-layer consolidation if Tak et al. replicates",
             fontsize=12, fontweight="bold")
plt.tight_layout()

for ext in ["png", "svg"]:
    fig.savefig(OUT_DIR / f"auroc_setc_vs_phase0_panel.{ext}",
                dpi=150, bbox_inches="tight")
    console.print(f"  ✓ auroc_setc_vs_phase0_panel.{ext}")
plt.close()

# ============================================================================
# SECTION 5 — SUMMARY TABLE & CSV
# ============================================================================

console.print("\n[bold cyan]Section 5: Summary...[/bold cyan]")

table = Table(title="Set C — Tak Replication Summary")
table.add_column("Model", style="cyan")
table.add_column("P0 peak (norm)", justify="right")
table.add_column("Set C best layer", justify="right")
table.add_column("Set C norm depth", justify="right")
table.add_column("Set C AUROC", justify="right")
table.add_column("In mid-zone?", justify="center")

summary_rows = []
for m in MODELS:
    if m.get("setc_aurocs") is None:
        table.add_row(m["label"], "—", "—", "—", "—", "missing")
        continue
    gate = MID_ZONE_LO <= m["setc_best_norm"] <= MID_ZONE_HI
    gate_str = "[bold green]✓ YES[/bold green]" if gate else "[bold yellow]✗ NO[/bold yellow]"
    table.add_row(
        m["label"],
        f"{m['phase0_norm_peak']*100:.1f}%",
        f"L{m['setc_best_layer']}/{m['n_layers']-1}",
        f"{m['setc_best_norm']*100:.1f}%",
        f"{m['setc_best_auroc']:.4f}",
        gate_str,
    )
    summary_rows.append({
        "model":             m["label"],
        "phase0_norm_peak":  round(m["phase0_norm_peak"], 4),
        "setc_best_layer":   m["setc_best_layer"],
        "setc_n_layers":     m["n_layers"],
        "setc_norm_peak":    round(m["setc_best_norm"], 4),
        "setc_best_auroc":   round(m["setc_best_auroc"], 4),
        "in_mid_zone":       MID_ZONE_LO <= m["setc_best_norm"] <= MID_ZONE_HI,
    })

console.print(table)

if summary_rows:
    pl.DataFrame(summary_rows).write_csv(OUT_DIR / "summary.csv")
    console.print("✓ summary.csv")

# ============================================================================
# SECTION 6 — BEHAVIORAL CHECK SUMMARY
# ============================================================================

console.print("\n[bold cyan]Section 6: Behavioral check results...[/bold cyan]")

for m in MODELS:
    beh_path = m["beh_check"]
    if beh_path.exists():
        last_line = beh_path.read_text().strip().split("\n")[-1]
        console.print(f"  {m['label']:30s} — {last_line}")
    else:
        console.print(f"  {m['label']:30s} — [yellow]behavioral_check not found[/yellow]")

# ============================================================================
# SECTION 7 — AUTO-GENERATE README
# ============================================================================

console.print("\n[bold cyan]Section 7: Writing README...[/bold cyan]")

mid_zone_hits = sum(
    1 for m in MODELS
    if m.get("setc_best_norm") is not None and MID_ZONE_LO <= m["setc_best_norm"] <= MID_ZONE_HI
)
models_run = sum(1 for m in MODELS if m.get("setc_aurocs") is not None)

gate_decision = "PASS ✓" if mid_zone_hits >= 3 else ("PARTIAL" if mid_zone_hits >= 2 else "FAIL ✗")

# Build per-model results block
per_model_lines = []
for m in MODELS:
    if m.get("setc_aurocs") is None:
        per_model_lines.append(f"| {m['label']} | — | — | — | not run |")
        continue
    gate = "✓" if MID_ZONE_LO <= m["setc_best_norm"] <= MID_ZONE_HI else "✗"
    p0_norm = m["phase0_norm_peak"]
    shift = m["setc_best_norm"] - p0_norm
    shift_str = f"{shift*100:+.1f}pp"
    per_model_lines.append(
        f"| {m['label']} | {p0_norm*100:.1f}% | "
        f"{m['setc_best_norm']*100:.1f}% (L{m['setc_best_layer']}/{m['n_layers']-1}) | "
        f"{m['setc_best_auroc']:.4f} | {shift_str} {gate} |"
    )

readme = f"""# Experiment 10 — Set C: Tak Replication Sanity Check

**Date:** 2026-02-25
**Question:** Does prediction-task framing shift emotion consolidation from late to mid layers?
**Gate:** Set C peak normalized depth in [40%, 75%] (mid-layer zone)
**Status:** {gate_decision} ({mid_zone_hits}/{models_run} models in mid-zone)

## What This Tests

Internal validation experiment. One thing changes vs Phase 0:
- **Suffix appended:** `"\\n\\nEmotion:"` — forces next-token prediction framing
- **Extraction point:** final token `":"` of suffixed text (not end-of-narrative)
- Everything else identical (same stimuli, same model, same probe)

This replicates the task framing in Tak et al., where emotion consolidation was found
in mid-layers (~40–75% depth) under prediction-task conditions. Phase 0 used passive
reading (no suffix), which produced late-layer consolidation (~80–95% depth).

## Results

| Model | P0 peak (norm) | Set C peak (norm) | Set C AUROC | Shift + Gate |
|-------|---------------|------------------|-------------|--------------|
{chr(10).join(per_model_lines)}

**Mid-zone criterion:** normalized depth 40–75%

## Behavioral Check

Top-10 next-token predictions at ":" verified for one sample per emotion.
See `outputs/behavioral_check_*.txt` for full details.

## Outputs

- `outputs/auroc_setc_vs_phase0_panel.png/.svg` — 4-panel comparison per model
- `outputs/results_*.csv` — per-layer AUROC for each model
- `outputs/summary.csv` — summary table
- `outputs/behavioral_check_*.txt` — top-10 prediction checks

## Interpretation

If Set C peaks in the mid-zone [40–75%]:
The late-layer consolidation in Phase 0/1 is explained by task framing (passive reading vs
prediction). Both paradigms are valid but measure different things. Proceed with Phase 2 analysis
using Phase 0/1 data, noting this methodological distinction.

If Set C still peaks late (>75%):
Task framing alone does not shift the peak in these models. Possible explanations:
(a) The Tak et al. result doesn't generalize to Llama/Gemma architectures;
(b) Our suffix format differs from their stimuli in a critical way;
(c) The late-layer localization in Phase 0/1 is a genuine property of these models,
not a task-framing artifact. Document and proceed.
"""

(EXP_DIR / "README.md").write_text(readme.strip())
console.print("\n[bold green]✓ Experiment 10 complete![/bold green]")
console.print(f"Gate: {gate_decision} ({mid_zone_hits}/{models_run} models in mid-zone)")
console.print(f"Results → {EXP_DIR / 'README.md'}")
