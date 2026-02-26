"""
Experiment 13 — v2 Probes: Stage 1 Gate Check (Tak Replication)
Question: Does llama1b_inst Set A probe peak at normalized depth 0.50–0.75?
Date: 2026-02-25

CRITICAL GATE: If h probe peak is at normalized depth 0.50–0.75, Tak et al.
replication is confirmed. If not, stop and debug the extraction (Exp 11).

Expected result:
  - Tak et al. (instruct model, few-shot prompt): peak at ~60% normalized depth
  - Phase 0 (passive reading): peak at ~87% normalized depth (late layers)
  - A leftward shift in the probe curve confirms the prediction-task framing hypothesis.

Prerequisite: Run experiments/11_v2_extract_seta/extract_llama1b_inst.py first.

Run: uv run python experiments/13_v2_probes/run_gate_check.py
"""

import sys
import json
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

EXP_DIR  = Path(__file__).parent
ROOT     = EXP_DIR.parent.parent
OUT_DIR  = EXP_DIR / "outputs"
ACT_DIR  = ROOT / "experiments" / "11_v2_extract_seta" / "outputs" / "activations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODEL_KEY = "llama1b_inst"
MODEL_DIR = ACT_DIR / MODEL_KEY
N_LAYERS  = 16
D_MODEL   = 2048

# Gate zone: mid-layer consolidation expected if Tak replication holds
GATE_LO = 0.50
GATE_HI = 0.75

# ============================================================================
# CHECK PREREQUISITES
# ============================================================================

manifest_path = MODEL_DIR / "manifest.csv"
if not manifest_path.exists():
    console.print(f"[bold red]✗ Manifest not found: {manifest_path}[/bold red]")
    console.print("[bold red]  Run: uv run python experiments/11_v2_extract_seta/extract_llama1b_inst.py[/bold red]")
    sys.exit(1)

manifest = pl.read_csv(manifest_path)
console.print(f"[bold cyan]Loaded manifest: {len(manifest)} stimuli[/bold cyan]")
console.print(f"Emotions: {manifest['emotion'].unique().to_list()}")

# ============================================================================
# LOAD ACTIVATIONS — h, a, m for all 80 stimuli
# ============================================================================

def load_activations(model_dir: Path, act_type: str, manifest: pl.DataFrame):
    """
    Load activations of one type (h, a, or m) for all stimuli in manifest.
    Returns: (activations, labels)
      activations: np.ndarray (n_stimuli, n_layers, d_model)
      labels: np.ndarray of emotion strings
    """
    all_acts = []
    labels   = []

    for row in manifest.iter_rows(named=True):
        stim_id = row["id"]
        npz_path = model_dir / f"{stim_id}.npz"

        if not npz_path.exists():
            console.print(f"[yellow]Missing: {npz_path}[/yellow]")
            continue

        npz = np.load(npz_path, allow_pickle=True)
        act = npz[act_type]  # (n_layers, d_model)

        if not np.all(np.isfinite(act)):
            console.print(f"[yellow]Non-finite values in {stim_id} {act_type}[/yellow]")
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

        all_acts.append(act)
        labels.append(row["emotion"])

    activations = np.stack(all_acts)  # (n_stimuli, n_layers, d_model)
    labels      = np.array(labels)
    return activations, labels


console.print("\n[bold cyan]Loading activations (h, a, m) for llama1b_inst Set A...[/bold cyan]")
h_acts, labels = load_activations(MODEL_DIR, "h", manifest)
a_acts, _      = load_activations(MODEL_DIR, "a", manifest)
m_acts, _      = load_activations(MODEL_DIR, "m", manifest)

console.print(f"h shape: {h_acts.shape}")
console.print(f"a shape: {a_acts.shape}")
console.print(f"m shape: {m_acts.shape}")
console.print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ============================================================================
# 8-CLASS PROBING — 5-fold stratified CV, macro AUROC at each layer
# ============================================================================

def probe_layer(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """
    Fit 8-class logistic regression probe at a single layer.
    X: (n_stimuli, d_model)
    y: (n_stimuli,) — string labels
    Returns dict with auroc, accuracy (mean ± std across folds).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aurocs = []
    accs   = []

    # Encode labels to integers for sklearn
    classes   = sorted(np.unique(y))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int     = np.array([label_map[yi] for yi in y])

    for train_idx, test_idx in skf.split(X, y_int):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        # Normalize per fold (important for LR convergence)
        mu  = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train_n = (X_train - mu) / std
        X_test_n  = (X_test  - mu) / std

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train_n, y_train)

        y_prob = clf.predict_proba(X_test_n)  # (n_test, 8)
        y_pred = clf.predict(X_test_n)

        # One-vs-rest macro AUROC
        if len(np.unique(y_test)) > 1:
            auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        else:
            auroc = float("nan")

        acc = (y_pred == y_test).mean()
        aurocs.append(auroc)
        accs.append(acc)

    return {
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std":  float(np.nanstd(aurocs)),
        "acc_mean":   float(np.mean(accs)),
        "acc_std":    float(np.std(accs)),
    }


def run_probes(activations: np.ndarray, labels: np.ndarray, act_name: str) -> pl.DataFrame:
    """Run probes at all layers and return results DataFrame."""
    results = []
    console.print(f"\n  Running {act_name} probes ({N_LAYERS} layers × 5-fold CV)...")
    for l in range(N_LAYERS):
        X = activations[:, l, :]
        r = probe_layer(X, labels)
        norm_depth = (l + 1) / N_LAYERS
        results.append({
            "layer":      l,
            "norm_depth": round(norm_depth, 4),
            "act_type":   act_name,
            "auroc_mean": r["auroc_mean"],
            "auroc_std":  r["auroc_std"],
            "acc_mean":   r["acc_mean"],
            "acc_std":    r["acc_std"],
        })
        console.print(f"    L{l:02d} ({norm_depth:.3f}): AUROC={r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")
    return pl.DataFrame(results)


console.print("\n[bold cyan]Running 8-class probes (5-fold stratified CV)...[/bold cyan]")
h_results = run_probes(h_acts, labels, "h")
a_results = run_probes(a_acts, labels, "a")
m_results = run_probes(m_acts, labels, "m")

# ============================================================================
# SAVE PROBE RESULTS
# ============================================================================

h_results.write_csv(OUT_DIR / f"probe_results_{MODEL_KEY}_seta_h.csv")
a_results.write_csv(OUT_DIR / f"probe_results_{MODEL_KEY}_seta_a.csv")
m_results.write_csv(OUT_DIR / f"probe_results_{MODEL_KEY}_seta_m.csv")
console.print(f"\n[bold green]✓ Probe results saved[/bold green]")

# ============================================================================
# GATE CHECK — h probe peak at normalized depth 0.50–0.75
# ============================================================================

console.print("\n[bold cyan]Gate check: h probe peak normalized depth...[/bold cyan]")

h_aurocs   = h_results["auroc_mean"].to_numpy()
peak_layer = int(np.argmax(h_aurocs))
peak_auroc = float(h_aurocs[peak_layer])
peak_norm  = float(h_results["norm_depth"][peak_layer])

a_aurocs    = a_results["auroc_mean"].to_numpy()
m_aurocs    = m_results["auroc_mean"].to_numpy()
a_peak_layer = int(np.argmax(a_aurocs))
m_peak_layer = int(np.argmax(m_aurocs))

gate_pass = GATE_LO <= peak_norm <= GATE_HI

console.print(f"\n  h peak layer: L{peak_layer} (norm depth {peak_norm:.3f}), AUROC={peak_auroc:.4f}")
console.print(f"  a peak layer: L{a_peak_layer} (norm depth {a_results['norm_depth'][a_peak_layer]:.3f})")
console.print(f"  m peak layer: L{m_peak_layer} (norm depth {m_results['norm_depth'][m_peak_layer]:.3f})")

if gate_pass:
    console.print(f"\n[bold green]✓ TAK REPLICATION GATE: PASS[/bold green]")
    console.print(f"  h peak at {peak_norm:.3f} normalized depth — within [{GATE_LO}, {GATE_HI}]")
    console.print(f"  Confirms: few-shot prediction-task framing → mid-layer consolidation")
    console.print(f"  Compare: Phase 0 passive reading → late-layer consolidation (0.875)")
else:
    console.print(f"\n[bold red]✗ TAK REPLICATION GATE: FAIL[/bold red]")
    console.print(f"  h peak at {peak_norm:.3f} normalized depth — outside [{GATE_LO}, {GATE_HI}]")
    console.print(f"  STOP: Debug prompt format / extraction before proceeding to Stage 2")
    console.print(f"  Check: Is the prompt correctly built? Is ':' the final token?")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

table = Table(title=f"8-Class Probe Results — {MODEL_KEY} Set A")
table.add_column("Layer", style="cyan", justify="right")
table.add_column("Norm Depth", justify="right")
table.add_column("h AUROC", justify="right")
table.add_column("a AUROC", justify="right")
table.add_column("m AUROC", justify="right")

for i in range(N_LAYERS):
    style = "bold green" if i == peak_layer else ""
    table.add_row(
        f"L{i}",
        f"{h_results['norm_depth'][i]:.3f}",
        f"{h_aurocs[i]:.4f}",
        f"{a_aurocs[i]:.4f}",
        f"{m_aurocs[i]:.4f}",
        style=style,
    )

console.print(table)

# ============================================================================
# VISUALIZATION — probe curves for h, a, m
# ============================================================================

console.print("\n[bold cyan]Generating probe curves figure...[/bold cyan]")

norm_depths = h_results["norm_depth"].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"8-Class Probe Curves — {MODEL_KEY} Set A (Tak few-shot prompt)", fontsize=13)

colors  = {"h": "#e74c3c", "a": "#2980b9", "m": "#27ae60"}
labels_ = {"h": "Residual stream (h)", "a": "MHSA output (a)", "m": "FFN output (m)"}
all_res = {"h": h_results, "a": a_results, "m": m_results}

for idx, (act_type, res) in enumerate(all_res.items()):
    ax   = axes[idx]
    auroc = res["auroc_mean"].to_numpy()
    std   = res["auroc_std"].to_numpy()

    ax.plot(norm_depths, auroc, color=colors[act_type], linewidth=2.5,
            label=labels_[act_type], marker="o", markersize=5)
    ax.fill_between(norm_depths, auroc - std, auroc + std,
                    alpha=0.2, color=colors[act_type])

    # Mark peak
    pk_l = np.argmax(auroc)
    ax.axvline(norm_depths[pk_l], color=colors[act_type], linestyle="--", alpha=0.6)
    ax.annotate(f"L{pk_l}\n{norm_depths[pk_l]:.2f}",
                xy=(norm_depths[pk_l], auroc[pk_l]),
                xytext=(8, -20), textcoords="offset points",
                fontsize=9, color=colors[act_type])

    # Gate zone shading
    ax.axvspan(GATE_LO, GATE_HI, alpha=0.08, color="gold", label="Tak gate zone")
    ax.axhline(1/8, linestyle=":", color="gray", alpha=0.5, label="Chance (0.125)")

    ax.set_xlabel("Normalized depth (layer / n_layers)", fontsize=10)
    ax.set_ylabel("Macro AUROC (8-class, OvR)", fontsize=10)
    ax.set_title(labels_[act_type], fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Add gate result as overall title annotation
gate_str = f"GATE: {'PASS' if gate_pass else 'FAIL'} — h peak at {peak_norm:.3f} (expected 0.50–0.75)"
gate_color_mpl = "green" if gate_pass else "red"
fig.text(0.5, -0.02, gate_str, ha="center", fontsize=12,
         color=gate_color_mpl, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / f"probe_curves_{MODEL_KEY}_seta_gate.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"probe_curves_{MODEL_KEY}_seta_gate.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ Saved probe_curves_{MODEL_KEY}_seta_gate.png/.svg[/bold green]")

# ============================================================================
# WRITE GATE CHECK REPORT
# ============================================================================

report_lines = [
    f"# Gate Check Report — Exp 13 v2 Probes (Tak Replication)",
    f"",
    f"**Date:** 2026-02-25",
    f"**Model:** {MODEL_KEY} ({MODEL_ID if 'MODEL_ID' in dir() else 'meta-llama/Llama-3.2-1B-Instruct'})",
    f"**Stimuli:** 80 emotional Set A",
    f"**Prompt:** Tak et al. few-shot (2-shot classification)",
    f"",
    f"## Gate Result: {'✅ PASS' if gate_pass else '❌ FAIL'}",
    f"",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| h probe peak layer | L{peak_layer} |",
    f"| h probe peak norm depth | {peak_norm:.4f} |",
    f"| h probe peak AUROC | {peak_auroc:.4f} |",
    f"| Gate zone | [{GATE_LO}, {GATE_HI}] |",
    f"| Gate: {'PASS' if gate_pass else 'FAIL'} | Peak {'within' if gate_pass else 'OUTSIDE'} gate zone |",
    f"",
    f"## All Component Peaks",
    f"",
    f"| Component | Peak Layer | Norm Depth | AUROC |",
    f"|-----------|-----------|-----------|-------|",
    f"| h (residual) | L{peak_layer} | {peak_norm:.4f} | {peak_auroc:.4f} |",
    f"| a (MHSA out) | L{a_peak_layer} | {a_results['norm_depth'][a_peak_layer]:.4f} | {a_aurocs[a_peak_layer]:.4f} |",
    f"| m (FFN out)  | L{m_peak_layer} | {m_results['norm_depth'][m_peak_layer]:.4f} | {m_aurocs[m_peak_layer]:.4f} |",
    f"",
    f"## Interpretation",
    f"",
]

if gate_pass:
    report_lines += [
        f"The h probe peak at normalized depth {peak_norm:.3f} falls within the expected gate zone [0.50, 0.75].",
        f"This replicates Tak et al.'s finding that few-shot classification framing consolidates",
        f"emotion representations at mid-layer depth (~60%), earlier than passive reading (~87%).",
        f"",
        f"**Conclusion:** Tak replication confirmed. Proceed to Stage 2.",
    ]
else:
    report_lines += [
        f"The h probe peak at normalized depth {peak_norm:.3f} is OUTSIDE the expected gate zone [0.50, 0.75].",
        f"This means either (a) the prompt format is not correctly replicating Tak et al., or",
        f"(b) this model differs from what Tak et al. tested.",
        f"",
        f"**Action required:** Debug before proceeding.",
        f"  1. Check: Is the ':' token correctly the final token for all stimuli?",
        f"  2. Check: Does the behavioral check show plausible emotion predictions?",
        f"  3. Compare: Run Exp 10 (suffix '\n\nEmotion:') — if it also fails, model issue.",
    ]

report_lines += [
    f"",
    f"## Outputs",
    f"",
    f"- `outputs/probe_results_{MODEL_KEY}_seta_{{h,a,m}}.csv` — per-layer AUROC",
    f"- `outputs/probe_curves_{MODEL_KEY}_seta_gate.png/.svg` — visualization",
]

(OUT_DIR / f"gate_check_report_{MODEL_KEY}.md").write_text("\n".join(report_lines))
console.print(f"[bold green]✓ Gate check report saved[/bold green]")

console.print(f"\n[bold {'green' if gate_pass else 'red'}]Gate: {'PASS — proceed to Stage 2' if gate_pass else 'FAIL — debug before continuing'}[/bold {'green' if gate_pass else 'red'}]")
