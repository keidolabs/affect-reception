"""
Experiment 19 — Zero-Shot Control: Probing + Comparison
Question: Does removing 2 few-shot shots change probe AUROC on Set B?
Date: 2026-02-28

Loads zero-shot activations (this experiment) and few-shot activations (Exp 12),
runs binary and 8-class probes on both, compares directly.

Prerequisite: run extract_llama1b_inst.py first.
Run: uv run python experiments/19_zeroshot_control/probe.py
"""

import sys
import json
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

EXP_DIR      = Path(__file__).parent
ROOT         = EXP_DIR.parent.parent
OUT_DIR      = EXP_DIR / "outputs"
ZS_ACT_DIR   = EXP_DIR / "outputs" / "activations" / "llama1b_inst"
FS_ACT_DIR   = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations" / "llama1b_inst"

sys.path.insert(0, str(ROOT))

N_LAYERS = 16
MODEL_KEY = "llama1b_inst"

# ============================================================================
# LOAD ACTIVATIONS HELPER
# ============================================================================

def load_setb_activations(act_dir: Path, act_type: str):
    """
    Load Set B activations for one condition (zero-shot or few-shot).
    Returns:
      h_acts: (n_stimuli, n_layers, d_model)
      labels_8class: 8-class emotion strings (clinical only, n=96)
      labels_binary: "emotional" / "neutral" strings (all 192)
      meta_all: list of dicts
    """
    manifest_path = act_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pl.read_csv(manifest_path)
    rows = manifest.iter_rows(named=True)

    all_acts    = []
    all_labels  = []
    all_meta    = []

    for row in rows:
        npz_path = act_dir / f"{row['id']}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        act = npz[act_type].copy()  # (n_layers, d_model)
        if not np.all(np.isfinite(act)):
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        all_acts.append(act)
        all_labels.append(row["emotion"])
        all_meta.append(dict(row))

    if not all_acts:
        raise ValueError(f"No activations found in {act_dir}")

    acts   = np.stack(all_acts)    # (n_stimuli, n_layers, d_model)
    labels = np.array(all_labels)
    return acts, labels, all_meta


# ============================================================================
# PROBING FUNCTIONS
# ============================================================================

def probe_binary(acts: np.ndarray, meta_list: list, n_folds: int = 5) -> list:
    """
    Binary probe: emotional (B_clinical) vs neutral at each layer.
    Returns list of dicts with layer, auroc_mean, auroc_std, acc_mean.
    """
    binary_labels = np.array([
        "emotional" if m["stimulus_set"] == "B_clinical" else "neutral"
        for m in meta_list
    ])
    label_map = {"emotional": 1, "neutral": 0}
    y = np.array([label_map[l] for l in binary_labels])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows = []
    for l in range(N_LAYERS):
        X = acts[:, l, :]
        fold_aurocs = []
        fold_accs   = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            mu  = X_tr.mean(0, keepdims=True)
            std = X_tr.std(0, keepdims=True) + 1e-8
            X_tr_n = (X_tr - mu) / std
            X_te_n = (X_te - mu) / std
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X_tr_n, y_tr)
            y_prob = clf.predict_proba(X_te_n)[:, 1]
            y_pred = clf.predict(X_te_n)
            if len(np.unique(y_te)) > 1:
                fold_aurocs.append(roc_auc_score(y_te, y_prob))
            fold_accs.append((y_pred == y_te).mean())
        rows.append({
            "layer": l,
            "norm_depth": round((l + 1) / N_LAYERS, 4),
            "auroc_mean": float(np.nanmean(fold_aurocs)),
            "auroc_std":  float(np.nanstd(fold_aurocs)),
            "acc_mean":   float(np.mean(fold_accs)),
        })
    return rows


def probe_8class(acts: np.ndarray, labels: np.ndarray, meta_list: list,
                 n_folds: int = 5) -> list:
    """
    8-class probe: only clinical stimuli, macro AUROC.
    Returns list of dicts with layer, auroc_mean, auroc_std, acc_mean.
    """
    clinical_mask = np.array([m["stimulus_set"] == "B_clinical" for m in meta_list])
    X_clin = acts[clinical_mask]
    y_clin = labels[clinical_mask]

    classes   = sorted(np.unique(y_clin))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int     = np.array([label_map[yi] for yi in y_clin])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows = []
    for l in range(N_LAYERS):
        X = X_clin[:, l, :]
        fold_aurocs = []
        fold_accs   = []
        for train_idx, test_idx in skf.split(X, y_int):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_int[train_idx], y_int[test_idx]
            mu  = X_tr.mean(0, keepdims=True)
            std = X_tr.std(0, keepdims=True) + 1e-8
            X_tr_n = (X_tr - mu) / std
            X_te_n = (X_te - mu) / std
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X_tr_n, y_tr)
            y_prob = clf.predict_proba(X_te_n)
            y_pred = clf.predict(X_te_n)
            if len(np.unique(y_te)) > 1:
                auroc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
                fold_aurocs.append(auroc)
            fold_accs.append((y_pred == y_te).mean())
        rows.append({
            "layer": l,
            "norm_depth": round((l + 1) / N_LAYERS, 4),
            "auroc_mean": float(np.nanmean(fold_aurocs)),
            "auroc_std":  float(np.nanstd(fold_aurocs)),
            "acc_mean":   float(np.mean(fold_accs)),
        })
    return rows


# ============================================================================
# LOAD BOTH CONDITIONS
# ============================================================================

console.print("[bold cyan]Loading zero-shot activations (Exp 19)...[/bold cyan]")
zs_acts, zs_labels, zs_meta = load_setb_activations(ZS_ACT_DIR, act_type="h")
console.print(f"  Loaded {len(zs_acts)} stimuli × {N_LAYERS} layers × {zs_acts.shape[2]} dims")

console.print("[bold cyan]Loading few-shot activations (Exp 12)...[/bold cyan]")
fs_acts, fs_labels, fs_meta = load_setb_activations(FS_ACT_DIR, act_type="h")
console.print(f"  Loaded {len(fs_acts)} stimuli × {N_LAYERS} layers × {fs_acts.shape[2]} dims")

# ============================================================================
# RUN PROBES
# ============================================================================

console.print("\n[bold cyan]Running binary probes (emotional vs neutral)...[/bold cyan]")
zs_binary = probe_binary(zs_acts, zs_meta)
fs_binary = probe_binary(fs_acts, fs_meta)

console.print("[bold cyan]Running 8-class probes (clinical stimuli only)...[/bold cyan]")
zs_8class = probe_8class(zs_acts, zs_labels, zs_meta)
fs_8class = probe_8class(fs_acts, fs_labels, fs_meta)

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Flatten into one comparison CSV
comparison_rows = []
for l in range(N_LAYERS):
    comparison_rows.append({
        "layer":          l,
        "norm_depth":     round((l + 1) / N_LAYERS, 4),
        "zs_binary_auroc": zs_binary[l]["auroc_mean"],
        "fs_binary_auroc": fs_binary[l]["auroc_mean"],
        "delta_binary":    zs_binary[l]["auroc_mean"] - fs_binary[l]["auroc_mean"],
        "zs_8class_auroc": zs_8class[l]["auroc_mean"],
        "fs_8class_auroc": fs_8class[l]["auroc_mean"],
        "delta_8class":    zs_8class[l]["auroc_mean"] - fs_8class[l]["auroc_mean"],
    })

df = pl.DataFrame(comparison_rows)
df.write_csv(OUT_DIR / "probe_comparison.csv")
console.print(f"\n[bold green]✓ probe_comparison.csv saved[/bold green]")

# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

table = Table(title="Zero-Shot vs Few-Shot — Set B Probes (h component, Llama-1B Instruct)")
table.add_column("Layer", justify="right")
table.add_column("Depth", justify="right")
table.add_column("Binary ZS", justify="right", style="cyan")
table.add_column("Binary FS", justify="right", style="yellow")
table.add_column("Δ Binary", justify="right")
table.add_column("8-Class ZS", justify="right", style="cyan")
table.add_column("8-Class FS", justify="right", style="yellow")
table.add_column("Δ 8-Class", justify="right")

zs_bin_peak = int(np.argmax([r["auroc_mean"] for r in zs_binary]))
fs_bin_peak = int(np.argmax([r["auroc_mean"] for r in fs_binary]))
zs_8cl_peak = int(np.argmax([r["auroc_mean"] for r in zs_8class]))
fs_8cl_peak = int(np.argmax([r["auroc_mean"] for r in fs_8class]))

for row in comparison_rows:
    l = row["layer"]
    db = row["delta_binary"]
    d8 = row["delta_8class"]
    db_str = f"[green]{db:+.4f}[/green]" if db >= 0 else f"[red]{db:+.4f}[/red]"
    d8_str = f"[green]{d8:+.4f}[/green]" if d8 >= 0 else f"[red]{d8:+.4f}[/red]"
    table.add_row(
        str(l),
        f"{row['norm_depth']:.3f}",
        f"{row['zs_binary_auroc']:.4f}",
        f"{row['fs_binary_auroc']:.4f}",
        db_str,
        f"{row['zs_8class_auroc']:.4f}",
        f"{row['fs_8class_auroc']:.4f}",
        d8_str,
    )

console.print(table)

# Summary stats at peak layers
console.print("\n[bold cyan]Peak layer summary:[/bold cyan]")
console.print(f"  Binary  — ZS peak: L{zs_bin_peak} ({zs_binary[zs_bin_peak]['auroc_mean']:.4f}) | "
              f"FS peak: L{fs_bin_peak} ({fs_binary[fs_bin_peak]['auroc_mean']:.4f})")
console.print(f"  8-class — ZS peak: L{zs_8cl_peak} ({zs_8class[zs_8cl_peak]['auroc_mean']:.4f}) | "
              f"FS peak: L{fs_8cl_peak} ({fs_8class[fs_8cl_peak]['auroc_mean']:.4f})")

# ============================================================================
# VISUALIZATION — 2x2: binary and 8-class x within-layer curves and delta
# ============================================================================

console.print("\n[bold cyan]Generating comparison plot...[/bold cyan]")

norm_depths = np.array([r["norm_depth"] for r in comparison_rows])

zs_bin_auroc = np.array([r["auroc_mean"] for r in zs_binary])
fs_bin_auroc = np.array([r["auroc_mean"] for r in fs_binary])
zs_bin_std   = np.array([r["auroc_std"] for r in zs_binary])
fs_bin_std   = np.array([r["auroc_std"] for r in fs_binary])

zs_8cl_auroc = np.array([r["auroc_mean"] for r in zs_8class])
fs_8cl_auroc = np.array([r["auroc_mean"] for r in fs_8class])
zs_8cl_std   = np.array([r["auroc_std"] for r in zs_8class])
fs_8cl_std   = np.array([r["auroc_std"] for r in fs_8class])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Experiment 19: Zero-Shot vs Few-Shot — Set B Probes\n"
    "Llama-3.2-1B-Instruct · h (residual stream) · 5-fold CV",
    fontsize=12
)

# ---- Panel 1: Binary probe ----
ax = axes[0]
ax.plot(norm_depths, fs_bin_auroc, color="#e67e22", linewidth=2.5,
        label="Few-shot (Exp 12)", marker="o", markersize=4)
ax.fill_between(norm_depths, fs_bin_auroc - fs_bin_std, fs_bin_auroc + fs_bin_std,
                alpha=0.2, color="#e67e22")
ax.plot(norm_depths, zs_bin_auroc, color="#2980b9", linewidth=2.5,
        label="Zero-shot (Exp 19)", marker="s", markersize=4, linestyle="--")
ax.fill_between(norm_depths, zs_bin_auroc - zs_bin_std, zs_bin_auroc + zs_bin_std,
                alpha=0.2, color="#2980b9")
ax.axhline(0.5, linestyle=":", color="gray", alpha=0.5, label="Chance (binary)")
ax.set_title("Binary Probe: Emotional vs Neutral")
ax.set_xlabel("Normalized depth"); ax.set_ylabel("AUROC")
ax.set_xlim(0, 1); ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ---- Panel 2: 8-class probe ----
ax = axes[1]
ax.plot(norm_depths, fs_8cl_auroc, color="#e67e22", linewidth=2.5,
        label="Few-shot (Exp 12)", marker="o", markersize=4)
ax.fill_between(norm_depths, fs_8cl_auroc - fs_8cl_std, fs_8cl_auroc + fs_8cl_std,
                alpha=0.2, color="#e67e22")
ax.plot(norm_depths, zs_8cl_auroc, color="#2980b9", linewidth=2.5,
        label="Zero-shot (Exp 19)", marker="s", markersize=4, linestyle="--")
ax.fill_between(norm_depths, zs_8cl_auroc - zs_8cl_std, zs_8cl_auroc + zs_8cl_std,
                alpha=0.2, color="#2980b9")
ax.axhline(1 / 8, linestyle=":", color="gray", alpha=0.5, label="Chance (8-class)")
ax.set_title("8-Class Probe: Emotion Category (Clinical Only)")
ax.set_xlabel("Normalized depth"); ax.set_ylabel("Macro AUROC")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "probe_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "probe_comparison.svg", bbox_inches="tight")
plt.close()
console.print("[bold green]✓ probe_comparison.png/.svg saved[/bold green]")

# ============================================================================
# WRITE FINDINGS TO README
# ============================================================================

zs_bin_max = float(np.max(zs_bin_auroc))
fs_bin_max = float(np.max(fs_bin_auroc))
zs_8cl_max = float(np.max(zs_8cl_auroc))
fs_8cl_max = float(np.max(fs_8cl_auroc))

bin_delta  = zs_bin_max - fs_bin_max
cls_delta  = zs_8cl_max - fs_8cl_max

# Determine outcome
if abs(bin_delta) < 0.05 and abs(cls_delta) < 0.10:
    outcome = "**Outcome 1** — Both probes robust to shot removal. Shots do not drive the signal. Confound eliminated."
elif bin_delta < -0.15 and cls_delta < -0.15:
    outcome = "**Outcome 2** — Both probes drop significantly. Few-shot context was load-bearing. Warrants deeper investigation."
elif abs(bin_delta) < 0.05 and cls_delta < -0.10:
    outcome = "**Outcome 3** — Binary stays high; 8-class drops. Shots help categorization but not binary affect detection. Strengthens dissociation story."
else:
    outcome = f"**Mixed** — binary delta {bin_delta:+.4f}, 8-class delta {cls_delta:+.4f}. Requires interpretation."

readme_path = EXP_DIR / "README.md"
existing    = readme_path.read_text()

results_section = f"""
## Results

**Date:** 2026-02-28
**Model:** Llama-3.2-1B-Instruct
**Activation type:** h (residual stream)
**Probe CV:** 5-fold stratified

### Peak AUROC Comparison

| Probe | Few-shot (Exp 12) | Zero-shot (Exp 19) | Delta |
|-------|-------------------|-------------------|-------|
| Binary (emotional vs neutral) | {fs_bin_max:.4f} at L{fs_bin_peak} | {zs_bin_max:.4f} at L{zs_bin_peak} | {bin_delta:+.4f} |
| 8-class (clinical emotions)   | {fs_8cl_max:.4f} at L{fs_8cl_peak} | {zs_8cl_max:.4f} at L{zs_8cl_peak} | {cls_delta:+.4f} |

### Interpretation

{outcome}

### Outputs

- `outputs/probe_comparison.csv` — AUROC by layer, both conditions
- `outputs/probe_comparison.png/.svg` — comparison plot
"""

# Append results to existing README (replace placeholder if present)
if "*To be filled after run.*" in existing:
    updated = existing.replace("*To be filled after run.*", results_section.strip())
else:
    updated = existing + "\n" + results_section

readme_path.write_text(updated)
console.print("[bold green]✓ README.md updated with results[/bold green]")

console.print(f"\n[bold green]✓ Experiment 19 complete![/bold green]")
console.print(f"  Binary  delta (ZS vs FS): {bin_delta:+.4f}")
console.print(f"  8-class delta (ZS vs FS): {cls_delta:+.4f}")
console.print(f"  {outcome}")
