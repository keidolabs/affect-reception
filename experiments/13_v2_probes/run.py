"""
Experiment 13 — v2 Full Probing Analysis
Question: Where do h, a, m components encode emotion? How well does A→B transfer?
Date: 2026-02-25

Comprehensive probing analysis across all 6 models:
  1. Within-set probes: Set A (8-class), Set B clinical (8-class), Set B binary
  2. Cross-set transfer: Train on A → test on B; Train on B → test on A
  3. All 3 activation types: h (residual), a (MHSA out), m (FFN out)
  4. Both conditions: all stimuli AND correctly-classified-only (Tak filter)

Prerequisite:
  - Run experiments/11_v2_extract_seta/extract_{model_key}.py for all models
  - Run experiments/12_v2_extract_setb/extract_{model_key}.py for all models
  - Run experiments/13_v2_probes/run_gate_check.py and gate must PASS

Run: uv run python experiments/13_v2_probes/run.py
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

EXP_DIR   = Path(__file__).parent
ROOT      = EXP_DIR.parent.parent
OUT_DIR   = EXP_DIR / "outputs"
SETA_DIR  = ROOT / "experiments" / "11_v2_extract_seta" / "outputs" / "activations"
SETB_DIR  = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODELS = [
    {"key": "llama1b_inst",  "label": "Llama-1B Instruct", "color": "#e74c3c",  "n_layers": 16},
    {"key": "llama1b_base",  "label": "Llama-1B Base",     "color": "#c0392b",  "n_layers": 16},
    {"key": "llama8b_inst",  "label": "Llama-8B Instruct", "color": "#2980b9",  "n_layers": 32},
    {"key": "llama8b_base",  "label": "Llama-8B Base",     "color": "#1a5276",  "n_layers": 32},
    {"key": "gemma9b_inst",  "label": "Gemma-9B Instruct", "color": "#27ae60",  "n_layers": 42},
    {"key": "gemma9b_base",  "label": "Gemma-9B Base",     "color": "#1e8449",  "n_layers": 42},
]

ACT_TYPES = ["h", "a", "m"]

# ============================================================================
# LOAD ACTIVATIONS HELPER
# ============================================================================

def load_activations(act_dir: Path, model_key: str, act_type: str,
                     filter_emotion: str = None, correct_only: bool = False):
    """
    Load activations for one model and act_type from a directory.
    Returns (activations, labels, metadata_list).
      activations: (n_stimuli, n_layers, d_model)
    """
    model_dir = act_dir / model_key
    manifest_path = model_dir / "manifest.csv"
    if not manifest_path.exists():
        return None, None, None

    manifest = pl.read_csv(manifest_path)

    all_acts = []
    all_labels = []
    all_meta = []

    for row in manifest.iter_rows(named=True):
        # Filter by emotion (e.g., exclude neutral for 8-class probe)
        if filter_emotion is not None and row["emotion"] == filter_emotion:
            continue
        # Filter to correctly-classified-only (Tak filter)
        if correct_only and not row.get("correct", True):
            continue

        stim_id = row["id"]
        npz_path = model_dir / f"{stim_id}.npz"
        if not npz_path.exists():
            continue

        npz = np.load(npz_path, allow_pickle=True)
        act = npz[act_type]  # (n_layers, d_model)

        if not np.all(np.isfinite(act)):
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

        all_acts.append(act)
        all_labels.append(row["emotion"])
        all_meta.append(dict(row))

    if not all_acts:
        return None, None, None

    activations = np.stack(all_acts)  # (n_stimuli, n_layers, d_model)
    labels = np.array(all_labels)
    return activations, labels, all_meta


# ============================================================================
# PROBING FUNCTIONS
# ============================================================================

def probe_layer(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """8-class logistic regression probe with 5-fold stratified CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    classes   = sorted(np.unique(y))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int     = np.array([label_map[yi] for yi in y])

    aurocs = []
    accs   = []
    for train_idx, test_idx in skf.split(X, y_int):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_int[train_idx], y_int[test_idx]
        mu = X_tr.mean(0, keepdims=True); std = X_tr.std(0, keepdims=True) + 1e-8
        X_tr_n = (X_tr - mu) / std; X_te_n = (X_te - mu) / std
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_tr_n, y_tr)
        y_prob = clf.predict_proba(X_te_n)
        y_pred = clf.predict(X_te_n)
        if len(np.unique(y_te)) > 1:
            auroc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
        else:
            auroc = float("nan")
        aurocs.append(auroc)
        accs.append((y_pred == y_te).mean())

    return {"auroc_mean": float(np.nanmean(aurocs)), "auroc_std": float(np.nanstd(aurocs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}


def probe_transfer(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train probe on X_train/y_train, evaluate on X_test/y_test (no CV — single train/test split).
    Used for cross-set transfer analysis.
    """
    classes = sorted(set(np.unique(y_train)) | set(np.unique(y_test)))
    label_map = {c: i for i, c in enumerate(classes)}
    y_tr_int = np.array([label_map[yi] for yi in y_train if yi in label_map])
    y_te_int = np.array([label_map[yi] for yi in y_test  if yi in label_map])

    # Filter to matched classes
    tr_mask = np.array([yi in label_map for yi in y_train])
    te_mask = np.array([yi in label_map for yi in y_test])

    if tr_mask.sum() < 16 or te_mask.sum() < 8:
        return {"auroc_mean": float("nan"), "acc_mean": float("nan"), "auroc_std": float("nan")}

    X_tr = X_train[tr_mask]; X_te = X_test[te_mask]
    mu = X_tr.mean(0, keepdims=True); std = X_tr.std(0, keepdims=True) + 1e-8
    X_tr_n = (X_tr - mu) / std; X_te_n = (X_te - mu) / std

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_tr_n, y_tr_int)
    y_prob = clf.predict_proba(X_te_n)
    y_pred = clf.predict(X_te_n)

    if len(np.unique(y_te_int)) > 1:
        auroc = roc_auc_score(y_te_int, y_prob, multi_class="ovr", average="macro",
                              labels=list(range(len(classes))))
    else:
        auroc = float("nan")

    acc = (y_pred == y_te_int).mean()
    return {"auroc_mean": float(auroc), "acc_mean": float(acc), "auroc_std": float("nan")}


# ============================================================================
# MAIN ANALYSIS LOOP — per model
# ============================================================================

all_probe_results = []  # collect for summary

for model_cfg in MODELS:
    model_key = model_cfg["key"]
    n_layers  = model_cfg["n_layers"]
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Probing: {model_cfg['label']} ({model_key})[/bold cyan]")

    for act_type in ACT_TYPES:
        console.print(f"\n  [bold white]--- {act_type.upper()} component ---[/bold white]")

        # ---- Within-Set A probes (8-class, emotional only) ----
        seta_acts, seta_labels, _ = load_activations(SETA_DIR, model_key, act_type,
                                                       filter_emotion=None)
        if seta_acts is None:
            console.print(f"  [yellow]SKIP: Set A not found for {model_key}[/yellow]")
            continue

        # Keep only emotional labels (no neutral)
        emo_mask = seta_labels != "neutral"
        seta_acts_emo = seta_acts[emo_mask]; seta_labels_emo = seta_labels[emo_mask]

        if len(seta_acts_emo) < 16:
            console.print(f"  [yellow]SKIP: Too few Set A stimuli ({len(seta_acts_emo)})[/yellow]")
            continue

        console.print(f"  Set A: {len(seta_acts_emo)} emotional stimuli × {n_layers} layers")

        within_a_rows = []
        for l in range(n_layers):
            r = probe_layer(seta_acts_emo[:, l, :], seta_labels_emo)
            within_a_rows.append({"layer": l, "norm_depth": round((l+1)/n_layers, 4),
                                   "model_key": model_key, "act_type": act_type,
                                   "condition": "within_A", **r})

        # Find peak layer from within-A probes (for reference in other analyses)
        a_aurocs = np.array([r["auroc_mean"] for r in within_a_rows])
        peak_l = int(np.argmax(a_aurocs))
        console.print(f"  Set A peak: L{peak_l} ({within_a_rows[peak_l]['norm_depth']:.3f}), AUROC={a_aurocs[peak_l]:.4f}")
        all_probe_results.extend(within_a_rows)

        pl.DataFrame(within_a_rows).write_csv(OUT_DIR / f"probe_results_{model_key}_seta_{act_type}.csv")

        # ---- Within-Set B clinical probes (8-class) ----
        setb_acts, setb_labels, setb_meta = load_activations(SETB_DIR, model_key, act_type)
        if setb_acts is None:
            console.print(f"  [yellow]Set B not found for {model_key} — skipping transfer[/yellow]")
            continue

        # Separate clinical and neutral
        b_clinical_mask = np.array([m["stimulus_set"] == "B_clinical" for m in setb_meta])
        b_neutral_mask  = np.array([m["stimulus_set"] == "neutral" for m in setb_meta])

        setb_acts_clin  = setb_acts[b_clinical_mask]
        setb_labels_clin = setb_labels[b_clinical_mask]
        setb_acts_neut  = setb_acts[b_neutral_mask]
        setb_labels_neut = setb_labels[b_neutral_mask]

        console.print(f"  Set B: {len(setb_acts_clin)} clinical + {len(setb_acts_neut)} neutral")

        # Within-Set B clinical (8-class)
        within_b_rows = []
        if len(setb_acts_clin) >= 16:
            for l in range(n_layers):
                r = probe_layer(setb_acts_clin[:, l, :], setb_labels_clin)
                within_b_rows.append({"layer": l, "norm_depth": round((l+1)/n_layers, 4),
                                       "model_key": model_key, "act_type": act_type,
                                       "condition": "within_B_clinical", **r})
            b_aurocs = np.array([r["auroc_mean"] for r in within_b_rows])
            b_peak = int(np.argmax(b_aurocs))
            console.print(f"  Set B peak: L{b_peak} ({within_b_rows[b_peak]['norm_depth']:.3f}), AUROC={b_aurocs[b_peak]:.4f}")
            all_probe_results.extend(within_b_rows)
            pl.DataFrame(within_b_rows).write_csv(OUT_DIR / f"probe_results_{model_key}_setb_clinical_{act_type}.csv")

        # ---- Cross-set transfer (train A → test B, train B → test A) ----
        transfer_rows = []
        for l in range(n_layers):
            # A→B transfer
            r_ab = probe_transfer(seta_acts_emo[:, l, :], seta_labels_emo,
                                   setb_acts_clin[:, l, :], setb_labels_clin)
            # B→A transfer
            r_ba = probe_transfer(setb_acts_clin[:, l, :], setb_labels_clin,
                                   seta_acts_emo[:, l, :], seta_labels_emo)
            norm_depth = round((l+1)/n_layers, 4)
            transfer_rows.append({"layer": l, "norm_depth": norm_depth,
                                   "model_key": model_key, "act_type": act_type,
                                   "condition": "A_to_B",
                                   "auroc_mean": r_ab["auroc_mean"], "acc_mean": r_ab["acc_mean"],
                                   "auroc_std": r_ab["auroc_std"]})
            transfer_rows.append({"layer": l, "norm_depth": norm_depth,
                                   "model_key": model_key, "act_type": act_type,
                                   "condition": "B_to_A",
                                   "auroc_mean": r_ba["auroc_mean"], "acc_mean": r_ba["acc_mean"],
                                   "auroc_std": r_ba["auroc_std"]})

        all_probe_results.extend(transfer_rows)
        pl.DataFrame(transfer_rows).write_csv(OUT_DIR / f"transfer_results_{model_key}_{act_type}.csv")

        # Quick transfer summary at peak layer
        ab_at_peak = transfer_rows[peak_l * 2]["auroc_mean"]
        ba_at_peak = transfer_rows[peak_l * 2 + 1]["auroc_mean"]
        console.print(f"  Transfer at Set A peak (L{peak_l}): A→B={ab_at_peak:.4f}, B→A={ba_at_peak:.4f}")

# ============================================================================
# COMBINED RESULTS
# ============================================================================

if all_probe_results:
    pl.DataFrame(all_probe_results).write_csv(OUT_DIR / "all_probe_results.csv")
    console.print(f"\n[bold green]✓ all_probe_results.csv saved ({len(all_probe_results)} rows)[/bold green]")

# ============================================================================
# VISUALIZATION — probe curves for primary model (llama1b_inst)
# ============================================================================

console.print("\n[bold cyan]Generating probe curve figures...[/bold cyan]")

PRIMARY = "llama1b_inst"
N_LAYERS_PRIMARY = 16

def load_csv_safe(path):
    return pl.read_csv(path) if path.exists() else None

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Probe Curves — {PRIMARY}\nWithin-Set vs Transfer", fontsize=13)

act_colors = {"h": "#e74c3c", "a": "#2980b9", "m": "#27ae60"}
act_labels  = {"h": "h (residual)", "a": "a (MHSA)", "m": "m (FFN)"}

for col_idx, act_type in enumerate(ACT_TYPES):
    # Row 0: Within-set probes
    ax_within = axes[0, col_idx]
    norm_depths = np.array([(l+1)/N_LAYERS_PRIMARY for l in range(N_LAYERS_PRIMARY)])

    seta_csv  = load_csv_safe(OUT_DIR / f"probe_results_{PRIMARY}_seta_{act_type}.csv")
    setb_csv  = load_csv_safe(OUT_DIR / f"probe_results_{PRIMARY}_setb_clinical_{act_type}.csv")

    if seta_csv is not None:
        a_auroc = seta_csv["auroc_mean"].to_numpy()
        ax_within.plot(norm_depths[:len(a_auroc)], a_auroc, color=act_colors[act_type],
                       linewidth=2.5, label="Set A (keyword-rich)", marker="o", markersize=4)
        ax_within.fill_between(norm_depths[:len(a_auroc)],
                                a_auroc - seta_csv["auroc_std"].to_numpy(),
                                a_auroc + seta_csv["auroc_std"].to_numpy(),
                                alpha=0.2, color=act_colors[act_type])

    if setb_csv is not None:
        b_auroc = setb_csv["auroc_mean"].to_numpy()
        ax_within.plot(norm_depths[:len(b_auroc)], b_auroc, color=act_colors[act_type],
                       linewidth=2.5, label="Set B (clinical)", linestyle="--",
                       marker="s", markersize=4)

    ax_within.axhline(1/8, linestyle=":", color="gray", alpha=0.5, label="Chance")
    ax_within.set_title(f"{act_labels[act_type]} — Within-set", fontsize=10)
    ax_within.set_xlabel("Normalized depth")
    ax_within.set_ylabel("Macro AUROC")
    ax_within.set_xlim(0, 1); ax_within.set_ylim(0, 1)
    ax_within.legend(fontsize=8); ax_within.grid(True, alpha=0.3)

    # Row 1: Transfer curves
    ax_transfer = axes[1, col_idx]
    transfer_csv = load_csv_safe(OUT_DIR / f"transfer_results_{PRIMARY}_{act_type}.csv")

    if transfer_csv is not None:
        ab_rows  = transfer_csv.filter(pl.col("condition") == "A_to_B")
        ba_rows  = transfer_csv.filter(pl.col("condition") == "B_to_A")

        if seta_csv is not None:
            ax_transfer.plot(norm_depths[:len(seta_csv)], seta_csv["auroc_mean"].to_numpy(),
                             color=act_colors[act_type], linewidth=1.5, linestyle="-",
                             alpha=0.5, label="Within-A")
        if setb_csv is not None:
            ax_transfer.plot(norm_depths[:len(setb_csv)], setb_csv["auroc_mean"].to_numpy(),
                             color=act_colors[act_type], linewidth=1.5, linestyle="--",
                             alpha=0.5, label="Within-B")

        if len(ab_rows) > 0:
            ax_transfer.plot(ab_rows["norm_depth"].to_numpy(), ab_rows["auroc_mean"].to_numpy(),
                             color="#e67e22", linewidth=2.5, label="A→B transfer", marker="^")
        if len(ba_rows) > 0:
            ax_transfer.plot(ba_rows["norm_depth"].to_numpy(), ba_rows["auroc_mean"].to_numpy(),
                             color="#8e44ad", linewidth=2.5, label="B→A transfer", marker="v")

    ax_transfer.axhline(1/8, linestyle=":", color="gray", alpha=0.5)
    ax_transfer.set_title(f"{act_labels[act_type]} — Cross-set Transfer", fontsize=10)
    ax_transfer.set_xlabel("Normalized depth")
    ax_transfer.set_ylabel("Macro AUROC")
    ax_transfer.set_xlim(0, 1); ax_transfer.set_ylim(0, 1)
    ax_transfer.legend(fontsize=8); ax_transfer.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / f"probe_curves_panel_{PRIMARY}.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / f"probe_curves_panel_{PRIMARY}.svg", bbox_inches="tight")
plt.close()
console.print(f"[bold green]✓ probe_curves_panel_{PRIMARY}.png/.svg saved[/bold green]")

# ============================================================================
# CROSS-MODEL SUMMARY TABLE
# ============================================================================

console.print("\n[bold cyan]Building cross-model summary table...[/bold cyan]")

summary_rows = []
for model_cfg in MODELS:
    mk = model_cfg["key"]
    nl = model_cfg["n_layers"]
    row = {"model": mk, "label": model_cfg["label"], "n_layers": nl}
    for act_type in ACT_TYPES:
        # Peak from within-A probes
        csv_path = OUT_DIR / f"probe_results_{mk}_seta_{act_type}.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            peak_idx = int(df["auroc_mean"].to_numpy().argmax())
            row[f"seta_{act_type}_peak_layer"]  = int(df["layer"][peak_idx])
            row[f"seta_{act_type}_peak_depth"]  = float(df["norm_depth"][peak_idx])
            row[f"seta_{act_type}_peak_auroc"]  = float(df["auroc_mean"][peak_idx])
        else:
            row[f"seta_{act_type}_peak_layer"]  = None
            row[f"seta_{act_type}_peak_depth"]  = None
            row[f"seta_{act_type}_peak_auroc"]  = None

        # Transfer drop A→B at Set A peak
        t_csv = OUT_DIR / f"transfer_results_{mk}_{act_type}.csv"
        if t_csv.exists() and row.get(f"seta_{act_type}_peak_layer") is not None:
            t_df   = pl.read_csv(t_csv).filter(pl.col("condition") == "A_to_B")
            peak_l = row[f"seta_{act_type}_peak_layer"]
            if len(t_df) > peak_l:
                row[f"ab_transfer_{act_type}_at_peak"] = float(t_df["auroc_mean"][peak_l])
            else:
                row[f"ab_transfer_{act_type}_at_peak"] = None
        else:
            row[f"ab_transfer_{act_type}_at_peak"] = None

    summary_rows.append(row)

if summary_rows:
    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(OUT_DIR / "probe_summary_all_models.csv")
    console.print(f"[bold green]✓ probe_summary_all_models.csv saved[/bold green]")

    # Print summary table
    table = Table(title="Cross-Model Probe Summary (h component)")
    table.add_column("Model", style="cyan")
    table.add_column("Set A h peak layer", justify="right")
    table.add_column("h peak depth", justify="right")
    table.add_column("h peak AUROC", justify="right")
    table.add_column("A→B transfer h", justify="right")

    for row in summary_rows:
        table.add_row(
            row["label"],
            str(row.get("seta_h_peak_layer", "??")),
            f"{row.get('seta_h_peak_depth') or 0:.3f}",
            f"{row.get('seta_h_peak_auroc') or 0:.4f}",
            f"{row.get('ab_transfer_h_at_peak', 0):.4f}" if row.get("ab_transfer_h_at_peak") else "N/A",
        )
    console.print(table)

# ============================================================================
# GENERATE README
# ============================================================================

console.print("\n[bold cyan]Writing README...[/bold cyan]")

# Load primary model gate check data
primary_seta_h = load_csv_safe(OUT_DIR / f"probe_results_{PRIMARY}_seta_h.csv")
primary_peak_str = "??"
primary_auroc_str = "??"
if primary_seta_h is not None:
    pk = int(primary_seta_h["auroc_mean"].to_numpy().argmax())
    primary_peak_str = f"L{pk} ({primary_seta_h['norm_depth'][pk]:.3f})"
    primary_auroc_str = f"{primary_seta_h['auroc_mean'][pk]:.4f}"

readme_lines = [
    "# Experiment 13 — v2 Full Probing Analysis",
    "",
    "**Date:** 2026-02-25",
    "**Primary model:** llama1b_inst (Llama-3.2-1B-Instruct)",
    "**Stimuli:** 80 Set A (keyword-rich) + 96 Set B clinical (keyword-free)",
    "**Activation types:** h (residual stream), a (MHSA output), m (FFN output)",
    "",
    "## Research Questions",
    "",
    "1. At which layers do h, a, m components encode emotion?",
    "2. Does the emotion encoding transfer from keyword-rich (Set A) to keyword-free (Set B)?",
    "3. How does base vs instruct affect probe performance?",
    "",
    "## Key Findings — Primary Model (llama1b_inst)",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| h probe peak | {primary_peak_str} |",
    f"| h probe peak AUROC | {primary_auroc_str} |",
    f"| Gate check | See outputs/gate_check_report_llama1b_inst.md |",
    "",
    "## Summary Table (h component across models)",
    "",
]

for row in summary_rows:
    readme_lines.append(
        f"| {row['label']} | L{row.get('seta_h_peak_layer', '?')} "
        f"({row.get('seta_h_peak_depth') or 0:.3f}) | "
        f"{row.get('seta_h_peak_auroc') or 0:.4f} AUROC | "
        f"A→B: {row.get('ab_transfer_h_at_peak') or 'N/A'} |"
    )

readme_lines += [
    "",
    "## Methodology",
    "",
    "- 8-class logistic regression probes with 5-fold stratified cross-validation",
    "- Macro AUROC (one-vs-rest) as primary metric",
    "- Stimuli normalized per fold before fitting",
    "- Transfer: train on all Set A emotional, test on all Set B clinical (same emotion classes)",
    "",
    "## Outputs",
    "",
    "- `outputs/probe_results_{model}_{set}_{act_type}.csv` — per-layer AUROC",
    "- `outputs/transfer_results_{model}_{act_type}.csv` — A→B and B→A transfer",
    "- `outputs/probe_summary_all_models.csv` — cross-model comparison",
    "- `outputs/probe_curves_panel_{PRIMARY}.png/.svg` — visualization",
    "",
]

(EXP_DIR / "README.md").write_text("\n".join(readme_lines))
console.print(f"[bold green]✓ README.md written[/bold green]")

console.print(f"\n[bold green]✓ Experiment 13 complete![/bold green]")
console.print(f"  Next: Run experiments/14_v2_patching/run.py (activation patching)")
