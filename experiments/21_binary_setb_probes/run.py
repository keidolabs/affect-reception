"""
Experiment 21 — Binary (Emotional vs Neutral) Probes on Set B
Question: At which layer does binary affect detection saturate for each model?
Date: 2026-03-14

Fills Table 4 gap in manuscript — runs binary probes on Set B clinical vignettes
(96 emotional + 96 neutral) for all 6 models using v2 activations (Exp 12).
Methodology matches Exp 13 (per-fold z-normalization, 5-fold stratified CV).

Prerequisite: experiments/12_v2_extract_setb activations for all models
Run: uv run python experiments/21_binary_setb_probes/run.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table

console = Console()
np.random.seed(42)

SEED = 42
EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
SETB_DIR = ROOT / "experiments" / "12_v2_extract_setb" / "outputs" / "activations"
OUT_DIR = EXP_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL REGISTRY
# ============================================================

MODELS = [
    {"key": "llama1b_inst",  "label": "Llama-1B Instruct", "n_layers": 16},
    {"key": "llama1b_base",  "label": "Llama-1B Base",     "n_layers": 16},
    {"key": "llama8b_inst",  "label": "Llama-8B Instruct", "n_layers": 32},
    {"key": "llama8b_base",  "label": "Llama-8B Base",     "n_layers": 32},
    {"key": "gemma9b_inst",  "label": "Gemma-9B Instruct", "n_layers": 42},
    {"key": "gemma9b_base",  "label": "Gemma-9B Base",     "n_layers": 42},
]

# ============================================================
# LOAD ACTIVATIONS
# ============================================================

def load_setb_activations(model_key: str, act_type: str = "h"):
    """
    Load Set B activations (emotional + neutral) for binary probing.
    Returns (activations, binary_labels) where 1=emotional, 0=neutral.
    """
    model_dir = SETB_DIR / model_key
    manifest_path = model_dir / "manifest.csv"
    if not manifest_path.exists():
        console.print(f"  [red]Manifest not found: {manifest_path}[/red]")
        return None, None

    manifest = pl.read_csv(manifest_path)
    all_acts = []
    all_labels = []

    for row in manifest.iter_rows(named=True):
        stim_id = row["id"]
        npz_path = model_dir / f"{stim_id}.npz"
        if not npz_path.exists():
            continue

        npz = np.load(npz_path, allow_pickle=True)
        act = npz[act_type].copy()  # (n_layers, d_model)

        if not np.all(np.isfinite(act)):
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

        all_acts.append(act)
        # Binary label: emotional (B-*) = 1, neutral (N-*) = 0
        all_labels.append(1 if row["emotion"] != "neutral" else 0)

    if not all_acts:
        return None, None

    activations = np.stack(all_acts)  # (n_stimuli, n_layers, d_model)
    labels = np.array(all_labels)
    return activations, labels

# ============================================================
# BINARY PROBE — per-fold z-normalization, 5-fold stratified CV
# ============================================================

def probe_binary_layer(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """Binary logistic regression probe with per-fold z-normalization."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    aurocs = []
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Per-fold z-normalization (matches Exp 13 methodology)
        mu = X_tr.mean(0, keepdims=True)
        std = X_tr.std(0, keepdims=True) + 1e-8
        X_tr_n = (X_tr - mu) / std
        X_te_n = (X_te - mu) / std

        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        clf.fit(X_tr_n, y_tr)

        y_prob = clf.predict_proba(X_te_n)[:, 1]  # P(emotional)
        y_pred = clf.predict(X_te_n)

        if len(np.unique(y_te)) > 1:
            auroc = roc_auc_score(y_te, y_prob)
        else:
            auroc = float("nan")
        aurocs.append(auroc)
        accs.append((y_pred == y_te).mean())

    return {
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
    }

# ============================================================
# MAIN — run binary probes for all models, all layers
# ============================================================

console.print("\n[bold cyan]Experiment 21 — Binary Set B Probes (Affect Reception)[/bold cyan]")
console.print("Probing emotional vs neutral on v2 Set B activations, all 6 models\n")

summary_rows = []

for model_cfg in MODELS:
    model_key = model_cfg["key"]
    n_layers = model_cfg["n_layers"]
    console.print(f"[bold white]{'='*60}[/bold white]")
    console.print(f"[bold white]{model_cfg['label']} ({model_key})[/bold white]")

    acts, labels = load_setb_activations(model_key, act_type="h")
    if acts is None:
        console.print(f"  [yellow]SKIP — no activations found[/yellow]")
        continue

    n_emo = (labels == 1).sum()
    n_neu = (labels == 0).sum()
    console.print(f"  Loaded: {n_emo} emotional + {n_neu} neutral = {len(labels)} total")
    console.print(f"  Shape: {acts.shape}")

    # Probe every layer
    rows = []
    for l in range(n_layers):
        r = probe_binary_layer(acts[:, l, :], labels)
        r["layer"] = l
        rows.append(r)
        if (l + 1) % 5 == 0 or l == n_layers - 1:
            console.print(f"  L{l:2d}: AUROC={r['auroc_mean']:.4f} (±{r['auroc_std']:.4f})  "
                          f"Acc={r['acc_mean']:.3f}")

    # Save per-layer CSV
    df = pl.DataFrame(rows)
    csv_path = OUT_DIR / f"results_binary_{model_key}.csv"
    df.write_csv(csv_path)
    console.print(f"  Saved → {csv_path.name}")

    # Find peak and saturation layer
    aurocs = [r["auroc_mean"] for r in rows]
    peak_auroc = max(aurocs)
    peak_layer = aurocs.index(peak_auroc)

    # Saturation: first layer where AUROC >= 0.999
    sat_layer = None
    for l, a in enumerate(aurocs):
        if a >= 0.999:
            sat_layer = l
            break

    console.print(f"  [bold green]Peak: L{peak_layer} AUROC={peak_auroc:.4f}[/bold green]")
    if sat_layer is not None:
        console.print(f"  [bold green]Saturates (>=0.999): L{sat_layer}/{n_layers}[/bold green]")
    else:
        console.print(f"  [yellow]Does not reach 0.999[/yellow]")

    summary_rows.append({
        "model": model_cfg["label"],
        "model_key": model_key,
        "n_layers": n_layers,
        "peak_auroc": peak_auroc,
        "peak_layer": peak_layer,
        "saturation_layer": sat_layer,
        "norm_depth_peak": round(peak_layer / n_layers, 3),
        "norm_depth_sat": round(sat_layer / n_layers, 3) if sat_layer is not None else None,
    })

# ============================================================
# SUMMARY TABLE
# ============================================================

console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
console.print("[bold cyan]SUMMARY — Binary Detection AUROC on Set B[/bold cyan]\n")

table = Table(title="Table 4 (Full) — Binary Detection AUROC on Set B")
table.add_column("Model", style="white")
table.add_column("Binary AUROC", style="green")
table.add_column("Peak Layer", style="cyan")
table.add_column("Saturates at", style="cyan")
table.add_column("Norm. Depth", style="dim")

for r in summary_rows:
    sat_str = f"L{r['saturation_layer']}/{r['n_layers']}" if r["saturation_layer"] is not None else "—"
    table.add_row(
        r["model"],
        f"{r['peak_auroc']:.4f}",
        f"L{r['peak_layer']}",
        sat_str,
        f"{r['norm_depth_peak']:.2f}",
    )

console.print(table)

# Save summary
summary_df = pl.DataFrame(summary_rows)
summary_path = OUT_DIR / "summary_binary_setb.csv"
summary_df.write_csv(summary_path)
console.print(f"\nSaved summary → {summary_path.name}")

# ============================================================
# MANUSCRIPT TABLE 4 — formatted for easy copy-paste
# ============================================================

console.print("\n[bold cyan]Manuscript Table 4 (markdown):[/bold cyan]\n")
console.print("| Model | Binary AUROC | Saturates at Layer |")
console.print("|-------|-------------|-------------------|")
for r in summary_rows:
    sat_str = f"L{r['saturation_layer']}/{r['n_layers']}" if r["saturation_layer"] is not None else "—"
    console.print(f"| {r['model']} | {r['peak_auroc']:.3f} | {sat_str} |")

console.print("\n[bold green]Done.[/bold green]")
