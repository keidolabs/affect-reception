"""
Experiment 18 — v2 Cross-Model Summary
Question: How do emotion circuits compare across 6 models (3 base/instruct pairs)?
Date: 2026-02-25

Aggregates results from Experiments 13–17 into:
  1. Primary summary table (6 models × key metrics)
  2. Cross-model comparison figures (paper-ready)
  3. Full findings README

Prerequisite: Run all experiments 13–17 for each model.

Run: uv run python experiments/18_v2_summary/run.py
"""

import sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
np.random.seed(42)

EXP_DIR  = Path(__file__).parent
ROOT     = EXP_DIR.parent.parent
OUT_DIR  = EXP_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

MODELS = [
    {"key": "llama1b_inst",  "label": "Llama-1B Instruct", "family": "Llama-1B",  "type": "instruct", "color": "#e74c3c",  "n_layers": 16},
    {"key": "llama1b_base",  "label": "Llama-1B Base",     "family": "Llama-1B",  "type": "base",     "color": "#c0392b",  "n_layers": 16},
    {"key": "llama8b_inst",  "label": "Llama-8B Instruct", "family": "Llama-8B",  "type": "instruct", "color": "#2980b9",  "n_layers": 32},
    {"key": "llama8b_base",  "label": "Llama-8B Base",     "family": "Llama-8B",  "type": "base",     "color": "#1a5276",  "n_layers": 32},
    {"key": "gemma9b_inst",  "label": "Gemma-9B Instruct", "family": "Gemma-9B",  "type": "instruct", "color": "#27ae60",  "n_layers": 42},
    {"key": "gemma9b_base",  "label": "Gemma-9B Base",     "family": "Gemma-9B",  "type": "base",     "color": "#1e8449",  "n_layers": 42},
]

PROBES_DIR   = ROOT / "experiments" / "13_v2_probes" / "outputs"
PATCHING_DIR = ROOT / "experiments" / "14_v2_patching" / "outputs"
ATTENTION_DIR = ROOT / "experiments" / "15_v2_attention" / "outputs"
KNOCKOUT_DIR = ROOT / "experiments" / "16_v2_knockout" / "outputs"
GEOMETRY_DIR = ROOT / "experiments" / "17_v2_geometry" / "outputs"

# ============================================================================
# COLLECT METRICS FROM ALL EXPERIMENTS
# ============================================================================

def safe_read_csv(path):
    return pl.read_csv(path) if path.exists() else None

def get_peak_from_csv(path, auroc_col="auroc_mean", norm_col="norm_depth", layer_col="layer"):
    df = safe_read_csv(path)
    if df is None or len(df) == 0:
        return None, None, None
    peak_idx = int(df[auroc_col].to_numpy().argmax())
    return int(df[layer_col][peak_idx]), float(df[norm_col][peak_idx]), float(df[auroc_col][peak_idx])


console.print("[bold cyan]Collecting metrics from all experiments...[/bold cyan]")

summary_rows = []

for model_cfg in MODELS:
    mk = model_cfg["key"]
    nl = model_cfg["n_layers"]

    row = {
        "model_key": mk,
        "label": model_cfg["label"],
        "family": model_cfg["family"],
        "type": model_cfg["type"],
        "n_layers": nl,
    }

    # --- Exp 13: Probe peaks ---
    for act_type in ["h", "a", "m"]:
        peak_l, peak_d, peak_auroc = get_peak_from_csv(
            PROBES_DIR / f"probe_results_{mk}_seta_{act_type}.csv"
        )
        row[f"seta_{act_type}_peak_layer"]  = peak_l
        row[f"seta_{act_type}_peak_depth"]  = peak_d
        row[f"seta_{act_type}_peak_auroc"]  = peak_auroc

        b_peak_l, b_peak_d, b_peak_auroc = get_peak_from_csv(
            PROBES_DIR / f"probe_results_{mk}_setb_clinical_{act_type}.csv"
        )
        row[f"setb_{act_type}_peak_auroc"] = b_peak_auroc

        # Transfer: A→B at Set A h peak
        t_df = safe_read_csv(PROBES_DIR / f"transfer_results_{mk}_{act_type}.csv")
        if t_df is not None and row.get(f"seta_{act_type}_peak_layer") is not None:
            ab_df = t_df.filter(pl.col("condition") == "A_to_B")
            peak_l_local = row[f"seta_{act_type}_peak_layer"]
            if peak_l_local is not None and peak_l_local < len(ab_df):
                row[f"ab_transfer_{act_type}"] = float(ab_df["auroc_mean"][peak_l_local])
            else:
                row[f"ab_transfer_{act_type}"] = None
        else:
            row[f"ab_transfer_{act_type}"] = None

    # --- Exp 14: Patching peak success ---
    p_df = safe_read_csv(PATCHING_DIR / f"patching_success_by_layer_{mk}_seta_h.csv")
    if p_df is not None and len(p_df) > 0:
        peak_row = p_df.filter(pl.col("success_rate") == p_df["success_rate"].max())
        row["patching_seta_peak_layer"]   = int(peak_row["layer_start"][0])
        row["patching_seta_peak_success"] = float(peak_row["success_rate"][0])
    else:
        row["patching_seta_peak_layer"] = row["patching_seta_peak_success"] = None

    # --- Exp 16: Critical knockout layers ---
    ko_df = safe_read_csv(KNOCKOUT_DIR / f"critical_layers_{mk}.csv")
    if ko_df is not None:
        seta_critical = sorted(
            ko_df.filter((pl.col("act_type") == "a") & (pl.col("knockout_type") == "zero") &
                          (pl.col("stimulus_set") == "seta"))["layer"].to_list()
        )
        row["ko_critical_layers_seta"] = str(seta_critical)
    else:
        row["ko_critical_layers_seta"] = None

    # --- Exp 17: Geometry ---
    sil_df = safe_read_csv(GEOMETRY_DIR / f"silhouette_scores_{mk}.csv")
    if sil_df is not None:
        emo_row = sil_df.filter(pl.col("grouping") == "emotion")
        set_row = sil_df.filter(pl.col("grouping") == "set")
        row["sil_emotion"] = float(emo_row["silhouette"][0]) if len(emo_row) > 0 else None
        row["sil_set"]     = float(set_row["silhouette"][0]) if len(set_row) > 0 else None
    else:
        row["sil_emotion"] = row["sil_set"] = None

    summary_rows.append(row)

summary_df = pl.DataFrame(summary_rows)
summary_df.write_csv(OUT_DIR / "summary_table.csv")
console.print(f"[bold green]✓ summary_table.csv saved ({len(summary_rows)} models)[/bold green]")

# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

table = Table(title="v2 Cross-Model Summary — Primary Metrics")
table.add_column("Model", style="cyan")
table.add_column("h peak depth", justify="right")
table.add_column("h AUROC (A)", justify="right")
table.add_column("h AUROC (B)", justify="right")
table.add_column("A→B transfer", justify="right")
table.add_column("Sil: emo", justify="right")
table.add_column("Sil: set", justify="right")

for row in summary_rows:
    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if v is not None else "N/A"
    table.add_row(
        row["label"],
        fmt(row.get("seta_h_peak_depth"), 3),
        fmt(row.get("seta_h_peak_auroc")),
        fmt(row.get("setb_h_peak_auroc")),
        fmt(row.get("ab_transfer_h")),
        fmt(row.get("sil_emotion")),
        fmt(row.get("sil_set")),
    )
console.print(table)

# ============================================================================
# VISUALIZATION 1 — Probe curves comparison (base vs instruct per family)
# ============================================================================

console.print("\n[bold cyan]Generating comparison figures...[/bold cyan]")

families = [("Llama-1B", "llama1b_base", "llama1b_inst"),
            ("Llama-8B", "llama8b_base", "llama8b_inst"),
            ("Gemma-9B", "gemma9b_base", "gemma9b_inst")]

family_n_layers = {"Llama-1B": 16, "Llama-8B": 32, "Gemma-9B": 42}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("h Probe AUROC: Base vs Instruct (Set A)", fontsize=12)

for col_idx, (family_name, base_key, inst_key) in enumerate(families):
    ax = axes[col_idx]
    nl = family_n_layers[family_name]
    norm_depths = np.array([(l+1)/nl for l in range(nl)])

    for mk, label, color, ls in [(base_key, "Base", "#2980b9", "--"), (inst_key, "Instruct", "#e74c3c", "-")]:
        csv_path = PROBES_DIR / f"probe_results_{mk}_seta_h.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            auroc = df["auroc_mean"].to_numpy()
            ax.plot(norm_depths[:len(auroc)], auroc, color=color, linestyle=ls,
                    linewidth=2.5, marker="o", markersize=4, label=label)
            ax.fill_between(norm_depths[:len(auroc)],
                             auroc - df["auroc_std"].to_numpy(),
                             auroc + df["auroc_std"].to_numpy(),
                             alpha=0.15, color=color)

    ax.axhline(1/8, linestyle=":", color="gray", alpha=0.5, label="Chance")
    ax.set_title(family_name); ax.set_xlabel("Normalized depth"); ax.set_ylabel("Macro AUROC")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "figure_probe_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "figure_probe_comparison.svg", bbox_inches="tight")
plt.close()
console.print("[bold green]✓ figure_probe_comparison.png/.svg[/bold green]")

# ============================================================================
# VISUALIZATION 2 — Set A vs Set B probe curves (primary model)
# ============================================================================

PRIMARY = "llama1b_inst"; NL = 16
norm_depths_primary = np.array([(l+1)/NL for l in range(NL)])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Set A vs Set B Probe Curves — {PRIMARY}", fontsize=12)

for col_idx, act_type in enumerate(["h", "a", "m"]):
    ax = axes[col_idx]
    for set_key, color, ls, label in [
        (f"seta_{act_type}", "#e74c3c", "-", "Set A (keyword-rich)"),
        (f"setb_clinical_{act_type}", "#2980b9", "--", "Set B (clinical)"),
    ]:
        csv_path = PROBES_DIR / f"probe_results_{PRIMARY}_{set_key}.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            auroc = df["auroc_mean"].to_numpy()
            ax.plot(norm_depths_primary[:len(auroc)], auroc, color=color, linestyle=ls,
                    linewidth=2.5, marker="o", markersize=4, label=label)
    ax.axhline(1/8, linestyle=":", color="gray", alpha=0.5)
    ax.set_title(f"{act_type.upper()} component")
    ax.set_xlabel("Normalized depth"); ax.set_ylabel("Macro AUROC")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "figure_setA_vs_setB.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "figure_setA_vs_setB.svg", bbox_inches="tight")
plt.close()
console.print("[bold green]✓ figure_setA_vs_setB.png/.svg[/bold green]")

# ============================================================================
# VISUALIZATION 3 — Summary bar chart (peak AUROC per model per set)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(summary_rows))
width = 0.35

seta_aurocs  = [r.get("seta_h_peak_auroc")  or 0 for r in summary_rows]
setb_aurocs  = [r.get("setb_h_peak_auroc")  or 0 for r in summary_rows]
ab_transfers = [r.get("ab_transfer_h")       or 0 for r in summary_rows]

bars_a = ax.bar(x - width, seta_aurocs, width, label="Set A peak AUROC (h)", color="#e74c3c", alpha=0.8)
bars_b = ax.bar(x,          setb_aurocs, width, label="Set B peak AUROC (h)", color="#2980b9", alpha=0.8)
bars_t = ax.bar(x + width,  ab_transfers, width, label="A→B transfer AUROC (h)", color="#27ae60", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in summary_rows], rotation=20, ha="right")
ax.set_ylabel("Macro AUROC"); ax.set_ylim(0, 1)
ax.axhline(1/8, linestyle=":", color="gray", alpha=0.5, label="Chance (0.125)")
ax.set_title("Cross-Model Summary — Probe AUROC (h component)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "figure_cross_model_auroc.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "figure_cross_model_auroc.svg", bbox_inches="tight")
plt.close()
console.print("[bold green]✓ figure_cross_model_auroc.png/.svg[/bold green]")

# ============================================================================
# README
# ============================================================================

filled_rows = [r for r in summary_rows if r.get("seta_h_peak_auroc") is not None]

readme_lines = [
    "# Experiment 18 — v2 Cross-Model Summary",
    "",
    "**Date:** 2026-02-25",
    "**Models:** 6 (3 base/instruct pairs: Llama-1B, Llama-8B, Gemma-9B)",
    "",
    "## Summary Table",
    "",
    "| Model | h peak depth (A) | h AUROC (A) | h AUROC (B) | A→B transfer | Sil: emo | Sil: set |",
    "|-------|-----------------|------------|------------|-------------|---------|---------|",
]

for row in summary_rows:
    def fmt(v, d=4): return f"{v:.{d}f}" if v is not None else "N/A"
    readme_lines.append(
        f"| {row['label']} | {fmt(row.get('seta_h_peak_depth'), 3)} | {fmt(row.get('seta_h_peak_auroc'))} | "
        f"{fmt(row.get('setb_h_peak_auroc'))} | {fmt(row.get('ab_transfer_h'))} | "
        f"{fmt(row.get('sil_emotion'))} | {fmt(row.get('sil_set'))} |"
    )

readme_lines += [
    "",
    "## Research Questions Answered",
    "",
    "### 1. Do the same circuits process Set A and Set B?",
    "",
    "*Based on patching (Exp 14) and geometry (Exp 17) results.*",
    "",
    "### 2. Which components carry the emotion signal?",
    "",
    "*Based on h vs a vs m probe peaks (Exp 13).*",
    "",
    "### 3. What does instruction tuning do?",
    "",
    "*Based on base vs instruct probe AUROC comparison.*",
    "",
    "## Figures",
    "",
    "- `outputs/figure_probe_comparison.png/.svg` — Base vs instruct probe curves",
    "- `outputs/figure_setA_vs_setB.png/.svg` — Set A vs Set B within-primary-model",
    "- `outputs/figure_cross_model_auroc.png/.svg` — Summary bar chart",
    "- `outputs/summary_table.csv` — Machine-readable summary",
    "",
    "## Outputs",
    "",
    "All experiment outputs:",
    "- Exp 11/12: Per-stimulus .npz activations (h, a, m, attn)",
    "- Exp 13: Probe curves + transfer results per model",
    "- Exp 14: Activation patching success heatmaps",
    "- Exp 15: Attention token-type analysis",
    "- Exp 16: Knockout critical layer identification",
    "- Exp 17: Representational geometry (PCA, cosine, cross-topic)",
]

(EXP_DIR / "README.md").write_text("\n".join(readme_lines))
console.print(f"[bold green]✓ README.md written[/bold green]")
console.print(f"\n[bold green]✓ Experiment 18 (Summary) complete![/bold green]")
