"""
Phase 1 Cross-Model Comparison
Date: 2026-02-24

For each model: loads Phase 0 (Set A) and Phase 1 (Set B) 8-class AUROC results
and produces:
  Figure 1 — 2×2 panel: Phase 0 vs Phase 1 curves per model (normalized depth)
  Figure 2 — Decodability gap: (Phase 0 − Phase 1) by normalized depth, all models
  Summary table — Phase 0 AUROC, Phase 1 AUROC, gap, best normalized depth

Requires: all extract.py + run.py in 05-08 to have been run first.
Run: uv run python experiments/09_phase1_comparison/compare.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()
np.random.seed(42)

EXP_ROOT   = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODELS = [
    {
        "key":         "llama1b",
        "label":       "Llama-3.2-1B",
        "color":       "#95a5a6",
        "phase0_dir":  EXP_ROOT / "00_phase0_replication",
        "phase1_dir":  EXP_ROOT / "05_phase1_llama1b",
        "n_layers":    16,
    },
    {
        "key":         "llama8b",
        "label":       "Llama-3-8B (base)",
        "color":       "#3498db",
        "phase0_dir":  EXP_ROOT / "01_phase0a_llama3-8b",
        "phase1_dir":  EXP_ROOT / "06_phase1_llama3-8b",
        "n_layers":    32,
    },
    {
        "key":         "llama8b_instruct",
        "label":       "Llama-3-8B (instruct)",
        "color":       "#e74c3c",
        "phase0_dir":  EXP_ROOT / "02_phase0b_llama3-8b-instruct",
        "phase1_dir":  EXP_ROOT / "07_phase1_llama3-8b-instruct",
        "n_layers":    32,
    },
    {
        "key":         "gemma9b",
        "label":       "Gemma-2-9B",
        "color":       "#9b59b6",
        "phase0_dir":  EXP_ROOT / "03_phase0c_gemma2-9b",
        "phase1_dir":  EXP_ROOT / "08_phase1_gemma2-9b",
        "n_layers":    42,
    },
]

# ============================================================================
# LOAD RESULTS
# ============================================================================

console.print("[bold cyan]Loading Phase 0 + Phase 1 results...[/bold cyan]")

missing = []
for m in MODELS:
    p0_path = m["phase0_dir"] / "outputs" / "results.csv"
    p1_path = m["phase1_dir"] / "outputs" / "results_8class.csv"

    if not p0_path.exists():
        console.print(f"  [red]MISSING Phase 0: {p0_path}[/red]")
        missing.append(m["key"])
        continue
    if not p1_path.exists():
        console.print(f"  [red]MISSING Phase 1: {p1_path}[/red]")
        missing.append(m["key"])
        continue

    df0 = pl.read_csv(p0_path)
    df1 = pl.read_csv(p1_path)

    m["p0_means"] = df0["auroc_mean"].to_list()
    m["p0_stds"]  = df0["auroc_std"].to_list()
    m["p1_means"] = df1["auroc_mean"].to_list()
    m["p1_stds"]  = df1["auroc_std"].to_list()

    m["p0_best"]  = max(m["p0_means"])
    m["p1_best"]  = max(m["p1_means"])
    m["gap"]      = m["p0_best"] - m["p1_best"]

    m["p0_best_layer_norm"] = np.argmax(m["p0_means"]) / (len(m["p0_means"]) - 1)
    m["p1_best_layer_norm"] = np.argmax(m["p1_means"]) / (len(m["p1_means"]) - 1)

    console.print(f"  [green]✓[/green] {m['label']}: P0={m['p0_best']:.4f} | P1={m['p1_best']:.4f} | gap={m['gap']:.4f}")

available = [m for m in MODELS if "p0_means" in m]

if len(available) == 0:
    console.print("[red]No results found. Run all Phase 1 experiments first.[/red]")
    sys.exit(1)
if missing:
    console.print(f"\n[yellow]Missing: {missing}. Continuing with {len(available)} model(s).[/yellow]")

# ============================================================================
# FIGURE 1 — 2×2 PANEL: Phase 0 vs Phase 1 per model (normalized depth)
# ============================================================================

console.print("\n[bold cyan]Figure 1: 2×2 panel (Phase 0 vs Phase 1 per model)...[/bold cyan]")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)
chance = 0.125

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

for ax, m in zip(axes, available):
    n0 = len(m["p0_means"])
    n1 = len(m["p1_means"])
    rel0 = [li / (n0 - 1) for li in range(n0)]
    rel1 = [li / (n1 - 1) for li in range(n1)]

    # Phase 0 — solid
    ax.plot(rel0, m["p0_means"], "-", color=m["color"], linewidth=2.5,
            label=f"Phase 0 — Set A (peak: {m['p0_best']:.3f})")
    ax.fill_between(rel0,
        [a - s for a, s in zip(m["p0_means"], m["p0_stds"])],
        [a + s for a, s in zip(m["p0_means"], m["p0_stds"])],
        alpha=0.12, color=m["color"])

    # Phase 1 — dashed
    ax.plot(rel1, m["p1_means"], "--", color=m["color"], linewidth=2.5,
            label=f"Phase 1 — Set B (peak: {m['p1_best']:.3f})")
    ax.fill_between(rel1,
        [a - s for a, s in zip(m["p1_means"], m["p1_stds"])],
        [a + s for a, s in zip(m["p1_means"], m["p1_stds"])],
        alpha=0.12, color=m["color"])

    ax.axhline(chance, color="gray", linestyle=":", linewidth=1, label=f"Chance ({chance:.3f})")
    ax.axhline(0.40, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="Gate (0.40)")

    ax.set_xlabel("Normalized depth (0→1)", fontsize=9)
    ax.set_ylabel("Macro AUROC (8-class OvR)", fontsize=9)
    ax.set_title(m["label"], fontsize=10, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.2)

    # Annotate the gap
    gap_txt = f"Gap: {m['gap']:.3f} ({m['gap']/m['p0_best']*100:.1f}%)"
    ax.text(0.02, 0.05, gap_txt, transform=ax.transAxes, fontsize=8,
            color=m["color"], fontweight="bold")

# Hide unused axes if fewer than 4 models
for ax in axes[len(available):]:
    ax.set_visible(False)

fig.suptitle(
    "Phase 0 vs Phase 1: Emotion Probe AUROC — Set A (keyword-rich) vs Set B (clinical vignettes)\n"
    "8-class OvR, 5-fold CV, 80 emotional / 96 clinical stimuli",
    fontsize=11, y=1.01
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "phase0_vs_phase1_panel.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "phase0_vs_phase1_panel.svg", bbox_inches="tight")
plt.close(fig)
console.print("[bold green]✓ phase0_vs_phase1_panel.png + .svg[/bold green]")

# ============================================================================
# FIGURE 2 — DECODABILITY GAP: (Phase 0 − Phase 1) by normalized depth
# ============================================================================

console.print("[bold cyan]Figure 2: Decodability gap by normalized depth...[/bold cyan]")

fig, ax = plt.subplots(figsize=(10, 5))

for m in available:
    n0 = len(m["p0_means"])
    n1 = len(m["p1_means"])

    # Interpolate both to 100 points for alignment
    rel0 = np.linspace(0, 1, n0)
    rel1 = np.linspace(0, 1, n1)
    x    = np.linspace(0, 1, 100)
    p0_interp = np.interp(x, rel0, m["p0_means"])
    p1_interp = np.interp(x, rel1, m["p1_means"])
    gap = p0_interp - p1_interp

    ax.plot(x, gap, color=m["color"], linewidth=2.5, label=m["label"])
    ax.fill_between(x, 0, gap, alpha=0.08, color=m["color"])

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_xlabel("Normalized layer depth (0=input, 1=output)", fontsize=10)
ax.set_ylabel("AUROC gap (Phase 0 − Phase 1)", fontsize=10)
ax.set_title(
    "Decodability Gap: How much does emotion encoding drop from keyword-rich to clinical vignettes?\n"
    "Positive = Phase 0 better; 0 = equal; Negative = Phase 1 better (unexpected)",
    fontsize=10
)
ax.set_xlim(-0.02, 1.02)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "decodability_gap.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "decodability_gap.svg", bbox_inches="tight")
plt.close(fig)
console.print("[bold green]✓ decodability_gap.png + .svg[/bold green]")

# ============================================================================
# SUMMARY TABLE + CSV
# ============================================================================

console.print("\n[bold cyan]Summary table...[/bold cyan]")

table = Table(title="Phase 0 vs Phase 1 — Cross-Model Summary")
table.add_column("Model", style="bold")
table.add_column("P0 AUROC", justify="right")
table.add_column("P0 best layer (norm)", justify="right")
table.add_column("P1 AUROC", justify="right")
table.add_column("P1 best layer (norm)", justify="right")
table.add_column("Gap (abs)", justify="right")
table.add_column("Gap (%)", justify="right")

summary_rows = []
for m in available:
    gap_pct = m["gap"] / m["p0_best"] * 100
    table.add_row(
        m["label"],
        f"{m['p0_best']:.4f}",
        f"{m['p0_best_layer_norm']:.1%}",
        f"{m['p1_best']:.4f}",
        f"{m['p1_best_layer_norm']:.1%}",
        f"{m['gap']:.4f}",
        f"{gap_pct:.1f}%",
    )
    summary_rows.append({
        "model":              m["label"],
        "p0_best_auroc":      round(m["p0_best"], 6),
        "p0_best_layer_norm": round(m["p0_best_layer_norm"], 4),
        "p1_best_auroc":      round(m["p1_best"], 6),
        "p1_best_layer_norm": round(m["p1_best_layer_norm"], 4),
        "gap_abs":            round(m["gap"], 6),
        "gap_pct":            round(gap_pct, 2),
    })

console.print(table)
pl.DataFrame(summary_rows).write_csv(OUTPUT_DIR / "summary.csv")
console.print("[bold green]✓ summary.csv[/bold green]")

# ============================================================================
# WRITE README
# ============================================================================

summary_md = "| Model | P0 AUROC | P0 peak (norm) | P1 AUROC | P1 peak (norm) | Gap | Gap % |\n"
summary_md += "|-------|---------|---------------|---------|---------------|-----|-------|\n"
for row in summary_rows:
    summary_md += (
        f"| {row['model']} | {row['p0_best_auroc']:.4f} | {row['p0_best_layer_norm']:.1%} | "
        f"{row['p1_best_auroc']:.4f} | {row['p1_best_layer_norm']:.1%} | "
        f"{row['gap_abs']:.4f} | {row['gap_pct']:.1f}% |\n"
    )

readme = f"""# Phase 1 Cross-Model Comparison

**Date:** 2026-02-24
**Question:** Does emotion category information survive in the residual stream
when keyword-rich stimuli (Set A) are replaced by clinical vignettes (Set B)?

## Summary

{summary_md}

## Outputs

- `outputs/phase0_vs_phase1_panel.png/.svg` — 2×2 per-model overlay
- `outputs/decodability_gap.png/.svg` — gap by normalized depth, all models
- `outputs/summary.csv` — machine-readable summary table
""".strip()

(Path(__file__).parent / "README.md").write_text(readme)
console.print("[bold green]✓ README.md[/bold green]")

console.print(f"\n[bold green]✓ Phase 1 comparison complete — results in {OUTPUT_DIR}[/bold green]")
