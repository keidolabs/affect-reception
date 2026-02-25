"""
Phase 0 Cross-Model Comparison
Date: 2026-02-24

Loads AUROC results from all four Phase 0 experiments and produces a
cross-model comparison: overlaid AUROC curves, base vs instruct delta,
per-class breakdown, and a summary README.

Requires: all four extract.py + run.py scripts to have been run first.

Run: uv run python experiments/04_phase0_comparison/compare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

console = Console()
np.random.seed(42)

EXP_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL REGISTRY — all four Phase 0 experiments
# ============================================================================

MODELS = [
    {
        "key":      "llama1b",
        "label":    "Llama-3.2-1B",
        "model_id": "meta-llama/Llama-3.2-1B",
        "exp_dir":  EXP_ROOT / "00_phase0_replication",
        "n_layers": 16,
        "d_model":  2048,
        "color":    "#95a5a6",   # gray — reference baseline
        "dash":     "dot",
    },
    {
        "key":      "llama8b",
        "label":    "Llama-3-8B (base)",
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "exp_dir":  EXP_ROOT / "01_phase0a_llama3-8b",
        "n_layers": 32,
        "d_model":  4096,
        "color":    "#3498db",   # blue
        "dash":     "solid",
    },
    {
        "key":      "llama8b_instruct",
        "label":    "Llama-3-8B (instruct)",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "exp_dir":  EXP_ROOT / "02_phase0b_llama3-8b-instruct",
        "n_layers": 32,
        "d_model":  4096,
        "color":    "#e74c3c",   # red
        "dash":     "dash",
    },
    {
        "key":      "gemma9b",
        "label":    "Gemma-2-9B",
        "model_id": "google/gemma-2-9b",
        "exp_dir":  EXP_ROOT / "03_phase0c_gemma2-9b",
        "n_layers": 42,
        "d_model":  3584,
        "color":    "#9b59b6",   # purple
        "dash":     "solid",
    },
]

EMOTION_ORDER = ["rage", "grief", "terror", "ecstasy", "loathing", "amazement", "admiration", "vigilance"]
EMOTION_COLORS = {
    "rage": "#e74c3c", "grief": "#3498db", "terror": "#2c3e50",
    "ecstasy": "#f1c40f", "loathing": "#27ae60", "amazement": "#9b59b6",
    "admiration": "#e67e22", "vigilance": "#1abc9c",
}

# ============================================================================
# LOAD RESULTS FROM EACH EXPERIMENT
# ============================================================================

console.print("[bold cyan]Loading results from all experiments...[/bold cyan]")

missing = []
for m in MODELS:
    results_path = m["exp_dir"] / "outputs" / "results.csv"
    act_path     = m["exp_dir"] / "outputs" / "activations" / "set_a_residuals.npy"
    meta_path    = m["exp_dir"] / "outputs" / "activations" / "metadata.json"

    if not results_path.exists():
        console.print(f"  [yellow]MISSING: {results_path}[/yellow]")
        missing.append(m["key"])
        continue

    df = pl.read_csv(results_path)
    m["auroc_means"] = df["auroc_mean"].to_list()
    m["auroc_stds"]  = df["auroc_std"].to_list()
    m["best_layer"]  = int(np.argmax(m["auroc_means"]))
    m["best_auroc"]  = m["auroc_means"][m["best_layer"]]
    m["best_layer_pct"] = m["best_layer"] / (m["n_layers"] - 1)

    # Load activations for per-class AUROC recomputation
    if act_path.exists() and meta_path.exists():
        m["activations"] = np.load(act_path)
        with open(meta_path) as f:
            meta = json.load(f)
        m["labels"] = np.array([x["emotion"] for x in meta])
        m["has_activations"] = True
    else:
        m["has_activations"] = False

    console.print(f"  [green]✓[/green] {m['label']}: best L{m['best_layer']} AUROC={m['best_auroc']:.4f}")

available = [m for m in MODELS if "auroc_means" in m]

if len(available) == 0:
    console.print("[red]No results found. Run all extract.py + run.py first.[/red]")
    sys.exit(1)
if missing:
    console.print(f"\n[yellow]Missing results for: {missing}. Continuing with {len(available)} model(s).[/yellow]")

# ============================================================================
# FIGURE 1 — OVERLAID AUROC: ABSOLUTE + NORMALIZED (matplotlib, publication quality)
# ============================================================================

console.print("\n[bold cyan]Figure 1: Overlaid AUROC comparison (matplotlib)...[/bold cyan]")

fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 2, wspace=0.35)
ax_abs = fig.add_subplot(gs[0])
ax_rel = fig.add_subplot(gs[1])

chance = 0.125

for m in available:
    layers_abs = list(range(m["n_layers"]))
    layers_rel = [li / (m["n_layers"] - 1) for li in layers_abs]
    lw = 2.5 if m["key"] != "llama1b" else 1.5

    for ax, layers in [(ax_abs, layers_abs), (ax_rel, layers_rel)]:
        ax.plot(layers, m["auroc_means"], color=m["color"], linewidth=lw,
                linestyle="-" if m["dash"] == "solid" else "--" if m["dash"] == "dash" else ":",
                label=m["label"])
        ax.fill_between(
            layers,
            [a - s for a, s in zip(m["auroc_means"], m["auroc_stds"])],
            [a + s for a, s in zip(m["auroc_means"], m["auroc_stds"])],
            alpha=0.1, color=m["color"]
        )

for ax in [ax_abs, ax_rel]:
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=f"Chance ({chance:.3f})")
    ax.axhline(0.40,   color="orange", linestyle=":", linewidth=1, label="Gate (0.40)")
    ax.set_ylabel("Macro AUROC (8-class OvR, 5-fold CV)", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)

ax_abs.set_xlabel("Layer (absolute)", fontsize=10)
ax_abs.set_title("Absolute Layer Index", fontsize=11)

ax_rel.set_xlabel("Relative Depth (layer / n_layers)", fontsize=10)
ax_rel.set_title("Normalized Depth (0 = input, 1 = output)", fontsize=11)
ax_rel.set_xlim(-0.02, 1.02)

# One shared legend on the right panel
ax_rel.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle(
    "Phase 0: Emotion Probe AUROC Across Models — Set A (keyword-rich, n=80 emotional)",
    fontsize=12, y=1.01
)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "auroc_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "auroc_comparison.svg", bbox_inches="tight")
plt.close(fig)
console.print("[bold green]✓ auroc_comparison.png + .svg[/bold green]")

# ============================================================================
# FIGURE 2 — PLOTLY INTERACTIVE: ABSOLUTE + NORMALIZED (side-by-side subplots)
# ============================================================================

console.print("[bold cyan]Figure 2: Interactive AUROC comparison (Plotly)...[/bold cyan]")

fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Absolute Layer Index", "Normalized Depth (0→1)"),
    horizontal_spacing=0.1,
)

plotly_dash = {"solid": "solid", "dash": "dash", "dot": "dot"}

for m in available:
    layers_abs = list(range(m["n_layers"]))
    layers_rel = [li / (m["n_layers"] - 1) for li in layers_abs]

    for col, (layers, xlabel) in enumerate([(layers_abs, "Layer"), (layers_rel, "Relative depth")], start=1):
        # CI ribbon
        fig2.add_trace(go.Scatter(
            x=layers + layers[::-1],
            y=[a + s for a, s in zip(m["auroc_means"], m["auroc_stds"])] +
              [a - s for a, s in zip(m["auroc_means"], m["auroc_stds"])][::-1],
            fill="toself",
            fillcolor="rgba({},{},{},0.13)".format(
                int(m["color"][1:3], 16),
                int(m["color"][3:5], 16),
                int(m["color"][5:7], 16),
            ),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=col)

        # AUROC line
        fig2.add_trace(go.Scatter(
            x=layers,
            y=m["auroc_means"],
            mode="lines+markers",
            name=m["label"],
            showlegend=(col == 1),
            line=dict(color=m["color"], width=2.5 if m["key"] != "llama1b" else 1.5,
                      dash=plotly_dash[m["dash"]]),
            marker=dict(size=5, color=m["color"]),
            customdata=[(li, a, s, m["label"])
                        for li, a, s in zip(layers_abs, m["auroc_means"], m["auroc_stds"])],
            hovertemplate=(
                f"<b>{m['label']}</b><br>"
                "Layer %{customdata[0]} (rel: %{x:.3f})<br>"
                "AUROC: %{customdata[1]:.4f} ± %{customdata[2]:.4f}<extra></extra>"
            ) if col == 2 else (
                f"<b>{m['label']}</b><br>"
                "Layer %{x}<br>"
                "AUROC: %{customdata[1]:.4f} ± %{customdata[2]:.4f}<extra></extra>"
            ),
        ), row=1, col=col)

# Reference lines
for col in [1, 2]:
    fig2.add_hline(y=chance, line_dash="dot", line_color="gray",
                   annotation_text="chance" if col == 2 else None,
                   row=1, col=col)
    fig2.add_hline(y=0.40, line_dash="dash", line_color="orange",
                   annotation_text="gate" if col == 2 else None,
                   row=1, col=col)

fig2.update_layout(
    title=dict(
        text="Phase 0: Cross-Model AUROC Comparison<br>"
             "<sup>Set A (keyword-rich) · 8-class OvR · 5-fold CV · seed=42 · hover for details</sup>",
        x=0.5,
    ),
    plot_bgcolor="white",
    width=1100, height=480,
    legend=dict(x=0.38, y=0.05, bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc", borderwidth=1),
)
fig2.update_yaxes(range=[0, 1.05], gridcolor="#f0f0f0")
fig2.update_xaxes(gridcolor="#f0f0f0")

fig2.write_html(OUTPUT_DIR / "auroc_comparison.html")
console.print("[bold green]✓ auroc_comparison.html[/bold green]")

# ============================================================================
# FIGURE 3 — BASE VS INSTRUCT DELTA (only if both 8B models available)
# ============================================================================

base_m     = next((m for m in available if m["key"] == "llama8b"), None)
instruct_m = next((m for m in available if m["key"] == "llama8b_instruct"), None)

if base_m and instruct_m:
    console.print("[bold cyan]Figure 3: Base vs Instruct delta...[/bold cyan]")

    delta = [i - b for i, b in zip(instruct_m["auroc_means"], base_m["auroc_means"])]
    layers_rel = [li / (base_m["n_layers"] - 1) for li in range(base_m["n_layers"])]

    fig3, ax = plt.subplots(figsize=(9, 4))
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in delta]
    ax.bar(layers_rel, delta, width=0.025, color=colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("AUROC(instruct) − AUROC(base)")
    ax.set_title("Base → Instruct AUROC Delta (Llama-3-8B)\n"
                 "Red = instruct better | Blue = base better")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "base_vs_instruct_delta.png", dpi=150, bbox_inches="tight")
    fig3.savefig(OUTPUT_DIR / "base_vs_instruct_delta.svg", bbox_inches="tight")
    plt.close(fig3)
    console.print("[bold green]✓ base_vs_instruct_delta.png + .svg[/bold green]")

    # Plotly version
    fig3p = go.Figure()
    fig3p.add_trace(go.Bar(
        x=layers_rel, y=delta,
        marker_color=["#e74c3c" if d > 0 else "#3498db" for d in delta],
        marker_opacity=0.75,
        customdata=list(range(base_m["n_layers"])),
        hovertemplate="Layer %{customdata} (rel: %{x:.3f})<br>Delta: %{y:+.4f}<extra></extra>",
    ))
    fig3p.add_hline(y=0, line_color="black", line_width=0.8)
    fig3p.update_layout(
        title=dict(text="Base → Instruct AUROC Delta (Llama-3-8B)<br>"
                        "<sup>Positive = instruct encodes emotion better at that layer depth</sup>",
                   x=0.5),
        xaxis_title="Normalized Layer Depth",
        yaxis_title="AUROC(instruct) − AUROC(base)",
        plot_bgcolor="white",
        width=850, height=380,
    )
    fig3p.update_xaxes(gridcolor="#f0f0f0")
    fig3p.update_yaxes(gridcolor="#f0f0f0", zeroline=False)
    fig3p.write_html(OUTPUT_DIR / "base_vs_instruct_delta.html")
    console.print("[bold green]✓ base_vs_instruct_delta.html[/bold green]")
else:
    console.print("[yellow]  Skipping base vs instruct delta (one or both models not yet run)[/yellow]")

# ============================================================================
# FIGURE 4 — PER-CLASS AUROC AT BEST LAYER (grouped bar, all models)
# ============================================================================

models_with_acts = [m for m in available if m.get("has_activations")]

if models_with_acts:
    console.print("\n[bold cyan]Figure 4: Per-class AUROC at best layer...[/bold cyan]")

    # Compute OvR AUROC per emotion class at each model's best layer
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    per_class = {}  # model_key → {emotion: auroc}
    for m in models_with_acts:
        console.print(f"  Computing per-class AUROC for {m['label']}...")
        emotional_mask = m["labels"] != "neutral"
        X_emo = m["activations"][emotional_mask][:, m["best_layer"], :]
        y_emo = m["labels"][emotional_mask]

        per_class[m["key"]] = {}
        for emotion in EMOTION_ORDER:
            y_bin = (y_emo == emotion).astype(int)
            fold_aurocs = []
            for train_idx, test_idx in cv.split(X_emo, y_bin):
                clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=42)
                clf.fit(X_emo[train_idx], y_bin[train_idx])
                probs = clf.predict_proba(X_emo[test_idx])[:, 1]
                if len(np.unique(y_bin[test_idx])) < 2:
                    fold_aurocs.append(0.5)
                else:
                    fold_aurocs.append(roc_auc_score(y_bin[test_idx], probs))
            per_class[m["key"]][emotion] = np.mean(fold_aurocs)

    # Grouped bar chart (Plotly)
    n_models = len(models_with_acts)
    fig4 = go.Figure()

    for m in models_with_acts:
        if m["key"] not in per_class:
            continue
        scores = [per_class[m["key"]].get(emo, 0) for emo in EMOTION_ORDER]
        fig4.add_trace(go.Bar(
            name=m["label"],
            x=EMOTION_ORDER,
            y=scores,
            marker_color=m["color"],
            marker_opacity=0.82,
            hovertemplate=f"<b>{m['label']}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    fig4.add_hline(y=0.5, line_dash="dot", line_color="gray",
                   annotation_text="chance (0.5)", annotation_position="right")
    fig4.update_layout(
        barmode="group",
        title=dict(
            text="Per-Class OvR AUROC at Best Layer — All Models<br>"
                 "<sup>Binary OvR probe (that emotion vs rest) · 5-fold CV · chance = 0.50</sup>",
            x=0.5,
        ),
        xaxis_title="Emotion",
        yaxis=dict(title="OvR AUROC", range=[0.4, 1.05], gridcolor="#f0f0f0"),
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.05),
        width=950, height=450,
    )
    fig4.write_html(OUTPUT_DIR / "per_class_auroc_comparison.html")
    console.print("[bold green]✓ per_class_auroc_comparison.html[/bold green]")
else:
    console.print("[yellow]  Skipping per-class comparison (no activation files found)[/yellow]")

# ============================================================================
# SUMMARY TABLE + README
# ============================================================================

console.print("\n[bold cyan]Summary table...[/bold cyan]")

table = Table(title="Phase 0 — Cross-Model Summary")
table.add_column("Model", style="cyan")
table.add_column("Layers", justify="right")
table.add_column("d_model", justify="right")
table.add_column("Best Layer", justify="right")
table.add_column("Rel. Depth", justify="right")
table.add_column("Best AUROC", justify="right")
table.add_column("Gate", justify="center")

rows_md = []
for m in available:
    gate = "[bold green]PASS[/bold green]" if m["best_auroc"] > 0.40 else "[bold red]FAIL[/bold red]"
    table.add_row(
        m["label"], str(m["n_layers"]), str(m["d_model"]),
        str(m["best_layer"]),
        f"{m['best_layer_pct']:.2f}",
        f"{m['best_auroc']:.4f}",
        gate,
    )
    rows_md.append(
        f"| {m['label']} | {m['n_layers']} | {m['d_model']} | "
        f"{m['best_layer']} | {m['best_layer_pct']:.2f} | "
        f"{m['best_auroc']:.4f} | {'✓ PASS' if m['best_auroc'] > 0.40 else '✗ FAIL'} |"
    )

console.print(table)

# README
table_md = (
    "| Model | Layers | d_model | Best Layer | Rel. Depth | Best AUROC | Gate |\n"
    "|-------|--------|---------|-----------|-----------|-----------|------|\n"
    + "\n".join(rows_md)
)

output_files = [
    "- `outputs/auroc_comparison.png/.svg` — overlaid AUROC curves (absolute + normalized)",
    "- `outputs/auroc_comparison.html` — interactive version with hover",
]
if base_m and instruct_m:
    output_files += [
        "- `outputs/base_vs_instruct_delta.png/.svg` — base → instruct AUROC delta",
        "- `outputs/base_vs_instruct_delta.html` — interactive version",
    ]
if models_with_acts:
    output_files.append("- `outputs/per_class_auroc_comparison.html` — per-class grouped bar (interactive)")

readme = f"""# Phase 0 — Cross-Model Comparison

**Date:** 2026-02-24
**Stimuli:** Set A standard (n=90: 80 emotional + 10 neutral)
**Method:** Layer-wise linear probes, 8-class OvR, 5-fold CV, macro AUROC

## Summary

{table_md}

## Key Observations

- All models trained on keyword-rich stimuli (Set A) — probes have the easiest possible task
- Best layer depth (normalized) varies by model — compare across architectures in the AUROC plot
- Base vs Instruct delta shows where RLHF fine-tuning reshapes emotion representations
- Per-class comparison shows which emotions decode most reliably across all model families

## Outputs

{chr(10).join(output_files)}
""".strip()

(Path(__file__).parent / "README.md").write_text(readme)
console.print("\n[bold green]✓ README.md[/bold green]")
console.print(f"\n[bold green]All comparison outputs saved to: {OUTPUT_DIR}[/bold green]")
