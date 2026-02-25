"""
Token Count Verification
Question: Do clinical/neutral pairs have similar token lengths (within ±10%)?
Date: 2026-02-24

Flags pairs where |clinical_tokens - neutral_tokens| / max(clinical_tokens, neutral_tokens) > 0.10

Run: uv run python validation/token_counts.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.loader import load_all, to_polars

console = Console()

VALIDATION_DIR = Path(__file__).parent
OUTPUT_DIR = VALIDATION_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.2-1B"
TOKEN_RATIO_THRESHOLD = 0.10  # flag pairs differing by more than 10%

# ============================================================================
# LOAD HF TOKEN
# ============================================================================

env_path = Path(__file__).parent.parent / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break

# ============================================================================
# LOAD STIMULI
# ============================================================================

console.print("[bold cyan]Loading stimuli...[/bold cyan]")
stimuli = load_all()
console.print(f"Loaded {len(stimuli)} stimuli")

# ============================================================================
# LOAD TOKENIZER & COMPUTE TOKEN COUNTS
# ============================================================================

console.print(f"\n[bold cyan]Loading tokenizer: {MODEL_ID}...[/bold cyan]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
console.print("Tokenizer loaded")

console.print("\n[bold cyan]Computing token counts for all stimuli...[/bold cyan]")

# Compute token counts for each stimulus
token_counts = []
for s in stimuli:
    tokens = tokenizer.encode(s.text, add_special_tokens=False)
    s.token_count = len(tokens)
    token_counts.append(len(tokens))

df = to_polars(stimuli)

console.print(f"Token count range: {min(token_counts)} – {max(token_counts)}")
console.print(f"Mean token count:  {np.mean(token_counts):.1f}")
console.print(f"Median token count: {np.median(token_counts):.1f}")

# ============================================================================
# CLINICAL / NEUTRAL PAIR ANALYSIS
# ============================================================================

console.print("\n[bold cyan]Checking clinical/neutral pair token ratios...[/bold cyan]")

clinical_df = df.filter(pl.col("stimulus_set") == "B_clinical")
neutral_df = df.filter(pl.col("stimulus_set") == "neutral")

# Build a fast lookup: neutral_id → token_count
neutral_lookup = {
    row["id"]: row["token_count"]
    for row in neutral_df.iter_rows(named=True)
}

all_pairs = []
flagged_pairs = []

for row in clinical_df.iter_rows(named=True):
    clinical_id = row["id"]
    neutral_id = row["matched_control_id"]
    clinical_tokens = row["token_count"]
    neutral_tokens = neutral_lookup.get(neutral_id)

    if neutral_tokens is None:
        console.print(f"[yellow]  WARNING: No neutral found for {clinical_id} → {neutral_id}[/yellow]")
        continue

    ratio = abs(clinical_tokens - neutral_tokens) / max(clinical_tokens, neutral_tokens)
    flagged = ratio > TOKEN_RATIO_THRESHOLD

    pair = {
        "clinical_id": clinical_id,
        "neutral_id": neutral_id,
        "emotion": row["emotion"],
        "domain": row["domain"],
        "vignette_idx": int(clinical_id.split("-v")[-1]),
        "clinical_tokens": clinical_tokens,
        "neutral_tokens": neutral_tokens,
        "abs_diff": abs(clinical_tokens - neutral_tokens),
        "ratio": round(ratio, 4),
        "flagged": flagged,
    }
    all_pairs.append(pair)
    if flagged:
        flagged_pairs.append(pair)

pairs_df = pl.DataFrame(all_pairs)

# ============================================================================
# PRINT RESULTS
# ============================================================================

console.print(f"\n[bold]Pair analysis summary:[/bold]")
console.print(f"  Total clinical/neutral pairs checked: {len(all_pairs)}")
console.print(f"  Flagged (ratio > {TOKEN_RATIO_THRESHOLD:.0%}): {len(flagged_pairs)}")

if flagged_pairs:
    console.print(f"\n[bold yellow]Flagged pairs (sorted by ratio desc):[/bold yellow]")
    flag_table = Table()
    flag_table.add_column("Clinical ID", style="cyan")
    flag_table.add_column("Neutral ID", style="dim cyan")
    flag_table.add_column("Clinical Tokens", justify="right")
    flag_table.add_column("Neutral Tokens", justify="right")
    flag_table.add_column("Abs Diff", justify="right")
    flag_table.add_column("Ratio", justify="right", style="yellow")

    for p in sorted(flagged_pairs, key=lambda x: x["ratio"], reverse=True):
        flag_table.add_row(
            p["clinical_id"], p["neutral_id"],
            str(p["clinical_tokens"]), str(p["neutral_tokens"]),
            str(p["abs_diff"]),
            f"{p['ratio']:.1%}",
        )
    console.print(flag_table)
else:
    console.print("\n[green]  No flagged pairs — all within ±10% threshold.[/green]")

# Per-emotion summary
console.print("\n[bold]Mean ratio by emotion:[/bold]")
emotion_summary = (
    pairs_df.group_by("emotion")
    .agg(
        pl.col("ratio").mean().alias("mean_ratio"),
        pl.col("flagged").sum().alias("n_flagged"),
    )
    .sort("mean_ratio", descending=True)
)
for row in emotion_summary.iter_rows(named=True):
    flag_note = f"  ← [yellow]{row['n_flagged']} flagged[/yellow]" if row["n_flagged"] > 0 else ""
    console.print(f"  {row['emotion']:<12} mean ratio: {row['mean_ratio']:.1%}{flag_note}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

df.write_csv(OUTPUT_DIR / "token_counts.csv")
console.print(f"\n[bold green]✓ Saved: {OUTPUT_DIR / 'token_counts.csv'}[/bold green]")

pairs_df.write_csv(OUTPUT_DIR / "all_pairs.csv")
console.print(f"[bold green]✓ Saved: {OUTPUT_DIR / 'all_pairs.csv'}[/bold green]")

if flagged_pairs:
    flagged_df = pl.DataFrame(flagged_pairs)
    flagged_df.write_csv(OUTPUT_DIR / "flagged_pairs.csv")
    console.print(f"[bold green]✓ Saved: {OUTPUT_DIR / 'flagged_pairs.csv'}[/bold green]")

console.print("\n[bold green]✓ Token count verification complete![/bold green]")
