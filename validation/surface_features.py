"""
Surface Feature Matching Validation
Question: Are Set B clinical/neutral pairs matched on surface text features?
Date: 2026-02-24

Checks per pair:
  - Token count ratio (|clinical - neutral| / max)
  - Type-token ratio (TTR = unique words / total words) difference
  - Average sentence length difference

Flags pairs where: TTR diff > 15% OR avg sentence length diff > 20%

Run: uv run python validation/surface_features.py
"""

import re
import sys
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.loader import load_all, to_polars

console = Console()

VALIDATION_DIR = Path(__file__).parent
OUTPUT_DIR = VALIDATION_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TTR_DIFF_THRESHOLD = 0.15      # 15% difference in type-token ratio
SENT_LEN_DIFF_THRESHOLD = 0.20 # 20% difference in mean sentence length


# ============================================================================
# FEATURE COMPUTATION HELPERS
# ============================================================================

def compute_ttr(text: str) -> float:
    """
    Type-Token Ratio (TTR) = unique word count / total word count.
    All lowercase, basic whitespace splitting.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_sentence_length(text: str) -> float:
    """
    Average number of words per sentence.
    Splits on ., !, ? and filters empty segments.
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)


def compute_word_count(text: str) -> int:
    return len(text.split())


# ============================================================================
# LOAD STIMULI
# ============================================================================

console.print("[bold cyan]Loading stimuli...[/bold cyan]")
stimuli = load_all()
df = to_polars(stimuli)
console.print(f"Loaded {len(stimuli)} stimuli")

# ============================================================================
# COMPUTE SURFACE FEATURES FOR ALL STIMULI
# ============================================================================

console.print("\n[bold cyan]Computing surface features...[/bold cyan]")

ttrs = []
avg_sent_lens = []
word_counts = []

for s in stimuli:
    ttrs.append(compute_ttr(s.text))
    avg_sent_lens.append(compute_avg_sentence_length(s.text))
    word_counts.append(compute_word_count(s.text))

df = df.with_columns([
    pl.Series("ttr", ttrs),
    pl.Series("avg_sent_len", avg_sent_lens),
    pl.Series("word_count_computed", word_counts),
])

console.print(f"TTR range: {min(ttrs):.3f} – {max(ttrs):.3f}")
console.print(f"Avg sentence length range: {min(avg_sent_lens):.1f} – {max(avg_sent_lens):.1f} words")

# ============================================================================
# CLINICAL / NEUTRAL PAIR COMPARISON
# ============================================================================

console.print("\n[bold cyan]Comparing clinical/neutral pairs on surface features...[/bold cyan]")

clinical_df = df.filter(pl.col("stimulus_set") == "B_clinical")
neutral_df = df.filter(pl.col("stimulus_set") == "neutral")

# Build fast lookups
neutral_ttr = {r["id"]: r["ttr"] for r in neutral_df.iter_rows(named=True)}
neutral_sent = {r["id"]: r["avg_sent_len"] for r in neutral_df.iter_rows(named=True)}
neutral_wc = {r["id"]: r["word_count_computed"] for r in neutral_df.iter_rows(named=True)}

all_pairs = []
flagged_pairs = []

for row in clinical_df.iter_rows(named=True):
    clinical_id = row["id"]
    neutral_id = row["matched_control_id"]

    n_ttr = neutral_ttr.get(neutral_id)
    n_sent = neutral_sent.get(neutral_id)
    n_wc = neutral_wc.get(neutral_id)

    if n_ttr is None:
        console.print(f"[yellow]  WARNING: No neutral found for {clinical_id}[/yellow]")
        continue

    # Compute feature differences
    ttr_diff = abs(row["ttr"] - n_ttr)
    sent_len_diff = abs(row["avg_sent_len"] - n_sent)
    sent_len_diff_pct = sent_len_diff / max(row["avg_sent_len"], n_sent) if max(row["avg_sent_len"], n_sent) > 0 else 0
    wc_diff_pct = abs(row["word_count_computed"] - n_wc) / max(row["word_count_computed"], n_wc)

    flagged = (ttr_diff > TTR_DIFF_THRESHOLD) or (sent_len_diff_pct > SENT_LEN_DIFF_THRESHOLD)

    pair = {
        "clinical_id": clinical_id,
        "neutral_id": neutral_id,
        "emotion": row["emotion"],
        "domain": row["domain"],
        # Clinical features
        "clinical_ttr": round(row["ttr"], 4),
        "clinical_avg_sent_len": round(row["avg_sent_len"], 1),
        "clinical_word_count": row["word_count_computed"],
        # Neutral features
        "neutral_ttr": round(n_ttr, 4),
        "neutral_avg_sent_len": round(n_sent, 1),
        "neutral_word_count": n_wc,
        # Differences
        "ttr_diff": round(ttr_diff, 4),
        "sent_len_diff_pct": round(sent_len_diff_pct, 4),
        "word_count_diff_pct": round(wc_diff_pct, 4),
        "flagged_ttr": ttr_diff > TTR_DIFF_THRESHOLD,
        "flagged_sent_len": sent_len_diff_pct > SENT_LEN_DIFF_THRESHOLD,
        "flagged": flagged,
    }
    all_pairs.append(pair)
    if flagged:
        flagged_pairs.append(pair)

pairs_df = pl.DataFrame(all_pairs)

# ============================================================================
# PRINT RESULTS
# ============================================================================

console.print(f"\n[bold]Surface feature pair analysis:[/bold]")
console.print(f"  Total pairs checked: {len(all_pairs)}")
console.print(
    f"  Flagged (TTR diff > {TTR_DIFF_THRESHOLD:.0%} OR sent len diff > {SENT_LEN_DIFF_THRESHOLD:.0%}): "
    f"{len(flagged_pairs)}"
)

if flagged_pairs:
    console.print(f"\n[bold yellow]Flagged pairs:[/bold yellow]")
    flag_table = Table()
    flag_table.add_column("Clinical ID", style="cyan")
    flag_table.add_column("Neutral ID", style="dim cyan")
    flag_table.add_column("TTR Diff", justify="right")
    flag_table.add_column("Sent Len Diff%", justify="right")
    flag_table.add_column("WC Diff%", justify="right")
    flag_table.add_column("Issues", style="yellow")

    for p in sorted(flagged_pairs, key=lambda x: x["ttr_diff"] + x["sent_len_diff_pct"], reverse=True):
        issues = []
        if p["flagged_ttr"]:
            issues.append("TTR")
        if p["flagged_sent_len"]:
            issues.append("SENT_LEN")
        flag_table.add_row(
            p["clinical_id"],
            p["neutral_id"],
            f"{p['ttr_diff']:.3f}",
            f"{p['sent_len_diff_pct']:.1%}",
            f"{p['word_count_diff_pct']:.1%}",
            ", ".join(issues),
        )
    console.print(flag_table)
else:
    console.print("\n[green]  No flagged pairs — all within thresholds.[/green]")

# Summary statistics
console.print("\n[bold]Summary statistics across all pairs:[/bold]")
summary_stats = pairs_df.select([
    pl.col("ttr_diff").mean().alias("mean_ttr_diff"),
    pl.col("ttr_diff").max().alias("max_ttr_diff"),
    pl.col("sent_len_diff_pct").mean().alias("mean_sent_len_diff_pct"),
    pl.col("sent_len_diff_pct").max().alias("max_sent_len_diff_pct"),
    pl.col("word_count_diff_pct").mean().alias("mean_wc_diff_pct"),
])
for col in summary_stats.columns:
    console.print(f"  {col}: {summary_stats[col][0]:.4f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

df.write_csv(OUTPUT_DIR / "surface_features.csv")
console.print(f"\n[bold green]✓ Saved: {OUTPUT_DIR / 'surface_features.csv'}[/bold green]")

pairs_df.write_csv(OUTPUT_DIR / "surface_feature_pairs.csv")
console.print(f"[bold green]✓ Saved: {OUTPUT_DIR / 'surface_feature_pairs.csv'}[/bold green]")

console.print("\n[bold green]✓ Surface feature matching validation complete![/bold green]")
