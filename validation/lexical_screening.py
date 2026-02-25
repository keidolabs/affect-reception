"""
Lexical Sentiment Screening
Question: Do Set B clinical vignettes (keyword-free) score closer to neutral
          on a standard sentiment classifier than Set A (keyword-rich)?
Date: 2026-02-24

Method: RoBERTa sentiment classifier (cardiffnlp/twitter-roberta-base-sentiment-latest)
        Labels: 0=Negative, 1=Neutral, 2=Positive
        Hypothesis: Set B clinical → lower sentiment polarity than Set A

Run: uv run python validation/lexical_screening.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from scipy.special import softmax
from scipy.stats import mannwhitneyu
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.loader import load_all, to_polars

console = Console()

VALIDATION_DIR = Path(__file__).parent
OUTPUT_DIR = VALIDATION_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Labels from model card: 0=Negative, 1=Neutral, 2=Positive
LABEL_NAMES = ["negative", "neutral", "positive"]

# Consistent color scheme for stimulus sets
SET_COLORS = {
    "A_standard": "#e74c3c",   # red — keyword-rich
    "B_clinical": "#3498db",   # blue — keyword-free clinical
    "neutral": "#95a5a6",      # gray — neutral controls
}

# ============================================================================
# LOAD STIMULI
# ============================================================================

console.print("[bold cyan]Loading stimuli...[/bold cyan]")
stimuli = load_all()
df = to_polars(stimuli)
console.print(f"Loaded {len(stimuli)} stimuli")

# ============================================================================
# LOAD SENTIMENT MODEL
# ============================================================================

console.print(f"\n[bold cyan]Loading sentiment model: {SENTIMENT_MODEL_ID}...[/bold cyan]")
sent_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_ID)
sent_model.eval()
console.print("Sentiment model loaded (CPU)")


def preprocess_text(text: str) -> str:
    """Preprocess text for RoBERTa: replace @mentions → @user, URLs → http."""
    tokens = text.split()
    cleaned = []
    for t in tokens:
        if t.startswith("@") and len(t) > 1:
            cleaned.append("@user")
        elif t.startswith("http"):
            cleaned.append("http")
        else:
            cleaned.append(t)
    return " ".join(cleaned)


def get_sentiment_scores(text: str) -> tuple[float, float, float]:
    """
    Get sentiment probabilities [negative, neutral, positive] for a text.
    Truncates at 512 tokens as required by RoBERTa.
    """
    text_clean = preprocess_text(text)
    encoded = sent_tokenizer(
        text_clean,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        output = sent_model(**encoded)
    scores = softmax(output.logits[0].numpy())
    return float(scores[0]), float(scores[1]), float(scores[2])


# ============================================================================
# RUN SENTIMENT SCORING ON ALL STIMULI
# ============================================================================

console.print(f"\n[bold cyan]Scoring {len(stimuli)} stimuli (this takes ~5-10 min)...[/bold cyan]")

neg_scores, neu_scores, pos_scores = [], [], []

from rich.progress import track
for s in track(stimuli, description="Scoring..."):
    neg, neu, pos = get_sentiment_scores(s.text)
    neg_scores.append(neg)
    neu_scores.append(neu)
    pos_scores.append(pos)

# Polarity = positive - negative (signed sentiment score)
polarity = [p - n for p, n in zip(pos_scores, neg_scores)]

df = df.with_columns([
    pl.Series("sentiment_negative", neg_scores),
    pl.Series("sentiment_neutral", neu_scores),
    pl.Series("sentiment_positive", pos_scores),
    pl.Series("sentiment_polarity", polarity),
])

# ============================================================================
# STATISTICAL COMPARISON: Set A vs Set B clinical (emotional stimuli only)
# ============================================================================

console.print("\n[bold cyan]Comparing sentiment distributions...[/bold cyan]")

set_a_emo = df.filter(
    (pl.col("stimulus_set") == "A_standard") & (pl.col("emotion") != "neutral")
)["sentiment_polarity"].to_numpy()

set_b_clin = df.filter(pl.col("stimulus_set") == "B_clinical")["sentiment_polarity"].to_numpy()

set_neutral = df.filter(
    (pl.col("stimulus_set") == "A_standard") & (pl.col("emotion") == "neutral")
)["sentiment_polarity"].to_numpy()

# Mann-Whitney U: are Set A and Set B from different distributions?
stat_ab, p_ab = mannwhitneyu(set_a_emo, set_b_clin, alternative="two-sided")

# Mann-Whitney U: are Set B and neutral from different distributions?
stat_bn, p_bn = mannwhitneyu(set_b_clin, set_neutral, alternative="two-sided")

console.print(f"\n[bold]Sentiment polarity (positive − negative):[/bold]")
console.print(f"  Set A emotional (n={len(set_a_emo)}): mean={set_a_emo.mean():.3f}, std={set_a_emo.std():.3f}")
console.print(f"  Set B clinical  (n={len(set_b_clin)}): mean={set_b_clin.mean():.3f}, std={set_b_clin.std():.3f}")
console.print(f"  Set A neutral   (n={len(set_neutral)}): mean={set_neutral.mean():.3f}, std={set_neutral.std():.3f}")

console.print(f"\n[bold]Mann-Whitney U tests:[/bold]")
console.print(f"  Set A emo vs Set B clinical: U={stat_ab:.0f}, p={p_ab:.4f}")
console.print(f"  Set B clinical vs neutral:   U={stat_bn:.0f}, p={p_bn:.4f}")

# Interpretation
if p_ab < 0.05:
    console.print(
        "[green]  ✓ Set A and Set B have significantly different sentiment distributions "
        "(expected: keyword-rich vs keyword-free)[/green]"
    )
else:
    console.print(
        "[yellow]  ~ Set A and Set B do NOT differ significantly in sentiment "
        "(unexpected — may indicate Set A keyword control worked too well)[/yellow]"
    )

# ============================================================================
# PLOTS
# ============================================================================

console.print("\n[bold cyan]Generating plots...[/bold cyan]")

# --- Box plot: polarity by stimulus set ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sentiment Screening: Set A vs Set B Clinical vs Neutral Controls", fontsize=13)

# Box plot of polarity
data_groups = [
    ("Set A\nEmotional", set_a_emo, SET_COLORS["A_standard"]),
    ("Set B\nClinical", set_b_clin, SET_COLORS["B_clinical"]),
    ("Set A\nNeutral", set_neutral, SET_COLORS["neutral"]),
]

ax1 = axes[0]
bp = ax1.boxplot(
    [d[1] for d in data_groups],
    labels=[d[0] for d in data_groups],
    patch_artist=True,
    notch=False,
)
for patch, (_, _, color) in zip(bp["boxes"], data_groups):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_ylabel("Sentiment Polarity (Positive − Negative)")
ax1.set_title("Polarity Distribution by Stimulus Set")
ax1.set_ylim(-1.1, 1.1)

# Add p-value annotations
y_max = 0.85
ax1.annotate(
    f"p={p_ab:.4f}" + (" *" if p_ab < 0.05 else " ns"),
    xy=(1.5, y_max),
    ha="center",
    fontsize=9,
    color="red" if p_ab < 0.05 else "gray",
)

# Scatter plot: per-emotion mean polarity
ax2 = axes[1]
set_b_by_emo = (
    df.filter(pl.col("stimulus_set") == "B_clinical")
    .group_by("emotion")
    .agg(pl.col("sentiment_polarity").mean().alias("b_polarity"))
    .sort("emotion")
)
set_a_by_emo = (
    df.filter(
        (pl.col("stimulus_set") == "A_standard") & (pl.col("emotion") != "neutral")
    )
    .group_by("emotion")
    .agg(pl.col("sentiment_polarity").mean().alias("a_polarity"))
    .sort("emotion")
)

joined = set_a_by_emo.join(set_b_by_emo, on="emotion")
for row in joined.iter_rows(named=True):
    ax2.scatter(row["a_polarity"], row["b_polarity"], s=80, zorder=5)
    ax2.annotate(
        row["emotion"],
        (row["a_polarity"], row["b_polarity"]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=8,
    )

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axvline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.set_xlabel("Set A Mean Polarity (keyword-rich)")
ax2.set_ylabel("Set B Mean Polarity (keyword-free)")
ax2.set_title("Per-Emotion Mean Polarity: Set A vs Set B")
# Diagonal reference line (y=x)
lims = [-0.6, 0.6]
ax2.plot(lims, lims, "k:", linewidth=0.7, alpha=0.5, label="y=x")
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "sentiment_boxplot.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "sentiment_boxplot.svg", bbox_inches="tight")
plt.close(fig)
console.print(f"[bold green]✓ Saved: sentiment_boxplot.png + .svg[/bold green]")

# ============================================================================
# SAVE RESULTS
# ============================================================================

df.write_csv(OUTPUT_DIR / "lexical_screening.csv")
console.print(f"[bold green]✓ Saved: {OUTPUT_DIR / 'lexical_screening.csv'}[/bold green]")

console.print("\n[bold green]✓ Lexical sentiment screening complete![/bold green]")
