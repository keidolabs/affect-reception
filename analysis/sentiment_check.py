"""
Quick CLI tool: run classical sentiment/keyword analysis on a string.
Shows whether keyword-based NLP methods detect emotion in our stimuli.

Usage:
    uv run python analysis/sentiment_check.py "A kitchen table set for two..."
    uv run python analysis/sentiment_check.py -  # read from stdin
"""

import sys
import re
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ============================================================================
# LEXICONS — small curated lists covering major sentiment/emotion categories
# ============================================================================

# NRC-style emotion keywords (top ~30 per category, public domain selections)
EMOTION_LEXICON = {
    "anger": {
        "angry", "fury", "rage", "hate", "hostile", "furious", "irritated",
        "outraged", "resentment", "bitter", "aggressive", "annoyed", "mad",
        "enraged", "wrathful", "livid", "infuriated", "indignant", "vengeful",
        "spite", "contempt", "scorn", "loathe", "detest", "abhor",
    },
    "fear": {
        "afraid", "scared", "terrified", "anxious", "panic", "dread", "horror",
        "frightened", "fearful", "phobia", "nervous", "worried", "alarmed",
        "threatened", "trembling", "petrified", "startled", "uneasy", "tense",
        "apprehensive", "terror", "fright", "paranoid", "spooked", "shaken",
    },
    "sadness": {
        "sad", "grief", "sorrow", "mourning", "depressed", "melancholy",
        "heartbroken", "miserable", "gloomy", "despair", "hopeless", "lonely",
        "unhappy", "crying", "tears", "weeping", "loss", "tragic", "devastated",
        "anguish", "suffering", "pain", "woe", "forlorn", "desolate",
    },
    "joy": {
        "happy", "joy", "delighted", "cheerful", "elated", "ecstatic",
        "thrilled", "pleased", "glad", "content", "bliss", "euphoria",
        "excited", "wonderful", "celebrate", "laughter", "smile", "grin",
        "radiant", "jubilant", "overjoyed", "grateful", "thankful", "blessed",
    },
    "disgust": {
        "disgusted", "revolting", "repulsive", "sickening", "nauseating",
        "vile", "gross", "repelled", "appalled", "abhorrent", "foul",
        "offensive", "distasteful", "loathsome", "hideous", "grotesque",
        "repugnant", "stomach-turning", "wretched", "horrid",
    },
    "surprise": {
        "surprised", "shocked", "astonished", "amazed", "stunned",
        "bewildered", "startled", "unexpected", "sudden", "disbelief",
        "awestruck", "dumbfounded", "speechless", "flabbergasted",
        "astounded", "incredulous", "unbelievable", "remarkable",
    },
}

# Positive/negative valence words (VADER-style top keywords)
POSITIVE_WORDS = {
    "good", "great", "love", "happy", "excellent", "best", "beautiful",
    "wonderful", "amazing", "nice", "awesome", "fantastic", "perfect",
    "brilliant", "superb", "outstanding", "magnificent", "delightful",
    "pleasant", "favorable", "positive", "enjoy", "hope", "bright",
    "warm", "kind", "gentle", "peaceful", "calm", "comfort", "success",
    "triumph", "victory", "proud", "confidence", "strength", "brave",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "worst", "hate", "ugly",
    "disgusting", "dreadful", "nasty", "pathetic", "miserable", "cruel",
    "painful", "suffering", "death", "dead", "kill", "die", "destroy",
    "broken", "failed", "failure", "lost", "alone", "empty", "dark",
    "cold", "sick", "hurt", "wound", "damage", "ruin", "decay", "rot",
    "toxic", "poison", "threat", "danger", "victim", "abuse", "neglect",
}

# Negation words that flip sentiment
NEGATORS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "can't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't",
}

# Intensifiers
INTENSIFIERS = {
    "very", "extremely", "incredibly", "absolutely", "completely", "totally",
    "utterly", "deeply", "profoundly", "immensely", "terribly", "awfully",
    "really", "so", "quite", "rather",
}


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z']+", text.lower())


def analyze(text: str) -> dict:
    """Run all keyword-based analyses on text."""
    tokens = tokenize(text)
    token_set = set(tokens)
    n_tokens = len(tokens)

    # --- Emotion keyword hits ---
    emotion_hits = {}
    for emotion, lexicon in EMOTION_LEXICON.items():
        found = token_set & lexicon
        emotion_hits[emotion] = sorted(found)

    # --- Valence ---
    pos_found = token_set & POSITIVE_WORDS
    neg_found = token_set & NEGATIVE_WORDS

    pos_count = sum(tokens.count(w) for w in pos_found)
    neg_count = sum(tokens.count(w) for w in neg_found)

    if (pos_count + neg_count) > 0:
        polarity = (pos_count - neg_count) / (pos_count + neg_count)
    else:
        polarity = 0.0

    # --- Negation ---
    neg_words_found = token_set & NEGATORS

    # --- Intensifiers ---
    intens_found = token_set & INTENSIFIERS

    return {
        "tokens": tokens,
        "n_tokens": n_tokens,
        "emotion_hits": emotion_hits,
        "positive_words": sorted(pos_found),
        "negative_words": sorted(neg_found),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "polarity": polarity,
        "negators": sorted(neg_words_found),
        "intensifiers": sorted(intens_found),
    }


def display(text: str, results: dict):
    """Pretty-print results with rich."""
    console.print()
    console.print(Panel(text, title="Input Text", border_style="cyan", width=90))

    # --- Token stats ---
    console.print(f"\n[bold]Tokens:[/bold] {results['n_tokens']}")

    # --- Emotion keywords ---
    emo_table = Table(title="Emotion Keyword Hits", show_lines=True)
    emo_table.add_column("Emotion", style="bold")
    emo_table.add_column("Count", justify="right")
    emo_table.add_column("Keywords Found")

    total_emo = 0
    for emotion, words in results["emotion_hits"].items():
        count = len(words)
        total_emo += count
        style = "red" if count > 0 else "dim"
        emo_table.add_row(
            emotion,
            str(count),
            ", ".join(words) if words else "—",
            style=style,
        )
    console.print(emo_table)

    # --- Valence ---
    val_table = Table(title="Valence (Keyword Polarity)", show_lines=True)
    val_table.add_column("Category", style="bold")
    val_table.add_column("Count", justify="right")
    val_table.add_column("Words Found")

    val_table.add_row(
        "Positive", str(results["pos_count"]),
        ", ".join(results["positive_words"]) if results["positive_words"] else "—",
        style="green" if results["pos_count"] > 0 else "dim",
    )
    val_table.add_row(
        "Negative", str(results["neg_count"]),
        ", ".join(results["negative_words"]) if results["negative_words"] else "—",
        style="red" if results["neg_count"] > 0 else "dim",
    )
    console.print(val_table)

    # --- Polarity score ---
    pol = results["polarity"]
    if pol > 0.1:
        pol_style = "green"
        pol_label = "POSITIVE"
    elif pol < -0.1:
        pol_style = "red"
        pol_label = "NEGATIVE"
    else:
        pol_style = "yellow"
        pol_label = "NEUTRAL"

    console.print(f"\n[bold]Polarity score:[/bold] [{pol_style}]{pol:+.3f} ({pol_label})[/{pol_style}]")

    # --- Modifiers ---
    if results["negators"]:
        console.print(f"[bold]Negators:[/bold] {', '.join(results['negators'])}")
    if results["intensifiers"]:
        console.print(f"[bold]Intensifiers:[/bold] {', '.join(results['intensifiers'])}")

    # --- Verdict ---
    console.print()
    if total_emo == 0 and results["pos_count"] == 0 and results["neg_count"] == 0:
        console.print(Panel(
            "[bold green]INVISIBLE TO KEYWORD ANALYSIS[/bold green]\n"
            "No emotion keywords, no valence words detected.\n"
            "A classical sentiment analyzer would classify this as neutral.",
            border_style="green",
        ))
    elif total_emo <= 1 and (results["pos_count"] + results["neg_count"]) <= 1:
        console.print(Panel(
            "[bold yellow]MOSTLY INVISIBLE[/bold yellow]\n"
            f"Only {total_emo} emotion keyword(s) and {results['pos_count'] + results['neg_count']} valence word(s).\n"
            "Classical analyzers would likely miss the emotional content.",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[bold red]DETECTABLE BY KEYWORD ANALYSIS[/bold red]\n"
            f"{total_emo} emotion keyword(s) and {results['pos_count'] + results['neg_count']} valence word(s) found.\n"
            "Classical analyzers would likely flag emotional content.",
            border_style="red",
        ))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: uv run python analysis/sentiment_check.py \"your text here\"[/red]")
        console.print("[dim]  or pipe from stdin: echo 'text' | uv run python analysis/sentiment_check.py -[/dim]")
        sys.exit(1)

    if sys.argv[1] == "-":
        text = sys.stdin.read().strip()
    else:
        text = " ".join(sys.argv[1:])

    results = analyze(text)
    display(text, results)
