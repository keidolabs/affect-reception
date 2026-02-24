"""
Curate Set A (standard/keyword-rich) stimuli from crowd-enVENT 2023.

Selection criteria:
- intensity >= 4 (high emotional intensity)
- word count >= 10 (no ultra-short fragments)
- top 10 by word count per emotion (prefer longer, richer examples)
- verify emotion keyword presence

Maps crowd-enVENT 13 emotions → Plutchik 8 primaries:
  anger → rage, sadness → grief, fear → terror, joy → ecstasy,
  disgust → loathing, surprise → amazement, trust → admiration
  no-emotion → neutral controls

Note: Vigilance (anticipation peak) has no crowd-enVENT match.
"""

import csv
import json
import random
from pathlib import Path

random.seed(42)

CROWD_ENVENT_PATH = Path("/Users/keeman/Downloads/crowd-enVent2023/corpus/crowd-enVent_generation.tsv")
OUTPUT_DIR = Path(__file__).parent / "set-a-standard"
JSONL_PATH = Path(__file__).parent / "set-a-standard.jsonl"

# crowd-enVENT emotion → our Plutchik primary
EMOTION_MAP = {
    "anger": "rage",
    "sadness": "grief",
    "fear": "terror",
    "joy": "ecstasy",
    "disgust": "loathing",
    "surprise": "amazement",
    "trust": "admiration",
    "no-emotion": "neutral",
}

# Emotion keywords we expect to find in Set A (confirmation check)
# Broad list: includes common synonyms, derivatives, and sentiment phrases
EMOTION_KEYWORDS = {
    # Anger / Rage family
    "anger", "angry", "furious", "rage", "mad", "irritated", "annoyed", "enraged", "livid",
    "infuriated", "infuriating", "outraged", "outrage", "fuming", "seething", "irate",
    "frustrated", "frustrating", "frustration", "resentful", "resentment", "bitter",
    "hostile", "aggravated", "incensed", "wrathful", "wrath", "temper", "fury",
    # Sadness / Grief family
    "sad", "sadness", "grief", "grieving", "devastated", "heartbroken", "miserable",
    "depressed", "depressing", "depression", "upset", "sorrowful", "sorrow", "mourning",
    "melancholy", "despairing", "despair", "hopeless", "lonely", "loneliness",
    "bereft", "forlorn", "desolate", "anguish", "heartbreak", "crying", "cried", "tears",
    "weeping", "wept", "sobbing", "sobbed",
    # Fear / Terror family
    "fear", "afraid", "scared", "terrified", "terrifying", "frightened", "frightening",
    "anxious", "anxiety", "panic", "panicked", "panicking", "dread", "dreaded", "dreading",
    "horror", "horrified", "horrifying", "alarmed", "alarming", "petrified", "terror",
    "worried", "worrying", "worry", "nervous", "shaking", "trembling",
    # Joy / Ecstasy family
    "joy", "joyful", "joyous", "happy", "happiness", "delighted", "thrilled", "ecstatic",
    "glad", "pleased", "elated", "euphoric", "blissful", "overjoyed", "jubilant",
    "cheerful", "excited", "excitement", "exciting", "wonderful", "fantastic", "amazing",
    "love", "loved", "loving", "grateful", "gratitude", "blessed", "proud",
    # Disgust / Loathing family
    "disgust", "disgusted", "disgusting", "revolted", "revolting", "repulsed", "repulsive",
    "sickened", "sickening", "appalled", "appalling", "loathing", "loathe", "abhorrent",
    "vile", "repugnant", "nauseated", "nauseating", "gross", "horrendous", "despicable",
    "contempt", "contemptuous",
    # Surprise / Amazement family
    "surprise", "surprised", "surprising", "shocked", "shocking", "astonished", "astonishing",
    "amazed", "amazing", "stunned", "stunning", "unexpected", "unexpectedly",
    "disbelief", "bewildered", "flabbergasted", "startled", "awestruck", "gobsmacked",
    "speechless", "unbelievable", "incredible", "incredulous",
    # Trust / Admiration family
    "trust", "trusted", "trusting", "trustworthy", "reliable", "faith", "confident",
    "confidence", "belief", "believe", "believed", "loyal", "loyalty", "devoted", "devotion",
    "dependable", "honest", "honesty", "integrity", "admire", "admired", "admiration",
    "respect", "respected",
    # General emotion/sentiment phrases
    "hurtful", "hurt", "painful", "pain", "suffering", "agony", "agonizing",
    "emotional", "overwhelmed", "overwhelming", "distraught", "distressed", "distressing",
}


def has_emotion_keyword(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & EMOTION_KEYWORDS)


def load_and_filter():
    with open(CROWD_ENVENT_PATH, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    candidates = {}
    for emotion_src, plutchik in EMOTION_MAP.items():
        pool = [
            r for r in rows
            if r["emotion"] == emotion_src
            and r["intensity"].isdigit()
            and int(r["intensity"]) >= 4
            and len(r["generated_text"].split()) >= 10
        ]
        # Sort by word count descending, take top 30 as candidate pool
        pool.sort(key=lambda r: len(r["generated_text"].split()), reverse=True)
        candidates[plutchik] = pool[:30]

    return candidates


def select_stimuli(candidates: dict) -> dict:
    selected = {}
    for plutchik, pool in candidates.items():
        if plutchik == "neutral":
            # For neutral: no intensity requirement, prefer longer, NO emotion keywords
            with open(CROWD_ENVENT_PATH, "r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                neutral_pool = [
                    r for r in reader
                    if r["emotion"] == "no-emotion"
                    and len(r["generated_text"].split()) >= 10
                    and not has_emotion_keyword(r["generated_text"])
                ]
            neutral_pool.sort(key=lambda r: len(r["generated_text"].split()), reverse=True)
            chosen = random.sample(neutral_pool[:20], min(10, len(neutral_pool[:20])))
            selected[plutchik] = chosen
            continue

        # For emotional stimuli: PRIORITIZE those with emotion keywords
        with_kw = [r for r in pool if has_emotion_keyword(r["generated_text"])]
        without_kw = [r for r in pool if not has_emotion_keyword(r["generated_text"])]

        # Take from keyword pool first, fill remainder from non-keyword if needed
        if len(with_kw) >= 10:
            chosen = random.sample(with_kw[:20], 10)
        else:
            chosen = with_kw + random.sample(without_kw, min(10 - len(with_kw), len(without_kw)))

        selected[plutchik] = chosen
    return selected


def write_markdown(selected: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for plutchik, stimuli in selected.items():
        if plutchik == "neutral":
            filename = "neutral-controls.md"
        else:
            filename = f"{plutchik}.md"

        lines = [
            "---",
            f"tags: [research, ei-mechanics, stimuli, {plutchik}, set-a, standard]",
            f"emotion: {plutchik}",
            "stimulus_set: A_standard",
            f"source: crowd-enVENT (Troiano et al., 2023)",
            "status: curated",
            "created: 2026-02-23",
            "---",
            "",
            f"# Set A Standard — {plutchik.title()}",
            "",
            f"Source: crowd-enVENT 2023 (emotion: {[k for k,v in EMOTION_MAP.items() if v==plutchik][0]})",
            f"Selection: intensity >= 4, word count >= 10, top by length",
            "",
        ]

        for i, row in enumerate(stimuli, 1):
            text = row["generated_text"].strip()
            wc = len(text.split())
            intensity = row["intensity"]
            has_kw = has_emotion_keyword(text)
            text_id = row["text_id"]

            lines.extend([
                f"## Stimulus {plutchik[0].upper()}{i:02d}",
                "",
                text,
                "",
                f"- **text_id**: {text_id}",
                f"- **words**: {wc}",
                f"- **intensity**: {intensity}/5",
                f"- **has_emotion_keyword**: {has_kw}",
                "",
            ])

        filepath = OUTPUT_DIR / filename
        filepath.write_text("\n".join(lines))
        print(f"  {filepath.name}: {len(stimuli)} stimuli")


def write_jsonl(selected: dict):
    records = []
    for plutchik, stimuli in selected.items():
        for i, row in enumerate(stimuli, 1):
            text = row["generated_text"].strip()
            records.append({
                "id": f"A-{plutchik}-{i:02d}",
                "text": text,
                "emotion": plutchik,
                "stimulus_set": "A_standard",
                "source_emotion": row["emotion"],
                "source_text_id": row["text_id"],
                "intensity": int(row["intensity"]) if row["intensity"].isdigit() else None,
                "word_count": len(text.split()),
                "has_emotion_keyword": has_emotion_keyword(text),
            })

    with open(JSONL_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  {JSONL_PATH.name}: {len(records)} records")


def print_summary(selected: dict):
    print("\n=== Set A Curation Summary ===\n")
    for plutchik, stimuli in selected.items():
        wcs = [len(r["generated_text"].split()) for r in stimuli]
        kw_count = sum(1 for r in stimuli if has_emotion_keyword(r["generated_text"]))
        print(
            f"  {plutchik:12s}  n={len(stimuli):2d}  "
            f"words: {min(wcs):3d}-{max(wcs):3d} (med {sorted(wcs)[len(wcs)//2]:3d})  "
            f"keywords: {kw_count}/{len(stimuli)}"
        )

    total = sum(len(s) for s in selected.values())
    print(f"\n  Total: {total} stimuli")

    # Check for missing keywords
    no_kw = []
    for plutchik, stimuli in selected.items():
        if plutchik == "neutral":
            continue
        for i, r in enumerate(stimuli, 1):
            if not has_emotion_keyword(r["generated_text"]):
                no_kw.append(f"  {plutchik}-{i:02d}: {r['generated_text'][:80]}...")
    if no_kw:
        print(f"\n  WARNING: {len(no_kw)} emotional stimuli without detected keywords:")
        for s in no_kw:
            print(s)
    else:
        print("\n  All emotional stimuli contain emotion keywords.")


if __name__ == "__main__":
    print("Loading crowd-enVENT 2023...")
    candidates = load_and_filter()

    print("Selecting stimuli...")
    selected = select_stimuli(candidates)

    print("\nWriting markdown files...")
    write_markdown(selected)

    print("\nWriting JSONL...")
    write_jsonl(selected)

    print_summary(selected)
