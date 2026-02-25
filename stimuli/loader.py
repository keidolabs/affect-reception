"""
Stimulus Loader — Unified loader for all stimulus sets.

Parses Set A (JSONL), Set B Clinical (markdown), and Set B Neutral (markdown)
into a unified Stimulus dataclass.

Run: uv run python stimuli/loader.py
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml
import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()

STIMULI_DIR = Path(__file__).parent

# 8 Plutchik primary emotions in this experiment
EMOTIONS = [
    "rage", "grief", "terror", "ecstasy",
    "loathing", "amazement", "admiration", "vigilance",
]


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class Stimulus:
    id: str                           # e.g., "A-rage-01", "B-grief-d1-v1", "N-grief-d1-c1"
    text: str                         # raw vignette/stimulus text
    emotion: str                      # plutchik primary or "neutral"
    intensity: str                    # "peak" for Set B, "high" for Set A emotional, "none" for neutral
    domain: Optional[int]             # 1, 2, 3 for Set B; None for Set A
    domain_label: str                 # e.g., "Bereavement (death of loved one)"
    stimulus_set: str                 # "A_standard", "B_clinical", "neutral"
    matched_control_id: Optional[str] # B-clinical → matched N-neutral ID (and vice versa)
    word_count: int                   # len(text.split())
    token_count: Optional[int]        # computed later with Llama-3 tokenizer
    has_emotion_keyword: Optional[bool]  # from Set A JSONL; None for Set B


# ============================================================================
# SET A LOADER — parse set-a-standard.jsonl
# ============================================================================

def load_set_a() -> list[Stimulus]:
    """Load Set A standard stimuli from JSONL file. Returns 90 stimuli."""
    path = STIMULI_DIR / "set-a-standard.jsonl"
    stimuli = []

    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)

        # Map intensity integer to string label
        if rec["emotion"] == "neutral":
            intensity = "none"
        else:
            intensity = "high" if rec.get("intensity", 4) >= 4 else "low"

        s = Stimulus(
            id=rec["id"],
            text=rec["text"],
            emotion=rec["emotion"],
            intensity=intensity,
            domain=None,
            domain_label="",
            stimulus_set="A_standard",
            matched_control_id=None,
            word_count=rec.get("word_count", len(rec["text"].split())),
            token_count=None,
            has_emotion_keyword=rec.get("has_emotion_keyword"),
        )
        stimuli.append(s)

    return stimuli


# ============================================================================
# SET B CLINICAL LOADER — parse markdown vignettes with ## Vignette N.M headers
# ============================================================================

def _parse_clinical_file(filepath: Path) -> list[Stimulus]:
    """
    Parse one clinical vignette markdown file (one emotion + one domain).

    File format:
      ---  (YAML frontmatter with emotion, intensity, domain fields)
      ---
      # Title
      ## Vignette 1.1
      {text}
      ## Vignette 1.2
      {text}
      ...
      ---  (design notes separator — stop here)
      ## Design Notes
      ...
    """
    content = filepath.read_text()

    # Split on --- to get frontmatter and body
    # content starts with "---\n" so split gives ['', frontmatter, body]
    parts = content.split("---", 2)
    if len(parts) < 3:
        console.print(f"[red]WARNING: Could not parse frontmatter in {filepath}[/red]")
        return []

    meta = yaml.safe_load(parts[1])
    body = parts[2]

    # Extract domain number and label from YAML "domain: N - Label" format
    # e.g., "1 - Bereavement (death of loved one)"
    domain_str = str(meta.get("domain", "1 - Unknown"))
    domain_num = int(domain_str.split("-")[0].strip())
    domain_label = domain_str.split("-", 1)[1].strip()

    emotion = meta.get("emotion", filepath.parent.name)

    # Stop at the design notes separator "\n---\n" in body
    # The body starts with "\n" (after the second ---), so rfind "\n---\n"
    design_notes_idx = body.rfind("\n---\n")
    if design_notes_idx > 0:
        body = body[:design_notes_idx]

    # Split on "## Vignette N.M" headers; skip pre-header text (index 0)
    vignette_blocks = re.split(r"## Vignette \d+\.\d+", body)[1:]
    vignette_texts = [v.strip() for v in vignette_blocks if v.strip()]

    stimuli = []
    for v_idx, vtext in enumerate(vignette_texts, start=1):
        stim_id = f"B-{emotion}-d{domain_num}-v{v_idx}"
        matched_id = f"N-{emotion}-d{domain_num}-c{v_idx}"

        s = Stimulus(
            id=stim_id,
            text=vtext,
            emotion=emotion,
            intensity="peak",
            domain=domain_num,
            domain_label=domain_label,
            stimulus_set="B_clinical",
            matched_control_id=matched_id,
            word_count=len(vtext.split()),
            token_count=None,
            has_emotion_keyword=None,
        )
        stimuli.append(s)

    return stimuli


def load_set_b_clinical() -> list[Stimulus]:
    """Load all Set B clinical vignettes. Returns 96 stimuli (8 × 3 × 4)."""
    stimuli = []
    clinical_dir = STIMULI_DIR / "set-b-clinical"

    for emotion in EMOTIONS:
        for domain_num in range(1, 4):  # domains 1, 2, 3
            filepath = clinical_dir / emotion / f"{emotion}-peak-domain{domain_num}.md"
            if filepath.exists():
                parsed = _parse_clinical_file(filepath)
                if len(parsed) != 4:
                    console.print(
                        f"[yellow]WARNING: {filepath.name} produced {len(parsed)} vignettes "
                        f"(expected 4)[/yellow]"
                    )
                stimuli.extend(parsed)
            else:
                console.print(f"[red]ERROR: Missing file {filepath}[/red]")

    return stimuli


# ============================================================================
# SET B NEUTRAL CONTROLS LOADER — parse markdown with ## Domain N / ### Control N.M headers
# ============================================================================

def _parse_neutral_file(filepath: Path) -> list[Stimulus]:
    """
    Parse one neutral controls markdown file (one emotion, all 3 domains × 4 controls = 12).

    File format:
      ---  (YAML frontmatter with emotion: neutral, matched_to: {emotion} (peak intensity))
      ---
      # Title
      ## Domain N: {label}
      ### Control N.M
      {text}
      ### Control N.M
      {text}
      ...
      ## Domain N+1: {label}
      ...
      ---  (matching notes separator)
      ## Matching Notes
      ...
    """
    content = filepath.read_text()

    parts = content.split("---", 2)
    if len(parts) < 3:
        console.print(f"[red]WARNING: Could not parse frontmatter in {filepath}[/red]")
        return []

    meta = yaml.safe_load(parts[1])
    body = parts[2]

    # Extract the matched emotion from "matched_to: grief (peak intensity)"
    matched_to_str = str(meta.get("matched_to", "unknown"))
    matched_emotion = matched_to_str.split("(")[0].strip()

    # Stop at the "## Matching Notes" section
    matching_notes_idx = body.find("\n## Matching Notes")
    if matching_notes_idx > 0:
        body = body[:matching_notes_idx]

    # Also stop at "\n---\n" if present before matching notes
    alt_sep_idx = body.rfind("\n---\n")
    if alt_sep_idx > 0:
        body = body[:alt_sep_idx]

    stimuli = []

    # Split on "## Domain N: {label}" headers using capturing groups
    # re.split with groups returns: [pre, num, label, content, num, label, content, ...]
    domain_blocks = re.split(r"## Domain (\d+): ([^\n]+)", body)
    # domain_blocks[0] = text before first domain header (title etc.)
    # Then triplets: (domain_num_str, domain_label, domain_content)

    i = 1
    while i + 2 < len(domain_blocks):
        domain_num = int(domain_blocks[i].strip())
        domain_label = domain_blocks[i + 1].strip()
        domain_content = domain_blocks[i + 2]

        # Split on "### Control N.M" using capturing group for the vignette index
        control_blocks = re.split(r"### Control \d+\.(\d+)", domain_content)
        # control_blocks[0] = text before first control header
        # Then pairs: (control_num_str, control_text)

        j = 1
        while j + 1 <= len(control_blocks) - 1:
            control_num = int(control_blocks[j].strip())
            control_text = control_blocks[j + 1].strip()

            stim_id = f"N-{matched_emotion}-d{domain_num}-c{control_num}"
            matched_id = f"B-{matched_emotion}-d{domain_num}-v{control_num}"

            s = Stimulus(
                id=stim_id,
                text=control_text,
                emotion="neutral",
                intensity="none",
                domain=domain_num,
                domain_label=domain_label,
                stimulus_set="neutral",
                matched_control_id=matched_id,
                word_count=len(control_text.split()),
                token_count=None,
                has_emotion_keyword=None,
            )
            stimuli.append(s)
            j += 2

        i += 3

    return stimuli


def load_set_b_neutral() -> list[Stimulus]:
    """Load all Set B neutral controls. Returns 96 stimuli (8 × 3 × 4)."""
    stimuli = []
    neutral_dir = STIMULI_DIR / "neutral-controls"

    for emotion in EMOTIONS:
        filepath = neutral_dir / f"{emotion}-controls.md"
        if filepath.exists():
            parsed = _parse_neutral_file(filepath)
            if len(parsed) != 12:
                console.print(
                    f"[yellow]WARNING: {filepath.name} produced {len(parsed)} controls "
                    f"(expected 12)[/yellow]"
                )
            stimuli.extend(parsed)
        else:
            console.print(f"[red]ERROR: Missing file {filepath}[/red]")

    return stimuli


# ============================================================================
# COMBINED LOADER
# ============================================================================

def load_all() -> list[Stimulus]:
    """Load all stimulus sets and return combined list of 282 stimuli."""
    set_a = load_set_a()
    set_b_clinical = load_set_b_clinical()
    set_b_neutral = load_set_b_neutral()
    return set_a + set_b_clinical + set_b_neutral


def export_jsonl(stimuli: list[Stimulus], path: Path) -> None:
    """Write stimuli list to JSONL file (one JSON object per line)."""
    with open(path, "w") as f:
        for s in stimuli:
            f.write(json.dumps(asdict(s)) + "\n")


def to_polars(stimuli: list[Stimulus]) -> pl.DataFrame:
    """Convert stimuli list to polars DataFrame."""
    records = [asdict(s) for s in stimuli]
    return pl.DataFrame(records)


# ============================================================================
# MAIN — Run as script to validate and generate stimuli.jsonl
# ============================================================================

if __name__ == "__main__":
    console.print("[bold cyan]Loading all stimulus sets...[/bold cyan]\n")

    set_a = load_set_a()
    console.print(f"Set A: {len(set_a)} stimuli")

    set_b_clinical = load_set_b_clinical()
    console.print(f"Set B clinical: {len(set_b_clinical)} stimuli")

    set_b_neutral = load_set_b_neutral()
    console.print(f"Set B neutral: {len(set_b_neutral)} stimuli")

    all_stimuli = set_a + set_b_clinical + set_b_neutral

    # ---- Summary table ----
    df = to_polars(all_stimuli)
    summary = (
        df.group_by(["stimulus_set", "emotion"])
        .agg(pl.len().alias("count"))
        .sort(["stimulus_set", "emotion"])
    )

    table = Table(title="Stimulus Counts by Set and Emotion")
    table.add_column("Stimulus Set", style="cyan")
    table.add_column("Emotion", style="white")
    table.add_column("Count", justify="right", style="green")

    for row in summary.iter_rows(named=True):
        table.add_row(row["stimulus_set"], row["emotion"], str(row["count"]))

    console.print(table)

    # ---- Totals ----
    set_totals = (
        df.group_by("stimulus_set")
        .agg(pl.len().alias("count"))
        .sort("stimulus_set")
    )
    console.print("\n[bold]Totals by set:[/bold]")
    for row in set_totals.iter_rows(named=True):
        console.print(f"  {row['stimulus_set']}: {row['count']}")

    console.print(f"\n[bold green]Total stimuli: {len(all_stimuli)}[/bold green]")

    if len(all_stimuli) != 282:
        console.print(
            f"[bold red]FAIL: Expected 282, got {len(all_stimuli)}![/bold red]"
        )
        sys.exit(1)

    console.print("[bold green]PASS: 282 stimuli loaded correctly[/bold green]")

    # ---- Export ----
    output_path = STIMULI_DIR / "stimuli.jsonl"
    export_jsonl(all_stimuli, output_path)
    console.print(f"\n[bold green]✓ Exported to {output_path}[/bold green]")
