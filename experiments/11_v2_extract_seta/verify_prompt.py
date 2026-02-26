"""
Experiment 11 — v2 Set A Extraction: Prompt Tokenization Verification
Question: Is the final token of the Tak et al. few-shot prompt ":" for all 6 tokenizers?
Date: 2026-02-25

Run FIRST before any extraction. Confirms that the Tak et al. few-shot prompt
correctly ends at ":" for all models. If any tokenizer splits ":" differently,
we need to adjust before extraction.

Run: uv run python experiments/11_v2_extract_seta/verify_prompt.py
"""

import os
import sys
import torch
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

torch.manual_seed(42)

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent

sys.path.insert(0, str(ROOT))

# ============================================================================
# LOAD HF TOKEN
# ============================================================================

env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if line.startswith("HF_ACCESS_TOKEN"):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break
console.print("[bold cyan]HF token loaded[/bold cyan]")

# ============================================================================
# TAK ET AL. FEW-SHOT PROMPT
# ============================================================================

SHOT_1_TEXT    = "My dog died last week. I miss him every day."
SHOT_1_EMOTION = "sadness"
SHOT_2_TEXT    = "I got promoted and my boss praised my work in front of everyone."
SHOT_2_EMOTION = "joy"

def build_tak_prompt(target_text: str) -> str:
    """Build Tak et al. few-shot classification prompt."""
    return (
        "What are the inferred emotions in the following contexts?\n\n"
        f"Context: {SHOT_1_TEXT}\n"
        f"Answer: {SHOT_1_EMOTION}\n\n"
        f"Context: {SHOT_2_TEXT}\n"
        f"Answer: {SHOT_2_EMOTION}\n\n"
        f"Context: {target_text}\n"
        "Answer:"
    )

# Sample target text for verification
SAMPLE_TARGET = "She couldn't stop crying after hearing the news about her father."

SAMPLE_PROMPT = build_tak_prompt(SAMPLE_TARGET)
console.print(f"\n[bold cyan]Tak prompt (first 300 chars):[/bold cyan]")
console.print(SAMPLE_PROMPT[:300])
console.print("...")

# ============================================================================
# MODEL REGISTRY — just tokenizer IDs for verification
# ============================================================================

TOKENIZER_REGISTRY = [
    ("llama1b_base",  "meta-llama/Llama-3.2-1B",              "transformerlens"),
    ("llama1b_inst",  "meta-llama/Llama-3.2-1B-Instruct",     "transformerlens"),
    ("llama8b_base",  "meta-llama/Meta-Llama-3-8B",           "hf_hooks"),
    ("llama8b_inst",  "meta-llama/Meta-Llama-3-8B-Instruct",  "hf_hooks"),
    ("gemma9b_base",  "google/gemma-2-9b",                    "hf_hooks"),
    ("gemma9b_inst",  "google/gemma-2-9b-it",                 "hf_hooks"),
]

# ============================================================================
# TOKENIZATION CHECK FOR EACH MODEL
# ============================================================================

console.print(f"\n[bold cyan]Checking tokenization for all 6 models...[/bold cyan]")

from transformers import AutoTokenizer

results = []
all_pass = True

for model_key, hf_id, framework in TOKENIZER_REGISTRY:
    console.print(f"\n[bold white]Checking {model_key} ({hf_id})...[/bold white]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        ids = tokenizer(SAMPLE_PROMPT, return_tensors="pt")["input_ids"]
        n_tokens = ids.shape[1]
        final_tok_id = ids[0, -1].item()
        final_tok_str = tokenizer.decode([final_tok_id])
        tok_ok = ":" in final_tok_str

        # Also check second-to-last token
        penult_tok_str = tokenizer.decode([ids[0, -2].item()])

        # Show last 5 tokens
        last5 = [tokenizer.decode([ids[0, i].item()]) for i in range(-5, 0)]

        results.append({
            "model_key": model_key,
            "hf_id": hf_id,
            "n_tokens": n_tokens,
            "final_tok": repr(final_tok_str),
            "penult_tok": repr(penult_tok_str),
            "last5": last5,
            "pass": tok_ok,
        })

        status = "[bold green]PASS[/bold green]" if tok_ok else "[bold red]FAIL[/bold red]"
        console.print(f"  {status} | n_tokens={n_tokens} | final={repr(final_tok_str)} | last5={last5}")

        if not tok_ok:
            all_pass = False
            console.print(f"  [bold red]WARNING: Final token {repr(final_tok_str)} does not contain ':' !")
            console.print(f"  Full prompt (last 50 chars): {repr(SAMPLE_PROMPT[-50:])}")

        del tokenizer

    except Exception as e:
        console.print(f"  [bold red]ERROR: {e}[/bold red]")
        results.append({
            "model_key": model_key,
            "hf_id": hf_id,
            "n_tokens": -1,
            "final_tok": "ERROR",
            "penult_tok": "ERROR",
            "last5": [],
            "pass": False,
        })
        all_pass = False

# ============================================================================
# SUMMARY TABLE
# ============================================================================

console.print("\n[bold cyan]Summary:[/bold cyan]")
table = Table(title="Tak Prompt Tokenization Verification")
table.add_column("Model", style="cyan")
table.add_column("n_tokens", justify="right")
table.add_column("Final token", style="white")
table.add_column("Last 5 tokens")
table.add_column("Status", justify="center")

for r in results:
    status = "[bold green]PASS[/bold green]" if r["pass"] else "[bold red]FAIL[/bold red]"
    table.add_row(
        r["model_key"],
        str(r["n_tokens"]),
        r["final_tok"],
        str(r["last5"]),
        status,
    )

console.print(table)

if all_pass:
    console.print("\n[bold green]✓ All models tokenize Tak prompt with ':' as final token[/bold green]")
    console.print("[bold green]✓ Safe to proceed with extraction[/bold green]")
else:
    console.print("\n[bold red]✗ SOME MODELS FAILED — investigate before extraction[/bold red]")
    import sys; sys.exit(1)
