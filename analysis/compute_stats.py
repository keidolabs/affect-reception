"""
Experiment: Statistics & aggregation for v2 paper
Question: Compute all AGG and STAT outputs (see .concepts/stats-emo-circ.md)
Date: 2026-02-28

Reads from:  results/
Writes to:   results/paper_data/
Run:         uv run python analysis/compute_stats.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats as scipy_stats
from rich.console import Console

console = Console()
np.random.seed(42)
rng = np.random.default_rng(42)

ROOT       = Path(__file__).parent.parent
RESULTS    = ROOT / "results"
PAPER_DATA = RESULTS / "paper_data"
PAPER_DATA.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONSTANTS
# ============================================================================

V2_MODELS  = ["llama1b_inst", "llama1b_base", "llama8b_inst", "llama8b_base", "gemma9b_inst", "gemma9b_base"]
ACT_TYPES  = ["h", "a", "m"]
CHANCE     = 1 / 8   # 8-class uniform
N_CV_FOLDS = 5

# Phase 1 per-model directories (Exps 05–08)
PHASE1_DIRS = {
    "llama1b":      RESULTS / "05_phase1_llama1b",
    "llama8b_base": RESULTS / "06_phase1_llama3-8b",
    "llama8b_inst": RESULTS / "07_phase1_llama3-8b-instruct",
    "gemma9b":      RESULTS / "08_phase1_gemma2-9b",
}

# Label → key mapping for phase1 comparison summary CSV
PHASE1_LABEL_TO_KEY = {
    "Llama-3.2-1B":       "llama1b",
    "Llama-3-8B (base)":  "llama8b_base",
    "Llama-3-8B (instruct)": "llama8b_inst",
    "Gemma-2-9B":         "gemma9b",
}

FAMILIES = [
    ("Llama-1B", "llama1b_base", "llama1b_inst"),
    ("Llama-8B", "llama8b_base", "llama8b_inst"),
    ("Gemma-9B", "gemma9b_base", "gemma9b_inst"),
]

# ============================================================================
# HELPERS
# ============================================================================

def safe_read(path):
    p = Path(path)
    return pl.read_csv(p) if p.exists() else None

def peak(df, col="auroc_mean"):
    """Return (idx, row_dict) of the max value in col."""
    if df is None or len(df) == 0:
        return None, None
    idx = int(df[col].to_numpy().argmax())
    return idx, df.row(idx, named=True)

def t_ci(mean, std, n=N_CV_FOLDS, alpha=0.05):
    """95% CI via t-distribution (df = n−1). Used when only mean/std available."""
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
    se = std / np.sqrt(n)
    return mean - t_crit * se, mean + t_crit * se

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a proportion p = successes/n."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)

def cohens_d(mean1, std1, mean2, std2):
    """Cohen's d with pooled SD."""
    pooled = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled if pooled > 0 else 0.0

def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

# ============================================================================
# AGG-1: Binary–8-class dissociation table
# ============================================================================

console.print("\n[bold cyan]AGG-1  Binary–8-class dissociation table...[/bold cyan]")

# Set A 8-class ceiling from Phase 1 cross-model comparison
p1_comp = safe_read(RESULTS / "09_phase1_comparison" / "summary.csv")
seta_ceiling = {}
if p1_comp is not None:
    for row in p1_comp.iter_rows(named=True):
        key = PHASE1_LABEL_TO_KEY.get(row["model"])
        if key:
            seta_ceiling[key] = row["p0_best_auroc"]

agg1_rows = []
for mk, exp_dir in PHASE1_DIRS.items():
    bin_df = safe_read(exp_dir / "results_binary.csv")
    cls_df = safe_read(exp_dir / "results_8class.csv")
    if bin_df is None or cls_df is None:
        continue
    _, bin_peak = peak(bin_df)
    _, cls_peak = peak(cls_df)
    agg1_rows.append({
        "model_key":             mk,
        "binary_peak_auroc":     float(bin_peak["auroc_mean"]),
        "binary_peak_layer":     int(bin_peak["layer"]),
        "eightclass_peak_auroc": float(cls_peak["auroc_mean"]),
        "eightclass_peak_layer": int(cls_peak["layer"]),
        "gap":                   float(bin_peak["auroc_mean"]) - float(cls_peak["auroc_mean"]),
        "seta_eightclass_auroc": seta_ceiling.get(mk),
    })

agg1_df = pl.DataFrame(agg1_rows)
agg1_df.write_csv(PAPER_DATA / "dissociation_gap.csv")
console.print(f"  ✓ dissociation_gap.csv  ({len(agg1_rows)} rows)")

# ============================================================================
# AGG-2: B→A transfer peak summary
# ============================================================================

console.print("[bold cyan]AGG-2  B→A transfer peaks...[/bold cyan]")

agg2_rows = []
for mk in V2_MODELS:
    for at in ACT_TYPES:
        df = safe_read(RESULTS / "13_v2_probes" / f"transfer_results_{mk}_{at}.csv")
        if df is None:
            continue
        ab_df = df.filter(pl.col("condition") == "A_to_B")
        ba_df = df.filter(pl.col("condition") == "B_to_A")
        _, ab_pk = peak(ab_df)
        _, ba_pk = peak(ba_df)
        agg2_rows.append({
            "model_key":        mk,
            "act_type":         at,
            "a_to_b_peak_auroc": float(ab_pk["auroc_mean"]) if ab_pk else None,
            "a_to_b_peak_layer": int(ab_pk["layer"])         if ab_pk else None,
            "b_to_a_peak_auroc": float(ba_pk["auroc_mean"]) if ba_pk else None,
            "b_to_a_peak_layer": int(ba_pk["layer"])         if ba_pk else None,
        })

agg2_df = pl.DataFrame(agg2_rows)
agg2_df.write_csv(PAPER_DATA / "transfer_peaks.csv")
console.print(f"  ✓ transfer_peaks.csv  ({len(agg2_rows)} rows)")

# ============================================================================
# AGG-3: Cross-set patching success by layer
# ============================================================================

console.print("[bold cyan]AGG-3  Cross-set patching by layer...[/bold cyan]")

agg3_rows = []
for mk in V2_MODELS:
    for at in ACT_TYPES:
        df = safe_read(RESULTS / "14_v2_patching" / f"patching_crossset_results_{mk}_{at}.csv")
        if df is None:
            continue
        for cond_raw, cond_label in [
            ("crossset_same_emotion", "same"),
            ("crossset_diff_emotion", "diff"),
        ]:
            sub = df.filter(pl.col("condition") == cond_raw)
            if len(sub) == 0:
                continue
            for layer_val in sorted(sub["layer"].unique().to_list()):
                layer_sub  = sub.filter(pl.col("layer") == layer_val)
                n_pairs    = len(layer_sub)
                n_success  = int(layer_sub["success"].cast(pl.Int32).sum())
                norm_depth = float(layer_sub["norm_depth"][0])
                agg3_rows.append({
                    "model_key":    mk,
                    "act_type":     at,
                    "layer":        int(layer_val),
                    "norm_depth":   norm_depth,
                    "condition":    cond_label,
                    "success_rate": n_success / n_pairs,
                    "n_pairs":      n_pairs,
                })

agg3_df = pl.DataFrame(agg3_rows)
agg3_df.write_csv(PAPER_DATA / "crossset_patching_by_layer.csv")
console.print(f"  ✓ crossset_patching_by_layer.csv  ({len(agg3_rows)} rows)")

# ============================================================================
# AGG-4: Knockout critical layer count summary
# ============================================================================

console.print("[bold cyan]AGG-4  Knockout critical layer counts...[/bold cyan]")

agg4_rows = []
for mk in V2_MODELS:
    df = safe_read(RESULTS / "16_v2_knockout" / f"critical_layers_{mk}.csv")
    if df is None:
        continue
    critical = df.filter(pl.col("is_critical") == True)
    for sset in ["seta", "setb"]:
        for at in ACT_TYPES:
            for ko_type in ["zero", "random"]:
                sub = critical.filter(
                    (pl.col("stimulus_set") == sset) &
                    (pl.col("act_type") == at) &
                    (pl.col("knockout_type") == ko_type)
                )
                layers   = sorted(sub["layer"].to_list())
                max_drop = float(sub["accuracy_drop"].max()) if len(sub) > 0 else 0.0
                agg4_rows.append({
                    "model_key":           mk,
                    "stimulus_set":        sset,
                    "act_type":            at,
                    "knockout_type":       ko_type,
                    "n_critical_layers":   len(layers),
                    "critical_layers_list": ",".join(str(l) for l in layers),
                    "max_accuracy_drop":   max_drop,
                })

agg4_df = pl.DataFrame(agg4_rows)
agg4_df.write_csv(PAPER_DATA / "knockout_counts.csv")
console.print(f"  ✓ knockout_counts.csv  ({len(agg4_rows)} rows)")

# ============================================================================
# AGG-5: Base vs instruct comparison table
# ============================================================================

console.print("[bold cyan]AGG-5  Base vs instruct comparison...[/bold cyan]")

summary_df = safe_read(RESULTS / "18_v2_summary" / "summary_table.csv")

# Build lookup dicts
sil_lookup    = {}   # (mk, grouping) → silhouette
perm_lookup   = {}   # (mk, emotion)  → p_permutation
cosine_lookup = {}   # mk → mean (within_emo − cross_emo) cosine gap

for mk in V2_MODELS:
    sil_df = safe_read(RESULTS / "17_v2_geometry" / f"silhouette_scores_{mk}.csv")
    if sil_df is not None:
        for row in sil_df.iter_rows(named=True):
            sil_lookup[(mk, row["grouping"])] = row["silhouette"]

    perm_df = safe_read(RESULTS / "17_v2_geometry" / f"permutation_test_{mk}.csv")
    if perm_df is not None:
        gaps = []
        for row in perm_df.iter_rows(named=True):
            perm_lookup[(mk, row["emotion"])] = row["p_permutation"]
            gaps.append(row["within_emo_cross_domain_sim"] - row["cross_emo_sim"])
        cosine_lookup[mk] = float(np.mean(gaps))

# B→A transfer h-component from AGG-2
ba_lookup = {
    row["model_key"]: row["b_to_a_peak_auroc"]
    for row in agg2_df.filter(pl.col("act_type") == "h").iter_rows(named=True)
}

# n_critical Set B (h, zero) from AGG-4
setb_critical_lookup = {
    row["model_key"]: row["n_critical_layers"]
    for row in agg4_df.filter(
        (pl.col("act_type") == "h") &
        (pl.col("knockout_type") == "zero") &
        (pl.col("stimulus_set") == "setb")
    ).iter_rows(named=True)
}

def summary_val(mk, col):
    if summary_df is None:
        return None
    rows = summary_df.filter(pl.col("model_key") == mk)
    return float(rows[col][0]) if len(rows) > 0 and col in rows.columns else None

METRICS = [
    ("setb_h_peak_auroc", lambda mk: summary_val(mk, "setb_h_peak_auroc")),
    ("ab_transfer_h",     lambda mk: summary_val(mk, "ab_transfer_h")),
    ("ba_transfer_h",     lambda mk: ba_lookup.get(mk)),
    ("cosine_gap",        lambda mk: cosine_lookup.get(mk)),
    ("sil_emotion",       lambda mk: sil_lookup.get((mk, "emotion"))),
    ("sil_set",           lambda mk: sil_lookup.get((mk, "set"))),
    ("grief_p",           lambda mk: perm_lookup.get((mk, "grief"))),
    ("n_critical_setb",   lambda mk: setb_critical_lookup.get(mk)),
]

agg5_rows = []
for family, base_mk, inst_mk in FAMILIES:
    for metric_name, extractor in METRICS:
        bv = extractor(base_mk)
        iv = extractor(inst_mk)
        delta = (iv - bv) if (iv is not None and bv is not None) else None
        agg5_rows.append({
            "family":         family,
            "metric":         metric_name,
            "base_value":     bv,
            "instruct_value": iv,
            "delta":          delta,
        })

pl.DataFrame(agg5_rows).write_csv(PAPER_DATA / "base_vs_instruct.csv")
console.print(f"  ✓ base_vs_instruct.csv  ({len(agg5_rows)} rows)")

# ============================================================================
# STAT-1: 95% CIs on probe AUROC (t-interval from 5-fold CV)
# ============================================================================

console.print("\n[bold cyan]STAT-1  Probe AUROC confidence intervals...[/bold cyan]")

# NOTE: actual condition name for Set B is "within_B_clinical" (not "within_B")
CONDITION_MAP = {
    "within_A":          "within_A",
    "within_B_clinical": "within_B",   # normalise label for paper
    "A_to_B":            "A_to_B",
    "B_to_A":            "B_to_A",
}

probe_df = safe_read(RESULTS / "13_v2_probes" / "all_probe_results.csv")

stat1_rows = []
if probe_df is not None:
    for mk in V2_MODELS:
        for at in ACT_TYPES:
            for cond_raw, cond_label in CONDITION_MAP.items():
                sub = probe_df.filter(
                    (pl.col("model_key") == mk) &
                    (pl.col("act_type") == at) &
                    (pl.col("condition") == cond_raw)
                )
                if len(sub) == 0:
                    continue
                _, pk = peak(sub)
                mean_val = float(pk["auroc_mean"])
                std_val  = float(pk["auroc_std"])
                ci_lo, ci_hi = t_ci(mean_val, std_val)
                stat1_rows.append({
                    "model_key":  mk,
                    "act_type":   at,
                    "condition":  cond_label,
                    "auroc_mean": mean_val,
                    "auroc_std":  std_val,
                    "ci_lower":   ci_lo,
                    "ci_upper":   ci_hi,
                })

stat1_df = pl.DataFrame(stat1_rows)
stat1_df.write_csv(PAPER_DATA / "probe_confidence_intervals.csv")
console.print(f"  ✓ probe_confidence_intervals.csv  ({len(stat1_rows)} rows)")

# ============================================================================
# STAT-2: Significance test — Set A vs Set B AUROC drop
# ============================================================================

console.print("[bold cyan]STAT-2  Set A vs Set B significance tests...[/bold cyan]")

stat2_rows = []
if probe_df is not None:
    for mk in V2_MODELS:
        for at in ACT_TYPES:
            sub_a = probe_df.filter(
                (pl.col("model_key") == mk) & (pl.col("act_type") == at) &
                (pl.col("condition") == "within_A")
            )
            sub_b = probe_df.filter(
                (pl.col("model_key") == mk) & (pl.col("act_type") == at) &
                (pl.col("condition") == "within_B_clinical")
            )
            if len(sub_a) == 0 or len(sub_b) == 0:
                continue
            _, pk_a = peak(sub_a)
            _, pk_b = peak(sub_b)
            mean_a, std_a = float(pk_a["auroc_mean"]), float(pk_a["auroc_std"])
            mean_b, std_b = float(pk_b["auroc_mean"]), float(pk_b["auroc_std"])

            # Welch's t using summary stats, n = N_CV_FOLDS per condition
            n   = N_CV_FOLDS
            se  = np.sqrt(std_a**2 / n + std_b**2 / n)
            t_s = (mean_a - mean_b) / se if se > 0 else 0.0
            # Welch–Satterthwaite df
            df_num = (std_a**2 / n + std_b**2 / n) ** 2
            df_den = (std_a**2 / n) ** 2 / (n - 1) + (std_b**2 / n) ** 2 / (n - 1)
            df_w   = df_num / df_den if df_den > 0 else n - 1
            p_val  = float(2 * scipy_stats.t.sf(abs(t_s), df=df_w))

            stat2_rows.append({
                "model_key":     mk,
                "act_type":      at,
                "seta_auroc":    mean_a,
                "setb_auroc":    mean_b,
                "drop":          mean_a - mean_b,
                "t_statistic":   round(t_s, 4),
                "df":            round(df_w, 2),
                "p_value":       p_val,
                "significant_05": p_val < 0.05,
            })

pl.DataFrame(stat2_rows).write_csv(PAPER_DATA / "set_comparison_tests.csv")
console.print(f"  ✓ set_comparison_tests.csv  ({len(stat2_rows)} rows)")

# ============================================================================
# STAT-3: Binomial CIs on patching success (Wilson score)
# ============================================================================

console.print("[bold cyan]STAT-3  Patching confidence intervals...[/bold cyan]")

stat3_rows = []
for mk in V2_MODELS:
    for at in ACT_TYPES:
        # Within-set (seta, setb)
        for sset in ["seta", "setb"]:
            df = safe_read(RESULTS / "14_v2_patching" / f"patching_success_by_layer_{mk}_{sset}_{at}.csv")
            if df is None or len(df) == 0:
                continue
            _, pk = peak(df, col="success_rate")
            sr    = float(pk["success_rate"])
            n     = int(pk["n_pairs"])
            layer = int(pk["layer"])
            ci_lo, ci_hi = wilson_ci(round(sr * n), n)
            stat3_rows.append({
                "model_key":    mk,
                "condition":    f"within_{sset}",
                "act_type":     at,
                "peak_layer":   layer,
                "success_rate": sr,
                "n_pairs":      n,
                "ci_lower":     ci_lo,
                "ci_upper":     ci_hi,
                "above_chance": ci_lo > CHANCE,
            })

        # Cross-set (from AGG-3)
        for cond_label in ["same", "diff"]:
            sub = agg3_df.filter(
                (pl.col("model_key") == mk) & (pl.col("act_type") == at) &
                (pl.col("condition") == cond_label)
            )
            if len(sub) == 0:
                continue
            _, pk = peak(sub, col="success_rate")
            sr    = float(pk["success_rate"])
            n     = int(pk["n_pairs"])
            layer = int(pk["layer"])
            ci_lo, ci_hi = wilson_ci(round(sr * n), n)
            stat3_rows.append({
                "model_key":    mk,
                "condition":    f"crossset_{cond_label}",
                "act_type":     at,
                "peak_layer":   layer,
                "success_rate": sr,
                "n_pairs":      n,
                "ci_lower":     ci_lo,
                "ci_upper":     ci_hi,
                "above_chance": ci_lo > CHANCE,
            })

pl.DataFrame(stat3_rows).write_csv(PAPER_DATA / "patching_confidence_intervals.csv")
console.print(f"  ✓ patching_confidence_intervals.csv  ({len(stat3_rows)} rows)")

# ============================================================================
# STAT-4: Multiple comparison correction — Benjamini-Hochberg FDR
# ============================================================================

console.print("[bold cyan]STAT-4  Permutation test FDR correction...[/bold cyan]")

stat4_rows = []
for mk in V2_MODELS:
    df = safe_read(RESULTS / "17_v2_geometry" / f"permutation_test_{mk}.csv")
    if df is None:
        continue
    for row in df.iter_rows(named=True):
        stat4_rows.append({
            "model_key":                   mk,
            "emotion":                     row["emotion"],
            "p_raw":                       row["p_permutation"],
            "within_emo_cross_domain_sim": row["within_emo_cross_domain_sim"],
            "cross_emo_sim":               row["cross_emo_sim"],
        })

if stat4_rows:
    stat4_df = pl.DataFrame(stat4_rows)
    p_vals   = stat4_df["p_raw"].to_numpy()
    n_tests  = len(p_vals)
    order    = np.argsort(p_vals)
    ranks    = np.empty_like(order)
    ranks[order] = np.arange(1, n_tests + 1)
    # BH: q = p * n / rank, enforce monotonicity from the right
    q_raw = p_vals * n_tests / ranks
    q_bh  = np.minimum.accumulate(q_raw[order[::-1]])[::-1][np.argsort(order)]
    q_bh  = np.minimum(q_bh, 1.0)
    stat4_df = stat4_df.with_columns([
        pl.Series("q_bh", q_bh),
        pl.Series("significant_05", q_bh < 0.05),
    ])
    stat4_df.write_csv(PAPER_DATA / "permutation_corrected.csv")
    n_sig = int((q_bh < 0.05).sum())
    console.print(f"  ✓ permutation_corrected.csv  ({n_tests} tests, {n_sig} survive FDR q<0.05)")

# ============================================================================
# STAT-5: Effect sizes
# ============================================================================

console.print("[bold cyan]STAT-5  Effect sizes...[/bold cyan]")

stat5_rows = []

# 1. Set A vs Set B AUROC drop — Cohen's d
if probe_df is not None:
    for mk in V2_MODELS:
        for at in ACT_TYPES:
            sub_a = probe_df.filter(
                (pl.col("model_key") == mk) & (pl.col("act_type") == at) & (pl.col("condition") == "within_A")
            )
            sub_b = probe_df.filter(
                (pl.col("model_key") == mk) & (pl.col("act_type") == at) & (pl.col("condition") == "within_B_clinical")
            )
            if len(sub_a) == 0 or len(sub_b) == 0:
                continue
            _, pa = peak(sub_a)
            _, pb = peak(sub_b)
            d = cohens_d(float(pa["auroc_mean"]), float(pa["auroc_std"]),
                         float(pb["auroc_mean"]), float(pb["auroc_std"]))
            stat5_rows.append({
                "comparison":    "setA_vs_setB_AUROC_drop",
                "model_key":     mk,
                "act_type":      at,
                "value1":        float(pa["auroc_mean"]),
                "value2":        float(pb["auroc_mean"]),
                "raw_diff":      float(pa["auroc_mean"]) - float(pb["auroc_mean"]),
                "effect_size":   d,
                "effect_metric": "cohens_d",
            })

# 2. Binary vs 8-class gap — raw difference (already on AUROC scale)
for row in agg1_rows:
    stat5_rows.append({
        "comparison":    "binary_vs_8class_gap",
        "model_key":     row["model_key"],
        "act_type":      "n/a",
        "value1":        row["binary_peak_auroc"],
        "value2":        row["eightclass_peak_auroc"],
        "raw_diff":      row["gap"],
        "effect_size":   row["gap"],
        "effect_metric": "raw_difference",
    })

# 3. Within-emotion vs cross-emotion cosine similarity — Cohen's d
for mk in V2_MODELS:
    perm_mk = safe_read(RESULTS / "17_v2_geometry" / f"permutation_test_{mk}.csv")
    if perm_mk is None:
        continue
    within_vals = perm_mk["within_emo_cross_domain_sim"].to_numpy()
    cross_vals  = perm_mk["cross_emo_sim"].to_numpy()
    d = cohens_d(float(np.mean(within_vals)), float(np.std(within_vals) or 1e-9),
                 float(np.mean(cross_vals)),  float(np.std(cross_vals) or 1e-9))
    stat5_rows.append({
        "comparison":    "within_vs_cross_emotion_cosine",
        "model_key":     mk,
        "act_type":      "h",
        "value1":        float(np.mean(within_vals)),
        "value2":        float(np.mean(cross_vals)),
        "raw_diff":      float(np.mean(within_vals - cross_vals)),
        "effect_size":   d,
        "effect_metric": "cohens_d",
    })

# 4. Patching success vs chance — Cohen's h
stat3_df = pl.DataFrame(stat3_rows)
for row in stat3_df.filter(
    (pl.col("condition") == "within_seta") & (pl.col("act_type") == "h")
).iter_rows(named=True):
    h = cohens_h(row["success_rate"], CHANCE)
    stat5_rows.append({
        "comparison":    "patching_success_vs_chance",
        "model_key":     row["model_key"],
        "act_type":      "h",
        "value1":        row["success_rate"],
        "value2":        CHANCE,
        "raw_diff":      row["success_rate"] - CHANCE,
        "effect_size":   h,
        "effect_metric": "cohens_h",
    })

pl.DataFrame(stat5_rows).write_csv(PAPER_DATA / "effect_sizes.csv")
console.print(f"  ✓ effect_sizes.csv  ({len(stat5_rows)} rows)")

# ============================================================================
# STAT-6: Power analysis for permutation tests
# ============================================================================

console.print("[bold cyan]STAT-6  Permutation test power analysis (simulation)...[/bold cyan]")

# Observed cosine similarities are in the range 0.975–0.985 with tiny gaps (~0.002–0.005).
# Simulate: 3 within-emotion cross-domain domain pairs per emotion (from 4 vignettes × 3 domains).
# True effect = the additional cosine advantage within-emotion vs cross-emotion.

BASE_SIM   = 0.978    # approximate observed mean cosine sim
BASE_STD   = 0.006    # approximate observed within-model SD
N_DOMAINS  = 3        # domain pairs per emotion
N_CROSS    = 21       # cross-emotion pairs (7 other emotions × 3)
N_SIM      = 1000     # simulated samples per effect size
N_PERM     = 500      # permutation draws per simulated test
ALPHA      = 0.05

def permutation_test_power(true_effect):
    n_reject = 0
    for _ in range(N_SIM):
        within = rng.normal(BASE_SIM + true_effect, BASE_STD, N_DOMAINS)
        cross  = rng.normal(BASE_SIM,               BASE_STD, N_CROSS)
        obs    = np.mean(within) - np.mean(cross)
        combined = np.concatenate([within, cross])
        null = np.array([
            np.mean(rng.permutation(combined)[:N_DOMAINS]) - np.mean(rng.permutation(combined)[N_DOMAINS:])
            for _ in range(N_PERM)
        ])
        if (null >= obs).mean() < ALPHA:
            n_reject += 1
    return n_reject / N_SIM

# Effect sizes from 0.001 to 0.020 (scale of observed cosine gaps)
effect_sizes = np.round(np.arange(0.001, 0.021, 0.001), 3)
stat6_rows = []
for eff in effect_sizes:
    power = permutation_test_power(float(eff))
    stat6_rows.append({
        "true_effect":      float(eff),
        "power":            round(power, 3),
        "detectable_at_80": power >= 0.80,
    })
    console.print(f"    effect={eff:.3f} → power={power:.3f}")

stat6_df = pl.DataFrame(stat6_rows)
stat6_df.write_csv(PAPER_DATA / "permutation_power.csv")
mde = next((r["true_effect"] for r in stat6_rows if r["detectable_at_80"]), None)
console.print(f"  ✓ permutation_power.csv  — min detectable effect @ 80% power: {mde}")

# ============================================================================
# SUMMARY
# ============================================================================

console.print("\n[bold green]All done![/bold green]")
console.print(f"Outputs → {PAPER_DATA}\n")
for f in sorted(PAPER_DATA.glob("*.csv")):
    n = pl.read_csv(f).shape[0]
    console.print(f"  {f.name:<48} {n:>4} rows")
