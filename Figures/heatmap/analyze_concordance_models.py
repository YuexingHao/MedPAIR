#!/usr/bin/env python3
"""Statistical analysis for concordance across Human/SR/CC models.

This script reconstructs per-Origin high-sentence sets from the original
upstream files used by update_cc_concordance_and_heatmap.py, then computes:

1) Pairwise Jaccard summary among all 10 labelers (mean, SD, 95% bootstrap CI).
2) Model-level summaries (average agreement with all others and with Human).
3) Pairwise significance tests for Human-alignment differences
   (paired t-test with Holm correction).
"""
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path("/orcd/home/002/yuexing/NeuRIPS25")
MV_PATH = ROOT / "Physician_Labels/Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv"

CC_SUMMARY = {
    "Qwen-14B (CC)": ROOT / "Merge/attribution/14b_contextcite_analysis/qwen14b_contextcite_topk_summary.csv",
    "Qwen-72B (CC)": ROOT / "Merge/attribution/qwen72b_contextcite/qwen72b_contextcite_topk_summary.csv",
    "Llama-70B (CC)": ROOT / "Merge/attribution/llama70b_contextcite/llama70b_contextcite_topk_summary.csv",
}
CC_TOPK_COL = {
    "Qwen-14B (CC)": "qwen14b_topK_sentence_ids",
    "Qwen-72B (CC)": "qwen72b_topK_sentence_ids",
    "Llama-70B (CC)": "llama70b_topK_sentence_ids",
}

SR_REFS = {
    "GPT-4o (SR)": (ROOT / "Physician_Labels/results/GPT4o_MatchRate.csv", "ID", "label_"),
    "GPT-5 (SR)": (ROOT / "Physician_Labels/results/GPT5_MatchRate.csv", "ID_corr", "q"),
    "Qwen-14B (SR)": (ROOT / "Physician_Labels/results/[SR]Qwen14B_annotated_MedPAIR_relevancy.csv", "ID", "q"),
    "Qwen-72B (SR)": (ROOT / "Physician_Labels/results/[SR]Qwen72B_annotated_MedPAIR_relevancy.csv", "ID", "q"),
    "Llama-70B (SR)": (ROOT / "Physician_Labels/results/[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv", "ID", "q"),
    "MedGemma-27B (SR)": (
        ROOT / "After_PT_Removal/MedGemma-27b-text-it/data/raw/MedGemma_SR_Match_Rate.csv",
        "ID",
        "q",
    ),
}

ORDER = [
    "Human",
    "GPT-4o (SR)",
    "GPT-5 (SR)",
    "MedGemma-27B (SR)",
    "Qwen-14B (SR)",
    "Qwen-72B (SR)",
    "Llama-70B (SR)",
    "Qwen-14B (CC)",
    "Qwen-72B (CC)",
    "Llama-70B (CC)",
]


def parse_ids(text: str) -> set[int]:
    if not isinstance(text, str) or not text.strip():
        return set()
    out = set()
    for tok in re.split(r"[,;.\s]+", text):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out


def load_human_high() -> dict[str, set[int]]:
    mv = pd.read_csv(MV_PATH, low_memory=False)
    sent_cols = [c for c in mv.columns if re.fullmatch(r"Sentence \d+", c)]
    out: dict[str, set[int]] = {}
    for _, row in mv.iterrows():
        origin = str(row["Origin"]).strip()
        high = set()
        for c in sent_cols:
            val = row[c]
            if pd.notna(val):
                try:
                    if float(val) > 0.66:
                        high.add(int(c.split()[-1]))
                except Exception:
                    pass
        out[origin] = high
    return out


def load_cc_topk(path: Path, topk_col: str) -> dict[str, set[int]]:
    df = pd.read_csv(path, low_memory=False)
    out = {}
    for _, row in df.iterrows():
        out[str(row["Origin"]).strip()] = parse_ids(row.get(topk_col, ""))
    return out


def load_sr_high(path: Path, id_col: str, col_prefix: str) -> dict[str, set[int]]:
    df = pd.read_csv(path, low_memory=False)
    cols = [f"{col_prefix}{i}" for i in range(1, 22) if f"{col_prefix}{i}" in df.columns]
    if id_col not in df.columns:
        alt = [c for c in ("ID", "ID_corr", "Origin") if c in df.columns]
        id_col = alt[0] if alt else id_col

    out = {}
    for _, row in df.iterrows():
        origin = str(row[id_col]).strip()
        high = set()
        for i, col in enumerate(cols, start=1):
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str) and "HIGH" in val.upper() and "IRREL" not in val.upper():
                high.add(i)
        out[origin] = high
    return out


def bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 3000) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "Figures/heatmap/_analysis_outputs"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed.")
    parser.add_argument("--n-boot", type=int, default=3000, help="Bootstrap samples for 95% CI.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Build high-sentence sets
    labeler_high: dict[str, dict[str, set[int]]] = {"Human": load_human_high()}
    for model, path in CC_SUMMARY.items():
        labeler_high[model] = load_cc_topk(path, CC_TOPK_COL[model])
    for model, (path, id_col, prefix) in SR_REFS.items():
        labeler_high[model] = load_sr_high(path, id_col, prefix)

    common_origins = set(labeler_high["Human"].keys())
    for model in ORDER:
        if model == "Human":
            continue
        common_origins &= set(labeler_high[model].keys())
    common_origins = sorted(common_origins)
    print(f"Common origins across all model files: {len(common_origins)}")

    # Pairwise per-origin Jaccard values
    pair_values: dict[tuple[str, str], np.ndarray] = {}
    pair_rows = []
    for a, b in combinations(ORDER, 2):
        vals = []
        for origin in common_origins:
            A = labeler_high[a].get(origin, set())
            B = labeler_high[b].get(origin, set())
            union = A | B
            if not union:
                continue
            vals.append(100.0 * len(A & B) / len(union))
        arr = np.array(vals, dtype=float)
        pair_values[(a, b)] = arr
        ci_lo, ci_hi = bootstrap_ci_mean(arr, rng, n_boot=args.n_boot)
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n_origins_used": len(arr),
                "mean_jaccard_pct": float(arr.mean()) if len(arr) else np.nan,
                "sd_jaccard_pct": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
                "ci95_low_jaccard_pct": ci_lo,
                "ci95_high_jaccard_pct": ci_hi,
            }
        )
    pair_df = pd.DataFrame(pair_rows).sort_values("mean_jaccard_pct", ascending=False)
    pair_df.to_csv(outdir / "pairwise_jaccard_summary.csv", index=False)

    # Model-level summary
    model_rows = []
    for model in ORDER:
        pair_means = []
        for other in ORDER:
            if model == other:
                continue
            key = (model, other) if (model, other) in pair_values else (other, model)
            pair_means.append(pair_values[key].mean())

        if model == "Human":
            human_agreement = 100.0
        else:
            hkey = ("Human", model) if ("Human", model) in pair_values else (model, "Human")
            human_agreement = float(pair_values[hkey].mean())

        model_rows.append(
            {
                "model": model,
                "avg_pairwise_jaccard_to_others_pct": float(np.mean(pair_means)),
                "jaccard_with_human_pct": human_agreement,
            }
        )
    model_df = pd.DataFrame(model_rows).sort_values("avg_pairwise_jaccard_to_others_pct", ascending=False)
    model_df.to_csv(outdir / "model_level_summary.csv", index=False)

    # Pairwise tests for Human alignment differences
    human_alignment = {}
    for model in ORDER:
        if model == "Human":
            continue
        key = ("Human", model) if ("Human", model) in pair_values else (model, "Human")
        human_alignment[model] = pair_values[key]

    test_rows = []
    model_keys = list(human_alignment.keys())
    for i, m1 in enumerate(model_keys):
        for m2 in model_keys[i + 1 :]:
            v1, v2 = human_alignment[m1], human_alignment[m2]
            if len(v1) != len(v2):
                n = min(len(v1), len(v2))
                v1 = v1[:n]
                v2 = v2[:n]
            diff = v1 - v2
            _, pval = ttest_rel(v1, v2, nan_policy="omit")
            ci_lo, ci_hi = bootstrap_ci_mean(diff, rng, n_boot=args.n_boot)
            test_rows.append(
                {
                    "model_better_on_human": m1 if diff.mean() > 0 else m2,
                    "model_worse_on_human": m2 if diff.mean() > 0 else m1,
                    "mean_diff_jaccard_with_human_pct": float(abs(diff.mean())),
                    "ci95_low_diff_pct": float(min(abs(ci_lo), abs(ci_hi))),
                    "ci95_high_diff_pct": float(max(abs(ci_lo), abs(ci_hi))),
                    "n_origins_used": len(diff),
                    "p_value_paired_t": float(pval),
                }
            )

    test_df = pd.DataFrame(test_rows).sort_values("p_value_paired_t").reset_index(drop=True)
    m = len(test_df)
    test_df["holm_alpha"] = 0.05 / (m - test_df.index)
    test_df["significant_holm_0p05"] = test_df["p_value_paired_t"] <= test_df["holm_alpha"]
    test_df.to_csv(outdir / "human_alignment_pairwise_tests_holm.csv", index=False)

    print(f"Wrote {outdir / 'pairwise_jaccard_summary.csv'}")
    print(f"Wrote {outdir / 'model_level_summary.csv'}")
    print(f"Wrote {outdir / 'human_alignment_pairwise_tests_holm.csv'}")


if __name__ == "__main__":
    main()
