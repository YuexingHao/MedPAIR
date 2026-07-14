"""Recompute concordance using the new top-K ContextCite labels.

Updates two artifacts:

1.  ``Figures/concordance_rate/concordance_933_by_origin_long.csv`` and
    ``..._summary.csv`` -- replaces rows for ``Qwen-14B_CC``, ``Qwen-72B_CC``,
    ``Llama-70B_CC`` with new per-Origin concordance values computed as
    ``|model_topK ∩ trainee_High| / |trainee_High| * 100``.  Other rows are
    preserved.

2.  ``Figures/heatmap/concordance_10x10_SR_CC_jaccard_high.csv`` -- rebuilds
    the full 10x10 pairwise Jaccard matrix on the High sentence sets across
    all shared Origins.  ``Human`` uses trainee MV > 0.66; CC labelers use
    the new top-K summaries; SR labelers use the existing q_/label_ columns.

After writing CSVs, re-runs the two plotting scripts to regenerate
``concordance_933_summary_bar.png`` and ``concordance_heatmap_transposed.png``.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/orcd/home/002/yuexing/NeuRIPS25")
MV_PATH = ROOT / "Physician_Labels/Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv"

CC_SUMMARY = {
    "Qwen-14B_CC":  ROOT / "Merge/attribution/14b_contextcite_analysis/qwen14b_contextcite_topk_summary.csv",
    "Qwen-72B_CC":  ROOT / "Merge/attribution/qwen72b_contextcite/qwen72b_contextcite_topk_summary.csv",
    "Llama-70B_CC": ROOT / "Merge/attribution/llama70b_contextcite/llama70b_contextcite_topk_summary.csv",
}
CC_TOPK_COL = {  # column name inside each summary CSV
    "Qwen-14B_CC":  "qwen14b_topK_sentence_ids",
    "Qwen-72B_CC":  "qwen72b_topK_sentence_ids",
    "Llama-70B_CC": "llama70b_topK_sentence_ids",
}

SR_REFS = {
    "GPT-4o (SR)":      (ROOT / "Physician_Labels/results/GPT4o_MatchRate.csv", "ID", "label_"),
    "GPT-5 (SR)":       (ROOT / "Physician_Labels/results/GPT5_MatchRate.csv", "ID_corr", "q"),
    "Qwen-14B (SR)":    (ROOT / "Physician_Labels/results/[SR]Qwen14B_annotated_MedPAIR_relevancy.csv", "ID", "q"),
    "Qwen-72B (SR)":    (ROOT / "Physician_Labels/results/[SR]Qwen72B_annotated_MedPAIR_relevancy.csv", "ID", "q"),
    "Llama-70B (SR)":   (ROOT / "Physician_Labels/results/[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv", "ID", "q"),
    "MedGemma-27B (SR)":(ROOT / "After_PT_Removal/MedGemma-27b-text-it/data/raw/MedGemma_SR_Match_Rate.csv", "ID", "q"),
}

CC_LABEL = {
    "Qwen-14B_CC":  "Qwen-14B (CC)",
    "Qwen-72B_CC":  "Qwen-72B (CC)",
    "Llama-70B_CC": "Llama-70B (CC)",
}

LONG_CSV = ROOT / "Figures/concordance_rate/concordance_933_by_origin_long.csv"
SUMMARY_CSV = ROOT / "Figures/concordance_rate/concordance_933_by_origin_summary.csv"
HEATMAP_CSV = ROOT / "Figures/heatmap/concordance_10x10_SR_CC_jaccard_high.csv"


def _parse_ids(s) -> set[int]:
    if not isinstance(s, str) or not s.strip():
        return set()
    out = set()
    for tok in re.split(r"[,;.\s]+", s):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out


# ---------------------------------------------------------------------------
# Build High sentence sets: Origin -> set[int]
# ---------------------------------------------------------------------------

def load_human_high(mv: pd.DataFrame) -> dict[str, set[int]]:
    out = {}
    sent_cols = [c for c in mv.columns if re.fullmatch(r"Sentence \d+", c)]
    for _, r in mv.iterrows():
        o = str(r["Origin"]).strip()
        highs = set()
        for c in sent_cols:
            v = r[c]
            if pd.notna(v):
                try:
                    if float(v) > 0.66:
                        highs.add(int(c.split()[-1]))
                except Exception:
                    pass
        out[o] = highs
    return out


def load_cc_topk(path: Path, topk_col: str) -> dict[str, set[int]]:
    df = pd.read_csv(path, low_memory=False)
    out = {}
    for _, r in df.iterrows():
        o = str(r["Origin"]).strip()
        out[o] = _parse_ids(r.get(topk_col, ""))
    return out


def load_sr_high(path: Path, id_col: str, col_prefix: str) -> dict[str, set[int]]:
    df = pd.read_csv(path, low_memory=False)
    cols = [f"{col_prefix}{i}" for i in range(1, 22) if f"{col_prefix}{i}" in df.columns]
    if id_col not in df.columns:
        alt = [c for c in ("ID", "ID_corr", "Origin") if c in df.columns]
        id_col = alt[0] if alt else id_col
    out = {}
    for _, r in df.iterrows():
        o = str(r[id_col]).strip()
        highs = set()
        for i, c in enumerate(cols, start=1):
            v = r[c]
            if pd.isna(v):
                continue
            if isinstance(v, str) and "HIGH" in v.upper() and "IRREL" not in v.upper():
                highs.add(i)
        out[o] = highs
    return out


# ---------------------------------------------------------------------------
# Update long + summary CSVs for CC rows
# ---------------------------------------------------------------------------

def update_long_and_summary(human_H: dict[str, set[int]],
                            cc_high: dict[str, dict[str, set[int]]]) -> None:
    long = pd.read_csv(LONG_CSV, low_memory=False)
    keep = long[~long["model"].isin(CC_SUMMARY.keys())].copy()

    new_rows = []
    for cc_key, cc_origin_H in cc_high.items():
        for o, cc_set in cc_origin_H.items():
            h_set = human_H.get(o, set())
            if not h_set:
                continue   # no trainee-High -> drop row (matches restrict_to_933)
            inter = len(cc_set & h_set)
            pct = 100.0 * inter / len(h_set)
            new_rows.append({
                "model": cc_key,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": len(h_set),
                "missing_model_row": False,
                "model_csv": str(CC_SUMMARY[cc_key]),
            })
    new_long = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True)
    new_long.to_csv(LONG_CSV, index=False)
    print(f"Wrote {LONG_CSV} ({len(new_long)} rows)")

    # Rebuild summary CSV
    summary = pd.read_csv(SUMMARY_CSV, low_memory=False)
    for cc_key in cc_high.keys():
        rows = new_long[new_long["model"] == cc_key]
        rows = rows[rows["concordance_pct"].notna()]
        n = len(rows)
        mean = rows["concordance_pct"].mean() if n else np.nan
        sd   = rows["concordance_pct"].std(ddof=1) if n > 1 else np.nan
        mean_pairs = rows["n_sentence_pairs"].mean() if n else np.nan
        mask = summary["model"] == cc_key
        if mask.any():
            summary.loc[mask, "n_origins_scored"] = n
            summary.loc[mask, "n_origins_no_valid_sentence_pairs"] = 933 - n
            summary.loc[mask, "mean_concordance_pct"] = mean
            summary.loc[mask, "sd_concordance_pct"] = sd
            summary.loc[mask, "mean_sentence_pairs_per_origin"] = mean_pairs
            summary.loc[mask, "model_csv"] = str(CC_SUMMARY[cc_key])
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}")


# ---------------------------------------------------------------------------
# Rebuild 10x10 heatmap CSV (Human + 9 LLM variants)
# ---------------------------------------------------------------------------

HEATMAP_ORDER = [
    "Human",
    "GPT-4o (SR)", "GPT-5 (SR)", "MedGemma-27B (SR)",
    "Qwen-14B (SR)", "Qwen-72B (SR)", "Llama-70B (SR)",
    "Qwen-14B (CC)", "Qwen-72B (CC)", "Llama-70B (CC)",
]


def rebuild_heatmap(labeler_high: dict[str, dict[str, set[int]]]) -> None:
    labelers = HEATMAP_ORDER
    common_origins = sorted(
        set.intersection(*[set(labeler_high[l].keys()) for l in labelers if l != "Human"]))
    common_origins = [o for o in common_origins if o in labeler_high["Human"]]
    # Keep only origins where every labeler has at least one High sentence
    # so pairwise Jaccard is defined.
    filtered = [o for o in common_origins
                if all(labeler_high[l].get(o) for l in labelers)]

    def pairwise_jaccard(a_name: str, b_name: str) -> float:
        a_map = labeler_high[a_name]
        b_map = labeler_high[b_name]
        vals = []
        for o in common_origins:
            A = a_map.get(o, set())
            B = b_map.get(o, set())
            union = A | B
            if not union:
                continue
            vals.append(len(A & B) / len(union))
        return 100.0 * float(np.mean(vals)) if vals else np.nan

    mat = np.full((len(labelers), len(labelers)), np.nan)
    for i, a in enumerate(labelers):
        for j, b in enumerate(labelers):
            if a == b:
                mat[i, j] = 100.0
            elif i < j:
                mat[i, j] = pairwise_jaccard(a, b)
            else:
                mat[i, j] = mat[j, i]

    df = pd.DataFrame(mat, index=labelers, columns=labelers).round(2)
    df.to_csv(HEATMAP_CSV)
    print(f"Wrote {HEATMAP_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading Human High (MV > 0.66) ...")
    mv = pd.read_csv(MV_PATH, low_memory=False)
    human_H = load_human_high(mv)
    print(f"  Human: {len(human_H)} origins")

    print("Loading CC top-K ...")
    cc_high = {k: load_cc_topk(p, CC_TOPK_COL[k]) for k, p in CC_SUMMARY.items()}
    for k, v in cc_high.items():
        print(f"  {k}: {len(v)} origins")

    print("Loading SR labelers ...")
    sr_high = {}
    for label, (path, id_col, prefix) in SR_REFS.items():
        sr_high[label] = load_sr_high(path, id_col, prefix)
        print(f"  {label}: {len(sr_high[label])} origins")

    # --- bar-chart (long + summary) ---
    update_long_and_summary(human_H, cc_high)

    # --- heatmap (10x10 pairwise Jaccard) ---
    labeler_high = {
        "Human": human_H,
        **{CC_LABEL[k]: v for k, v in cc_high.items()},
        **sr_high,
    }
    rebuild_heatmap(labeler_high)

    # --- rerun figure scripts ---
    print("Regenerating bar chart ...")
    subprocess.check_call([
        sys.executable,
        str(ROOT / "Figures/concordance_rate/make_performance_bar_charts.py"),
    ])

    print("Regenerating heatmap ...")
    subprocess.check_call([
        sys.executable,
        str(ROOT / "Figures/heatmap/make_concordance_heatmap_transposed.py"),
    ])


if __name__ == "__main__":
    main()
