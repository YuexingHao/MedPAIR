"""Paired significance testing for the combined MedPAIR figure.

The stars previously drawn on the figure came from an UNPAIRED two-proportion
z-test whose sample size was reverse-engineered from the rounded accuracy
printed on the plot, with no multiple-comparison correction, at thresholds
(p<0.05 for one star) that did not match the figure caption.

Original and best-labeler accuracy are measured on the same questions, so the
correct test is McNemar's, using the discordant pairs only. Significance is
Bonferroni-corrected at two levels:

    *   p < 0.01 / 7  = 1.43e-3   7 comparisons within a dataset (6 models + human)
    **  p < 0.01 / 28 = 3.57e-4   all 7 x 4 comparisons treated as one family
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medpair_accuracy_lib as L  # noqa: E402

ALPHA = 0.01
N_WITHIN_DATASET = 7  # 6 models + human
N_DATASETS = 4
ALPHA_WITHIN = ALPHA / N_WITHIN_DATASET
ALPHA_FAMILY = ALPHA / (N_WITHIN_DATASET * N_DATASETS)

ORIGINAL_FILES = {
    "GPT5": "gpt5_predictions_Original_Accuracy.csv",
    "GPT4o": "gpt4o_predictions_Original_Accuracy.csv",
    "Llama-70B": "Llama70B_ORIGINAL_predictions.csv",
    "Qwen-72B": "Qwen_72B_predictions_ORIGINAL.csv",
    "Qwen-14B": "Qwen_14B_predictions_ORIGINAL.csv",
    "MedGemma-27B": "MedGemma27B_predictions_ORIGINAL.csv",
}

# library model name -> prettified x-tick label used on the figure
MODEL_TO_PRETTY = {
    "GPT5": "GPT-5",
    "GPT4o": "GPT-4o",
    "Llama-70B": "Llama-70B",
    "Qwen-72B": "Qwen-72B",
    "Qwen-14B": "Qwen-14B",
    "MedGemma-27B": "MedGemma-27B",
}

CONDITION_BLOCKS = [
    "Trainee Removed (Low+Irr Labeled by Trainees)",
    "Qwen-14B Removed",
    "Qwen-72B Removed",
    "Llama-70B Removed",
    "GPT-4o Removed",
    "GPT-5 Removed",
    "MedGemma-27B Removed",
]


def mcnemar_p(orig_ok: pd.Series, best_ok: pd.Series) -> tuple[float, int, int]:
    """Exact two-sided McNemar on the discordant pairs."""
    common = orig_ok.index.intersection(best_ok.index)
    a = orig_ok.loc[common].astype(bool)
    b_ = best_ok.loc[common].astype(bool)
    b = int((a & ~b_).sum())   # original right, best wrong
    c = int((~a & b_).sum())   # original wrong, best right
    n = b + c
    if n == 0:
        return 1.0, b, c
    return float(binomtest(b, n, 0.5).pvalue), b, c


def stars_for(p: float) -> str:
    if p < ALPHA_FAMILY:
        return "**"
    if p < ALPHA_WITHIN:
        return "*"
    return ""


def compute(verbose: bool = False) -> dict[tuple[str, str], dict]:
    """{(scope, pretty_model): {p, stars, best, b, c, n_paired}}"""
    out: dict[tuple[str, str], dict] = {}
    scopes = ["mmlu", "jama", "medxpert", "medbullets", "overall"]

    for model, pretty in MODEL_TO_PRETTY.items():
        orig = L.per_question(model, ORIGINAL_FILES[model])
        if orig is None:
            continue
        conds: dict[str, pd.DataFrame] = {}
        for block in CONDITION_BLOCKS:
            if (block, model) in L.EXCLUDED_CELLS:
                continue
            f = L.BLOCKS[block].get(model)
            if not f:
                continue
            r = L.per_question(model, f)
            if r is not None:
                conds[block] = r
        if not conds:
            continue

        for scope in scopes:
            o = orig if scope == "overall" else orig[orig["source"] == scope]
            if o.empty:
                continue
            best_label, best_ok, best_acc = None, None, -1.0
            for label, r in conds.items():
                rr = r if scope == "overall" else r[r["source"] == scope]
                if rr.empty:
                    continue
                acc = 100.0 * rr["ok"].mean()
                if acc > best_acc:
                    best_label, best_ok, best_acc = label, rr["ok"], acc
            if best_ok is None:
                continue
            p, b, c = mcnemar_p(o["ok"], best_ok)
            out[(scope, pretty)] = {
                "p": p,
                "stars": stars_for(p),
                "best": best_label,
                "best_acc": best_acc,
                "orig_acc": 100.0 * o["ok"].mean(),
                "b": b,
                "c": c,
                "n_paired": int(o["ok"].index.intersection(best_ok.index).size),
            }
            if verbose:
                print(
                    f"{scope:11} {pretty:13} orig={out[(scope,pretty)]['orig_acc']:5.1f} "
                    f"best={best_acc:5.1f} ({best_label[:22]:22}) "
                    f"b={b:3} c={c:3} p={p:.3e} {out[(scope,pretty)]['stars'] or '-'}"
                )
    return out


if __name__ == "__main__":
    print(f"alpha within-dataset (0.01/7)  = {ALPHA_WITHIN:.3e}   -> *")
    print(f"alpha family (0.01/28)         = {ALPHA_FAMILY:.3e}   -> **\n")
    res = compute(verbose=True)
    n_star = sum(1 for v in res.values() if v["stars"] == "*")
    n_dstar = sum(1 for v in res.values() if v["stars"] == "**")
    print(f"\ncells={len(res)}  '*'={n_star}  '**'={n_dstar}  ns={len(res)-n_star-n_dstar}")
