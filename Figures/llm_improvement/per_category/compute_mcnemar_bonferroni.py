#!/usr/bin/env python3
"""McNemar exact test with Bonferroni correction for the MedPAIR combined figure.

Uses ``Physician_Labels/Mar2_2026_Data/933_llm_human_round1_round2_responses.csv``
which provides paired Round 1 vs Round 2 predictions for every model + human
majority on the same 933 items.

Test: McNemar's exact two-sided test on the (b, c) discordant-pair table,
``b`` = right in R1 and wrong in R2, ``c`` = wrong in R1 and right in R2.
Exact p = min(1, 2 * P(Binom(b + c, 0.5) <= min(b, c))).

Bonferroni correction: across all 35 cells (7 rows * 5 columns of the figure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent.parent
_SRC = (
    _WORKSPACE_ROOT
    / "Physician_Labels"
    / "Mar2_2026_Data"
    / "933_llm_human_round1_round2_responses.csv"
)
_OUT_CSV = _THIS_DIR / "medpair_mcnemar_bonferroni.csv"

_BASE_MODELS = [
    ("Physician + Trainee", "Human_Majority_Round1", "Human_Majority_Round2"),
    ("GPT4o", "GPT-4o_Round1", "GPT-4o_Round2"),
    ("Qwen-14B", "Qwen-14B_Round1", "Qwen-14B_Round2"),
    ("Qwen 72B", "Qwen-72B_Round1", "Qwen-72B_Round2"),
    ("Llama-70B", "Llama-70B_Round1", "Llama-70B_Round2"),
    ("MedGemma-27B", "MedGemma-27B_Round1", "MedGemma-27B_Round2"),
    ("GPT 5", "GPT-5_Round1", "GPT-5_Round2"),
]
_DATA_SOURCES = [
    ("Total", None),
    ("MMLU", "mmlu"),
    ("Jama", "jama"),
    ("MedXpert", "medxpert"),
    ("Medbullets", "medbullets"),
]


def _mcnemar_exact(r1_correct: np.ndarray, r2_correct: np.ndarray) -> tuple[int, int, float]:
    b = int(np.sum(r1_correct & ~r2_correct))
    c = int(np.sum(~r1_correct & r2_correct))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p_one = stats.binom.cdf(k, n, 0.5)
    return b, c, float(min(1.0, 2.0 * p_one))


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main() -> None:
    if not _SRC.is_file():
        print(f"Missing paired responses file: {_SRC}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(_SRC, low_memory=False)
    gt = df["correct_answer"].astype("string").str.strip().str.upper().str[:1]
    ds = df["data_source_corr"].astype("string").str.lower().str.strip()

    rows: list[dict] = []
    for base_model, r1_col, r2_col in _BASE_MODELS:
        if r1_col not in df.columns or r2_col not in df.columns:
            print(f"WARN: missing columns for {base_model}: {r1_col} / {r2_col}", file=sys.stderr)
            continue
        r1 = df[r1_col].astype("string").str.strip().str.upper().str[:1]
        r2 = df[r2_col].astype("string").str.strip().str.upper().str[:1]
        r1_correct_all = ((r1 == gt) & r1.notna() & gt.notna()).fillna(False).to_numpy()
        r2_correct_all = ((r2 == gt) & r2.notna() & gt.notna()).fillna(False).to_numpy()

        for col_name, ds_key in _DATA_SOURCES:
            mask = np.ones(len(df), dtype=bool) if ds_key is None else (ds == ds_key).to_numpy()
            n = int(mask.sum())
            r1c = r1_correct_all[mask]
            r2c = r2_correct_all[mask]
            b, c, p = _mcnemar_exact(r1c, r2c)
            rows.append({
                "base_model": base_model,
                "data_source": col_name,
                "n_paired": n,
                "r1_acc_pct": 100.0 * float(np.mean(r1c)) if n else float("nan"),
                "r2_acc_pct": 100.0 * float(np.mean(r2c)) if n else float("nan"),
                "delta_pp": 100.0 * float(np.mean(r2c) - np.mean(r1c)) if n else float("nan"),
                "b_R1right_R2wrong": b,
                "c_R1wrong_R2right": c,
                "p_value": p,
            })

    out = pd.DataFrame(rows)
    n_tests = len(out)
    alpha = 0.05
    out["bonferroni_p"] = (out["p_value"] * n_tests).clip(upper=1.0)
    out["signif"] = out["bonferroni_p"].map(_stars)
    out["n_comparisons"] = n_tests
    out = out[[
        "base_model", "data_source", "n_paired",
        "r1_acc_pct", "r2_acc_pct", "delta_pp",
        "b_R1right_R2wrong", "c_R1wrong_R2right",
        "p_value", "n_comparisons", "bonferroni_p", "signif",
    ]]
    out.to_csv(_OUT_CSV, index=False)
    print(f"Wrote {_OUT_CSV}")
    print(f"N tests = {n_tests}   Bonferroni alpha = {alpha}/{n_tests} = {alpha/n_tests:.5f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
