#!/usr/bin/env python3
"""Recompute the llama70b-removed SR accuracies for every model in one place.

Reads each model's [SR]_*_predictions_on_llama70b_removed.csv, parses answers with
the same sr_common.extract_letter every runner uses, and reports accuracy by
data source with the denominator always visible.

Coverage is reported separately from accuracy on purpose: a model that only
answered 646 of 1300 rows is not comparable to one that answered all 1300, and
collapsing the two hides exactly the failure that made the old numbers wrong.

    python summarize_sr_results.py [--csv out.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_common import extract_letter, GOLD_COL, KEY_COL  # noqa: E402

BASE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal")
SOURCE_COL = "data_source_df3"
SOURCES = ["mmlu", "jama", "medxpert", "medbullets"]

# (display name, prediction column, candidate csv paths in priority order)
#
# Several models have case-variant twins of the same filename holding DIFFERENT
# data - e.g. [SR]_GPT4o_... is an empty scaffold while [SR]_gpt4o_... holds the
# real predictions. Anything globbing case-insensitively silently picks one at
# random, so the candidates are listed explicitly and the file actually used is
# printed with the results.
MODELS = [
    ("GPT-4o", "gpt4o_direct_prediction", [
        "GPT4o/results/predictions/[SR]_GPT4o_predictions_on_llama70b_removed.csv",
        "GPT4o/results/predictions/[SR]_gpt4o_predictions_on_llama70b_removed.csv",
    ]),
    ("GPT-5", "gpt5_direct_prediction", [
        "GPT5/results/predictions/[SR]_GPT5_predictions_on_llama70b_removed.csv",
        "GPT5/results/predictions/[SR]_gpt5_predictions_on_llama70b_removed.csv",
    ]),
    ("Llama-70B", "llama70b_direct_prediction", [
        "Llama-70B/results/predictions/[SR]_Llama70B_predictions_on_llama70b_removed.csv",
    ]),
    ("MedGemma-27B", "medgemma27b_direct_prediction", [
        "MedGemma-27b-text-it/results/predictions/[SR]_MedGemma27B_predictions_on_llama70b_removed.csv",
    ]),
    ("Qwen-14B", "qwen14b_direct_prediction", [
        "Qwen2.5-14B-Instruct/results/predictions/[SR]_Qwen14B_predictions_on_llama70b_removed.csv",
    ]),
    ("Qwen-72B", "qwen72b_direct_prediction", [
        "Qwen2.5-72B-Instruct/results/predictions/[SR]_Qwen72B_predictions_on_llama70b_removed.csv",
    ]),
]


def score(path, pred_col):
    df = pd.read_csv(BASE / path, low_memory=False)
    if pred_col not in df.columns:
        return None, f"column {pred_col!r} absent"

    letters = df[pred_col].map(extract_letter)
    gold = df[GOLD_COL].astype(str).str.strip().str.upper()
    scored = letters.notna()
    correct = scored & (letters == gold)

    raw = df[pred_col].astype(str).str.strip()
    n_error = int(raw.str.startswith("Error").sum())
    n_blank = int((df[pred_col].isna()).sum())

    row = {
        "n": len(df),
        "scored": int(scored.sum()),
        "errors": n_error,
        "blank": n_blank,
        "coverage": scored.sum() / len(df) * 100,
        "acc_scored": correct.sum() / scored.sum() * 100 if scored.sum() else float("nan"),
        "acc_all": correct.sum() / len(df) * 100,
    }
    if SOURCE_COL in df.columns:
        src = df[SOURCE_COL].astype(str).str.lower()
        for s in SOURCES:
            m = src == s
            ns = int((m & scored).sum())
            row[s] = (correct[m].sum() / ns * 100) if ns else float("nan")
            row[f"{s}_n"] = ns
    return row, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="Also write the table here.")
    args = ap.parse_args()

    rows, notes = [], []
    for name, pred_col, candidates in MODELS:
        # Among the case-variant twins, use whichever actually has predictions.
        best, best_path, errs = None, None, []
        for path in candidates:
            if not (BASE / path).exists():
                errs.append(f"missing {path}")
                continue
            row, err = score(path, pred_col)
            if err:
                errs.append(f"{path}: {err}")
                continue
            if best is None or row["scored"] > best["scored"]:
                best, best_path = row, path

        if best is None:
            notes.append(f"{name}: no scorable file ({'; '.join(errs)})")
            continue

        if len(candidates) > 1:
            notes.append(f"{name}: used {Path(best_path).name} "
                         f"({best['scored']} scored) out of {len(candidates)} case-variants")
        rows.append({"Model": name, "File": Path(best_path).name, **best})

    if not rows:
        print("No scorable model files found.")
        return

    df = pd.DataFrame(rows)

    print("=" * 100)
    print("SR (Self-Report) accuracies - llama70b-removed reduced context")
    print("=" * 100)
    cols = ["Model", "scored", "n", "coverage", "acc_scored", "acc_all"] + SOURCES
    print(df[[c for c in cols if c in df.columns]].to_string(
        index=False, float_format=lambda x: f"{x:.1f}"))

    print()
    print("Per-source denominators (rows that produced a parseable answer):")
    dcols = ["Model"] + [f"{s}_n" for s in SOURCES if f"{s}_n" in df.columns]
    print(df[dcols].to_string(index=False))

    incomplete = df[df["coverage"] < 99.0]
    if len(incomplete):
        print()
        print("!! Incomplete coverage - these are NOT comparable to full-coverage rows:")
        for _, r in incomplete.iterrows():
            print(f"   {r['Model']:14s} {r['scored']:>5}/{r['n']} scored "
                  f"({r['coverage']:.1f}%)  errors={r['errors']}  blank={r['blank']}")

    if notes:
        print()
        print("Notes:")
        for nt in notes:
            print(f"   - {nt}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
