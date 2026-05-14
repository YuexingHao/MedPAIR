#!/usr/bin/env python3
"""
Majority vote on ``q1`` per ``Origin`` from ``Round2_933_Eval.csv``.

Ties (including three raters with three different answers): pick uniformly at random among
values tied for the highest count. Reproducible with ``--seed`` (default 42).

Writes ``Round2_933_MJ.csv`` with:

- ``Origin`` stripped of the ``_phase_3`` suffix (e.g. ``ID0002``) to align with Centaur exports.
- ``data_source_corr_x`` from ``Centaur_933_Clinician_Student_Majority_Vote.csv`` (join on ``Origin``).
- Majority ``q1``, correct answer, and match flag.

Also writes per-source accuracy (``Round2_933_MJ_accuracy_by_data_source.csv`` by default): counts and
accuracy of ``matches_correct`` within each ``data_source_corr_x``.

Prints **mean and sample standard deviation** of ``Duration (s)`` across all eval rows (when present).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_answer(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s.rstrip("'")


def majority_q1(series: pd.Series, rng: np.random.Generator) -> str | float:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) == 0:
        return np.nan
    counts = Counter(vals)
    best = max(counts.values())
    tied = [k for k, v in counts.items() if v == best]
    if len(tied) == 1:
        return tied[0]
    return str(rng.choice(tied))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "Round2_933_Eval.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Round2_933_MJ.csv",
    )
    p.add_argument(
        "--centaur",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "Mar2_2026_Data"
        / "Centaur_933_Clinician_Student_Majority_Vote.csv",
        help="933 Centaur table with Origin and data_source_corr_x.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--by-source-out",
        type=Path,
        default=None,
        help="CSV: accuracy by data_source_corr_x (default: alongside --output as "
        "Round2_933_MJ_accuracy_by_data_source.csv).",
    )
    args = p.parse_args()
    if args.by_source_out is None:
        args.by_source_out = args.output.with_name("Round2_933_MJ_accuracy_by_data_source.csv")

    df = pd.read_csv(args.input, low_memory=False)
    need = {"Origin", "q1", "Correct answer"}
    if not need.issubset(df.columns):
        raise ValueError(f"Need columns {need}; got {list(df.columns)}")

    if "Duration (s)" in df.columns:
        dur = pd.to_numeric(df["Duration (s)"], errors="coerce").dropna()
        if len(dur) > 1:
            print(
                "Duration (s) — all rows: "
                f"n={len(dur)}, mean={dur.mean():.6g}, std={dur.std(ddof=1):.6g}, "
                f"min={dur.min():.6g}, max={dur.max():.6g}",
                flush=True,
            )
        elif len(dur) == 1:
            print(f"Duration (s): n=1, value={dur.iloc[0]:.6g}", flush=True)

    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for origin, g in df.groupby("Origin", sort=False):
        mj = majority_q1(g["q1"], rng)
        correct = g["Correct answer"].iloc[0]
        match = _norm_answer(mj) == _norm_answer(correct) if pd.notna(mj) else False
        rows.append(
            {
                "Origin": origin,
                "q1_majority": mj,
                "Correct answer": correct,
                "matches_correct": bool(match),
            }
        )

    out = pd.DataFrame(rows)
    out["Origin"] = out["Origin"].astype(str).str.removesuffix("_phase_3")

    if not args.centaur.is_file():
        raise FileNotFoundError(f"Centaur CSV not found: {args.centaur}")
    centaur = pd.read_csv(args.centaur, usecols=["Origin", "data_source_corr_x"], low_memory=False)
    centaur["Origin"] = centaur["Origin"].astype(str).str.strip()
    out = out.merge(centaur, on="Origin", how="left")
    if out["data_source_corr_x"].isna().any():
        n_miss = int(out["data_source_corr_x"].isna().sum())
        print(f"Warning: {n_miss} Origins missing data_source_corr_x after merge", flush=True)

    cols = [
        "Origin",
        "data_source_corr_x",
        "q1_majority",
        "Correct answer",
        "matches_correct",
    ]
    out = out[cols]
    out.to_csv(args.output, index=False)
    n = len(out)
    acc = float(out["matches_correct"].mean()) if n else float("nan")
    print(f"Wrote {args.output} ({n} Origins)")
    print(f"Accuracy (majority vs correct): {acc * 100:.2f}% ({out['matches_correct'].sum()}/{n})")

    sub = out.dropna(subset=["data_source_corr_x"])
    by_src = (
        sub.groupby("data_source_corr_x", sort=True)
        .agg(
            n_origins=("matches_correct", "size"),
            n_correct=("matches_correct", "sum"),
        )
        .reset_index()
    )
    by_src["accuracy"] = by_src["n_correct"] / by_src["n_origins"]
    by_src["accuracy_pct"] = (100.0 * by_src["accuracy"]).round(2)
    by_src.to_csv(args.by_source_out, index=False)
    print(f"Wrote {args.by_source_out}")
    print("Accuracy by data_source_corr_x:")
    for _, r in by_src.iterrows():
        print(
            f"  {r['data_source_corr_x']!s:12s}  "
            f"{int(r['n_correct'])}/{int(r['n_origins'])}  "
            f"{r['accuracy_pct']:.2f}%"
        )
    if out["data_source_corr_x"].isna().any():
        nm = int(out["data_source_corr_x"].isna().sum())
        print(f"  (excluded {nm} Origins with missing data_source_corr_x from by-source table)")


if __name__ == "__main__":
    main()
