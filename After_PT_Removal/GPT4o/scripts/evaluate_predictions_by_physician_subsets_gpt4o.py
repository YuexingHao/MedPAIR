#!/usr/bin/env python3
"""
``evaluate_predictions_by_physician_subsets_gpt4o.py`` — GPT4o project: evaluate letter-match
accuracy on prediction CSVs under ``results/predictions/`` for:

1) Origins in 933_Clinician_Student_Majority_Vote.csv
2) IDs in HardQA_Clinician_Student_Majority_Vote.csv (column "ID")
3) Origins in Impossible_Clinician_Student_Majority_Vote.csv

Logic matches the pipeline notebooks: extract predicted letter from model output,
compare to answer_corr, then subset by Origin/ID and report overall + per-source stats.

Output:
  - Per-file details (unless ``--quiet``): accuracy lines and per-source breakdowns.
  - Summary tables: long-format metrics and a wide accuracy matrix across subsets.
  - ``results/eval_physician_subsets_summary.csv`` (long-format, one row per file × subset).

For GPT-5 predictions, use ``../GPT5/scripts/evaluate_predictions_by_physician_subsets_gpt5.py``.
For Qwen2.5-72B-Instruct predictions, use
``Qwen2.5-72B-Instruct/results/scripts/evaluate_predictions_by_physician_subsets_qwen72b.py``.
For MedGemma-27b-text-it predictions, use
``MedGemma-27b-text-it/results/scripts/evaluate_predictions_by_physician_subsets_medgemma27b.py``.

CLI (see ``--help``):
  - ``--predictions-dir DIR`` — evaluate CSVs in another directory (default: GPT4o ``results/predictions``).
  - ``--merge-reference REF.csv`` — join ``REF`` on ``Origin`` when predictions lack ``answer_corr`` / ``answer_df3``.
  - By default, skips filenames containing ``progress``; use ``--include-progress`` to include them.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Short labels for summary tables (order preserved in output)
SUBSET_SHORT = (
    ("933_Clinician_Student_Majority_Vote", "933"),
    ("HardQA_Clinician_Student_Majority_Vote", "HardQA"),
    ("Impossible_Clinician_Student_Majority_Vote", "Impossible"),
)
SUBSET_ORDER = {full: short for full, short in SUBSET_SHORT}

# GPT4o/results/predictions — scripts/ -> GPT4o root
GPT4O_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = GPT4O_ROOT / "results" / "predictions"
PHYSICIAN_DATA = GPT4O_ROOT.parent.parent / "Physician_Labels" / "Mar2_2026_Data"

CSV_933 = PHYSICIAN_DATA / "933_Clinician_Student_Majority_Vote.csv"
CSV_HARD = PHYSICIAN_DATA / "HardQA_Clinician_Student_Majority_Vote.csv"
CSV_IMPOSSIBLE = PHYSICIAN_DATA / "Impossible_Clinician_Student_Majority_Vote.csv"

PRED_COLS = [
    "gpt4o_direct_prediction",
    "gpt5_direct_prediction",
    "GPT4o_prediction",  # remove_low_irr / Llama-style exports
    "gpt_direct_prediction",  # trainee-removed notebooks
    "majority_vote",
    "GPT5_on_72B_SR",
    "Extracted_Answer",
]
ANSWER_COLS = ["answer_corr", "answer_df3"]
SOURCE_COLS = [
    "data_source_df3",
    "data_source_corr",
    "data_source_corr_trainee",
    "data_source",
]


def ensure_origin_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``Origin`` exists (copy from ``ID_corr`` when needed)."""
    out = df.copy()
    if "Origin" not in out.columns and "ID_corr" in out.columns:
        out["Origin"] = out["ID_corr"].astype(str).str.strip()
    return out


def _origin_key(df: pd.DataFrame) -> pd.Series:
    if "Origin" in df.columns:
        return df["Origin"].astype(str).str.strip()
    if "ID_corr" in df.columns:
        return df["ID_corr"].astype(str).str.strip()
    raise ValueError("Need `Origin` or `ID_corr` column for row alignment.")


def merge_with_reference(pred: pd.DataFrame, ref_path: Path) -> pd.DataFrame:
    """Inner-join predictions with a reference CSV that has answers and optional source columns."""
    ref = pd.read_csv(ref_path)
    ref = ensure_origin_column(ref)
    pred = ensure_origin_column(pred)
    if "Origin" not in ref.columns:
        raise ValueError("Reference CSV must have `Origin` or `ID_corr`")
    if not any(c in ref.columns for c in ANSWER_COLS):
        raise ValueError("Reference CSV must include `answer_corr` or `answer_df3`")
    ref_cols: list[str] = ["Origin"]
    for c in ANSWER_COLS + SOURCE_COLS:
        if c in ref.columns:
            ref_cols.append(c)
    ref_cols = list(dict.fromkeys(ref_cols))
    sub = ref[ref_cols].drop_duplicates(subset=["Origin"])
    return pred.merge(sub, on="Origin", how="inner")


def load_origin_set_933() -> set[str]:
    s = pd.read_csv(CSV_933, usecols=["Origin"])["Origin"].astype(str).str.strip()
    return set(s)


def load_origin_set_hardqa() -> set[str]:
    s = pd.read_csv(CSV_HARD, usecols=["ID"])["ID"].astype(str).str.strip()
    return set(s)


def load_origin_set_impossible() -> set[str]:
    s = pd.read_csv(CSV_IMPOSSIBLE, usecols=["Origin"])["Origin"].astype(str).str.strip()
    return set(s)


def extract_letter_from_text(x) -> str | None:
    if not isinstance(x, str):
        return None
    m = re.search(
        r"Option\s*\[?([A-J])\]?|^\s*([A-J])\s*$",
        x.strip(),
        flags=re.IGNORECASE,
    )
    if m:
        return (m.group(1) or m.group(2)).upper()
    return None


def pick_pred_col(df: pd.DataFrame, basename: str | None = None) -> str | None:
    """Pick prediction column; Original Accuracy export uses ``Round 1 Letter`` when present."""
    b = (basename or "").lower()
    if b == "gpt4o_predictions_original_accuracy.csv" and "Round 1 Letter" in df.columns:
        return "Round 1 Letter"
    for c in PRED_COLS:
        if c in df.columns:
            return c
    if "Round 1 Letter" in df.columns:
        return "Round 1 Letter"
    if "gpt_letter" in df.columns:
        return "gpt_letter"
    return None


def pick_answer_col(df: pd.DataFrame) -> str | None:
    for c in ANSWER_COLS:
        if c in df.columns:
            return c
    return None


def build_gpt_letter(df: pd.DataFrame, pred_col: str) -> pd.Series:
    if pred_col == "gpt_letter":
        return df[pred_col].astype(str).str.strip().str.upper()
    if pred_col == "GPT5_on_72B_SR":
        return df[pred_col].astype(str).str.strip().str.upper()
    if pred_col == "Extracted_Answer":
        return df[pred_col].astype(str).str.strip().str.upper()
    if pred_col in ("Round 1 Letter", "Round 2 Letter"):
        def _round_letter(x) -> str | None:
            if pd.isna(x):
                return None
            s = str(x).strip()
            if len(s) == 1:
                return s.upper()
            return extract_letter_from_text(s)

        return df[pred_col].map(_round_letter)
    if pred_col == "majority_vote" and pick_answer_col(df) is not None:
        return df[pred_col].astype(str).str.strip().str.upper()
    # gpt4o_direct_prediction / gpt5_direct_prediction: usually XML with Option letter
    return df[pred_col].apply(extract_letter_from_text)


def evaluate_subset(
    df: pd.DataFrame,
    subset_name: str,
    origins: set[str],
    pred_col: str,
) -> dict | None:
    key = _origin_key(df)
    mask = key.isin(origins)
    sub = df.loc[mask].copy()
    n = len(sub)
    if n == 0:
        return {
            "subset": subset_name,
            "n_rows": 0,
            "correct": 0,
            "total": 0,
            "accuracy": float("nan"),
            "per_source": [],
        }

    ans_col = pick_answer_col(sub)
    if ans_col is None:
        return None

    sub["answer_letter"] = sub[ans_col].astype(str).str.strip().str.upper()
    if pred_col not in sub.columns:
        return None

    sub["gpt_letter"] = build_gpt_letter(sub, pred_col)
    sub["gpt_letter_match"] = np.where(
        sub["gpt_letter"] == sub["answer_letter"],
        "Correct",
        "Incorrect",
    )
    sub["gpt_letter_binary"] = (sub["gpt_letter_match"] == "Correct").astype(int)

    correct = int(sub["gpt_letter_binary"].sum())
    total = int(sub["gpt_letter_binary"].notna().sum())
    acc = correct / total if total > 0 else float("nan")
    std = sub["gpt_letter_binary"].std(ddof=1) if total > 1 else float("nan")
    if total > 0 and acc == acc:
        se = np.sqrt(acc * (1.0 - acc) / total)
        ci_low = max(0.0, acc - 1.96 * se)
        ci_high = min(1.0, acc + 1.96 * se)
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    source_col = next((c for c in SOURCE_COLS if c in sub.columns), None)
    per_source = []
    if source_col:
        for src in sub[source_col].dropna().unique():
            part = sub[sub[source_col] == src]
            c = int(part["gpt_letter_binary"].sum())
            t = int(part["gpt_letter_binary"].notna().sum())
            a = c / t if t > 0 else float("nan")
            std_src = part["gpt_letter_binary"].std(ddof=1) if t > 1 else float("nan")
            per_source.append(
                {"source": str(src), "correct": c, "total": t, "accuracy": a, "std": std_src}
            )

    return {
        "subset": subset_name,
        "n_rows": n,
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_source": per_source,
    }


def _fmt_acc(x: float) -> str:
    if x != x:  # NaN
        return "—"
    return f"{100.0 * x:.2f}%"


def _fmt_sd(x: float) -> str:
    """SD of binary correctness; display on 0–1 scale (not as %)."""
    if x != x:  # NaN
        return "—"
    return f"{x:.4f}"


def _fmt_ci(low: float, high: float) -> str:
    """95% CI bounds on 0–1 scale (not as %)."""
    if low != low or high != high:  # NaN
        return "—"
    return f"[{low:.4f}, {high:.4f}]"


def process_file(
    path: Path,
    o933: set[str],
    hard: set[str],
    imp: set[str],
    verbose: bool,
    merge_reference: Path | None,
) -> list[dict[str, Any]]:
    """Return one record per subset evaluation; empty list if file skipped entirely."""
    records: list[dict[str, Any]] = []
    if verbose:
        print(f"\n{'='*80}\n{path.name}\n{'='*80}")

    try:
        df = ensure_origin_column(pd.read_csv(path))
    except Exception as e:
        if verbose:
            print(f"  SKIP read error: {e}")
        return [
            {
                "prediction_file": path.name,
                "subset": "—",
                "n_rows": 0,
                "correct": 0,
                "total": 0,
                "accuracy": float("nan"),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "pred_col": "",
                "note": f"read error: {e}",
            }
        ]

    if pick_answer_col(df) is None and merge_reference is not None and merge_reference.is_file():
        try:
            df = merge_with_reference(df, merge_reference)
        except Exception as e:
            if verbose:
                print(f"  SKIP merge with {merge_reference.name}: {e}")
            return [
                {
                    "prediction_file": path.name,
                    "subset": "—",
                    "n_rows": 0,
                    "correct": 0,
                    "total": 0,
                    "accuracy": float("nan"),
                    "std": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "pred_col": "",
                    "note": f"merge failed: {e}",
                }
            ]

    try:
        _origin_key(df)
    except ValueError as e:
        if verbose:
            print(f"  SKIP: {e}")
        return [
            {
                "prediction_file": path.name,
                "subset": "—",
                "n_rows": 0,
                "correct": 0,
                "total": 0,
                "accuracy": float("nan"),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "pred_col": "",
                "note": str(e),
            }
        ]

    if pick_answer_col(df) is None:
        if verbose:
            print(
                "  SKIP: no answer column (need answer_corr or answer_df3, "
                "or pass --merge-reference with a CSV that has them)"
            )
        return [
            {
                "prediction_file": path.name,
                "subset": "—",
                "n_rows": 0,
                "correct": 0,
                "total": 0,
                "accuracy": float("nan"),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "pred_col": "",
                "note": "no answer column",
            }
        ]

    pred_col = pick_pred_col(df, path.name)
    if pred_col is None:
        if verbose:
            print("  SKIP: no prediction column")
        return [
            {
                "prediction_file": path.name,
                "subset": "—",
                "n_rows": 0,
                "correct": 0,
                "total": 0,
                "accuracy": float("nan"),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "pred_col": "",
                "note": "no prediction column",
            }
        ]

    for label, origin_set in [
        ("933_Clinician_Student_Majority_Vote", o933),
        ("HardQA_Clinician_Student_Majority_Vote", hard),
        ("Impossible_Clinician_Student_Majority_Vote", imp),
    ]:
        short = SUBSET_ORDER[label]
        out = evaluate_subset(df, label, origin_set, pred_col)
        if out is None:
            if verbose:
                print(f"  [{label}] SKIP (could not evaluate)")
            records.append(
                {
                    "prediction_file": path.name,
                    "subset": short,
                    "n_rows": 0,
                    "correct": 0,
                    "total": 0,
                    "accuracy": float("nan"),
                    "std": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "pred_col": pred_col,
                    "note": "could not evaluate subset",
                }
            )
            continue
        if out["n_rows"] == 0:
            if verbose:
                print(f"  [{label}] no overlapping rows")
            records.append(
                {
                    "prediction_file": path.name,
                    "subset": short,
                    "n_rows": 0,
                    "correct": 0,
                    "total": 0,
                    "accuracy": float("nan"),
                    "std": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "pred_col": pred_col,
                    "note": "no overlapping rows",
                }
            )
            continue

        if verbose:
            print(
                f"  [{label}] rows={out['n_rows']}  correct={out['correct']}/{out['total']}  "
                f"accuracy={out['accuracy']:.2%}"
            )
            for row in out["per_source"]:
                print(
                    f"      {row['source']}: {row['correct']}/{row['total']} "
                    f"({row['accuracy']:.2%}) std={row['std']:.4f}"
                )

        records.append(
            {
                "prediction_file": path.name,
                "subset": short,
                "n_rows": out["n_rows"],
                "correct": out["correct"],
                "total": out["total"],
                "accuracy": out["accuracy"],
                "std": out["std"],
                "ci_low": out["ci_low"],
                "ci_high": out["ci_high"],
                "pred_col": pred_col,
                "note": "",
            }
        )

    return records


def print_summary_tables(long_df: pd.DataFrame) -> None:
    """Print long-format summary and a wide accuracy matrix for quick comparison."""
    if long_df.empty:
        return

    ok = long_df["note"].eq("") | long_df["note"].isna()
    work = long_df.loc[ok & long_df["subset"].ne("—")].copy()
    skips = long_df.loc[~ok | long_df["subset"].eq("—")]

    print("\n" + "=" * 100)
    print("SUMMARY — letter accuracy by prediction file and physician subset")
    print("=" * 100)

    if not work.empty:
        disp = work.assign(
            Accuracy_pct=work["accuracy"].map(_fmt_acc),
            SD_disp=work["std"].map(_fmt_sd),
            CI95=work.apply(lambda r: _fmt_ci(r["ci_low"], r["ci_high"]), axis=1),
            Score=work.apply(
                lambda r: f"{int(r['correct'])}/{int(r['total'])}", axis=1
            ),
        )[
            [
                "prediction_file",
                "subset",
                "n_rows",
                "Score",
                "Accuracy_pct",
                "SD_disp",
                "CI95",
                "pred_col",
            ]
        ]
        disp.columns = [
            "Prediction file",
            "Subset",
            "Rows (matched)",
            "Correct / total",
            "Accuracy",
            "SD",
            "95% CI",
            "Pred column",
        ]
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 200)
        print(disp.to_string(index=False))
        print()

        # Wide table: one row per file, accuracy columns per subset
        pivot = work.pivot_table(
            index="prediction_file",
            columns="subset",
            values="accuracy",
            aggfunc="first",
        )
        subset_cols = [s for _, s in SUBSET_SHORT]
        pivot = pivot.reindex(columns=subset_cols)
        try:
            wide = pivot.map(_fmt_acc)
        except AttributeError:
            wide = pivot.applymap(_fmt_acc)  # pandas < 2.1
        wide = wide.reset_index().rename(columns={"prediction_file": "Prediction file"})
        print("—" * 100)
        print("WIDE VIEW — accuracy % by subset (same row order as sorted filenames)")
        print("—" * 100)
        print(wide.to_string(index=False))
        print()

    if not skips.empty:
        print("—" * 100)
        print("SKIPPED OR FAILED FILES")
        print("—" * 100)
        sd = skips[["prediction_file", "note"]].drop_duplicates()
        sd = sd.rename(columns={"prediction_file": "Prediction file", "note": "Reason"})
        print(sd.to_string(index=False))
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GPT4o: letter-match accuracy on prediction CSVs vs physician subsets "
            "(933 / HardQA / Impossible)."
        )
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory of prediction CSVs (default: this project’s results/predictions).",
    )
    p.add_argument(
        "--merge-reference",
        type=Path,
        default=None,
        help=(
            "Reference CSV with Origin/ID_corr and answer_corr or answer_df3 "
            "(inner-join when predictions lack ground-truth columns)."
        ),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write eval_physician_subsets_summary.csv (default: next to predictions).",
    )
    p.add_argument(
        "--include-progress",
        action="store_true",
        help="Include filenames containing 'progress' (default: skip them).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-file detail output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for p in (CSV_933, CSV_HARD, CSV_IMPOSSIBLE):
        if not p.is_file():
            print(f"Missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    o933 = load_origin_set_933()
    hard = load_origin_set_hardqa()
    imp = load_origin_set_impossible()

    pred_dir = (args.predictions_dir or PREDICTIONS_DIR).resolve()
    merge_ref = args.merge_reference.resolve() if args.merge_reference else None

    print(f"GPT4O_ROOT: {GPT4O_ROOT}")
    print(f"Predictions dir: {pred_dir}")
    if merge_ref is not None:
        print(f"Merge reference: {merge_ref}")
    print(f"933 Origins: {len(o933)} | HardQA IDs: {len(hard)} | Impossible Origins: {len(imp)}")

    if not pred_dir.is_dir():
        print(f"Missing predictions directory: {pred_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(pred_dir.glob("*.csv"))
    if not args.include_progress:
        files = [f for f in files if "progress" not in f.name.lower()]
    if not files:
        print("No CSV files in predictions directory.")
        return

    verbose = not args.quiet
    all_records: list[dict[str, Any]] = []
    for f in files:
        all_records.extend(
            process_file(f, o933, hard, imp, verbose=verbose, merge_reference=merge_ref)
        )

    long_df = pd.DataFrame(all_records)
    print_summary_tables(long_df)

    if args.output_csv is not None:
        out_csv = args.output_csv.resolve()
    elif args.predictions_dir is not None:
        out_csv = pred_dir.parent / "eval_physician_subsets_summary.csv"
    else:
        out_csv = GPT4O_ROOT / "results" / "eval_physician_subsets_summary.csv"
    try:
        long_df.to_csv(out_csv, index=False)
        print(f"Wrote long-format table: {out_csv}")
    except OSError as e:
        print(f"Could not write {out_csv}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
