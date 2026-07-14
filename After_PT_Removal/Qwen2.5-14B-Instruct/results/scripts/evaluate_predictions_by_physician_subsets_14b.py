#!/usr/bin/env python3
"""
``evaluate_predictions_by_physician_subsets_14b.py`` — Qwen2.5-14B-Instruct: evaluate
letter-match accuracy on prediction CSVs in ``results/predictions/`` against physician subsets:

1) Origins in 933_Clinician_Student_Majority_Vote.csv
2) IDs in HardQA_Clinician_Student_Majority_Vote.csv (column "ID")
3) Origins in Impossible_Clinician_Student_Majority_Vote.csv

Same evaluation logic as ``GPT4o/scripts/evaluate_predictions_by_physician_subsets_gpt4o.py`` and
``Qwen2.5-72B-Instruct/results/scripts/evaluate_predictions_by_physician_subsets_qwen72b.py``.

Output:
  - Per-file details (unless ``--quiet``): accuracy lines and per-source breakdowns.
  - Summary tables: long-format metrics and a wide accuracy matrix across subsets.
  - ``results/eval_physician_subsets_summary.csv`` (long-format, one row per file × subset).
  - With ``--by-data-source``: ``results/eval_physician_subsets_by_data_source.csv`` (one row per
    file × subset × ``data_source`` value).

CLI (see ``--help``):
  - **Default inputs:** all ``*.csv`` in ``results/predictions/`` except filenames containing
    ``progress`` (checkpoint rows are often incomplete). Use ``--include-progress`` to add them.
  - **Default merge:** ``data/raw/14B_High_Low_Irr_Data_Source_.csv`` adds ``answer_corr`` when
    missing (override with ``--merge-reference`` / ``--no-default-merge``).
  - **Trainee progress:** ``Qwen_14B_predictions_on_Trainee_progress.csv`` may store the letter in
    ``Original``; it is copied to ``Extracted_Answer``.
  - **Empty Origin:** some exports (e.g. ``Qwen_14B_predictions_GPT5.csv``) omit ``Origin``; by
    default it is filled from ``Qwen_14B_predictions_GPT4o.csv`` via matching ``QA_ID`` (same row
    order / IDs as GPT4o). Override with ``--origin-lookup`` / ``--no-origin-lookup``.
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

# results/scripts/this_file -> project root (Qwen2.5-14B-Instruct)
QWEN14_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = QWEN14_ROOT / "results" / "predictions"
PHYSICIAN_DATA = QWEN14_ROOT.parent.parent / "Physician_Labels" / "Mar2_2026_Data"
DEFAULT_MERGE_REFERENCE = QWEN14_ROOT / "data" / "raw" / "14B_High_Low_Irr_Data_Source_.csv"
DEFAULT_ORIGIN_LOOKUP = PREDICTIONS_DIR / "Qwen_14B_predictions_GPT4o.csv"

CSV_933 = PHYSICIAN_DATA / "933_Clinician_Student_Majority_Vote.csv"
CSV_HARD = PHYSICIAN_DATA / "HardQA_Clinician_Student_Majority_Vote.csv"
CSV_IMPOSSIBLE = PHYSICIAN_DATA / "Impossible_Clinician_Student_Majority_Vote.csv"

PRED_COLS = [
    "Extracted_Answer",
    "gpt4o_direct_prediction",
    "gpt5_direct_prediction",
    "GPT4o_prediction",
    "gpt_direct_prediction",
    "majority_vote",
    "GPT5_on_72B_SR",
]
ANSWER_COLS = ["answer_corr", "answer_df3"]
SOURCE_COLS = [
    "data_source_df3",
    "data_source_corr",
    "data_source_corr_trainee",
    "data_source.1",  # pandas renames duplicate "data_source" headers in some exports
    "data_source",
]


def pick_source_col(df: pd.DataFrame) -> str | None:
    """Choose the data-source column with the most non-null values (handles duplicate headers)."""
    candidates: list[str] = []
    for c in SOURCE_COLS:
        if c in df.columns:
            candidates.append(c)
    for c in df.columns:
        if str(c).startswith("data_source") and c not in candidates:
            candidates.append(c)
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    best: str | None = None
    best_n = -1
    for c in uniq:
        n = int(df[c].notna().sum())
        if n > best_n:
            best_n = n
            best = c
    return best


def _origin_str(x: Any) -> str:
    """Normalize Origin for merges (avoid float NaN vs str dtype mismatch)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def fill_origin_from_qa_lookup(pred: pd.DataFrame, lookup_path: Path) -> pd.DataFrame:
    """Fill missing/empty ``Origin`` from ``QA_ID`` using a final export (``Origin`` or ``ID_corr``)."""
    if not lookup_path.is_file() or "QA_ID" not in pred.columns:
        return pred
    try:
        look = pd.read_csv(lookup_path)
    except OSError:
        return pred
    if "QA_ID" not in look.columns:
        return pred
    if "Origin" in look.columns and look["Origin"].map(_origin_str).ne("").any():
        oid = look["Origin"].map(_origin_str)
    elif "ID_corr" in look.columns:
        oid = look["ID_corr"].map(_origin_str)
    else:
        return pred
    look = pd.DataFrame({"QA_ID": look["QA_ID"].astype(str).str.strip(), "Origin": oid})
    look = look.dropna(subset=["QA_ID"])
    look = look[look["Origin"].ne("")]
    if look.empty:
        return pred
    mapping = look.drop_duplicates("QA_ID", keep="last").set_index("QA_ID")["Origin"]

    out = pred.copy()
    out["QA_ID"] = out["QA_ID"].astype(str).str.strip()
    if "Origin" not in out.columns:
        out["Origin"] = ""
    out["Origin"] = out["Origin"].astype(object)
    orig = out["Origin"].map(_origin_str)
    missing = orig.eq("")
    filled = out["QA_ID"].map(mapping)
    out.loc[missing, "Origin"] = filled[missing].values
    return out


def ensure_origin_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``Origin`` exists (copy from ``ID_corr`` when needed)."""
    out = df.copy()
    if "Origin" not in out.columns and "ID_corr" in out.columns:
        out["Origin"] = out["ID_corr"].astype(str).str.strip()
    return out


def ensure_extracted_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """Trainee *progress* rows use ``Original`` for the extracted letter; mirror ``Extracted_Answer``."""
    out = df.copy()
    if "Extracted_Answer" not in out.columns and "Original" in out.columns:
        out["Extracted_Answer"] = out["Original"]
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
    sub["Origin"] = sub["Origin"].map(_origin_str)
    pred = pred.copy()
    pred["Origin"] = pred["Origin"].map(_origin_str)
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


def pick_pred_col(df: pd.DataFrame) -> str | None:
    for c in PRED_COLS:
        if c in df.columns:
            return c
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
    if pred_col == "majority_vote" and pick_answer_col(df) is not None:
        return df[pred_col].astype(str).str.strip().str.upper()
    return df[pred_col].apply(extract_letter_from_text)


def evaluate_subset(
    df: pd.DataFrame,
    subset_name: str,
    origins: set[str],
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
            "source_column_used": "",
        }

    ans_col = pick_answer_col(sub)
    if ans_col is None:
        return None

    sub["answer_letter"] = sub[ans_col].astype(str).str.strip().str.upper()
    pred_col = pick_pred_col(sub)
    if pred_col is None:
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

    source_col = pick_source_col(sub)
    per_source = []
    if source_col:
        for src in sub[source_col].dropna().unique():
            part = sub[sub[source_col] == src]
            c = int(part["gpt_letter_binary"].sum())
            t = int(part["gpt_letter_binary"].notna().sum())
            a = c / t if t > 0 else float("nan")
            std_src = part["gpt_letter_binary"].std(ddof=1) if t > 1 else float("nan")
            if t > 0 and a == a:
                se = np.sqrt(a * (1.0 - a) / t)
                ci_lo = max(0.0, a - 1.96 * se)
                ci_hi = min(1.0, a + 1.96 * se)
            else:
                ci_lo = float("nan")
                ci_hi = float("nan")
            per_source.append(
                {
                    "source": str(src),
                    "correct": c,
                    "total": t,
                    "accuracy": a,
                    "std": std_src,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                }
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
        "source_column_used": source_col or "",
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
    origin_lookup: Path | None = None,
    by_source_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return one record per subset evaluation; empty list if file skipped entirely."""
    records: list[dict[str, Any]] = []
    if verbose:
        print(f"\n{'='*80}\n{path.name}\n{'='*80}")

    try:
        df = ensure_extracted_answer_column(ensure_origin_column(pd.read_csv(path)))
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

    if origin_lookup is not None:
        before = df["Origin"].map(_origin_str).ne("").sum() if "Origin" in df.columns else 0
        df = fill_origin_from_qa_lookup(df, origin_lookup)
        after = df["Origin"].map(_origin_str).ne("").sum() if "Origin" in df.columns else 0
        if verbose and after > before:
            print(
                f"  Filled Origin from {origin_lookup.name} (QA_ID): "
                f"{int(after - before)} rows"
            )

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

    pred_col = pick_pred_col(df)
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
        out = evaluate_subset(df, label, origin_set)
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

        if by_source_records is not None and out.get("per_source"):
            for ps in out["per_source"]:
                by_source_records.append(
                    {
                        "prediction_file": path.name,
                        "subset": short,
                        "data_source": ps["source"],
                        "source_column": out.get("source_column_used", ""),
                        "n_rows": ps["total"],
                        "correct": ps["correct"],
                        "total": ps["total"],
                        "accuracy": ps["accuracy"],
                        "std": ps["std"],
                        "ci_low": ps["ci_low"],
                        "ci_high": ps["ci_high"],
                        "pred_col": pred_col,
                        "note": "",
                    }
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
            "Qwen 14B: letter-match accuracy on prediction CSVs vs physician subsets "
            "(933 / HardQA / Impossible)."
        )
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory of prediction CSVs (default: <project>/results/predictions).",
    )
    p.add_argument(
        "--merge-reference",
        type=Path,
        default=None,
        help=(
            "Reference CSV with Origin/ID_corr and answer_corr or answer_df3 "
            "(inner-join when predictions lack ground truth). "
            f"Default: {DEFAULT_MERGE_REFERENCE.name} under project data/raw/ if present."
        ),
    )
    p.add_argument(
        "--no-default-merge",
        action="store_true",
        help="Do not use data/raw/14B_High_Low_Irr_Data_Source_.csv when --merge-reference is omitted.",
    )
    p.add_argument(
        "--origin-lookup",
        type=Path,
        default=None,
        help=(
            "CSV with QA_ID and Origin or ID_corr (default: Qwen_14B_predictions_GPT4o.csv if present). "
            "Fills empty Origin before merge (needed for GPT5-only exports, etc.)."
        ),
    )
    p.add_argument(
        "--no-origin-lookup",
        action="store_true",
        help="Do not fill Origin from QA_ID (disables default GPT5 lookup file).",
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
        help="Include filenames containing 'progress' (default: skip them, like GPT4o eval script).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-file detail output.")
    p.add_argument(
        "--by-data-source",
        action="store_true",
        help=(
            "Also write eval_physician_subsets_by_data_source.csv (breakdown by data_source "
            "within each physician subset)."
        ),
    )
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

    if args.merge_reference is not None:
        merge_ref = args.merge_reference.resolve()
    elif args.no_default_merge:
        merge_ref = None
    elif DEFAULT_MERGE_REFERENCE.is_file():
        merge_ref = DEFAULT_MERGE_REFERENCE.resolve()
    else:
        merge_ref = None

    if args.no_origin_lookup:
        origin_lookup: Path | None = None
    elif args.origin_lookup is not None:
        origin_lookup = args.origin_lookup.resolve()
    elif DEFAULT_ORIGIN_LOOKUP.is_file():
        origin_lookup = DEFAULT_ORIGIN_LOOKUP.resolve()
    else:
        origin_lookup = None

    print(f"QWEN14_ROOT: {QWEN14_ROOT}")
    print(f"Predictions dir: {pred_dir}")
    if args.include_progress:
        print("Input mode: including *progress*.csv files")
    else:
        print("Input mode: skipping *progress* in filenames (default); use --include-progress to add")
    if merge_ref is not None:
        print(f"Merge reference: {merge_ref}")
    elif not args.no_default_merge and not DEFAULT_MERGE_REFERENCE.is_file():
        print(
            f"Note: default merge file missing ({DEFAULT_MERGE_REFERENCE}); "
            "pass --merge-reference or add data/raw/14B_High_Low_Irr_Data_Source_.csv",
            file=sys.stderr,
        )
    if origin_lookup is not None:
        print(f"Origin lookup (QA_ID): {origin_lookup}")

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
    by_source: list[dict[str, Any]] = [] if args.by_data_source else []
    for f in files:
        all_records.extend(
            process_file(
                f,
                o933,
                hard,
                imp,
                verbose=verbose,
                merge_reference=merge_ref,
                origin_lookup=origin_lookup,
                by_source_records=by_source if args.by_data_source else None,
            )
        )

    long_df = pd.DataFrame(all_records)
    print_summary_tables(long_df)

    if args.by_data_source and by_source:
        by_df = pd.DataFrame(by_source)
        print("\n" + "=" * 100)
        print("BY data_source — letter accuracy (within each file × physician subset)")
        print("=" * 100)
        disp = by_df.assign(
            Accuracy_pct=by_df["accuracy"].map(_fmt_acc),
            Score=by_df.apply(lambda r: f"{int(r['correct'])}/{int(r['total'])}", axis=1),
        )[
            [
                "prediction_file",
                "subset",
                "data_source",
                "Score",
                "Accuracy_pct",
                "source_column",
            ]
        ]
        disp.columns = [
            "Prediction file",
            "Subset",
            "data_source",
            "Correct / total",
            "Accuracy",
            "Source column",
        ]
        pd.set_option("display.max_rows", 500)
        print(disp.to_string(index=False))
        print()

    if args.output_csv is not None:
        out_csv = args.output_csv.resolve()
    elif args.predictions_dir is not None:
        out_csv = pred_dir.parent / "eval_physician_subsets_summary.csv"
    else:
        out_csv = QWEN14_ROOT / "results" / "eval_physician_subsets_summary.csv"
    try:
        long_df.to_csv(out_csv, index=False)
        print(f"Wrote long-format table: {out_csv}")
    except OSError as e:
        print(f"Could not write {out_csv}: {e}", file=sys.stderr)

    if args.by_data_source:
        if args.output_csv is not None:
            by_path = args.output_csv.resolve().with_name(
                "eval_physician_subsets_by_data_source.csv"
            )
        elif args.predictions_dir is not None:
            by_path = pred_dir.parent / "eval_physician_subsets_by_data_source.csv"
        else:
            by_path = QWEN14_ROOT / "results" / "eval_physician_subsets_by_data_source.csv"
        try:
            pd.DataFrame(by_source).to_csv(by_path, index=False)
            print(f"Wrote by-data_source table: {by_path}")
        except OSError as e:
            print(f"Could not write {by_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
