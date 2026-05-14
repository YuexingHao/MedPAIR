#!/usr/bin/env python3
"""Rebuild merged_attribution_scores_14B.csv from a directory of Merge_Q*.csv files.

The repo no longer stores thousands of per-question CSVs under attribution; keep an
extracted tarball elsewhere and pass `--input-dir` to rebuild the merged CSV.

Usage:
  python merge_14b_per_question_csvs.py --input-dir /path/to/Merge_Q_folder
  python merge_14b_per_question_csvs.py --input-dir ... --check-only
  python merge_14b_per_question_csvs.py --input-dir ... --parquet out.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MERGE_ROOT = Path(__file__).resolve().parents[1]
if str(MERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(MERGE_ROOT))
import paths  # noqa: E402


def concat_merge_q(input_dir: Path) -> tuple[pd.DataFrame, int]:
    files = sorted(input_dir.glob("Merge_Q*.csv"))
    if not files:
        raise FileNotFoundError(f"No Merge_Q*.csv under {input_dir}")
    n_files = len(files)
    parts = [pd.read_csv(f) for f in files]
    out = pd.concat(parts, ignore_index=True)
    if "QA_ID" in out.columns:
        out["_n"] = out["QA_ID"].astype(str).str.extract(r"Q(\d+)", expand=False).astype(float)
        out = out.sort_values("_n", kind="mergesort").drop(columns="_n")
    return out.reset_index(drop=True), n_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Merge_Q*.csv (e.g. extracted archive)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=paths.MERGED_ATTRIBUTION_SCORES_14B,
        help="Output merged CSV path",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Compare row count to existing merged CSV; do not write",
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="If set, also write this Parquet path (single file, fast for pandas)",
    )
    args = ap.parse_args()

    df, n_files = concat_merge_q(args.input_dir.resolve())
    print(f"From {args.input_dir}: {n_files} files -> {len(df)} rows")

    if args.check_only:
        if not args.output.exists():
            print(f"check-only: {args.output} missing; cannot compare.")
            sys.exit(1)
        existing = pd.read_csv(args.output, low_memory=False)
        if len(existing) != len(df):
            print(f"Mismatch: existing merged has {len(existing)} rows, concat has {len(df)}")
            sys.exit(1)
        print("check-only: row counts match.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    if args.parquet:
        df.to_parquet(args.parquet, index=False)
        print(f"Wrote {args.parquet}")


if __name__ == "__main__":
    main()
