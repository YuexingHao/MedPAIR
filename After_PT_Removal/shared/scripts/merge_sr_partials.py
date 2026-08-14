#!/usr/bin/env python3
"""Merge sharded SR predictions into a model's prediction CSV.

Sharded jobs each write Origin,<pred_col> to their own partial file so that
parallel writers cannot clobber a shared CSV. This merges them in one pass.

    python merge_sr_partials.py \
        --target ".../[SR]_MedGemma27B_predictions_on_llama70b_removed.csv" \
        --pred-col medgemma27b_direct_prediction \
        --partials ".../partials/medgemma_shard_*.csv"
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_common import KEY_COL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--pred-col", required=True)
    ap.add_argument("--partials", required=True,
                    help="Glob for the shard partial CSVs.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.partials))
    if not paths:
        print(f"ERROR: no partials matched {args.partials}")
        sys.exit(1)
    print(f"Found {len(paths)} partial files")

    merged, dupes = {}, 0
    for p in paths:
        df = pd.read_csv(p)
        if KEY_COL not in df.columns or args.pred_col not in df.columns:
            print(f"ERROR: {p} lacks {KEY_COL}/{args.pred_col}; got {list(df.columns)}")
            sys.exit(1)
        for k, v in zip(df[KEY_COL], df[args.pred_col]):
            if pd.isna(v):
                continue
            if k in merged and merged[k] != v:
                # Shards are disjoint, so this means a shard was run twice with
                # different settings. Refuse rather than silently pick one.
                dupes += 1
            merged[k] = v
        print(f"  {Path(p).name}: {len(df)} rows")

    if dupes:
        print(f"ERROR: {dupes} keys appear in >1 partial with conflicting values. "
              f"Shards should be disjoint - check --num-shards/--shard-index.")
        sys.exit(1)

    print(f"\nTotal distinct predictions: {len(merged)}")

    if not args.target.exists():
        print(f"ERROR: target not found: {args.target}")
        sys.exit(1)
    out = pd.read_csv(args.target, low_memory=False)
    if args.pred_col not in out.columns:
        out[args.pred_col] = pd.NA
    # All-empty columns read back as float64; pandas 3 refuses to store letters.
    out[args.pred_col] = out[args.pred_col].astype("object")

    idx = {k: i for i, k in enumerate(out[KEY_COL])}
    updated, missing = 0, 0
    for k, v in merged.items():
        if k in idx:
            out.at[out.index[idx[k]], args.pred_col] = v
            updated += 1
        else:
            missing += 1

    print(f"Rows updated:        {updated}/{len(out)}")
    if missing:
        print(f"Keys not in target:  {missing}")

    still_blank = out[args.pred_col].isna().sum()
    print(f"Still blank/unfilled: {still_blank}")

    if args.dry_run:
        print("\n  • --dry-run: target left untouched")
    else:
        out.to_csv(args.target, index=False)
        print(f"\n  ✓ Wrote {args.target}")


if __name__ == "__main__":
    main()
