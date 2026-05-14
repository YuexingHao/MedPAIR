#!/usr/bin/env python3
"""
Concatenate every ``eval_physician_subsets_summary.csv`` under ``After_PT_Removal/*/results/``.

Each per-model script writes that file next to its predictions, e.g.:
  - ``GPT4o/scripts/evaluate_predictions_by_physician_subsets_gpt4o.py`` → ``GPT4o/results/eval_physician_subsets_summary.csv``
  - ``GPT5/scripts/..._gpt5.py`` → ``GPT5/results/...``
  - ``Qwen2.5-72B-Instruct/results/scripts/..._qwen72b.py`` → ``Qwen2.5-72B-Instruct/results/...``
  - ``MedGemma-27b-text-it/results/scripts/..._medgemma27b.py`` → ``MedGemma-27b-text-it/results/...``
  - ``Llama-70B/scripts/evaluate_predictions_by_physician_subsets_llama70b.py`` → ``Llama-70B/results/eval_physician_subsets_summary.csv``

Run (from repo root or anywhere):

  python After_PT_Removal/scripts/aggregate_physician_subset_evals.py

Options:
  --output PATH   Write combined long-format CSV
  --result-report PATH  Also write a wide CSV shaped like GPT4o ``Result_Report.csv`` (physician
                    subsets: 933→MMLU, HardQA→Jama, Impossible→MedXpert); see
                    ``physician_eval_to_result_report.py``.
  --wide          Print a compact pivot: project × subset → mean accuracy (numeric rows only)
  --quiet         Only print paths found + optional --output line
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# scripts/aggregate_... -> After_PT_Removal is parent
AFTER_PT_REMOVAL = Path(__file__).resolve().parent.parent


def find_summary_csvs(base: Path) -> list[tuple[str, Path]]:
    """Return (project_folder_name, csv_path) for each eval summary under base."""
    found: list[tuple[str, Path]] = []
    for path in sorted(base.glob("**/eval_physician_subsets_summary.csv")):
        rel = path.relative_to(base)
        if len(rel.parts) < 2:
            continue
        project = rel.parts[0]
        found.append((project, path))
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--base",
        type=Path,
        default=AFTER_PT_REMOVAL,
        help=f"Root to scan (default: {AFTER_PT_REMOVAL})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write combined table with column `project` to this CSV.",
    )
    p.add_argument(
        "--result-report",
        type=Path,
        default=None,
        help=(
            "Write Result_Report-shaped wide CSV (Base Model × labeler, physician subset accuracies). "
            "Uses physician_eval_to_result_report.long_to_result_report."
        ),
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="Print mean accuracy by project × subset (excludes error/skip rows).",
    )
    p.add_argument("--quiet", action="store_true", help="Minimal stdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    if not base.is_dir():
        print(f"Not a directory: {base}", file=sys.stderr)
        sys.exit(1)

    pairs = find_summary_csvs(base)
    if not pairs:
        print(f"No eval_physician_subsets_summary.csv under {base}", file=sys.stderr)
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for project, csv_path in pairs:
        df = pd.read_csv(csv_path)
        df.insert(0, "project", project)
        df["_source_path"] = str(csv_path.relative_to(base))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if args.result_report is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from physician_eval_to_result_report import long_to_result_report

        rr = long_to_result_report(combined, verbose=not args.quiet)
        rr_path = args.result_report.resolve()
        rr_path.parent.mkdir(parents=True, exist_ok=True)
        rr.to_csv(rr_path, index=False)
        if not args.quiet:
            print(f"Wrote Result_Report-style table: {rr_path} ({len(rr)} rows)")

    if args.output is not None:
        out = combined.drop(columns=["_source_path"], errors="ignore")
        out.to_csv(args.output.resolve(), index=False)
        if not args.quiet:
            print(f"Wrote {args.output.resolve()} ({len(out)} rows)")

    if args.quiet and (args.output is not None or args.result_report is not None):
        return

    if not args.quiet:
        print(f"Scanned: {base}")
        for project, csv_path in pairs:
            print(f"  [{project}] {csv_path.relative_to(base)}")
        print()

    _n = combined["note"].fillna("")
    ok = _n.eq("") | _n.str.startswith("gold=")
    work = combined.loc[ok].copy()

    if args.wide and not work.empty:
        pivot = work.pivot_table(
            index="project",
            columns="subset",
            values="accuracy",
            aggfunc="mean",
        )
        subset_order = ["933", "HardQA", "Impossible"]
        pivot = pivot.reindex(columns=[c for c in subset_order if c in pivot.columns])
        print(
            "Mean accuracy by project × subset (mean over all prediction CSV rows in that project; "
            "use long table below for per-file detail)"
        )
        print("—" * 80)
        print((pivot * 100).map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—").to_string())
        print()

    disp = combined.drop(columns=["_source_path"], errors="ignore")
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 55)
    print("Combined long-format (all rows)")
    print("—" * 80)
    print(disp.to_string(index=False))

    bad = combined.loc[~ok, ["project", "prediction_file", "note"]].drop_duplicates()
    if not bad.empty:
        print()
        print("Rows with notes (skipped / errors)")
        print("—" * 80)
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
