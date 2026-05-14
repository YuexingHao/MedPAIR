#!/usr/bin/env python3
"""
Convert combined long-format ``eval_physician_subsets_summary`` tables (from
``aggregate_physician_subset_evals.py``) into a wide CSV shaped like ``Result_Report.csv``:

- ``MMLU``  ← accuracy on subset **933** (percent 0–100)
- ``Jama``  ← **HardQA**
- ``MedXpert`` ← **Impossible**
- ``Medbullets`` ← unweighted mean of the three subset accuracies (distinct from **Total**)
- ``Total`` ← accuracy pooled by counts (weighted) across the three subsets

Each row is one (Base Model, Low+Irr Labelers) pair, inferred from ``project`` + ``prediction_file``.

Unrecognized files are skipped (see stderr when ``--verbose``).

CLI: With no positional ``input_csv``, the script scans ``--base`` like
``aggregate_physician_subset_evals.py`` (default: After_PT_Removal). ``-o/--output`` is optional;
default path is ``<NeuRIPS25>/Figures/PhysicianEval_Result_Report.csv``. With a positional
``input_csv`` and no ``-o``, writes ``PhysicianEval_Result_Report.csv`` next to that file.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# project folder name -> x-axis label in the MedPAIR figure
PROJECT_TO_BASE_MODEL: dict[str, str] = {
    "GPT4o": "GPT4o",
    "GPT5": "GPT 5",
    "Qwen2.5-14B-Instruct": "Qwen-14B",
    "Qwen2.5-72B-Instruct": "Qwen 72B",
    "MedGemma-27b-text-it": "MedGemma-27B",
    "Llama-70B": "Llama-70B",
}


def _strip_sr_prefix(name: str) -> str:
    if "[SR]" in name:
        name = name.split("]", 1)[-1].lstrip("_")
    return name.strip()


def infer_labeler(project: str, prediction_file: str) -> str | None:
    """Map prediction filename to Result_Report ``Low+Irr Labelers`` label."""
    b = _strip_sr_prefix(prediction_file).lower()

    # e.g. Qwen_14B_predictions_ORIGINAL.csv (*_original.csv),
    # Llama70B_annotated_ORIGINAL_Accuracy.csv (annotated_original_accuracy),
    # GPT5: gpt5_predictions_Original_Accuracy.csv (majority_vote runs)
    if re.search(
        r"_original\.csv|original_predictions\.csv|predictions_original|original_progress\.csv",
        b,
    ):
        return "Original Accuracy"
    if re.search(r"annotated_original_accuracy\.csv$", b):
        return "Original Accuracy"
    if "gpt5_predictions_original_accuracy" in b:
        return "Original Accuracy"
    if "gpt4o_predictions_original_accuracy" in b:
        return "Original Accuracy"
    if "trainee_irr" in b:
        return "Trainee Removed (IRR)"
    if "trainee" in b:
        return "Trainee Removed"
    if "on_gpt5_removed" in b or "on_gpt5_progress" in b:
        return "GPT5 Removed"
    if "on_gpt4o" in b or b.endswith("gpt4o.csv") or "_gpt4o_progress" in b:
        return "GPT-4o Removed"
    # Llama-70B exports are named ``Llama70B_predictions_on_<X>.csv`` — do **not** use
    # ``"llama70b" in b`` (that matches every file and steals GPT5 / MedGemma / etc.).
    if re.search(r"on_gpt5\.csv$|on_gpt5_progress", b) or (
        "on_gpt5" in b and "removed" in b
    ):
        return "GPT5 Removed"
    if "on_medgemma" in b or "medgemma_removed" in b:
        return "MedGemma-27B Removed"
    if "_medgemma.csv" in b or b.endswith("medgemma.csv"):
        return "MedGemma-27B Removed"
    if "gemma_sr" in b or "predictions_gemma" in b or re.search(
        r"medgemma27b_predictions_gemma", b
    ):
        return "MedGemma-27B Removed"
    if "qwen72b" in b or "on_72b" in b or "on_qwen72b" in b:
        return "Qwen-72B Removed"
    if (
        "on_14b_removed" in b
        or "on_qwen14b" in b
        or "predictions_on_14b" in b
        or re.search(r"qwen_14b_predictions_14b\.csv$", b)
        or re.search(r"predictions_on_qwen14b", b)
        or re.search(r"predictions_qwen14b\.csv$", b)
    ):
        return "Qwen-14B Removed"
    if str(project) == "GPT5" and "gpt4o_predictions" in b and "14b" in b:
        return "Qwen-14B Removed (GPT4o CSV)"
    if project == "Qwen2.5-14B-Instruct" and b.endswith("gpt5.csv"):
        return "GPT5 Removed"
    if project == "Qwen2.5-14B-Instruct" and (
        "predictions_gpt5" in b.replace("-", "_") or "_gpt5.csv" in b
    ):
        return "GPT5 Removed"
    # Qwen-14B Llama-70B PT removal: ``Qwen_14B_predictions_70B.csv`` (not ``on_70b`` / ``on_llama``).
    if project == "Qwen2.5-14B-Instruct" and re.search(
        r"_70b\.csv$|_70b_progress\.csv$", b
    ):
        return "Llama-70b Removed"
    # Other models: ``*_on_Llama70B_removed.csv`` etc.
    if "on_llama" in b:
        return "Llama-70b Removed"
    # Llama-70B self: ``Llama70B_predictions_on_70B.csv``
    if re.search(r"on_70b\.csv$|on_70b_progress\.csv$", b) or "predictions_on_70b" in b:
        return "Llama-70b Removed"
    return None


def infer_base_model(project: str) -> str:
    return PROJECT_TO_BASE_MODEL.get(project, project)


def long_to_result_report(
    combined: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Pivot long physician-eval rows into Result_Report-shaped wide rows."""
    _n = combined["note"].fillna("")
    ok = _n.eq("") | _n.str.startswith("gold=")
    work = combined.loc[ok].copy()
    if work.empty:
        return pd.DataFrame()

    rows_out: list[dict[str, float | str]] = []
    skipped: list[tuple[str, str, str]] = []

    for (project, pred_file), g in work.groupby(["project", "prediction_file"]):
        labeler = infer_labeler(str(project), str(pred_file))
        if labeler is None:
            skipped.append((str(project), str(pred_file), "unmapped labeler"))
            continue

        by_sub = g.set_index("subset")
        need = {"933", "HardQA", "Impossible"}
        if not need.issubset(set(by_sub.index.astype(str))):
            skipped.append((str(project), str(pred_file), "missing subset row"))
            continue

        base = infer_base_model(str(project))
        acc: dict[str, float] = {}
        std: dict[str, float] = {}
        correct_t = 0
        n_t = 0
        acc_list: list[float] = []

        for sub in ("933", "HardQA", "Impossible"):
            r = by_sub.loc[sub]
            p = float(r["accuracy"])
            if math.isnan(p):
                skipped.append((str(project), str(pred_file), f"nan accuracy {sub}"))
                acc = {}
                break
            acc[sub] = p
            std[sub] = float(r["std"]) if pd.notna(r["std"]) else float("nan")
            c = int(r["correct"])
            n = int(r["total"])
            correct_t += c
            n_t += n
            acc_list.append(p)
        if not acc:
            continue

        total_p = correct_t / n_t if n_t > 0 else float("nan")
        total_se = (
            math.sqrt(total_p * (1.0 - total_p) / n_t)
            if n_t > 0 and total_p == total_p
            else float("nan")
        )

        macro_p = float(np.mean(acc_list))
        macro_se = (
            float(np.nanstd(acc_list, ddof=1) / math.sqrt(len(acc_list)))
            if len(acc_list) > 1
            else float("nan")
        )

        def pct(x: float) -> float:
            return 100.0 * x

        def pct_std(x: float) -> float:
            return 100.0 * x if x == x else float("nan")

        row: dict[str, float | str] = {
            "Base Model": base,
            "Low+Irr Labelers": labeler,
            "Total": pct(total_p),
            "MMLU": pct(acc["933"]),
            "Jama": pct(acc["HardQA"]),
            "MedXpert": pct(acc["Impossible"]),
            "Medbullets": pct(macro_p),
            "Total_STD": pct_std(total_se),
            "MMLU_STD": pct_std(std["933"]),
            "JAMA_STD": pct_std(std["HardQA"]),
            "MedXpert_STD": pct_std(std["Impossible"]),
            "MedBullets_STD": pct_std(macro_se),
            "_weight_n": float(n_t),
        }
        rows_out.append(row)

    rows_out = _dedupe_weighted_by_n(rows_out)

    if verbose and skipped:
        print("Skipped groups:", file=sys.stderr)
        for t in skipped[:50]:
            print(f"  {t}", file=sys.stderr)
        if len(skipped) > 50:
            print(f"  ... and {len(skipped) - 50} more", file=sys.stderr)

    if not rows_out:
        return pd.DataFrame()

    out = pd.DataFrame(rows_out)
    sort_order = [
        "Trainee",
        "Physician",
        "GPT4o",
        "GPT 5",
        "Qwen-14B",
        "Qwen 72B",
        "Llama-70B",
        "MedGemma-27B",
    ]
    labeler_order = [
        "Original Accuracy",
        "Trainee Removed",
        "Trainee Removed (IRR)",
        "Trainee Removed (248 QAs)",
        "Qwen-14B Removed",
        "Qwen-14B Removed (GPT4o CSV)",
        "Qwen-72B Removed",
        "Llama-70b Removed",
        "GPT-4o Removed",
        "GPT5 Removed",
        "MedGemma-27B Removed",
    ]

    def base_key(x: str) -> int:
        return sort_order.index(x) if x in sort_order else len(sort_order)

    def lab_key(x: str) -> int:
        return labeler_order.index(x) if x in labeler_order else len(labeler_order)

    out["_bk"] = out["Base Model"].map(base_key)
    out["_lk"] = out["Low+Irr Labelers"].map(lab_key)
    out = out.sort_values(["_bk", "_lk"]).drop(columns=["_bk", "_lk"])
    out = out.drop(columns=["_weight_n"], errors="ignore")
    return out.reset_index(drop=True)


def _dedupe_weighted_by_n(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate (Base Model, Labeler) rows via sample-size-weighted means."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["Base Model"]), str(r["Low+Irr Labelers"]))].append(r)

    numeric_cols = [
        "Total",
        "MMLU",
        "Jama",
        "MedXpert",
        "Medbullets",
        "Total_STD",
        "MMLU_STD",
        "JAMA_STD",
        "MedXpert_STD",
        "MedBullets_STD",
    ]
    out: list[dict[str, Any]] = []
    for grp in groups.values():
        if len(grp) == 1:
            out.append(grp[0])
            continue
        weights = np.array([float(r.get("_weight_n", 1.0)) for r in grp], dtype=float)
        s = float(weights.sum())
        if s <= 0:
            out.append(grp[0])
            continue
        w = weights / s
        merged: dict[str, Any] = {
            "Base Model": grp[0]["Base Model"],
            "Low+Irr Labelers": grp[0]["Low+Irr Labelers"],
            "_weight_n": s,
        }
        for c in numeric_cols:
            vals = np.array([float(r[c]) for r in grp], dtype=float)
            merged[c] = float(np.dot(w, vals))
        out.append(merged)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=None,
        help="Combined long CSV (default: stdin or use --from-aggregate)",
    )
    p.add_argument(
        "--from-aggregate",
        action="store_true",
        help="Explicitly scan --base for summaries (same as running with no positional input_csv).",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Root for --from-aggregate (default: After_PT_Removal).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output wide CSV path. If omitted when scanning projects: "
            "NeuRIPS25/Figures/PhysicianEval_Result_Report.csv. "
            "If omitted with a positional input_csv: same directory as input."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Print skipped files to stderr.")
    return p.parse_args()


def _find_summaries(base: Path) -> list[Path]:
    return sorted(base.glob("**/eval_physician_subsets_summary.csv"))


def main() -> None:
    args = parse_args()
    after_pt = Path(__file__).resolve().parent.parent
    workspace_root = after_pt.parent
    base = (args.base or after_pt).resolve()

    def _default_output_path() -> Path:
        if use_aggregate:
            return (workspace_root / "Figures" / "PhysicianEval_Result_Report.csv").resolve()
        assert args.input_csv is not None
        return Path(args.input_csv).resolve().parent / "PhysicianEval_Result_Report.csv"

    # No positional CSV => scan --base for eval_physician_subsets_summary.csv (same as aggregate).
    use_aggregate = args.input_csv is None

    if use_aggregate:
        paths = _find_summaries(base)
        if not paths:
            print(f"No eval_physician_subsets_summary.csv under {base}", file=sys.stderr)
            sys.exit(1)
        frames: list[pd.DataFrame] = []
        for csv_path in paths:
            df = pd.read_csv(csv_path)
            proj = csv_path.relative_to(base).parts[0]
            df.insert(0, "project", proj)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
    elif args.input_csv is not None:
        combined = pd.read_csv(args.input_csv)
    else:
        print("Internal error: no input source.", file=sys.stderr)
        sys.exit(1)

    if "project" not in combined.columns:
        print(
            "Input must include a `project` column (use combined output from "
            "aggregate_physician_subset_evals.py or --from-aggregate).",
            file=sys.stderr,
        )
        sys.exit(1)

    out = long_to_result_report(combined, verbose=args.verbose)
    out_path = (args.output or _default_output_path()).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
