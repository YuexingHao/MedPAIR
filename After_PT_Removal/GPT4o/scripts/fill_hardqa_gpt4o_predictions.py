#!/usr/bin/env python3
"""
Fill missing **Hard QA (733)** rows in ``gpt4o_predictions_Original_Accuracy.csv``.

``eval_physician_subsets_summary.csv`` only shows ``total=33`` for HardQA because the prediction
CSV shares ``Origin`` with the 933 + Impossible pool (1300 rows) and intersects HardQA IDs in only
33 places. This script:

1. Loads ``HardQA_Clinician_Student_Majority_Vote.csv`` (733 ``ID`` values).
2. Finds ``ID`` values absent from the prediction file's ``Origin`` column (~700 rows).
3. Calls **GPT-4o** with the same prompt shape as the GPT4o notebooks (context + ``question_options``).
   Context is ``step1_excerpts`` from the HardQA table (same clinical text as ``answer_corr`` / keys).
4. Merges new rows into the predictions CSV (default: writes a new file; use ``--in-place`` carefully).

Requirements:

- **API key:** set ``export OPENAI_API_KEY='sk-...'`` *or* pass ``--api-key-file /path/to/key.txt``
  (first non-empty, non-``#`` line). Do not commit keys.
- Optional: ``data/raw/merged_2k_with_4k_ID.csv`` for ``4k_ID`` (all 733 HardQA IDs are present).

After updating predictions, re-run::

  python After_PT_Removal/GPT4o/scripts/evaluate_predictions_by_physician_subsets_gpt4o.py

Examples::

  # Count gaps only
  python After_PT_Removal/GPT4o/scripts/fill_hardqa_gpt4o_predictions.py --dry-run

  # Test two calls, write merged file next to original
  python After_PT_Removal/GPT4o/scripts/fill_hardqa_gpt4o_predictions.py --limit 2

  # Full run (~700 API calls)
  python After_PT_Removal/GPT4o/scripts/fill_hardqa_gpt4o_predictions.py --in-place

  # Key in a file (e.g. on a cluster)
  python After_PT_Removal/GPT4o/scripts/fill_hardqa_gpt4o_predictions.py --api-key-file ~/.openai_api_key
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GPT4O_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRED = GPT4O_ROOT / "results" / "predictions" / "gpt4o_predictions_Original_Accuracy.csv"
DEFAULT_HARD = (
    GPT4O_ROOT.parent.parent
    / "Physician_Labels"
    / "Mar2_2026_Data"
    / "HardQA_Clinician_Student_Majority_Vote.csv"
)
DEFAULT_MERGED = GPT4O_ROOT / "data" / "raw" / "merged_2k_with_4k_ID.csv"


def resolve_openai_api_key(api_key_file: Path | None) -> str:
    """Prefer ``OPENAI_API_KEY``, then first non-comment line of ``api_key_file``."""
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    if api_key_file is not None:
        path = Path(api_key_file).expanduser().resolve()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return line.strip().strip("'\"")
    return ""


def extract_letter_from_text(x: str | float | None) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
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


def build_prompt(context: str, question: str) -> str:
    return f"""You are given some context and a multiple-choice question.

Select the most appropriate answer from the options provided.

{context}

{question}

Provide your response in the following format:
<answer>Option [letter]</answer>"""


def call_gpt4o(
    prompt: str,
    *,
    api_key: str,
    model: str,
    sleep_s: float,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("Install openai: pip install openai") from e

    if not api_key:
        raise SystemExit(
            "No API key: set environment variable OPENAI_API_KEY or use --api-key-file PATH "
            "(file with the key on the first non-empty line).\n"
            "Example: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    if sleep_s > 0:
        time.sleep(sleep_s)
    return text


def row_from_hardqa(
    template: pd.Series,
    hard: pd.Series,
    four_k: str | float,
    pred_raw: str,
) -> dict[str, object]:
    """Shape one output row like ``gpt4o_predictions_Original_Accuracy.csv``."""
    letter = extract_letter_from_text(pred_raw)
    hid = str(hard["ID"]).strip()

    out: dict[str, object] = {}
    for col in template.index:
        out[col] = np.nan

    # Optional numeric / string columns from HardQA + merge
    sn = hard.get("sentence_number")
    try:
        sn_int = int(sn) if pd.notna(sn) else np.nan
    except (TypeError, ValueError):
        sn_int = np.nan

    out["Origin"] = hid
    out["ID_corr"] = hid
    out["sentence_number_corr"] = sn_int
    out["answer_corr"] = str(hard["answer"]).strip().upper()
    out["data_source_corr"] = str(hard["data_source"]).strip()
    out["step1_excerpts"] = str(hard["step1_excerpts"])
    out["question_options"] = str(hard["question_options"])
    out["Filtered_Sentences"] = ""
    out["New_Sentences"] = ""
    out["gpt_direct_prediction"] = pred_raw
    out["Round 1"] = pred_raw
    out["Round 2"] = ""
    out["Round 1 Letter"] = letter if letter else ""
    out["Round 2 Letter"] = ""
    out["Correct Letter"] = out["answer_corr"]
    out["4k_ID"] = "" if four_k is None or (isinstance(four_k, float) and np.isnan(four_k)) else str(four_k)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    p.add_argument("--hardqa", type=Path, default=DEFAULT_HARD)
    p.add_argument("--merged-2k", type=Path, default=DEFAULT_MERGED)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Merged CSV path (default: <predictions_stem>_hardqa_filled.csv).",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --predictions (backup recommended).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts and exit.")
    p.add_argument("--limit", type=int, default=None, help="Max new API calls (for testing).")
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep after each API call (rate limits).",
    )
    p.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Append each new row to this CSV as you go (resume not automatic; use for safety).",
    )
    p.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Read OpenAI API key from first non-empty line if OPENAI_API_KEY is unset.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = args.predictions.resolve()
    hard_path = args.hardqa.resolve()
    merged_path = args.merged_2k.resolve()

    if not pred_path.is_file():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        sys.exit(1)
    if not hard_path.is_file():
        print(f"HardQA CSV not found: {hard_path}", file=sys.stderr)
        sys.exit(1)

    hard_df = pd.read_csv(hard_path)
    if "ID" not in hard_df.columns:
        print("HardQA CSV must have an ID column.", file=sys.stderr)
        sys.exit(1)

    merged_4k: pd.Series | None = None
    if merged_path.is_file():
        mdf = pd.read_csv(merged_path)
        if "ID" in mdf.columns and "4k_ID" in mdf.columns:
            merged_4k = mdf.set_index(mdf["ID"].astype(str).str.strip())["4k_ID"]
    else:
        print(f"Note: {merged_path} missing; 4k_ID will be empty.", file=sys.stderr)

    pred_df = pd.read_csv(pred_path, low_memory=False)
    if "Origin" not in pred_df.columns:
        print("Predictions CSV must have Origin.", file=sys.stderr)
        sys.exit(1)

    hard_ids = hard_df["ID"].astype(str).str.strip()
    have = set(pred_df["Origin"].astype(str).str.strip())
    missing_mask = ~hard_ids.isin(have)
    missing_df = hard_df.loc[missing_mask].copy()
    n_missing = len(missing_df)

    print(f"HardQA rows: {len(hard_df)} | Already in predictions: {len(hard_df) - n_missing} | Missing: {n_missing}")

    if args.dry_run:
        return

    if n_missing == 0:
        print("Nothing to do.")
        return

    api_key = resolve_openai_api_key(args.api_key_file)
    if not api_key:
        print(
            "No API key: set OPENAI_API_KEY or pass --api-key-file PATH\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "  echo 'sk-...' > ~/.openai_api_key && chmod 600 ~/.openai_api_key\n"
            "  python .../fill_hardqa_gpt4o_predictions.py --api-key-file ~/.openai_api_key",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit is not None:
        missing_df = missing_df.head(args.limit).copy()
        print(f"Processing (--limit) {len(missing_df)} rows.")
    else:
        print(f"Processing {len(missing_df)} rows (API calls).")

    template = pred_df.iloc[0]
    new_rows: list[dict[str, object]] = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # type: ignore

    for _, hr in tqdm(missing_df.iterrows(), total=len(missing_df)):
        hid = str(hr["ID"]).strip()
        four_k = merged_4k[hid] if merged_4k is not None and hid in merged_4k.index else np.nan

        context = str(hr["step1_excerpts"])
        question = str(hr["question_options"])
        prompt = build_prompt(context, question)
        pred_raw = call_gpt4o(
            prompt,
            api_key=api_key,
            model=args.model,
            sleep_s=args.sleep,
        )
        row = row_from_hardqa(template, hr, four_k, pred_raw)
        new_rows.append(row)

        if args.progress is not None:
            chunk = pd.DataFrame([row])
            header = not args.progress.is_file()
            chunk.to_csv(
                args.progress,
                mode="a",
                index=False,
                header=header,
            )

    new_df = pd.DataFrame(new_rows, columns=pred_df.columns)
    merged = pd.concat([pred_df, new_df], ignore_index=True)

    out_path = pred_path if args.in_place else (args.out or pred_path.with_name(f"{pred_path.stem}_hardqa_filled{pred_path.suffix}"))
    out_path = out_path.resolve()
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(merged)} rows).")

    if args.in_place:
        print("Re-run: python After_PT_Removal/GPT4o/scripts/evaluate_predictions_by_physician_subsets_gpt4o.py")


if __name__ == "__main__":
    main()
