#!/usr/bin/env python3
"""
Generate GPT-5 SR predictions on the llama70b-removed reduced context.

Prompt, input columns and answer parsing are imported from
shared/scripts/sr_common.py so that every model in Table 6 is scored identically.

Input:  SR_Predictions/Llama-70B_Removed/Llama_70B_[SR]_predictions.csv
        context = [SR]High, question = question_options, key = Origin
Output: Updates gpt5_direct_prediction in
  /home/yuexing/NeuRIPS25/After_PT_Removal/GPT5/results/predictions/[SR]_GPT5_predictions_on_llama70b_removed.csv

GENERATED FILE - edit scratchpad/gen_api.py or sr_common.py, not this copy.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError

sys.path.insert(0, "/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts")
from sr_common import (  # noqa: E402
    CONTEXT_COL, QUESTION_COL, KEY_COL,
    chat_messages, extract_letter, load_sr_input, summarize,
)

TITLE = "GPT-5"
MODEL_ID = "gpt-5"
PRED_COL = "gpt5_direct_prediction"
OUTPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/GPT5/results/predictions/[SR]_GPT5_predictions_on_llama70b_removed.csv")

# Generation budget. Kept generous so truncation is never what separates the
# models; GPT-5 also pays for hidden reasoning tokens out of this budget.
TOKEN_PARAM = "max_completion_tokens"
TOKEN_BUDGET = 3000
EXTRA_PARAMS = {"reasoning_effort": "low"}

MAX_RETRIES = 5
INITIAL_BACKOFF = 2
MAX_BACKOFF = 60


def generate_prediction(context, question, client):
    """One API call with exponential backoff. Returns raw text or an Error: string."""
    kwargs = dict(
        model=MODEL_ID,
        messages=chat_messages(context, question),
        **{TOKEN_PARAM: TOKEN_BUDGET},
        **EXTRA_PARAMS,
    )

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content or not content.strip():
                return (f"Error: empty_content "
                        f"(finish_reason={response.choices[0].finish_reason})")
            return content

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = min(backoff, MAX_BACKOFF)
                print(f"  Rate limited; retrying in {wait}s "
                      f"({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                backoff *= 2
            else:
                return "Error: rate_limit_exceeded"

        except Exception as e:
            msg = str(e)
            if "rate" in msg.lower() and attempt < MAX_RETRIES - 1:
                wait = min(backoff, MAX_BACKOFF)
                time.sleep(wait)
                backoff *= 2
                continue
            return f"Error: {msg[:200]}"


def parse_args():
    p = argparse.ArgumentParser(description=f"{TITLE} SR inference")
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N rows (pilot runs, keeps API cost down).")
    p.add_argument("--dry-run", action="store_true",
                   help="Score and report without writing the output CSV.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(f"{TITLE} SR INFERENCE - REDUCED CONTEXT (OpenAI API)")
    print("=" * 80)
    print(f"Start Time:      {datetime.now()}")
    print(f"Model:           {MODEL_ID}")
    print(f"Output File:     {OUTPUT_FILE}")
    print(f"Prediction col:  {PRED_COL}")
    print(f"Context col:     {CONTEXT_COL}   Question col: {QUESTION_COL}")
    print(f"{TOKEN_PARAM}: {TOKEN_BUDGET}  extra: {EXTRA_PARAMS}")
    print()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    df_input = load_sr_input(limit=args.limit)
    print(f"Loaded {len(df_input)} input rows")

    if not OUTPUT_FILE.exists():
        print(f"ERROR: output file not found: {OUTPUT_FILE}")
        sys.exit(1)
    df_output = pd.read_csv(OUTPUT_FILE, low_memory=False)
    print(f"Loaded {len(df_output)} output rows")
    if PRED_COL not in df_output.columns:
        print(f"  • {PRED_COL} absent from output file; it will be created")
        df_output[PRED_COL] = pd.NA
    # An all-empty column reads back as float64, and pandas 3 refuses to store a
    # letter in it ("Invalid value 'D' for dtype 'float64'"). Force object first.
    df_output[PRED_COL] = df_output[PRED_COL].astype("object")

    print("\n" + "=" * 80)
    print(f"Running inference on {len(df_input)} samples...")
    print("=" * 80 + "\n")

    raw_responses = []
    for i, (_, row) in enumerate(tqdm(df_input.iterrows(), total=len(df_input))):
        context = str(row.get(CONTEXT_COL, "")).strip()
        question = str(row.get(QUESTION_COL, "")).strip()
        if not context or not question:
            raw_responses.append("Error: missing_context_or_question")
            continue

        response = generate_prediction(context, question, client)
        raw_responses.append(response)
        if i < 2:
            print(f"  [sample raw response row {i}]: {response[:300]!r}")

    stats = summarize(df_input, raw_responses, PRED_COL)

    letters = [extract_letter(r) for r in raw_responses]
    new_by_key = {k: v for k, v in zip(df_input[KEY_COL], letters) if v}

    updated = 0
    key_to_idx = {k: i for i, k in enumerate(df_output[KEY_COL])}
    for key, letter in new_by_key.items():
        if key in key_to_idx:
            df_output.at[df_output.index[key_to_idx[key]], PRED_COL] = letter
            updated += 1

    if args.dry_run:
        print("\n  • --dry-run: output CSV left untouched")
    else:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(OUTPUT_FILE, index=False)
        print(f"\n  ✓ Saved {updated} predictions to: {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples:         {stats['n']}")
    print(f"Parsed predictions:    {stats['scored']}")
    print(f"Unparsed responses:    {stats['unparsed']}")
    print(f"Coverage:              {stats['coverage']:.1f}% ({stats['scored']}/{stats['n']})")
    print(f"Accuracy (parsed):     {stats['acc_scored']:.1f}% ({stats['correct']}/{stats['scored']})")
    print(f"Accuracy (all rows):   {stats['acc_all']:.1f}% ({stats['correct']}/{stats['n']})")
    print(f"Output rows updated:   {updated}")
    print(f"End Time:              {datetime.now()}")
    print("=" * 80)

    # Per-row error handling means a run where every row failed still reaches
    # this point. Exit non-zero so Slurm reports FAILED instead of COMPLETED and
    # a total failure cannot be mistaken for a finished run.
    if stats["n"] and stats["scored"] == 0:
        print("\nERROR: no row produced a parseable answer - treating as failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
