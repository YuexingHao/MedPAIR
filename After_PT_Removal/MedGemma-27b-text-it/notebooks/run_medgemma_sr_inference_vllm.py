#!/usr/bin/env python3
"""
MedGemma-27B SR inference on the llama70b-removed reduced context, via vLLM.

Why a separate engine for this model: MedGemma reasons for ~1900 tokens before it
emits <answer>, so the 2048-token budget is genuinely needed, and HF generate()
with device_map="auto" (pipeline-parallel, one GPU active at a time) ran at
~81 s/row => ~29 h for 1300 rows. vLLM does tensor-parallel + continuous
batching, which is the same greedy decode at a fraction of the wall-clock.

Everything that determines the *result* is unchanged and still comes from
shared/scripts/sr_common.py: the same prompt, the same [SR]High / question_options
columns, the same answer parser, greedy decoding. Only the execution engine differs.

Usage:
    python run_medgemma_sr_inference_vllm.py [--limit N] [--dry-run]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, "/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts")
from sr_common import (  # noqa: E402
    CONTEXT_COL, QUESTION_COL, KEY_COL,
    chat_messages, extract_letter, load_sr_input, summarize,
)

TITLE = "MEDGEMMA-27B-TEXT-IT (vLLM)"
MODEL_ID = ("/orcd/compute/mghassem/001/gobi1/huggingface/hub/"
            "models--google--medgemma-27b-text-it/snapshots/"
            "5b667cf2ddcf064085bc90952edb35a0edbfb79c")
PRED_COL = "medgemma27b_direct_prediction"
OUTPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/"
                   "results/predictions/[SR]_MedGemma27B_predictions_on_llama70b_removed.csv")

MAX_NEW_TOKENS = 2048
# Longest SR prompt is 829 tokens; 3072 covers prompt + full output and leaves
# the rest of GPU memory for KV cache, which is what buys the batching.
MAX_MODEL_LEN = 3072


def parse_args():
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N rows (pilot runs).")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="Score and report without writing the output CSV.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(f"{TITLE} SR INFERENCE - REDUCED CONTEXT")
    print("=" * 80)
    print(f"Start Time:      {datetime.now()}")
    print(f"Model:           {MODEL_ID}")
    print(f"Output File:     {OUTPUT_FILE}")
    print(f"Prediction col:  {PRED_COL}")
    print(f"Context col:     {CONTEXT_COL}   Question col: {QUESTION_COL}")
    print(f"max_new_tokens:  {args.max_new_tokens}   TP size: {args.tensor_parallel_size}")
    print()

    df_input = load_sr_input(limit=args.limit)
    print(f"Loaded {len(df_input)} input rows")

    if not OUTPUT_FILE.exists():
        print(f"ERROR: output file not found: {OUTPUT_FILE}")
        sys.exit(1)
    df_output = pd.read_csv(OUTPUT_FILE, low_memory=False)
    print(f"Loaded {len(df_output)} output rows")
    if PRED_COL not in df_output.columns:
        df_output[PRED_COL] = pd.NA
    # An all-empty column reads back as float64 and pandas 3 refuses to store a
    # letter in it; force object before assigning.
    df_output[PRED_COL] = df_output[PRED_COL].astype("object")

    # Build prompts with the model's own chat template - byte-identical to what
    # the transformers path fed the model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    prompts, keys = [], []
    for _, row in df_input.iterrows():
        context = str(row.get(CONTEXT_COL, "")).strip()
        question = str(row.get(QUESTION_COL, "")).strip()
        if not context or not question:
            continue
        prompts.append(tokenizer.apply_chat_template(
            chat_messages(context, question), tokenize=False, add_generation_prompt=True))
        keys.append(row[KEY_COL])
    print(f"Built {len(prompts)} prompts")

    from vllm import LLM, SamplingParams

    print(f"\nLoading model into vLLM...")
    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
    )
    print("  ✓ Model loaded")

    sampling = SamplingParams(
        temperature=0.0,      # greedy, matches do_sample=False
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 80)
    print(f"Running inference on {len(prompts)} samples...")
    print("=" * 80 + "\n")

    outputs = llm.generate(prompts, sampling)

    # vLLM returns results in submission order, but sort by request id defensively.
    by_key = {}
    for key, out in zip(keys, outputs):
        by_key[key] = out.outputs[0].text.strip()

    raw_responses = [by_key.get(k, "Error: no_output") for k in df_input[KEY_COL]]
    for i, r in enumerate(raw_responses[:2]):
        print(f"  [sample raw response row {i}]: {r[:300]!r}")

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


if __name__ == "__main__":
    main()
