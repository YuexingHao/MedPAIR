#!/usr/bin/env python3
"""
MedGemma-27B SR inference on the llama70b-removed reduced context, batched.

Why batching: MedGemma reasons for ~1900 tokens before emitting <answer>, so the
2048-token budget is genuinely required (it is what took coverage from 5% to
100%). One sequence at a time ran at ~81 s/row => ~29 h for 1300 rows. Decoding
is memory-bandwidth bound, so batching many sequences through the same weight
reads costs little extra time per row.

Everything that determines the *result* is unchanged and still comes from
shared/scripts/sr_common.py: same prompt, same chat template, same
[SR]High / question_options columns, same parser, same greedy decode. This keeps
the model on transformers, matching the completed Qwen-14B run.

Usage:
    python run_medgemma_sr_inference_batched.py [--limit N] [--batch-size 16] [--dry-run]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts")
from sr_common import (  # noqa: E402
    CONTEXT_COL, QUESTION_COL, KEY_COL,
    chat_messages, extract_letter, load_sr_input, summarize,
)

TITLE = "MEDGEMMA-27B-TEXT-IT (batched)"
MODEL_ID = ("/orcd/compute/mghassem/001/gobi1/huggingface/hub/"
            "models--google--medgemma-27b-text-it/snapshots/"
            "5b667cf2ddcf064085bc90952edb35a0edbfb79c")
PRED_COL = "medgemma27b_direct_prediction"
OUTPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/"
                   "results/predictions/[SR]_MedGemma27B_predictions_on_llama70b_removed.csv")

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_BATCH_SIZE = 16


def parse_args():
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N rows (pilot runs).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="Attention kernel. Fall back to eager if sdpa misbehaves.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the input into this many shards (Slurm array).")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Which shard this job handles, 0-based.")
    p.add_argument("--partial-out", type=Path, default=None,
                   help="Write Origin,prediction for this shard here instead of "
                        "editing the shared CSV. Required when sharding: parallel "
                        "jobs writing the same file would clobber each other.")
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
    print(f"max_new_tokens:  {args.max_new_tokens}   batch_size: {args.batch_size}")
    print()

    df_input = load_sr_input(limit=args.limit)
    print(f"Loaded {len(df_input)} input rows")

    if args.num_shards > 1:
        if args.partial_out is None:
            print("ERROR: --partial-out is required with --num-shards > 1, "
                  "otherwise parallel shards overwrite each other's results.")
            sys.exit(1)
        # Strided slice keeps each shard's prompt-length mix similar, so the
        # shards finish in comparable time.
        df_input = df_input.iloc[args.shard_index::args.num_shards].copy()
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(df_input)} rows")

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

    # On H200 nodes with cuDNN 9.19 the cuDNN attention backend cannot build an
    # execution plan for Gemma3's batched padded attention and every batch dies
    # with "No valid execution plans built". Flash / mem-efficient handle it fine,
    # so drop cuDNN out of the SDPA backend list. Harmless on other GPUs.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("cuDNN SDPA backend disabled (flash/mem-efficient remain enabled)")

    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    # Decoder-only batched generation requires LEFT padding, otherwise the
    # generated continuation starts after pad tokens and comes out garbage.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        attn_implementation=args.attn,
    )
    model.eval()
    print(f"  ✓ Model loaded (attn={args.attn})")

    # Sort by prompt length so each batch pads to a similar width - less wasted
    # compute on padding. Results are mapped back by Origin, so order is safe.
    prompts, keys = [], []
    for _, row in df_input.iterrows():
        context = str(row.get(CONTEXT_COL, "")).strip()
        question = str(row.get(QUESTION_COL, "")).strip()
        if not context or not question:
            continue
        prompts.append(tokenizer.apply_chat_template(
            chat_messages(context, question), tokenize=False, add_generation_prompt=True))
        keys.append(row[KEY_COL])

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    prompts = [prompts[i] for i in order]
    keys = [keys[i] for i in order]
    print(f"Built {len(prompts)} prompts (length-sorted for efficient padding)")

    print("\n" + "=" * 80)
    print(f"Running inference on {len(prompts)} samples...")
    print("=" * 80 + "\n")

    by_key = {}
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for b in tqdm(range(n_batches), total=n_batches, desc="batches"):
        lo = b * args.batch_size
        chunk = prompts[lo:lo + args.batch_size]
        chunk_keys = keys[lo:lo + args.batch_size]

        try:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(42)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Left padding makes the prompt width uniform across the batch, so a
            # single slice point is correct for every row.
            gen = outputs[:, inputs["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for k, t in zip(chunk_keys, texts):
                by_key[k] = t.strip()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  batch {b}: OOM ({e}); retry these rows with a smaller --batch-size")
            torch.cuda.empty_cache()
            for k in chunk_keys:
                by_key[k] = "Error: oom"
        except Exception as e:
            print(f"  batch {b}: {type(e).__name__}: {e}")
            for k in chunk_keys:
                by_key[k] = f"Error: {e}"

        if b == 0:
            for k in chunk_keys[:2]:
                print(f"  [sample raw response {k}]: {by_key[k][:300]!r}")

    raw_responses = [by_key.get(k, "Error: no_output") for k in df_input[KEY_COL]]
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
        print("\n  • --dry-run: nothing written")
    elif args.partial_out is not None:
        args.partial_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({KEY_COL: list(new_by_key.keys()),
                      PRED_COL: list(new_by_key.values())}).to_csv(
            args.partial_out, index=False)
        print(f"\n  ✓ Wrote {len(new_by_key)} predictions to partial: {args.partial_out}")
        print("     merge into the main CSV with shared/scripts/merge_sr_partials.py")
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

    # Per-batch error handling means a run where every batch died still reaches
    # this point. Exit non-zero so Slurm reports FAILED instead of COMPLETED and
    # a total failure cannot be mistaken for a finished run.
    if stats["n"] and stats["scored"] == 0:
        print("\nERROR: no row produced a parseable answer - treating as failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
