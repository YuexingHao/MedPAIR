#!/usr/bin/env python3
"""
Generate QWEN2.5-14B-INSTRUCT SR predictions on the llama70b-removed reduced context.

Prompt, input columns and answer parsing are imported from
shared/scripts/sr_common.py so that every model in Table 6 is scored identically.

Input:  SR_Predictions/Llama-70B_Removed/Llama_70B_[SR]_predictions.csv
        context = [SR]High, question = question_options, key = Origin
Output: Updates qwen14b_direct_prediction in
  /home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/[SR]_Qwen14B_predictions_on_llama70b_removed.csv

GENERATED FILE - edit scratchpad/gen_local.py or sr_common.py, not this copy.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts")
from sr_common import (  # noqa: E402
    CONTEXT_COL, QUESTION_COL, KEY_COL, GOLD_COL,
    build_prompt, chat_messages, extract_letter, load_sr_input, summarize,
)

TITLE = "QWEN2.5-14B-INSTRUCT"
MODEL_ID = "/orcd/compute/mghassem/001/gobi1/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
PRED_COL = "qwen14b_direct_prediction"
OUTPUT_FILE = Path("/home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/[SR]_Qwen14B_predictions_on_llama70b_removed.csv")

# 2048 tokens for every local model: MedGemma-27B emits a long reasoning
# preamble before its answer, and a shared budget keeps truncation from being
# the thing that differentiates the models.
DEFAULT_MAX_NEW_TOKENS = 2048

# NOTE: an early-stop on "</answer>" was tried and measured to give no speedup -
# MedGemma-27B only emits the tag near the end of its ~1900-token reasoning, so
# there is nothing to cut. Throughput has to come from batching instead; see
# MedGemma-27b-text-it/notebooks/run_medgemma_sr_inference_batched.py.


def generate_prediction(context, question, model, tokenizer, seed=42,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        use_chat_template=True):
    """Greedy decode one answer.

    The chat template is on by default. Feeding these instruction-tuned models a
    bare prompt makes them drop the <answer> tags entirely - Qwen-14B scored 0/20
    that way - so the template is part of the aligned setup, not a per-model tweak.
    """
    try:
        if use_chat_template:
            text = tokenizer.apply_chat_template(
                chat_messages(context, question),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = build_prompt(context, question)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.manual_seed(seed)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy, deterministic
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens; slicing the decoded string by prompt length
        # is wrong once a chat template has been applied.
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response or "Error: empty_response"
    except Exception as e:
        return f"Error: {e}"


def parse_args():
    p = argparse.ArgumentParser(description=f"{TITLE} SR inference")
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N rows (pilot runs).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--no-chat-template", action="store_true",
                   help="Feed the bare prompt instead. Diagnostics only - it breaks alignment.")
    p.add_argument("--dry-run", action="store_true",
                   help="Score and report without writing the output CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    use_chat_template = not args.no_chat_template

    print("=" * 80)
    print(f"{TITLE} SR INFERENCE - REDUCED CONTEXT")
    print("=" * 80)
    print(f"Start Time:      {datetime.now()}")
    print(f"Model:           {MODEL_ID}")
    print(f"Output File:     {OUTPUT_FILE}")
    print(f"Prediction col:  {PRED_COL}")
    print(f"Context col:     {CONTEXT_COL}   Question col: {QUESTION_COL}")
    print(f"Chat template:   {use_chat_template}   max_new_tokens: {args.max_new_tokens}")
    print()

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
    if KEY_COL not in df_output.columns:
        print(f"ERROR: '{KEY_COL}' not found in output file")
        sys.exit(1)

    # On the H200 nodes (cuDNN 9.19) the cuDNN SDPA backend fails to build an
    # execution plan for Gemma3 attention. Flash / mem-efficient handle it fine.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    print(f"\nLoading model: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,        # transformers 5.x renamed torch_dtype
            device_map="auto",
            local_files_only=True,
        )
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        sys.exit(1)

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

        response = generate_prediction(
            context, question, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=use_chat_template,
        )
        raw_responses.append(response)
        if i < 2:
            print(f"  [sample raw response row {i}]: {response[:300]!r}")

    stats = summarize(df_input, raw_responses, PRED_COL)

    # Merge: only parseable letters overwrite the output column, so a failed row
    # never masquerades as a prediction.
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
