#!/usr/bin/env python3
"""
Generate answer + attribution files for every row in a CSV of
(context, question) pairs, with automatic resume support.

Example:
    python generate_attributions.py \
        --csv_path merged_llm_4k_questions.csv \
        --model_id meta-llama/Llama-3.1-70B-Instruct \
        --dtype float16 \
        --load_in_8bit \
        --device_map cuda:0 \
        --output_dir llama70b_attribution_scores \
        --flush_every 25 > logs/llama70b_attribution.log 2>&1 &

You can optionally specify a range of rows to process:
Example: rows 1 - 2 000
python generate_attributions.py \
    --csv_path merged_llm_4k_questions.csv \
    --model_id Qwen/Qwen2.5-72B-Instruct \
    --dtype float16 \
    --load_in_8bit \
    --device_map cuda:0 \
    --output_dir qwen72b_attribution_scores \
    --flush_every 25 \
    --start_id 1 \
    --end_id 2000 \
    > logs/qwen72b_attribution_1_2000.log 2>&1 &


Example: rows 2 001 - end
python generate_attributions.py \
    --csv_path merged_llm_4k_questions.csv \
    --model_id Qwen/Qwen2.5-72B-Instruct \
    --dtype float16 \
    --load_in_8bit \
    --device_map cuda:0 \
    --output_dir qwen72b_attribution_scores \
    --flush_every 25 \
    --start_id 2001 \
    > logs/qwen72b_attribution_2001_end.log 2>&1 &
"""
import argparse
import os
import re
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from context_cite import ContextCiter


# --------------------------------------------------------------------------- #
#                               argument parsing                              #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM attribution generator")
    parser.add_argument(
        "--csv_path",
        default="merged_llm_4k_questions.csv",
        help="Input CSV with at least columns `context` and `question`",
    )
    parser.add_argument(
        "--model_id",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="🤗 model repo or local path",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype to load the model with (ignored if --load_in_8bit is set)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model with 8-bit weight-only quantization (bitsandbytes).",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help=(
            "Device placement strategy.  Common values:  "
            "`auto` (HF will shard), "
            "`cuda:0` (entire model on a single GPU), "
            "`balanced_low_0`, `sequential`, or `cpu`.\n"
            "See https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained"  # noqa: E501
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="llama70b_attribution_scores",
        help="Folder where per-question CSVs and the running summary are saved",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=25,
        help="Flush buffered results to disk every N rows (safer restarts)",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="1-based index of the first QA row to process (default: 1)",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="1-based index of the last QA row to process, inclusive "
             "(omit to run through the end of the CSV)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Maximum number of extraction attempts per question "
             "before recording None",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                              helper functions                               #
# --------------------------------------------------------------------------- #
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    model_id: str,
    dtype: str,
    device_map: str,
    load_in_8bit: bool,
):
    """
    Load model & tokenizer, optionally with 8-bit weight-only quantisation.
    """
    model_kwargs = {"device_map": device_map}

    if load_in_8bit:
        # weight-only Int8 quantisation (bitsandbytes)
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,       # HF defaults
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        # keep matmuls in BF16/FP16 so we still pass torch_dtype
        model_kwargs["torch_dtype"] = _DTYPE_MAP.get(dtype, torch.bfloat16)
    else:
        model_kwargs["torch_dtype"] = _DTYPE_MAP[dtype]

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def extract_answer(raw_response: str, llama_mode: bool = False) -> str:
    """
    Extract the answer choice letter (A-J) from the model's textual reply.

    The model is *supposed* to wrap the answer like:
        <answer>Option B</answer>

    But we also try a couple of fall-back patterns, just in case.
    Returns
    -------
    letter : str
        The uppercase letter (A-J).

    Raises
    ------
    ValueError
        If no valid answer letter can be found.
    """
    # Guard against NaN / non-string inputs ----------------------------------
    if pd.isna(raw_response):
        raise ValueError("Raw response is NaN / missing")
    raw_response = str(raw_response)

    patterns = [
        # 1) Well-formed tag, allowing optional brackets: <answer>Option [D]</answer>
        r"<answer>\s*Option\s*\[?\s*([A-J])\s*\]?\s*</answer>",
        # 2) Tag present but missing the "Option" word, allowing brackets: <answer>[D]</answer>
        r"<answer>\s*\[?\s*([A-J])\s*\]?\s*</answer>",
        # 3) Loose “Option X” anywhere, with optional brackets: “Option [D]”
        r"Option\s*\[?\s*([A-J])\s*\]?",
    ]

    if llama_mode:
        patterns.extend(
            [
                # Option (C)   or   Option (F)</answer>
                r"Option\s*\(?\s*([A-J])\s*\)?",
                # The best answer is B.   Best answer is: C
                r"\bbest\s+answer\s+is\s*([A-J])\b",
                # The correct answer is: (E) …
                r"\bcorrect\s+answer\s+(?:is|:)\s*\(?\s*([A-J])\s*\)?",
                # answer is: $\boxed{I}$      (handles optional $ … $)
                r"answer\s+is\s*:?\s*\$?\\boxed\{\s*([A-J])\s*\}\$?",
            ]
        )

    for pat in patterns:
        m = re.search(pat, raw_response, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()

    # If we get here, nothing matched
    raise ValueError(
        "Could not extract an answer letter (A-J) from the model's response:\n"
        f"{raw_response[:200]}…"
    )


def flush_buffer(
    rows: List[pd.DataFrame], summary_path: Path, header_if_new: bool
) -> None:
    """Append accumulated results to the running summary CSV on disk."""
    if not rows:
        return
    df_out = pd.concat(rows, ignore_index=True)
    df_out.to_csv(
        summary_path,
        mode="a",
        header=header_if_new,
        index=False,
    )
    rows.clear()


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Give each run its own summary file
    range_tag = f"{args.start_id}_{args.end_id}"
    summary_path = Path(args.output_dir) / f"attributions_summary_{range_tag}.csv"

    # ------------------------------------------------------------------ Model
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, args.dtype, args.device_map, args.load_in_8bit
    )

    # ------------------------------------------------------------------ Data
    df = pd.read_csv(args.csv_path)

    # --------------------------- Optional range filtering (1-based, inclusive)
    #    e.g. --start_id 1 --end_id 2000  ➜  processes Merge Q1 … Merge Q2000
    #    Omitting --end_id means “process until the end of the CSV”.
    start_id = max(args.start_id, 1)
    end_id = args.end_id if args.end_id is not None else len(df)
    df = df.iloc[start_id - 1 : end_id]

    print(
        f"📄 Loaded {len(df):,} rows from {args.csv_path} "
        f"(processing IDs {start_id}-{end_id})"
    )

    processed_ids: set[str] = set()
    if summary_path.exists():
        processed_ids = set(pd.read_csv(summary_path)["QA_ID"])
        print(f"🔄 Resuming — {len(processed_ids):,} rows already done.")

    buffer: List[pd.DataFrame] = []
    header_written = summary_path.exists()

    # ------------------------------------------------------------- Main loop
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Attributing MCQ Answers",
        unit="qa",
    ):
        qa_id = f"Merge Q{idx + 1}"
        if qa_id in processed_ids:
            continue  # already done

        context_text = row["context"]
        question = row["question"]

        prompt = (
            f"{question}\n"
            "Select the most appropriate answer from the options provided.\n\n"
            "Provide your response in the following format:\n<answer>Option [letter]</answer>"
        )

        answer_letter: str | None = None
        last_result: pd.DataFrame | None = None
        last_raw_response: str = ""

        # ------------------------------------------------- Attempt extraction
        for attempt in range(1, args.max_attempts + 1):
            try:
                cc = ContextCiter(
                    model,
                    tokenizer,
                    context=context_text,
                    query=prompt,
                    generate_kwargs={
                        "max_new_tokens": 2048,
                        "do_sample": False,
                        "pad_token_id": tokenizer.eos_token_id,
                    },
                )

                last_raw_response = cc.response
                candidate = extract_answer(last_raw_response)

                # Attribution table →
                result = cc.get_attributions(as_dataframe=True)
                if isinstance(result, pd.io.formats.style.Styler):
                    result = result.data
                if isinstance(result, pd.DataFrame):
                    last_result = result  # keep the most recent attribution table

                if candidate is not None:
                    answer_letter = candidate  # ✅ success
                    break

            except Exception as exc:
                # keep the exception text as the raw response for transparency
                last_raw_response = f"ERROR on attempt {attempt}: {exc}"

        # -------------------------------------------- Assemble final record
        if last_result is None:
            # If we never received an attribution table, create a minimal one
            last_result = pd.DataFrame(
                {
                    "QA_ID": [qa_id],
                    "Extracted_Answer": [answer_letter],
                    "Raw_Response": [last_raw_response],
                }
            )
        else:
            last_result["QA_ID"] = qa_id
            last_result["Extracted_Answer"] = answer_letter
            last_result["Raw_Response"] = last_raw_response

        # ---------- per-question CSV with attribution scores
        per_question_path = Path(args.output_dir) / f"{qa_id.replace(' ', '_')}.csv"
        last_result.to_csv(per_question_path, index=False)

        # ---------- add to in-memory buffer for the running summary
        buffer.append(last_result)

        status = f"(answer={answer_letter})" if answer_letter else "(answer=None)"
        print(f"✅ {qa_id}  {status}")

        # ---------- flush to disk periodically
        if len(buffer) >= args.flush_every:
            flush_buffer(buffer, summary_path, not header_written)
            header_written = True

    # ---------------------------------------------------------------- done
    flush_buffer(buffer, summary_path, not header_written)
    print("🎉 Finished!")


if __name__ == "__main__":
    main()
