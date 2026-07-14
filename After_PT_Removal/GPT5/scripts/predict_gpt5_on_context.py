#!/usr/bin/env python3
"""Run GPT-5 predictions on a chosen context column and export a CSV."""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


PROMPT_TEMPLATE = """
You are a clinical reasoning assistant. You will receive a patient case summary and a multiple-choice question.
Read the question and state your answer.
Patient Context:
{context}

Question and Options:
{question}

Please select the single most appropriate answer. Respond only in the following format:
Answer: [LETTER]
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-5 predictions on a context column and write output CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument(
        "--context-column",
        type=str,
        default="Random_Sentences",
        help="Input column used as context for prediction (default: Random_Sentences).",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        default="question_options",
        help="Column containing the MCQ options/prompt block.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="gpt5_direct_prediction",
        help="Output column to store model predictions.",
    )
    parser.add_argument(
        "--answer-column",
        type=str,
        default="answer_corr",
        help="Answer column used to compute gpt_letter_match fields.",
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://yuexing-may-26-resource.services.ai.azure.com/openai/v1",
        ),
        help="Azure OpenAI endpoint base URL ending with /openai/v1",
    )
    parser.add_argument(
        "--azure-key-env",
        type=str,
        default="AZURE_OPENAI_API_KEY",
        help="Environment variable name containing Azure OpenAI API key.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds for chat completion calls.",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel request workers (1 = sequential).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute predictions even when prediction column is already present.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Debug/testing subset size.",
    )
    return parser.parse_args()


def build_client(
    azure_endpoint: str,
    azure_key_env: str,
) -> OpenAI:
    # Preferred: Azure AD token provider; fallback: API key.
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default",
        )
        return OpenAI(base_url=azure_endpoint, api_key=token_provider)
    except Exception:  # noqa: BLE001
        azure_key = os.getenv(azure_key_env) or os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_key:
            raise RuntimeError(
                f"Azure credentials unavailable. Install azure-identity for AAD auth, "
                f"or set {azure_key_env}/AZURE_OPENAI_API_KEY."
            )
        return OpenAI(base_url=azure_endpoint, api_key=azure_key)


def call_model(
    client: OpenAI,
    model_name: str,
    context: str,
    question: str,
    max_retries: int,
    request_timeout: float,
) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=request_timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
    return f"Error: {last_error}"


def extract_letter_from_text(x: object) -> str | None:
    if not isinstance(x, str):
        return None

    text = x.strip()

    # Current expected format.
    m = re.search(
        r"\bAnswer\s*:\s*(?:Option\s*)?\[?\s*([A-J])\s*\]?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Legacy formats from older runs.
    legacy_patterns = [
        r"<answer>\s*Option\s*\[?\s*([A-J])\s*\]?\s*</answer>",
        r"^\s*Option\s*\[?\s*([A-J])\s*\]?\s*$",
        r"^\s*([A-J])\s*$",
    ]
    for pattern in legacy_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def add_match_columns(df: pd.DataFrame, prediction_col: str, answer_col: str) -> pd.DataFrame:
    out = df.copy()
    out["gpt_letter"] = out[prediction_col].apply(extract_letter_from_text)
    if answer_col in out.columns:
        out["answer_letter"] = out[answer_col].astype(str).str.strip().str.upper()
        out.loc[out["answer_letter"].isin(["", "NAN", "NONE"]), "answer_letter"] = pd.NA
        valid = out["gpt_letter"].notna() & out["answer_letter"].notna()
        out["gpt_letter_match"] = pd.NA
        out.loc[valid, "gpt_letter_match"] = np.where(
            out.loc[valid, "gpt_letter"] == out.loc[valid, "answer_letter"],
            "Correct",
            "Incorrect",
        )
        out["gpt_letter_binary"] = out["gpt_letter_match"].map(
            {"Correct": 1, "Incorrect": 0}
        ).astype("Int64")
    else:
        out["answer_letter"] = pd.NA
        out["gpt_letter_match"] = pd.NA
        out["gpt_letter_binary"] = pd.NA
    return out


def main() -> None:
    args = parse_args()
    client = build_client(args.azure_endpoint, args.azure_key_env)

    df = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    if "Origin" not in df.columns:
        raise ValueError("Input CSV must include 'Origin' column.")
    if args.context_column not in df.columns:
        raise ValueError(f"Input CSV must include '{args.context_column}'.")
    if args.question_column not in df.columns:
        raise ValueError(f"Input CSV must include '{args.question_column}'.")

    existing_by_origin: dict[str, str] = {}
    if args.resume and args.output_csv.exists():
        out_df = pd.read_csv(args.output_csv)
        if "Origin" in out_df.columns and args.prediction_column in out_df.columns:
            tmp = out_df[["Origin", args.prediction_column]].dropna()
            existing_by_origin = dict(
                zip(tmp["Origin"].astype(str), tmp[args.prediction_column])
            )

    if args.prediction_column not in df.columns:
        df[args.prediction_column] = pd.NA

    missing_work: list[tuple[int, str, str]] = []
    for idx, row in df.iterrows():
        origin = str(row["Origin"])
        if (not args.force_rerun) and origin in existing_by_origin:
            df.at[idx, args.prediction_column] = existing_by_origin[origin]
            continue
        if (not args.force_rerun) and pd.notna(df.at[idx, args.prediction_column]):
            continue
        missing_work.append(
            (idx, str(row[args.context_column]), str(row[args.question_column]))
        )

    if args.workers <= 1:
        wrote = 0
        for idx, context, question in missing_work:
            prediction = call_model(
                client=client,
                model_name=args.model,
                context=context,
                question=question,
                max_retries=args.max_retries,
                request_timeout=args.request_timeout,
            )
            df.at[idx, args.prediction_column] = prediction
            wrote += 1

            if args.save_every > 0 and wrote % args.save_every == 0:
                tmp_df = add_match_columns(df, args.prediction_column, args.answer_column)
                args.output_csv.parent.mkdir(parents=True, exist_ok=True)
                tmp_df.to_csv(args.output_csv, index=False)
                print(f"[progress] {wrote}/{len(missing_work)} new rows completed")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_idx = {
                ex.submit(
                    call_model,
                    client=client,
                    model_name=args.model,
                    context=context,
                    question=question,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout,
                ): idx
                for idx, context, question in missing_work
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    prediction = fut.result()
                except Exception as exc:  # noqa: BLE001
                    prediction = f"Error: {exc}"
                df.at[idx, args.prediction_column] = prediction
                completed += 1
                if args.save_every > 0 and completed % args.save_every == 0:
                    tmp_df = add_match_columns(df, args.prediction_column, args.answer_column)
                    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
                    tmp_df.to_csv(args.output_csv, index=False)
                    print(f"[progress] {completed}/{len(missing_work)} new rows completed")

    out_df = add_match_columns(df, args.prediction_column, args.answer_column)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    final_non_null = int(out_df[args.prediction_column].notna().sum())
    print(
        f"[done] Wrote {len(out_df)} rows to {args.output_csv} "
        f"(predictions present: {final_non_null})"
    )


if __name__ == "__main__":
    main()
