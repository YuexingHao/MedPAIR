#!/usr/bin/env python3
"""Re-run GPT-4o predictions on MJ_LowIRR rerun input."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI


PROMPT_TEMPLATE = """
You are given some context and a multiple-choice question.

Select the most appropriate answer from the options provided.

{context}

{question}

Provide your response in the following format:
<answer>Option [letter]</answer>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run GPT-4o on MJ_LowIRR input (expert-933 subset)."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--context-column",
        type=str,
        default="Filtered_Sentences",
        help="Input column used as context for prediction (default: Filtered_Sentences).",
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
        help="Recompute predictions even when gpt_direct_prediction is already present.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Debug/testing subset size.",
    )
    return parser.parse_args()


def ensure_openai_key() -> None:
    return


def build_client(
    azure_endpoint: str,
    azure_key_env: str,
) -> OpenAI:
    # Preferred path: Azure AD token provider (DefaultAzureCredential).
    # Fallback path: AZURE_OPENAI_API_KEY style key auth.
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
) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
    return f"Error: {last_error}"


def main() -> None:
    args = parse_args()
    ensure_openai_key()
    client = build_client(args.azure_endpoint, args.azure_key_env)

    df = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    if "Origin" not in df.columns:
        raise ValueError("Input CSV must include 'Origin' column.")
    if args.context_column not in df.columns or "question_options" not in df.columns:
        raise ValueError(
            f"Input CSV must include {args.context_column} and question_options."
        )

    out_df: pd.DataFrame
    existing_by_origin: dict[str, str] = {}
    if args.resume and args.output_csv.exists():
        out_df = pd.read_csv(args.output_csv)
        if "Origin" in out_df.columns and "gpt_direct_prediction" in out_df.columns:
            tmp = out_df[["Origin", "gpt_direct_prediction"]].dropna()
            existing_by_origin = dict(zip(tmp["Origin"].astype(str), tmp["gpt_direct_prediction"]))
        else:
            out_df = pd.DataFrame()
    else:
        out_df = pd.DataFrame()

    if "gpt_direct_prediction" not in df.columns:
        df["gpt_direct_prediction"] = pd.NA

    total = len(df)
    missing_work: list[tuple[int, str, str]] = []
    for idx, row in df.iterrows():
        origin = str(row["Origin"])
        if (not args.force_rerun) and origin in existing_by_origin:
            df.at[idx, "gpt_direct_prediction"] = existing_by_origin[origin]
            continue
        if (not args.force_rerun) and pd.notna(df.at[idx, "gpt_direct_prediction"]):
            continue
        missing_work.append(
            (idx, str(row[args.context_column]), str(row["question_options"]))
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
            )
            df.at[idx, "gpt_direct_prediction"] = prediction
            wrote += 1

            if args.save_every > 0 and wrote % args.save_every == 0:
                args.output_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(args.output_csv, index=False)
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
                ): idx
                for idx, context, question in missing_work
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    prediction = fut.result()
                except Exception as exc:  # noqa: BLE001
                    prediction = f"Error: {exc}"
                df.at[idx, "gpt_direct_prediction"] = prediction
                completed += 1
                if args.save_every > 0 and completed % args.save_every == 0:
                    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(args.output_csv, index=False)
                    print(f"[progress] {completed}/{len(missing_work)} new rows completed")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    final_non_null = int(df["gpt_direct_prediction"].notna().sum())
    print(f"[done] Wrote {len(df)} rows to {args.output_csv} (predictions present: {final_non_null})")


if __name__ == "__main__":
    main()
