#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_on_trainee_irr_removed.csv"
GPT5_OUT = ROOT / "GPT5" / "results" / "predictions" / "gpt5_predictions_on_trainee_irr_removed.csv"
MEDGEMMA_OUT = ROOT / "MedGemma-27b-text-it" / "results" / "predictions" / "MedGemma27B_predictions_on_trainee_irr_removed.csv"
LLAMA70B_OUT = ROOT / "Llama-70B" / "results" / "predictions" / "Llama70B_predictions_on_trainee_irr_removed.csv"
QWEN14B_OUT = ROOT / "Qwen2.5-14B-Instruct" / "results" / "predictions" / "Qwen_14B_predictions_trainee_irr_removed.csv"
QWEN72B_OUT = ROOT / "Qwen2.5-72B-Instruct" / "results" / "predictions" / "Qwen_72B_predictions_trainee_irr_removed.csv"
DEFAULT_MEDGEMMA_SNAPSHOT = Path(
    "/orcd/compute/mghassem/001/gobi1/huggingface/hub/models--google--medgemma-27b-text-it/snapshots/5b667cf2ddcf064085bc90952edb35a0edbfb79c"
)
DEFAULT_LLAMA70B_SNAPSHOT = Path(
    "/orcd/compute/mghassem/001/gobi1/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
)
DEFAULT_QWEN14B_SNAPSHOT = Path(
    "/orcd/compute/mghassem/001/gobi1/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
)
DEFAULT_QWEN72B_SNAPSHOT = Path(
    "/orcd/compute/mghassem/001/gobi1/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
)
EXPERT_933_CSV = ROOT.parent / "Physician_Labels" / "Mar2_2026_Data" / "933_Clinician_Student_Majority_Vote.csv"


def build_prompt(context: str, question: str) -> str:
    return (
        "You are a clinical reasoning assistant. You will receive a patient case summary "
        "and a multiple-choice question.\n\n"
        f"{context}\n\n"
        f"{question}\n\n"
        "Please select the single most appropriate answer. Respond only in the following format:\n\n"
        "<answer>Option X</answer>"
    )


def extract_letter(text: Any) -> Optional[str]:
    s = str(text or "")
    m = re.search(r"Option\s*([A-J])", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-J])\b", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def format_answer_xml(text: str) -> str:
    letter = extract_letter(text)
    return f"<answer>Option {letter}</answer>" if letter else text.strip()


def get_gpt5_client():
    # Azure path (preferred when endpoint/version are available)
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    if azure_key and azure_endpoint and api_version:
        from openai import AzureOpenAI

        deployment = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "gpt-5")
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return client, deployment

    # OpenAI public API path
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from openai import OpenAI

        return OpenAI(api_key=openai_key), "gpt-5"

    raise RuntimeError(
        "No usable GPT-5 credentials found. Set either "
        "(AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + OPENAI_API_VERSION) "
        "or OPENAI_API_KEY."
    )


def run_gpt5(df: pd.DataFrame, out_path: Path, max_rows: Optional[int], max_tokens: int) -> None:
    client, model = get_gpt5_client()
    is_azure = client.__class__.__name__.lower().startswith("azure")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        done_df = pd.read_csv(out_path)
        rows = done_df.to_dict("records")
        done_origins = set(done_df.get("Origin", pd.Series([], dtype=str)).astype(str).tolist())
        print(f"[gpt5] Resuming from {len(done_origins)} existing Origins: {out_path}")
    else:
        rows = []
        done_origins = set()
        print(f"[gpt5] Starting new run: {out_path}")

    todo_df = df.copy()
    todo_df["__origin"] = todo_df["Origin"].astype(str)
    todo_df = todo_df[~todo_df["__origin"].isin(done_origins)]
    if max_rows is not None:
        todo_df = todo_df.head(max_rows)
    print(f"[gpt5] Remaining rows to process: {len(todo_df)}")

    for _, r in tqdm(todo_df.iterrows(), total=len(todo_df), desc="gpt5"):
        row = r.to_dict()
        row.pop("__origin", None)
        context = str(row.get("New_Sentences", "")).strip()
        question = str(row.get("question_options", "")).strip()
        prompt = build_prompt(context, question)
        try:
            if is_azure:
                # Azure often uses chat completions deployments.
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                )
                pred_raw = (resp.choices[0].message.content or "").strip()
            else:
                # OpenAI public endpoint: GPT-5 works best via Responses API.
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "minimal"},
                )
                pred_raw = str(getattr(resp, "output_text", "") or "").strip()
                if not pred_raw:
                    # Fallback parse for cases where output_text is empty.
                    chunks = []
                    for item in getattr(resp, "output", []) or []:
                        if getattr(item, "type", None) != "message":
                            continue
                        for c in getattr(item, "content", []) or []:
                            txt = getattr(c, "text", None)
                            if txt:
                                chunks.append(str(txt))
                    pred_raw = "\n".join(chunks).strip()
            pred_xml = format_answer_xml(pred_raw)
        except Exception as e:
            pred_raw = f"ERROR: {type(e).__name__}: {e}"
            pred_xml = pred_raw

        row["gpt5_direct_prediction"] = pred_xml
        row["gpt5_raw_response"] = pred_raw
        row["gpt_letter"] = extract_letter(pred_xml)
        row["answer_letter"] = str(row.get("answer_corr", "")).strip().upper() or None
        if row["gpt_letter"] and row["answer_letter"]:
            row["gpt_letter_match"] = "Correct" if row["gpt_letter"] == row["answer_letter"] else "Incorrect"
            row["gpt_letter_binary"] = 1 if row["gpt_letter"] == row["answer_letter"] else 0
        else:
            row["gpt_letter_match"] = None
            row["gpt_letter_binary"] = None

        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"[gpt5] Wrote {len(rows)} rows to {out_path}")


def run_open_model(
    model_key: str,
    df: pd.DataFrame,
    out_path: Path,
    snapshot: Path,
    max_rows: Optional[int],
    allow_cpu: bool,
    max_new_tokens: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not snapshot.exists():
        raise FileNotFoundError(f"[{model_key}] snapshot not found: {snapshot}")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(f"[{model_key}] CUDA is not available. Re-run on a GPU node, or pass --allow-cpu (very slow).")

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[{model_key}] Loading tokenizer/model from: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        done_df = pd.read_csv(out_path)
        rows = done_df.to_dict("records")
        done_origins = set(done_df.get("Origin", pd.Series([], dtype=str)).astype(str).tolist())
        print(f"[{model_key}] Resuming from {len(done_origins)} existing Origins: {out_path}")
    else:
        rows = []
        done_origins = set()
        print(f"[{model_key}] Starting new run: {out_path}")

    todo_df = df.copy()
    todo_df["__origin"] = todo_df["Origin"].astype(str)
    todo_df = todo_df[~todo_df["__origin"].isin(done_origins)]
    if max_rows is not None:
        todo_df = todo_df.head(max_rows)
    print(f"[{model_key}] Remaining rows to process: {len(todo_df)}")

    for _, r in tqdm(todo_df.iterrows(), total=len(todo_df), desc=model_key):
        row = r.to_dict()
        row.pop("__origin", None)
        context = str(row.get("New_Sentences", "")).strip()
        question = str(row.get("question_options", "")).strip()
        # One standardized prompt template for all rerun models.
        prompt = build_prompt(context, question)
        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt")
            if torch.cuda.is_available():
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
            raw = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            pred_xml = format_answer_xml(raw)
        except Exception as e:
            raw = f"ERROR: {type(e).__name__}: {e}"
            pred_xml = raw

        answer = extract_letter(pred_xml)
        row[f"{model_key}_raw_response"] = raw
        row[f"{model_key}_direct_prediction"] = pred_xml
        row[f"{model_key}_extracted_answer"] = answer
        # Keep a standardized extracted-answer column for downstream evaluators.
        row["Extracted_Answer"] = answer
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"[{model_key}] Wrote {len(rows)} rows to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run predictions on physician-labeler irrelevant-sentence subset.")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--model",
        choices=["gpt5", "medgemma", "llama70b", "qwen14b", "qwen72b", "both", "all_open", "all"],
        default="both",
    )
    p.add_argument("--gpt5-out", type=Path, default=GPT5_OUT)
    p.add_argument("--medgemma-out", type=Path, default=MEDGEMMA_OUT)
    p.add_argument("--llama70b-out", type=Path, default=LLAMA70B_OUT)
    p.add_argument("--qwen14b-out", type=Path, default=QWEN14B_OUT)
    p.add_argument("--qwen72b-out", type=Path, default=QWEN72B_OUT)
    p.add_argument("--medgemma-snapshot", type=Path, default=DEFAULT_MEDGEMMA_SNAPSHOT)
    p.add_argument("--llama70b-snapshot", type=Path, default=DEFAULT_LLAMA70B_SNAPSHOT)
    p.add_argument("--qwen14b-snapshot", type=Path, default=DEFAULT_QWEN14B_SNAPSHOT)
    p.add_argument("--qwen72b-snapshot", type=Path, default=DEFAULT_QWEN72B_SNAPSHOT)
    p.add_argument("--max-rows", type=int, default=None, help="Debug/testing subset size.")
    p.add_argument("--max-tokens", type=int, default=400, help="Max output tokens for GPT-5 call.")
    p.add_argument(
        "--local-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for local open models (MedGemma/Llama/Qwen).",
    )
    p.add_argument("--allow-cpu", action="store_true", help="Allow MedGemma run without CUDA (very slow).")
    p.add_argument("--expert-933-only", action="store_true", help="Restrict run to the expert 933 Origin subset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    needed = {"New_Sentences", "question_options", "answer_corr"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if args.expert_933_only:
        if not EXPERT_933_CSV.exists():
            raise FileNotFoundError(f"Expert 933 CSV not found: {EXPERT_933_CSV}")
        ref = pd.read_csv(EXPERT_933_CSV, usecols=["Origin"]).drop_duplicates()
        keep = set(ref["Origin"].astype(str))
        df = df[df["Origin"].astype(str).isin(keep)].copy()
        print(f"Filtered to expert 933 subset: {len(df)} rows")
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    if args.model in {"gpt5", "both", "all"}:
        run_gpt5(df, args.gpt5_out, args.max_rows, args.max_tokens)
    if args.model in {"medgemma", "both", "all_open", "all"}:
        run_open_model(
            "medgemma",
            df,
            args.medgemma_out,
            args.medgemma_snapshot,
            args.max_rows,
            args.allow_cpu,
            args.local_max_new_tokens,
        )
    if args.model in {"llama70b", "all_open", "all"}:
        run_open_model(
            "llama70b",
            df,
            args.llama70b_out,
            args.llama70b_snapshot,
            args.max_rows,
            args.allow_cpu,
            args.local_max_new_tokens,
        )
    if args.model in {"qwen14b", "all_open", "all"}:
        run_open_model(
            "qwen14b",
            df,
            args.qwen14b_out,
            args.qwen14b_snapshot,
            args.max_rows,
            args.allow_cpu,
            args.local_max_new_tokens,
        )
    if args.model in {"qwen72b", "all_open", "all"}:
        run_open_model(
            "qwen72b",
            df,
            args.qwen72b_out,
            args.qwen72b_snapshot,
            args.max_rows,
            args.allow_cpu,
            args.local_max_new_tokens,
        )


if __name__ == "__main__":
    main()
