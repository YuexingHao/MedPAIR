"""
Query runner: format prompt, call model, parse response, save results incrementally.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import ToolkitConfig

RELEVANCE_LABELS = {"High Relevance", "Low Relevance", "Irrelevant"}
VALID_ANSWER_LETTERS = set("ABCDEFGHIJ")


# ---------------------------------------------------------------------------
# Model client helpers
# ---------------------------------------------------------------------------

def _get_client(config: ToolkitConfig):
    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Set the '{config.api_key_env}' environment variable."
        )
    if config.provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    raise ValueError(f"Unknown provider: {config.provider!r}")


def _call_model(client, config: ToolkitConfig, prompt: str) -> str:
    if config.provider == "openai":
        resp = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
        )
        return resp.choices[0].message.content.strip()
    raise ValueError(f"Unknown provider: {config.provider!r}")


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Try to extract and parse the first JSON object from text."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _clean_labels(raw: list, n_sentences: int) -> List[Optional[str]]:
    """Validate label list against expected sentence count; return [] on mismatch."""
    cleaned = [str(l).strip() for l in raw]
    if len(cleaned) != n_sentences:
        return []
    return [l if l in RELEVANCE_LABELS else None for l in cleaned]


def _parse_single(content: str, n_sentences: int, max_sentences: int
                  ) -> Tuple[Optional[str], List[Optional[str]]]:
    """
    Returns (llm_answer, padded_sentence_labels).
    llm_answer is a single uppercase letter or None.
    """
    llm_answer: Optional[str] = None
    sentence_labels: List[Optional[str]] = []

    data = _extract_json(content)
    if data:
        raw_ans = data.get("Answer")
        if isinstance(raw_ans, str) and raw_ans.strip().upper() in VALID_ANSWER_LETTERS:
            llm_answer = raw_ans.strip().upper()
        raw_labels = data.get("Sentence_Relevance", [])
        if isinstance(raw_labels, list):
            sentence_labels = _clean_labels(raw_labels, n_sentences)

    # Regex fallbacks
    if llm_answer is None:
        m = re.search(r"\bAnswer:\s*([A-J])\b", content, re.IGNORECASE)
        if m:
            llm_answer = m.group(1).upper()

    if not sentence_labels:
        found = re.findall(
            r"\[(High Relevance|Low Relevance|Irrelevant)\]", content, re.IGNORECASE
        )
        found = [f.title() for f in found]
        sentence_labels = _clean_labels(found, n_sentences)

    # Pad / truncate to max_sentences
    sentence_labels = (sentence_labels or [])[:max_sentences]
    sentence_labels += [None] * (max_sentences - len(sentence_labels))
    return llm_answer, sentence_labels


def _parse_multi(content: str, question_names: List[str], n_sentences: int, max_sentences: int
                 ) -> Tuple[Dict[str, Optional[str]], List[Optional[str]]]:
    """
    Returns (answers_dict, padded_sentence_labels).
    answers_dict maps each question name to a letter or None.
    """
    answers: Dict[str, Optional[str]] = {name: None for name in question_names}
    sentence_labels: List[Optional[str]] = []

    data = _extract_json(content)
    if data:
        for name in question_names:
            val = data.get(name)
            if isinstance(val, str) and val.strip().upper() in VALID_ANSWER_LETTERS:
                answers[name] = val.strip().upper()
        raw_labels = data.get("Sentence_Relevance", [])
        if isinstance(raw_labels, list):
            sentence_labels = _clean_labels(raw_labels, n_sentences)

    sentence_labels = (sentence_labels or [])[:max_sentences]
    sentence_labels += [None] * (max_sentences - len(sentence_labels))
    return answers, sentence_labels


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_query(config: ToolkitConfig) -> None:
    df = pd.read_csv(config.input_path)
    prompt_template = config.get_prompt_template()
    label_cols = [f"label_{i + 1}" for i in range(config.max_sentences)]

    # Build the extra column names that will be appended to each row
    if config.mode == "single":
        extra_cols = ["Raw_Response", "LLM_answer"] + label_cols
    else:
        q_names = [q.name for q in config.questions]
        extra_cols = ["Raw_Response"] + q_names + label_cols

    # Resume: collect already-processed IDs
    processed_ids: set = set()
    if not os.path.exists(config.output_path):
        header = df.columns.tolist() + extra_cols
        pd.DataFrame(columns=header).to_csv(config.output_path, index=False)
    else:
        try:
            existing = pd.read_csv(config.output_path)
            if config.id_col and config.id_col in existing.columns:
                for v in existing[config.id_col].dropna().unique():
                    processed_ids.add(str(v))
                print(f"Resuming: {len(processed_ids)} rows already processed.")
        except Exception:
            pass

    client = _get_client(config)

    for i in range(len(df)):
        row = df.iloc[i]

        if config.id_col:
            row_id = str(row[config.id_col])
            if row_id in processed_ids:
                print(f"  Skipping row {i} (id={row_id})")
                continue

        try:
            sentences = str(row[config.sentences_col]).strip()
            options = str(row[config.options_col]).strip()
            n_sentences = int(row[config.sentence_count_col])

            prompt = prompt_template.format(sentences=sentences, options=options)
            content = _call_model(client, config, prompt)

            if config.mode == "single":
                llm_answer, sentence_labels = _parse_single(content, n_sentences, config.max_sentences)
                row_extra = [content, llm_answer] + sentence_labels
            else:
                q_names = [q.name for q in config.questions]
                answers, sentence_labels = _parse_multi(content, q_names, n_sentences, config.max_sentences)
                row_extra = [content] + [answers[n] for n in q_names] + sentence_labels

        except Exception as e:
            print(f"  Error at row {i}: {e}")
            content = ""
            if config.mode == "single":
                row_extra = [content, None] + [None] * config.max_sentences
            else:
                q_names = [q.name for q in config.questions]
                row_extra = [content] + [None] * len(q_names) + [None] * config.max_sentences

        out_row = pd.concat(
            [df.iloc[[i]].reset_index(drop=True),
             pd.DataFrame([row_extra], columns=extra_cols)],
            axis=1,
        )
        out_row.to_csv(config.output_path, mode="a", index=False, header=False)
        print(f"  Row {i} processed.")
        time.sleep(0.1)

    print(f"\nDone. Results saved to: {config.output_path}")
