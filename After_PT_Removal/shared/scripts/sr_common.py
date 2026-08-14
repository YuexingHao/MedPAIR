#!/usr/bin/env python3
"""Single source of truth for the SR (Self-Report) reduced-context experiment.

Every model - GPT-4o, GPT-5, Llama-70B, MedGemma-27B, Qwen-14B, Qwen-72B - must
read the same columns, see the same prompt, and have its answer parsed the same
way, or the per-model accuracies in Table 6 are not comparable.

Import from here rather than re-defining any of it in a per-model script:

    sys.path.insert(0, "/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts")
    from sr_common import (INPUT_FILE, CONTEXT_COL, QUESTION_COL, KEY_COL,
                           GOLD_COL, SYSTEM_PROMPT, build_prompt, chat_messages,
                           extract_letter, load_sr_input)
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- input schema

INPUT_FILE = Path(
    "/home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/"
    "Llama-70B_Removed/Llama_70B_[SR]_predictions.csv"
)

CONTEXT_COL = "[SR]High"        # reduced-relevance context (llama70b-removed)
QUESTION_COL = "question_options"
KEY_COL = "Origin"              # join key between input and per-model output
GOLD_COL = "answer_corr"

REQUIRED_COLS = [KEY_COL, CONTEXT_COL, QUESTION_COL, GOLD_COL]

# ---------------------------------------------------------------------- prompt

SYSTEM_PROMPT = (
    "You are a medical expert assistant helping with multiple-choice "
    "medical questions."
)

PROMPT_TEMPLATE = """You are given some context and a multiple-choice question.

Select the most appropriate answer from the options provided.

{context}

{question}

Provide your response in the following format:
<answer>Option [letter]</answer>"""


def build_prompt(context, question):
    """The one SR user prompt. Do not reformat per model."""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def chat_messages(context, question):
    """Canonical chat turns, used for both API models and local chat templates."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(context, question)},
    ]


# ------------------------------------------------------------------- parsing

# Ordered most-specific first. Deliberately stops before "any capital A-J
# anywhere in the text": that fallback turns prose like "Analysis suggests..."
# into a confident answer of "A", which silently fabricates predictions.
import re

_PATTERNS = [
    # 1. The requested format.
    re.compile(r"<answer>\s*(?:Option\s*)?\[?\s*([A-J])\s*\]?\s*</answer>", re.I | re.S),
    # 2. "Option [C]" / "Option C" anywhere (model dropped the tags).
    re.compile(r"\bOption\s*\[?\s*([A-J])\s*\]?", re.I),
    # 3. "Answer: C" style.
    re.compile(r"\bAnswer\s*:\s*\[?\s*([A-J])\s*\]?", re.I),
    # 4. A bare letter on its own line.
    re.compile(r"(?:^|\n)\s*\(?\[?([A-J])\]?\)?[\.\)]?\s*(?:\n|$)"),
]


def extract_letter(pred_text):
    """Return the predicted option letter, or None if nothing parseable.

    None means "no answer recovered" and must be counted as unparsed coverage
    loss - never silently scored as wrong or guessed at.
    """
    if not isinstance(pred_text, str) or not pred_text.strip():
        return None
    if pred_text.startswith("Error"):
        return None

    for pat in _PATTERNS:
        m = pat.search(pred_text)
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------- loading


def load_sr_input(limit=None):
    """Load the shared SR input, validating the columns every model relies on."""
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR: input {INPUT_FILE} is missing required columns {missing}.\n"
            f"Available: {list(df.columns)}"
        )

    if limit:
        df = df.head(limit).copy()
    return df


def summarize(df_input, predictions, pred_col):
    """Coverage/accuracy over the rows actually scored, with the denominator shown."""
    letters = [extract_letter(p) if not (p or "").startswith("Error") else None
               for p in (str(x) if x is not None else "" for x in predictions)]
    gold = df_input[GOLD_COL].astype(str).str.strip().str.upper().tolist()

    n = len(df_input)
    scored = sum(1 for x in letters if x)
    correct = sum(1 for x, g in zip(letters, gold) if x and x == g)
    return {
        "n": n,
        "scored": scored,
        "unparsed": n - scored,
        "correct": correct,
        "acc_scored": (correct / scored * 100) if scored else float("nan"),
        "acc_all": correct / n * 100 if n else float("nan"),
        "coverage": scored / n * 100 if n else float("nan"),
        "pred_col": pred_col,
    }
