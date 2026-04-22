"""
Config dataclass + interactive setup + JSON save/load.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

SINGLE_PROMPT_TEMPLATE = """\
You are given a list of sentences and a multiple-choice question.

Your task is twofold:
(1) Select the most appropriate answer from the given options.
(2) Label each sentence as [High Relevance], [Low Relevance], or [Irrelevant]
    based on its contribution to answering the question.

Definitions:
- [High Relevance]: Directly supports the correct answer with essential information.
- [Low Relevance]: Provides useful context but is not critical to answering.
- [Irrelevant]: Unrelated to the question or not useful for reasoning.

Question and Options:
{options}

Sentences:
{sentences}

Return ONLY a JSON object in this exact format:
{{
  "Answer": "<letter>",
  "Sentence_Relevance": ["High Relevance", "Low Relevance", ...]
}}"""


def _make_multi_prompt_template(question_names: List[str]) -> str:
    n = len(question_names)
    keys_str = "\n".join(f'  "{name}": "<letter>",' for name in question_names)
    return (
        f"You are given a list of sentences and {n} multiple-choice question(s).\n\n"
        "Your task is twofold:\n"
        "(1) Select the single best answer for EACH question.\n"
        "(2) Label each sentence as [High Relevance], [Low Relevance], or [Irrelevant]\n"
        "    based on its contribution to answering the questions.\n\n"
        "Questions and Options:\n{options}\n\n"
        "Sentences:\n{sentences}\n\n"
        "Return ONLY a JSON object in this exact format:\n"
        "{\n"
        + keys_str + "\n"
        '  "Sentence_Relevance": ["High Relevance", "Low Relevance", ...]\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuestionDef:
    """One question in multi-question mode."""
    name: str        # output column name, e.g. "household_income"
    correct_col: str  # input column with the ground-truth answer (empty = unknown)


@dataclass
class ToolkitConfig:
    # I/O
    input_path: str = ""
    output_path: str = ""

    # Dataset columns
    id_col: Optional[str] = None          # unique ID for resume support
    sentences_col: str = ""               # pre-formatted sentences text
    sentence_count_col: str = ""          # integer count of sentences in vignette
    options_col: str = ""                 # question + answer options text
    groupby_col: Optional[str] = None     # optional: group rows for per-group stats

    # Task mode
    mode: str = "single"                  # "single" or "multi"
    correct_answer_col: Optional[str] = None   # single-mode ground truth
    questions: List[QuestionDef] = field(default_factory=list)  # multi-mode

    # Model
    provider: str = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    api_key_env: str = "OPENAI_API_KEY"

    # Prompt
    custom_prompt_path: Optional[str] = None   # override default template

    # Output
    max_sentences: int = 30

    # ---------------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "ToolkitConfig":
        with open(path) as f:
            data = json.load(f)
        questions = [QuestionDef(**q) for q in data.pop("questions", [])]
        return cls(**data, questions=questions)

    def get_prompt_template(self) -> str:
        if self.custom_prompt_path:
            with open(self.custom_prompt_path) as f:
                return f.read()
        if self.mode == "single":
            return SINGLE_PROMPT_TEMPLATE
        return _make_multi_prompt_template([q.name for q in self.questions])


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------

def _ask(msg: str, default: Optional[str] = None) -> str:
    hint = f" [{default}]" if default else ""
    val = input(f"{msg}{hint}: ").strip()
    return val if val else (default or "")


def _ask_optional(msg: str) -> Optional[str]:
    val = input(f"{msg} (Enter to skip): ").strip()
    return val or None


def _show_cols(cols: List[str]) -> None:
    if cols:
        print(f"  Columns: {cols}")


def interactive_setup(config_path: str, input_path: Optional[str] = None) -> ToolkitConfig:
    print("\n=== Toolkit Setup ===\n")
    cfg = ToolkitConfig()

    # --- I/O ---
    cfg.input_path = input_path or _ask("Input CSV path")
    cfg.output_path = _ask("Output CSV path")

    # Peek at columns so the user can see what's available
    cols: List[str] = []
    try:
        sample = pd.read_csv(cfg.input_path, nrows=1)
        cols = sample.columns.tolist()
    except Exception as e:
        print(f"  (Could not read input file to show columns: {e})")

    # --- Dataset columns ---
    print("\n-- Dataset columns --")
    _show_cols(cols)
    cfg.id_col = _ask_optional("Unique ID column (for resume support)")
    cfg.sentences_col = _ask("Pre-formatted sentences column")
    cfg.sentence_count_col = _ask("Sentence count column")
    cfg.options_col = _ask("Question + answer options column")
    cfg.groupby_col = _ask_optional("Group-by column for per-group analysis (e.g. data_source)")

    # --- Task mode ---
    print("\n-- Question mode --")
    print("  1. Single question per row")
    print("  2. Multiple questions per row")
    mode_choice = _ask("Choice", "1")
    cfg.mode = "multi" if mode_choice.strip() == "2" else "single"

    if cfg.mode == "single":
        _show_cols(cols)
        cfg.correct_answer_col = _ask_optional("Correct answer column (for accuracy analysis)")
    else:
        n = int(_ask("How many questions per row"))
        questions = []
        for i in range(n):
            print(f"\n  Question {i + 1}:")
            _show_cols(cols)
            name = _ask("    Output column name (e.g. 'household_income')")
            correct_col = _ask_optional("    Correct answer column for this question")
            questions.append(QuestionDef(name=name, correct_col=correct_col or ""))
        cfg.questions = questions

    # --- Model ---
    print("\n-- Model --")
    cfg.provider = "openai"
    cfg.api_key_env = "OPENAI_API_KEY"
    cfg.model_name = _ask("Model name", "gpt-4o")

    cfg.temperature = float(_ask("Temperature", "0"))
    cfg.max_sentences = int(_ask("Max sentences to label", "30"))
    cfg.custom_prompt_path = _ask_optional("Custom prompt template file path (optional)")

    cfg.save(config_path)
    return cfg
