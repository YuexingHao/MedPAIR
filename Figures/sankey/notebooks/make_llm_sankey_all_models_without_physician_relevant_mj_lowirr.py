#!/usr/bin/env python3
"""Generate llm_sankey_all_models_without_physician_relevant using MJ_LowIRR rerun files.

This wrapper loads the existing compiled Sankey generator and overrides only the
round-2 (rerun) file mapping to use *_on_MJ_LowIRR.csv for all six models.
For in-progress reruns, it also aligns round-2 rows to the canonical 933 MJ
Origin IDs and marks missing predictions as explicit "__MISSING__" answers.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


def _load_base_module():
    pyc = (
        Path(__file__).resolve().parent
        / "__pycache__"
        / "make_llm_sankey_all_models_without_physician_relevant.cpython-313.pyc"
    )
    if not pyc.exists():
        raise FileNotFoundError(f"Missing compiled generator: {pyc}")
    loader = importlib.machinery.SourcelessFileLoader("sankey_base_mj", str(pyc))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_base_module()
    orig_load_round_letters = mod.load_round_letters
    orig_make_subplots = mod.make_subplots
    expected_origins_cache: pd.DataFrame | None = None

    def robust_extract_letter(x):
        """Extract a single answer letter from structured or lightly free-form output."""
        if pd.isna(x):
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace("<answer>", "").replace("</answer>", "")
        s = s.replace("<ANSWER>", "").replace("</ANSWER>", "")
        su = s.upper().strip()

        # Prefer explicit option patterns first.
        m = re.search(r"\bOPTION\s*[\(\[]?\s*([A-J])\s*[\)\]]?\b", su)
        if m:
            return m.group(1)

        # Common short forms like "Answer: D", "(C)", or a bare single-letter answer.
        m = re.search(r"\bANSWER\s*[:\-]?\s*[\(\[]?\s*([A-J])\s*[\)\]]?\b", su)
        if m:
            return m.group(1)
        m = re.fullmatch(r"[\(\[]?\s*([A-J])\s*[\)\]]?", su)
        if m:
            return m.group(1)

        # Last-resort token-level letter extraction (avoid scanning characters in prose).
        m = re.search(r"\b([A-J])\b", su)
        if m:
            return m.group(1)
        return None

    def make_letters(df: pd.DataFrame, primary_col: str, fallback_col: str | None = None) -> pd.Series:
        letters = df[primary_col].apply(robust_extract_letter)
        if fallback_col and fallback_col in df.columns:
            fallback = df[fallback_col].apply(robust_extract_letter)
            letters = letters.where(letters.notna(), fallback)
        # Treat missing predictions as explicit wrong answers (not dropped).
        letters = letters.fillna("__MISSING__")
        return letters

    def expected_mj_origins(repo: Path) -> pd.DataFrame:
        nonlocal expected_origins_cache
        if expected_origins_cache is None:
            expected_path = (
                repo
                / "After_PT_Removal"
                / "shared"
                / "data"
                / "Centaur_Lab_First_Round_933_MJ_LowIRR_as_NewSentences_for_rerun.csv"
            )
            expected_origins_cache = (
                mod.pd.read_csv(expected_path, usecols=["Origin"])
                .dropna(subset=["Origin"])
                .copy()
            )
            expected_origins_cache["Origin"] = (
                expected_origins_cache["Origin"].astype(str).str.strip()
            )
            expected_origins_cache = expected_origins_cache[
                ["Origin"]
            ].drop_duplicates(subset=["Origin"])
        return expected_origins_cache.copy()

    def load_round_letters_mj(repo: Path, *, model_name: str, round_kind: str) -> pd.DataFrame:
        # Keep round-1 behavior from the original script, except Llama-70B:
        # recompute R1 from the Trainee file per the updated mapping.
        if round_kind == "r1":
            if model_name != "Llama-70B":
                return orig_load_round_letters(repo, model_name=model_name, round_kind=round_kind)
            p = (
                repo
                / "After_PT_Removal"
                / "Llama-70B"
                / "results"
                / "predictions"
                / "Llama70B_predictions_on_Trainee.csv"
            )
            df = mod.pd.read_csv(
                p,
                usecols=["Origin", "Extracted_Answer"],
            ).copy()
            df["letter"] = df["Extracted_Answer"].apply(robust_extract_letter)
            df = df[["Origin", "letter"]].dropna(subset=["Origin"]).copy()
            df["Origin"] = df["Origin"].astype(str).str.strip()
            return df.drop_duplicates(subset=["Origin"])

        if round_kind != "r2":
            raise ValueError(f"Unsupported round_kind: {round_kind}")

        if model_name == "Qwen-72B":
            p = repo / "After_PT_Removal" / "Qwen2.5-72B-Instruct" / "results" / "predictions" / "Qwen_72B_predictions_on_MJ_LowIRR.csv"
            df = mod.pd.read_csv(p, usecols=["Origin", "qwen72b_direct_prediction", "qwen72b_extracted_answer"]).copy()
            df["letter"] = make_letters(df, "qwen72b_direct_prediction", "qwen72b_extracted_answer")
        elif model_name == "Qwen-14B":
            p = repo / "After_PT_Removal" / "Qwen2.5-14B-Instruct" / "results" / "predictions" / "Qwen_14B_predictions_on_MJ_LowIRR.csv"
            df = mod.pd.read_csv(p, usecols=["Origin", "qwen14b_direct_prediction", "qwen14b_extracted_answer"]).copy()
            df["letter"] = make_letters(df, "qwen14b_direct_prediction", "qwen14b_extracted_answer")
        elif model_name == "Llama-70B":
            p = repo / "After_PT_Removal" / "Llama-70B" / "results" / "predictions" / "Llama70B_predictions_on_MJ_LowIRR.csv"
            df = mod.pd.read_csv(p, usecols=["Origin", "llama70b_direct_prediction", "llama70b_extracted_answer"]).copy()
            df["letter"] = make_letters(df, "llama70b_direct_prediction", "llama70b_extracted_answer")
        elif model_name == "MedGemma-27B":
            p = repo / "After_PT_Removal" / "MedGemma-27b-text-it" / "results" / "predictions" / "MedGemma27B_predictions_on_MJ_LowIRR.csv"
            df = mod.pd.read_csv(p, usecols=["Origin", "medgemma_direct_prediction", "medgemma_extracted_answer"]).copy()
            df["letter"] = make_letters(df, "medgemma_direct_prediction", "medgemma_extracted_answer")
        elif model_name == "GPT4o":
            p = (
                repo
                / "After_PT_Removal"
                / "GPT4o"
                / "results"
                / "predictions"
                / "gpt4o_predictions_on_MJ_LowIRR_expert933_subset_from_existing1300.csv"
            )
            df = mod.pd.read_csv(p, usecols=["Origin", "gpt_direct_prediction"]).copy()
            df["letter"] = make_letters(df, "gpt_direct_prediction")
        elif model_name == "GPT 5":
            p = repo / "After_PT_Removal" / "GPT5" / "results" / "predictions" / "gpt5_predictions_on_MJ_LowIRR.csv"
            df = mod.pd.read_csv(p, usecols=["Origin", "gpt5_direct_prediction", "answer_letter"]).copy()
            df["letter"] = make_letters(df, "gpt5_direct_prediction", "answer_letter")
        else:
            raise ValueError(f"Unsupported model for MJ_LowIRR mapping: {model_name}")

        df = df[["Origin", "letter"]].dropna(subset=["Origin"]).copy()
        df["Origin"] = df["Origin"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["Origin"])

        # Force round-2 to canonical 933 MJ Origins. Any missing rerun predictions
        # are represented as explicit "__MISSING__" letters (counted as wrong).
        canonical = expected_mj_origins(repo)
        df = canonical.merge(df, on="Origin", how="left")
        df["letter"] = df["letter"].fillna("__MISSING__")
        return df

    mod.load_round_letters = load_round_letters_mj
    # Keep full wording while preventing horizontal clipping/overlap.
    def panel_title_wrapped(model_name: str, r1_correct: int, r2_correct: int) -> str:
        return (
            f"<b>{model_name}  ({r1_correct} QAs)</b><br>"
            f"R1 Correct: {r1_correct}<br>"
            f"R2 Correct: {r2_correct}"
        )

    mod.panel_title = panel_title_wrapped
    # Base script applies a +90 y-shift to subplot annotations; pre-offset so
    # multi-line subtitles stay fully within the image canvas.
    def make_subplots_patched(*args, **kwargs):
        fig = orig_make_subplots(*args, **kwargs)
        anns = fig.layout.annotations or []
        for ann in anns:
            base = ann.yshift if ann.yshift is not None else 0
            ann.yshift = base - 42
        return fig

    mod.make_subplots = make_subplots_patched
    # Base generator excludes Llama-70B by default unless --include-llama is set.
    # Force inclusion for this MJ_LowIRR rerun figure unless user explicitly
    # provided the flag already.
    if "--include-llama" not in sys.argv:
        sys.argv.append("--include-llama")
    mod.main()


if __name__ == "__main__":
    main()
