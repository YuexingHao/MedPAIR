#!/usr/bin/env python3
"""Compute Sankey/barplot counts directly from raw prediction CSVs.

Definition used for every model and every setting:
- Universe: 933 Origin IDs from Physician_Labels/Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv
- R1-correct set: examples where model R1 prediction matches gold answer.
- R2-correct: within that R1-correct set, examples where model R2 prediction still matches gold.

This avoids mixing overall-accuracy counts with R1-conditioned counts.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

MODELS = ["Qwen-72B", "Llama-70B", "Qwen-14B", "MedGemma-27B", "GPT4o", "GPT 5"]
SETTINGS = ["Relevant", "Random", "Irrelevant"]
SOURCES = ["MMLU", "JAMA", "MedXpert", "MedBullets"]
SOURCE_MAP = {
    "mmlu": "MMLU",
    "jama": "JAMA",
    "medxpert": "MedXpert",
    "medbullets": "MedBullets",
}


R1_CONFIG: dict[str, dict[str, Any]] = {
    "Qwen-72B": {
        "path": "After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_ORIGINAL.csv",
        "origin_col": "Origin",
        "pred_cols": ["Extracted_Answer", "qwen72b_direct_prediction", "qwen72b_extracted_answer"],
    },
    "Llama-70B": {
        "path": "After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_Trainee.csv",
        "origin_col": "Origin",
        "pred_cols": ["Extracted_Answer", "llama70b_direct_prediction", "llama70b_extracted_answer"],
    },
    "Qwen-14B": {
        "path": "After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_ORIGINAL.csv",
        "origin_col": "Origin",
        "pred_cols": ["Extracted_Answer", "qwen14b_direct_prediction", "qwen14b_extracted_answer"],
    },
    "MedGemma-27B": {
        "path": "After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_ORIGINAL.csv",
        "origin_col": "Origin",
        "pred_cols": ["Extracted_Answer", "medgemma_direct_prediction", "medgemma_extracted_answer"],
    },
    "GPT4o": {
        "path": "After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_Original_Accuracy.csv",
        "origin_col": "Origin",
        "pred_cols": ["gpt_direct_prediction", "gpt4o_direct_prediction", "gpt_letter"],
    },
    "GPT 5": {
        "path": "After_PT_Removal/GPT5/results/predictions/gpt5_predictions_Original_Accuracy.csv",
        "origin_col": "ID_corr",
        "pred_cols": ["majority_vote", "gpt5_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
    },
}


R2_CONFIG: dict[str, dict[str, dict[str, Any]]] = {
    "Relevant": {
        "Qwen-72B": {
            "path": "After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_trainee_irr_removed.csv",
            "origin_col": "Origin",
            "pred_cols": ["qwen72b_direct_prediction", "qwen72b_extracted_answer", "Extracted_Answer"],
        },
        "Llama-70B": {
            "path": "After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_trainee_irr_removed.csv",
            "origin_col": "Origin",
            "pred_cols": ["llama70b_direct_prediction", "llama70b_extracted_answer", "Extracted_Answer"],
        },
        "Qwen-14B": {
            "path": "After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_trainee_irr_removed.csv",
            "origin_col": "Origin",
            "pred_cols": ["qwen14b_direct_prediction", "qwen14b_extracted_answer", "Extracted_Answer"],
        },
        "MedGemma-27B": {
            "path": "After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_trainee_irr_removed.csv",
            "origin_col": "Origin",
            "pred_cols": ["medgemma_direct_prediction", "medgemma_extracted_answer", "Extracted_Answer"],
        },
        "GPT4o": {
            "path": "After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_trainee_irr_removed.csv",
            "origin_col": "Origin",
            "pred_cols": ["gpt4o_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
        },
        "GPT 5": {
            "path": "After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_trainee_irr_removed.csv",
            "origin_col": "Origin",
            "pred_cols": ["gpt5_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
        },
    },
    "Irrelevant": {
        "Qwen-72B": {
            "path": "After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_MJ_LowIRR.csv",
            "origin_col": "Origin",
            "pred_cols": ["qwen72b_direct_prediction", "qwen72b_extracted_answer", "Extracted_Answer"],
        },
        "Llama-70B": {
            "path": "After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_MJ_LowIRR.csv",
            "origin_col": "Origin",
            "pred_cols": ["llama70b_direct_prediction", "llama70b_extracted_answer", "Extracted_Answer"],
        },
        "Qwen-14B": {
            "path": "After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_on_MJ_LowIRR.csv",
            "origin_col": "Origin",
            "pred_cols": ["qwen14b_direct_prediction", "qwen14b_extracted_answer", "Extracted_Answer"],
        },
        "MedGemma-27B": {
            "path": "After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_MJ_LowIRR.csv",
            "origin_col": "Origin",
            "pred_cols": ["medgemma_direct_prediction", "medgemma_extracted_answer", "Extracted_Answer"],
        },
        "GPT4o": {
            "path": "After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_MJ_LowIRR_expert933_subset_from_existing1300.csv",
            "origin_col": "Origin",
            "pred_cols": ["gpt4o_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
        },
        "GPT 5": {
            "path": "After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_MJ_LowIRR.csv",
            "origin_col": "Origin",
            "pred_cols": ["gpt5_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
        },
    },
    "Random": {
        "Qwen-72B": {
            "path": "After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_Random.csv",
            "origin_col": "Origin",
            "pred_cols": ["qwen72b_direct_prediction", "qwen72b_extracted_answer", "Extracted_Answer"],
        },
        "Llama-70B": {
            "path": "After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_Random.csv",
            "origin_col": "Origin",
            "pred_cols": ["llama70b_direct_prediction", "llama70b_extracted_answer", "Extracted_Answer"],
        },
        "Qwen-14B": {
            "path": "After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_on_Random.csv",
            "origin_col": "Origin",
            "pred_cols": ["qwen14b_direct_prediction", "qwen14b_extracted_answer", "Extracted_Answer"],
        },
        "MedGemma-27B": {
            "path": "After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_Random.csv",
            "origin_col": "Origin",
            "pred_cols": ["medgemma_direct_prediction", "medgemma_extracted_answer", "Extracted_Answer"],
        },
        "GPT4o": {
            "path": "After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_Random.csv",
            "origin_col": "Origin",
            "pred_cols": ["gpt4o_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
        },
        "GPT 5": {
            "path": "After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_Random.csv",
            "origin_col": "Origin",
            "pred_cols": ["gpt5_direct_prediction", "gpt_direct_prediction", "gpt_letter"],
        },
    },
}


def extract_letter(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    su = s.upper().replace("<ANSWER>", "").replace("</ANSWER>", "")
    for pat in (
        r"\bOPTION\s*[\(\[]?\s*([A-J])\s*[\)\]]?\b",
        r"\bANSWER\s*[:\-]?\s*[\(\[]?\s*([A-J])\s*[\)\]]?\b",
        r"^[\(\[]?\s*([A-J])\s*[\)\]]?$",
        r"\b([A-J])\b",
    ):
        m = re.search(pat, su)
        if m:
            return m.group(1)
    return None


def _read_csv_safe(path: Path, retries: int = 4) -> pd.DataFrame:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0)
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed reading {path}: {last_err or exc}") from exc


def _resolve_prediction_path(repo: Path, rel_path: str) -> Path:
    p = repo / rel_path
    # Fallback candidates: latest backup snapshot for this file name.
    fname = Path(rel_path).name
    results_dir = repo / "After_PT_Removal" / "results"
    backup_candidates = sorted(
        results_dir.glob(f"predictions_backup_before_true_rerun_*/{fname}.*.bak"),
        key=lambda x: x.as_posix(),
    )
    latest_backup = backup_candidates[-1] if backup_candidates else None

    if p.exists():
        # If canonical file is currently being rewritten and much smaller than
        # the latest backup, prefer backup to avoid mid-run partial reads.
        if latest_backup is not None:
            try:
                if p.stat().st_size < latest_backup.stat().st_size * 0.8:
                    return latest_backup
            except OSError:
                pass
        return p

    if backup_candidates:
        return latest_backup

    raise FileNotFoundError(f"Missing prediction file: {p}")


def _choose_origin_column(df: pd.DataFrame, preferred: str) -> str:
    for c in (preferred, "Origin", "ID_corr"):
        if c in df.columns:
            return c
    raise ValueError("Missing origin column; tried: preferred/Origin/ID_corr")


def _load_prediction_map(
    repo: Path,
    config: dict[str, Any],
    valid_origins: set[str],
) -> dict[str, str | None]:
    p = _resolve_prediction_path(repo, str(config["path"]))

    df = _read_csv_safe(p)
    origin_col = _choose_origin_column(df, str(config.get("origin_col", "Origin")))
    pred_cols = [c for c in config.get("pred_cols", []) if c in df.columns]
    if not pred_cols:
        raise ValueError(f"No configured prediction columns found in {p}")

    out = pd.DataFrame()
    out["Origin"] = df[origin_col].astype(str).str.strip()
    out = out[out["Origin"].isin(valid_origins)].copy()

    pred = pd.Series([None] * len(out), index=out.index, dtype=object)
    for c in pred_cols:
        parsed = df.loc[out.index, c].apply(extract_letter)
        pred = pred.where(pred.notna(), parsed)
    out["pred"] = pred

    out = out[["Origin", "pred"]].drop_duplicates(subset=["Origin"])
    return dict(zip(out["Origin"], out["pred"]))


def load_ref_933_and_sources(repo: Path) -> tuple[set[str], dict[str, str]]:
    p = repo / "Physician_Labels" / "2026-03_March" / "933_Clinician_Student_Majority_Vote.csv"
    df = pd.read_csv(p, usecols=["Origin", "data_source_corr"]).copy()
    df["Origin"] = df["Origin"].astype(str).str.strip()
    df["data_source_corr"] = df["data_source_corr"].astype(str).str.strip().str.lower()

    valid_origins = set(df["Origin"])
    source_by_origin: dict[str, str] = {}
    for _, r in df.iterrows():
        src = SOURCE_MAP.get(str(r["data_source_corr"]))
        if src:
            source_by_origin[str(r["Origin"])] = src
    return valid_origins, source_by_origin


def load_gold_map(repo: Path, valid_origins: set[str]) -> dict[str, str]:
    p = repo / "After_PT_Removal" / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_Original_Accuracy.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing gold source file: {p}")

    df = _read_csv_safe(p)
    origin_col = _choose_origin_column(df, "Origin")

    answer_col = None
    for c in ("answer_corr", "answer_letter", "Correct_Answer"):
        if c in df.columns:
            answer_col = c
            break
    if answer_col is None:
        raise ValueError(f"No answer column found in gold file: {p}")

    out = pd.DataFrame()
    out["Origin"] = df[origin_col].astype(str).str.strip()
    out = out[out["Origin"].isin(valid_origins)].copy()
    out["gold"] = df.loc[out.index, answer_col].apply(extract_letter)
    out = out[out["gold"].notna()]
    out = out[["Origin", "gold"]].drop_duplicates(subset=["Origin"])
    return dict(zip(out["Origin"], out["gold"]))


def _count_by_source(origins: list[str], source_by_origin: dict[str, str]) -> dict[str, int]:
    counts = {s: 0 for s in SOURCES}
    for o in origins:
        s = source_by_origin.get(o)
        if s in counts:
            counts[s] += 1
    return counts


def compute_all_summaries(repo: Path) -> dict[str, dict[str, dict[str, Any]]]:
    valid_origins, source_by_origin = load_ref_933_and_sources(repo)
    gold_by_origin = load_gold_map(repo, valid_origins)

    summaries: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in SETTINGS}

    for model in MODELS:
        r1_map = _load_prediction_map(repo, R1_CONFIG[model], valid_origins)

        r1_correct_origins = [
            o
            for o in valid_origins
            if o in gold_by_origin and r1_map.get(o) is not None and r1_map.get(o) == gold_by_origin[o]
        ]
        r1_by_source = _count_by_source(r1_correct_origins, source_by_origin)

        for setting in SETTINGS:
            r2_map = _load_prediction_map(repo, R2_CONFIG[setting][model], valid_origins)
            r2_correct_origins = [
                o
                for o in r1_correct_origins
                if r2_map.get(o) is not None and r2_map.get(o) == gold_by_origin[o]
            ]
            r2_by_source = _count_by_source(r2_correct_origins, source_by_origin)

            by_source = {}
            for src in SOURCES:
                r1s = int(r1_by_source[src])
                r2s = int(r2_by_source[src])
                by_source[src] = {
                    "r1_correct": r1s,
                    "spurious": r1s - r2s,
                    "r2_correct": r2s,
                }

            r1_total = len(r1_correct_origins)
            r2_total = len(r2_correct_origins)
            spurious = r1_total - r2_total
            pct = (spurious / r1_total * 100.0) if r1_total > 0 else 0.0

            summaries[setting][model] = {
                "model": model,
                "by_source": by_source,
                "round1_correct": int(r1_total),
                "spurious": int(spurious),
                "round2_correct": int(r2_total),
                "pct_spurious": float(pct),
            }

    return summaries


def counts_tuple_map(summaries: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, tuple[int, int]]]:
    out: dict[str, dict[str, tuple[int, int]]] = {m: {} for m in MODELS}
    for setting in SETTINGS:
        for model in MODELS:
            s = summaries[setting][model]
            out[model][setting] = (int(s["round1_correct"]), int(s["round2_correct"]))
    return out
