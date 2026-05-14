#!/usr/bin/env python3
"""
Recompute Sankey inputs for Expert QA (933): per MedPAIR source, per model.

**Per-QA definition (no shortcut from pooled totals):** for each ``Origin`` in the 933 expert
cohort we align Round 1 (original) and Round 2 (trainee-removed) predictions, then:

- ``r1_correct[Origin]`` — letter match vs gold for that QA in round 1.
- ``r2_correct[Origin]`` — letter match vs gold for that QA in round 2.
- **Spurious (Sankey red flow)** — count of QAs where ``r1_correct & ~r2_correct``.
- **Stayed correct (green)** — ``r1_correct & r2_correct`` (Sankey green flow mass = that count).
- **Recovered** — ``~r1_correct & r2_correct`` (not drawn as red; explains why
  ``(sum r1_correct) - (sum r2_correct)`` can differ from the spurious count).

The script never sets spurious to ``total_R1_correct - total_R2_correct``; every number is a
sum of per-QA boolean flags after an ``Origin``-keyed join.

Uses the same letter-matching logic as ``evaluate_predictions_by_physician_subsets_*.py``.
Writes:

- ``Figures/sankey/data/sankey_spurious_expert933.json`` (for notebooks /
  ``make_llm_sankey_all_models.py``). If
  ``Physician_Labels/Apr1_2026_Data/Round2_933_MJ_accuracy_by_data_source.csv`` exists, the JSON
  also includes ``physicians_round2_mj`` (per-source cohort size and MJ-wrong vs gold for the
  Physicians Sankey).
- ``Figures/sankey/data/sankey_spurious_expert933_results.csv`` (tidy counts + subplot grid;
  physician rows are prefixed when that MJ file is present)

Repo root is detected by walking parents until ``933_Clinician_Student_Majority_Vote.csv`` is
found (``Path(__file__).resolve()`` is used, so a symlinked clone resolves to its target, e.g.
``/orcd/...`` — that is normal). Run from any cwd under the repo::

  python Figures/sankey/compute_sankey_spurious_data.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    """NeuRIPS25 root whether this file lives under ``Figures/sankey/`` or ``Figures/sankey/notebooks/``."""
    for p in [_THIS_DIR, *_THIS_DIR.parents]:
        if (
            p / "Physician_Labels" / "Mar2_2026_Data" / "933_Clinician_Student_Majority_Vote.csv"
        ).is_file():
            return p
    raise FileNotFoundError(
        "Could not find repo root (933_Clinician_Student_Majority_Vote.csv). "
        f"Searched upward from {_THIS_DIR}"
    )


_WORKSPACE_ROOT = _find_repo_root()
_AFTER_PT = _WORKSPACE_ROOT / "After_PT_Removal"
# Round-2 clinician majority vote vs gold, by MedPAIR source (933 expert cohort).
_MJ_ACC_BY_SOURCE = (
    _WORKSPACE_ROOT
    / "Physician_Labels"
    / "Apr1_2026_Data"
    / "Round2_933_MJ_accuracy_by_data_source.csv"
)
_CSV_933 = (
    _WORKSPACE_ROOT
    / "Physician_Labels"
    / "Mar2_2026_Data"
    / "933_Clinician_Student_Majority_Vote.csv"
)
# Gold MCQ letters (same file used across eval scripts); 933 CSV has no answer column.
_GPT4O_ORIG_FOR_ANSWERS = (
    _AFTER_PT / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_Original_Accuracy.csv"
)

MODEL_ORDER = ["GPT4o", "GPT 5", "Qwen-14B", "Qwen 72B", "Llama-70B", "MedGemma-27B"]

# Same order as ``make_llm_sankey_all_models.build_figure`` (rows 2–3 of the 3×3 grid).
SANKEY_FIGURE_MODEL_ORDER = [
    "Qwen 72B",
    "Llama-70B",
    "Qwen-14B",
    "MedGemma-27B",
    "GPT4o",
    "GPT 5",
]


def _sankey_subplot_rc(model_name: str) -> tuple[int, int]:
    """1-based row/col in Plotly figure (row 1 = Physicians; LLM panels start row 2)."""
    i = SANKEY_FIGURE_MODEL_ORDER.index(model_name)
    return (i // 3) + 2, (i % 3) + 1

# Notebook / figure keys (must match Spurious_Rate_Manuscript_Figure.ipynb)
DATASET_KEYS = ["MMLU", "JAMA", "MedXpert", "MedBullets"]

SOURCE_NORMALIZE = {
    "mmlu": "MMLU",
    "jama": "JAMA",
    "medxpert": "MedXpert",
    "medbullets": "MedBullets",
}

PRED_COLS_GPT4O_STYLE = [
    "Extracted_Answer",
    "Llama70B_answer",
    "gpt4o_direct_prediction",
    "gpt5_direct_prediction",
    "GPT4o_prediction",
    "gpt_direct_prediction",
    "majority_vote",
    "GPT5_on_72B_SR",
]

PRED_COLS_GPT5_STYLE = [
    "Extracted_Answer",
    "gpt5_direct_prediction",
    "gpt4o_direct_prediction",
    "majority_vote",
    "GPT5_on_72B_SR",
]

ANSWER_COLS = ["answer_corr", "answer_df3"]


# (display name, original CSV, trainee-removed CSV)
_MODEL_FILES: list[tuple[str, Path, Path]] = [
    (
        "GPT4o",
        _AFTER_PT / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_Original_Accuracy.csv",
        _AFTER_PT / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_on_trainee_removed.csv",
    ),
    (
        "GPT 5",
        _AFTER_PT / "GPT5" / "results" / "predictions" / "gpt5_predictions_Original_Accuracy.csv",
        _AFTER_PT / "GPT5" / "results" / "predictions" / "gpt5_predictions_on_trainee_removed.csv",
    ),
    (
        "Qwen-14B",
        _AFTER_PT
        / "Qwen2.5-14B-Instruct"
        / "results"
        / "predictions"
        / "Qwen_14B_predictions_ORIGINAL.csv",
        _AFTER_PT
        / "Qwen2.5-14B-Instruct"
        / "results"
        / "predictions"
        / "Qwen_14B_predictions_trainee_removed.csv",
    ),
    (
        "Qwen 72B",
        _AFTER_PT
        / "Qwen2.5-72B-Instruct"
        / "results"
        / "predictions"
        / "Qwen_72B_predictions_ORIGINAL.csv",
        _AFTER_PT
        / "Qwen2.5-72B-Instruct"
        / "results"
        / "predictions"
        / "Qwen_72B_predictions_Trainee.csv",
    ),
    (
        "Llama-70B",
        _AFTER_PT
        / "Llama-70B"
        / "results"
        / "predictions"
        / "Llama70B_annotated_ORIGINAL_Accuracy.csv",
        _AFTER_PT / "Llama-70B" / "results" / "predictions" / "Llama70B_predictions_on_Trainee.csv",
    ),
    (
        "MedGemma-27B",
        _AFTER_PT
        / "MedGemma-27b-text-it"
        / "results"
        / "predictions"
        / "MedGemma27B_predictions_ORIGINAL.csv",
        _AFTER_PT
        / "MedGemma-27b-text-it"
        / "results"
        / "predictions"
        / "MedGemma27B_predictions_on_Trainee.csv",
    ),
]


def _llama_round1_annotated_path() -> Path:
    """
    Primary path is under ``After_PT_Removal``; also check ``Physician_Labels/results/`` where
    exports sometimes land (e.g. ``[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv``). All are merged
    to the 933 Origins via ``prepare_with_ref``. Do **not** use ``Llama70B_ORIGINAL_predictions.csv``
    (different row universe than the expert cohort).
    """
    candidates = [
        _AFTER_PT
        / "Llama-70B"
        / "results"
        / "predictions"
        / "Llama70B_annotated_ORIGINAL_Accuracy.csv",
        _WORKSPACE_ROOT
        / "Physician_Labels"
        / "results"
        / "Llama70B_annotated_ORIGINAL_Accuracy.csv",
        _WORKSPACE_ROOT
        / "Physician_Labels"
        / "results"
        / "[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Llama-70B round-1 annotated predictions missing. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
        + f"\n\nRepo root: {_WORKSPACE_ROOT} (symlink targets like /orcd/... are normal)."
    )


def _model_files_resolved() -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for name, p_o, p_t in _MODEL_FILES:
        if name == "Llama-70B":
            p_o = _llama_round1_annotated_path()
        out.append((name, p_o, p_t))
    return out


def extract_letter_from_text(x) -> str | None:
    if not isinstance(x, str):
        return None
    m = re.search(
        r"Option\s*\[?([A-J])\]?|^\s*([A-J])\s*$",
        x.strip(),
        flags=re.IGNORECASE,
    )
    if m:
        return (m.group(1) or m.group(2)).upper()
    return None


def augment_extracted_answer(df: pd.DataFrame) -> pd.DataFrame:
    """Qwen/Llama/MedGemma trainee exports often store the letter in ``Original`` only."""
    out = df.copy()
    if "Extracted_Answer" not in out.columns and "Original" in out.columns:
        out["Extracted_Answer"] = out["Original"]
    return out


def ensure_origin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Origin" not in out.columns and "ID_corr" in out.columns:
        out["Origin"] = out["ID_corr"].astype(str).str.strip()
    return out


def pick_answer_col(df: pd.DataFrame) -> str | None:
    for c in ANSWER_COLS:
        if c in df.columns:
            return c
    return None


def pick_pred_col(df: pd.DataFrame, basename: str, style: str) -> str | None:
    b = basename.lower()
    if style == "gpt5":
        if b == "gpt5_predictions_original_accuracy.csv" and "majority_vote" in df.columns:
            return "majority_vote"
        for c in PRED_COLS_GPT5_STYLE:
            if c in df.columns:
                return c
    else:
        # For GPT-4o Original Accuracy, use model prediction text instead of Round 1 Letter.
        # Round 1 Letter in this file reflects a separate notebook export and undercounts R1.
        if b == "gpt4o_predictions_original_accuracy.csv":
            for c in ("gpt_direct_prediction", "gpt4o_direct_prediction"):
                if c in df.columns:
                    return c
        for c in PRED_COLS_GPT4O_STYLE:
            if c in df.columns:
                return c
        if "Round 1 Letter" in df.columns:
            return "Round 1 Letter"
    if "gpt_letter" in df.columns:
        return "gpt_letter"
    return None


def build_gpt_letter(df: pd.DataFrame, pred_col: str, *, gpt5_style: bool) -> pd.Series:
    if pred_col == "gpt_letter":
        return df[pred_col].astype(str).str.strip().str.upper()
    if pred_col in ("GPT5_on_72B_SR", "Extracted_Answer", "Llama70B_answer"):
        return df[pred_col].astype(str).str.strip().str.upper()
    if pred_col in ("Round 1 Letter", "Round 2 Letter"):

        def _one(x) -> str | None:
            if pd.isna(x):
                return None
            s = str(x).strip()
            if len(s) == 1 and s.isalpha():
                return s.upper()
            return extract_letter_from_text(s)

        return df[pred_col].map(_one)
    if (
        not gpt5_style
        and pred_col == "majority_vote"
        and pick_answer_col(df) is not None
    ):
        return df[pred_col].astype(str).str.strip().str.upper()
    return df[pred_col].apply(extract_letter_from_text)


def _correct_series(
    df: pd.DataFrame, pred_col: str, *, gpt5_style: bool
) -> pd.Series:
    ac = pick_answer_col(df)
    if ac is None:
        raise ValueError("No answer column after merge")
    ans = df[ac].astype(str).str.strip().str.upper()
    gl = build_gpt_letter(df, pred_col, gpt5_style=gpt5_style)
    return gl.notna() & ans.notna() & (gl == ans)


def _qa_id_sort_key(qa: Any) -> int:
    if pd.isna(qa):
        return 0
    s = str(qa).strip()
    if s.lower().startswith("merge q"):
        try:
            return int(s.split("Q", 1)[1].strip())
        except (ValueError, IndexError):
            return 0
    return 0


def _one_row_per_origin(df: pd.DataFrame, *, model_name: str, path: Path, role: str) -> pd.DataFrame:
    """If a prediction file has duplicate ``Origin`` rows, keep one row per QA (avoid merge blow-up)."""
    if df.empty or not df["Origin"].duplicated().any():
        return df
    extra = int(df.duplicated(subset=["Origin"], keep=False).sum())
    print(
        f"  [{model_name}] {path.name} ({role}): {extra} rows share an Origin with another row; "
        f"keeping one row per Origin (last by QA_ID order).",
        file=sys.stderr,
    )
    out = df.copy()
    if "QA_ID" in out.columns:
        out["_sort_q"] = out["QA_ID"].map(_qa_id_sort_key)
        out = out.sort_values("_sort_q").drop(columns=["_sort_q"])
    return out.drop_duplicates(subset=["Origin"], keep="last")


def prepare_with_ref(pred: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    pred = ensure_origin(pred)
    ref = ensure_origin(ref)
    r = ref[["Origin", "answer_corr", "data_source_corr"]].drop_duplicates(subset=["Origin"])
    r["Origin"] = r["Origin"].astype(str).str.strip()
    pred = pred.copy()
    pred["Origin"] = pred["Origin"].astype(str).str.strip()
    for c in ("answer_corr", "data_source_corr"):
        if c in pred.columns:
            pred = pred.drop(columns=[c])
    return pred.merge(r, on="Origin", how="inner")


def counts_for_model(
    name: str,
    path_orig: Path,
    path_tr: Path,
    ref: pd.DataFrame,
    pred_style: str,
) -> dict[str, Any]:
    if not path_orig.is_file():
        raise FileNotFoundError(f"Missing original predictions: {path_orig}")
    if not path_tr.is_file():
        raise FileNotFoundError(f"Missing trainee predictions: {path_tr}")

    o = prepare_with_ref(augment_extracted_answer(pd.read_csv(path_orig)), ref)
    t = prepare_with_ref(augment_extracted_answer(pd.read_csv(path_tr)), ref)
    o = _one_row_per_origin(o, model_name=name, path=path_orig, role="round1")
    t = _one_row_per_origin(t, model_name=name, path=path_tr, role="round2")

    pc_o = pick_pred_col(o, path_orig.name, pred_style)
    pc_t = pick_pred_col(t, path_tr.name, pred_style)
    if pc_o is None:
        raise ValueError(f"{name}: no prediction column in {path_orig.name}")
    if pc_t is None:
        raise ValueError(f"{name}: no prediction column in {path_tr.name}")

    gs = pred_style == "gpt5"
    o = o.copy()
    t = t.copy()
    o["_r1_correct"] = _correct_series(o, pc_o, gpt5_style=gs)
    t["_r2_correct"] = _correct_series(t, pc_t, gpt5_style=gs)

    o["_panel"] = o["data_source_corr"].astype(str).str.strip().str.lower()
    joined = o[["Origin", "data_source_corr", "_panel", "_r1_correct"]].merge(
        t[["Origin", "_r2_correct"]],
        on="Origin",
        how="inner",
        validate="one_to_one",
    )

    joined["_panel"] = joined["_panel"].map(lambda x: SOURCE_NORMALIZE.get(x, None))
    bad = joined["_panel"].isna()
    if bad.any():
        n_bad = int(bad.sum())
        print(f"  [{name}] warning: {n_bad} rows with unknown data_source_corr (skipped)", file=sys.stderr)

    qa = joined.loc[~bad].copy()
    qa["_spurious"] = qa["_r1_correct"] & ~qa["_r2_correct"]

    n_r1 = int(qa["_r1_correct"].sum())
    n_r2 = int(qa["_r2_correct"].sum())
    n_spurious = int(qa["_spurious"].sum())
    n_recovered = int((~qa["_r1_correct"] & qa["_r2_correct"]).sum())
    if n_r2 != n_r1 - n_spurious + n_recovered:
        raise ValueError(
            f"{name}: internal mismatch R2 correct {n_r2} vs "
            f"{n_r1}-{n_spurious}+{n_recovered}"
        )

    out: dict[str, dict[str, list[int]]] = {}
    for ds in DATASET_KEYS:
        sub = qa[qa["_panel"] == ds]
        r1 = int(sub["_r1_correct"].sum())
        r2_spur = int(sub["_spurious"].sum())
        if r2_spur > r1:
            raise ValueError(f"{name} {ds}: spurious {r2_spur} > round1_correct {r1}")
        out[ds] = {"round1_correct": r1, "round2_incorrect": r2_spur}

    return {
        "name": name,
        "n_origins_joined": int(len(qa)),
        "by_dataset": out,
        "pairwise_expert933": {
            "n_correct_round1": n_r1,
            "n_correct_round2": n_r2,
            "n_spurious_r1_correct_r2_wrong": n_spurious,
            "n_recovered_r1_wrong_r2_correct": n_recovered,
            "net_drop_in_correct_answers": n_r1 - n_r2,
        },
    }


def load_ref933() -> pd.DataFrame:
    if not _CSV_933.is_file():
        raise FileNotFoundError(_CSV_933)
    if not _GPT4O_ORIG_FOR_ANSWERS.is_file():
        raise FileNotFoundError(_GPT4O_ORIG_FOR_ANSWERS)
    meta = pd.read_csv(_CSV_933, usecols=["Origin", "data_source_corr"])
    meta["Origin"] = meta["Origin"].astype(str).str.strip()
    ans = pd.read_csv(
        _GPT4O_ORIG_FOR_ANSWERS, usecols=["Origin", "answer_corr"]
    ).drop_duplicates(subset=["Origin"])
    ans["Origin"] = ans["Origin"].astype(str).str.strip()
    df = meta.merge(ans, on="Origin", how="left")
    if df["answer_corr"].isna().any():
        raise ValueError(
            "Some 933 Origins lack answer_corr in GPT-4o Original predictions export"
        )
    return df


def build_datasets_payload(ref: pd.DataFrame) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    max_length: dict[str, int] = {}

    for ds in DATASET_KEYS:
        sub = ref[
            ref["data_source_corr"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(SOURCE_NORMALIZE.get)
            == ds
        ]
        max_length[ds] = int(len(sub))

    for ds in DATASET_KEYS:
        datasets[ds] = {
            "max_length": max_length[ds],
            "round1_correct": [],
            "round2_incorrect": [],
        }

    details = []
    for name, p_o, p_t in _model_files_resolved():
        if name not in MODEL_ORDER:
            raise ValueError(name)
        style = "gpt5" if name == "GPT 5" else "gpt4o"
        row = counts_for_model(name, p_o, p_t, ref, style)
        details.append(row)
        for ds in DATASET_KEYS:
            b = row["by_dataset"][ds]
            datasets[ds]["round1_correct"].append(b["round1_correct"])
            datasets[ds]["round2_incorrect"].append(b["round2_incorrect"])

    return {"datasets": datasets, "model_details": details}


def load_physicians_round2_mj(path: Path) -> dict[str, dict[str, int]]:
    """
    Per MedPAIR source: cohort size (Sankey R1 mass per source) and R2 MJ-wrong vs gold
    (red flow = n_origins − n_correct). Keys match ``DATASET_KEYS``.
    """
    df = pd.read_csv(path)
    out: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        key = str(row["data_source_corr_x"]).strip().lower()
        ds = SOURCE_NORMALIZE.get(key)
        if ds is None:
            continue
        n_o = int(row["n_origins"])
        n_c = int(row["n_correct"])
        out[ds] = {
            "round1_correct": n_o,
            "round2_incorrect": n_o - n_c,
        }
    missing = [ds for ds in DATASET_KEYS if ds not in out]
    if missing:
        raise ValueError(f"{path}: missing sources {missing} (have {list(out.keys())})")
    return out


def write_results_csv(payload: dict, csv_path: Path) -> None:
    """
    Tidy table for manuscript / sanity checks. Matches subplot titles and Sankey link %s:

    - ``round1_correct`` / ``round2_incorrect_spurious`` per source: Sankey red flows (R1 correct
      but R2 wrong), same as JSON.
    - **Pooled row** — ``eval_*`` are sums of the same per-QA flags as the Sankey:
      ``eval_n_spurious_r1c_r2w`` = count of QAs with (R1 correct ∧ R2 wrong).
      ``eval_net_drop_in_correct_answers`` = ``n_correct_round1`` − ``n_correct_round2`` (sum of
      per-QA correct flags), equal to ``spurious − recovered``, **not** the red Sankey mass.
    """
    rows: list[dict[str, Any]] = []
    models: list[str] = payload["models"]
    ds_block: dict[str, Any] = payload["datasets"]
    details: list[dict[str, Any]] = payload["model_details"]

    pw_empty = {
        "eval_n_correct_round1": "",
        "eval_n_correct_round2": "",
        "eval_n_spurious_r1c_r2w": "",
        "eval_n_recovered_r1w_r2c": "",
        "eval_net_drop_in_correct_answers": "",
    }

    phy_block = payload.get("physicians_round2_mj")
    if phy_block:
        phy_row, phy_col = 1, 2
        r1_tot = 0
        r2_tot = 0
        for ds in DATASET_KEYS:
            b = phy_block[ds]
            r1 = int(b["round1_correct"])
            r2i = int(b["round2_incorrect"])
            r1_tot += r1
            r2_tot += r2i
            stay = r1 - r2i
            pct = (100.0 * r2i / r1) if r1 > 0 else 0.0
            rows.append(
                {
                    "row_type": "physicians_by_medpair_source",
                    "model": "Physicians",
                    "model_index_json": "",
                    "sankey_subplot_row": phy_row,
                    "sankey_subplot_col": phy_col,
                    "medpair_source": ds,
                    "n_expert_qa_in_source": r1,
                    "round1_correct": r1,
                    "round2_incorrect_spurious": r2i,
                    "round2_still_correct": stay,
                    "pct_spurious_among_round1_correct": round(pct, 4),
                    **pw_empty,
                }
            )
        stay_tot = r1_tot - r2_tot
        pct_all = (100.0 * r2_tot / r1_tot) if r1_tot > 0 else 0.0
        rows.append(
            {
                "row_type": "physicians_pooled_over_sources",
                "model": "Physicians",
                "model_index_json": "",
                "sankey_subplot_row": phy_row,
                "sankey_subplot_col": phy_col,
                "medpair_source": "ALL_SOURCES_SUM",
                "n_expert_qa_in_source": int(payload["overall_max_length"]),
                "round1_correct": r1_tot,
                "round2_incorrect_spurious": r2_tot,
                "round2_still_correct": stay_tot,
                "pct_spurious_among_round1_correct": round(pct_all, 4),
                **pw_empty,
            }
        )

    for mi, model_name in enumerate(models):
        sr, sc = _sankey_subplot_rc(model_name)
        for ds in DATASET_KEYS:
            d = ds_block[ds]
            r1 = int(d["round1_correct"][mi])
            r2 = int(d["round2_incorrect"][mi])
            n_cohort = int(d["max_length"])
            stay = r1 - r2
            pct = (100.0 * r2 / r1) if r1 > 0 else 0.0
            row = {
                "row_type": "by_medpair_source",
                "model": model_name,
                "model_index_json": mi,
                "sankey_subplot_row": sr,
                "sankey_subplot_col": sc,
                "medpair_source": ds,
                "n_expert_qa_in_source": n_cohort,
                "round1_correct": r1,
                "round2_incorrect_spurious": r2,
                "round2_still_correct": stay,
                "pct_spurious_among_round1_correct": round(pct, 4),
            }
            row.update(pw_empty)
            rows.append(row)

    for mi, model_name in enumerate(models):
        sr, sc = _sankey_subplot_rc(model_name)
        r1_tot = sum(int(ds_block[ds]["round1_correct"][mi]) for ds in DATASET_KEYS)
        r2_tot = sum(int(ds_block[ds]["round2_incorrect"][mi]) for ds in DATASET_KEYS)
        stay_tot = r1_tot - r2_tot
        pct_all = (100.0 * r2_tot / r1_tot) if r1_tot > 0 else 0.0
        pw = details[mi]["pairwise_expert933"]
        rows.append(
            {
                "row_type": "model_pooled_over_sources",
                "model": model_name,
                "model_index_json": mi,
                "sankey_subplot_row": sr,
                "sankey_subplot_col": sc,
                "medpair_source": "ALL_SOURCES_SUM",
                "n_expert_qa_in_source": int(payload["overall_max_length"]),
                "round1_correct": r1_tot,
                "round2_incorrect_spurious": r2_tot,
                "round2_still_correct": stay_tot,
                "pct_spurious_among_round1_correct": round(pct_all, 4),
                "eval_n_correct_round1": pw["n_correct_round1"],
                "eval_n_correct_round2": pw["n_correct_round2"],
                "eval_n_spurious_r1c_r2w": pw["n_spurious_r1_correct_r2_wrong"],
                "eval_n_recovered_r1w_r2c": pw["n_recovered_r1_wrong_r2_correct"],
                "eval_net_drop_in_correct_answers": pw["net_drop_in_correct_answers"],
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main() -> None:
    ref = load_ref933()
    ref["Origin"] = ref["Origin"].astype(str).str.strip()
    payload = build_datasets_payload(ref)
    payload["models"] = MODEL_ORDER
    payload["overall_max_length"] = int(len(ref))
    payload["source_csv_933"] = str(_CSV_933)
    payload["answer_letters_from"] = str(_GPT4O_ORIG_FOR_ANSWERS)

    if _MJ_ACC_BY_SOURCE.is_file():
        payload["physicians_round2_mj"] = load_physicians_round2_mj(_MJ_ACC_BY_SOURCE)
        payload["physicians_round2_mj_csv"] = str(_MJ_ACC_BY_SOURCE)
    else:
        print(
            f"Note: {_MJ_ACC_BY_SOURCE.name} not found — JSON/CSV will omit Physicians R2 MJ block.",
            file=sys.stderr,
        )

    out_dir = _WORKSPACE_ROOT / "Figures" / "sankey" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sankey_spurious_expert933.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")

    csv_path = out_dir / "sankey_spurious_expert933_results.csv"
    write_results_csv(payload, csv_path)
    print(f"Wrote {csv_path}")

    n_joined = payload["model_details"][0]["n_origins_joined"]
    print(f"Expert QA (933): {n_joined} Origins joined on ref (same for all models)\n")
    for d in payload["model_details"]:
        name = d["name"]
        print(f"{name}:")
        r1_tot = 0
        r2_tot = 0
        for ds in DATASET_KEYS:
            b = d["by_dataset"][ds]
            r1 = b["round1_correct"]
            r2 = b["round2_incorrect"]
            r1_tot += r1
            r2_tot += r2
            print(f"  {ds}: round1_correct={r1}  round2_incorrect={r2}")
        print(
            f"  Total (sum over sources): round1_correct={r1_tot}  "
            f"round2_incorrect_spurious={r2_tot}"
        )
        pw = d["pairwise_expert933"]
        print(
            "  Pooled 933 (matches eval CSV correct counts & origin logic): "
            f"correct_R1={pw['n_correct_round1']} correct_R2={pw['n_correct_round2']} "
            f"net_drop={pw['net_drop_in_correct_answers']} | "
            f"spurious_R1c_R2w={pw['n_spurious_r1_correct_r2_wrong']} "
            f"recovered_R1w_R2c={pw['n_recovered_r1_wrong_r2_correct']}"
        )
        print()


if __name__ == "__main__":
    main()
