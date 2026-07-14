from __future__ import annotations

import collections
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


_THIS_DIR = Path(__file__).resolve().parent
_LEGACY_PYC_CANDIDATES = [
    _THIS_DIR / "make_medpair_combined_figure.pyc",
    _THIS_DIR / "__pycache__" / "make_medpair_combined_figure.cpython-313.pyc",
]
_MAY27_RAW = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "May27_2026_Data"
    / "Text Relevance Analysis Case View gpt5 phase - 052626.csv"
)
_R1_Q1_BY_SRC = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "Mar2_2026_Data"
    / "Clinician_Student_q1_MJ_accuracy_by_data_source.csv"
)
_R2_MJ_BY_SRC = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "Apr1_2026_Data"
    / "Round2_933_MJ_accuracy_by_data_source.csv"
)
_QWEN72_RAW = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "Jun19_2026_Data"
    / "Qwen72B_Predicted_High.csv"
)
_PT_SOURCE_MAP = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "May27_2026_Data"
    / "May27_2026_Origin_Summary_933.csv"
)
_LLAMA70B_SR_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Llama-70B"
    / "results"
    / "predictions"
    / "[SR]_Llama70B_predictions_on_14B.csv"
)
_QWEN14B_SR_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-14B-Instruct"
    / "results"
    / "predictions"
    / "[SR]_Qwen_14B_predictions_Qwen14B.csv"
)
_QWEN72B_SR_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-72B-Instruct"
    / "results"
    / "predictions"
    / "[SR]_Qwen_72B_predictions_14B_SR.csv"
)
_LLAMA70B_CC_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Llama-70B"
    / "results"
    / "predictions"
    / "Llama70B_predictions_on_trainee_irr_removed.csv"
)
_LLAMA70B_SR_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Llama-70B"
    / "results"
    / "predictions"
    / "Llama70B_predictions_on_trainee_irr_removed.csv"
)
_QWEN14B_CC_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-14B-Instruct"
    / "results"
    / "predictions"
    / "Qwen_14B_predictions_trainee_irr_removed.csv"
)
_QWEN14B_SR_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-14B-Instruct"
    / "results"
    / "predictions"
    / "Qwen_14B_predictions_trainee_irr_removed.csv"
)
_QWEN72B_CC_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-72B-Instruct"
    / "results"
    / "predictions"
    / "Qwen_72B_predictions_trainee_irr_removed.csv"
)
_QWEN72B_SR_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-72B-Instruct"
    / "results"
    / "predictions"
    / "Qwen_72B_predictions_trainee_irr_removed.csv"
)
_GPT4O_SR_QWEN72B_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "GPT4o"
    / "results"
    / "predictions"
    / "[SR]_gpt4o_predictions_on_qwen72b_removed.csv"
)
_GPT5_SR_QWEN72B_REMOVED_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "GPT5"
    / "results"
    / "predictions"
    / "[SR]_gpt5_predictions_on_72B_removed.csv"
)
_MEDGEMMA27B_SR_QWEN72B_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "MedGemma-27b-text-it"
    / "results"
    / "predictions"
    / "[SR]_MedGemma27B_predictions_72B.csv"
)
_QWEN14B_SR_QWEN72B_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Qwen2.5-14B-Instruct"
    / "results"
    / "predictions"
    / "[SR]_Qwen_14B_predictions_Qwen72B.csv"
)
_LLAMA70B_SR_QWEN72B_RAW = (
    _THIS_DIR.parent.parent.parent
    / "After_PT_Removal"
    / "Llama-70B"
    / "results"
    / "predictions"
    / "[SR]_Llama70B_predictions_on_72B.csv"
)
_PT_MODEL = "Physician + Trainee"
_GPT5_REMOVED = "GPT5 Removed"
_QWEN72_REMOVED = "Qwen-72B Removed"
_LLAMA70B_SR = "Llama-70B (SR)"
_LLAMA70B_CC_REMOVED = "Llama-70B Removed (CC)"
_LLAMA70B_SR_REMOVED = "Llama-70B Removed (SR)"
_QWEN14B_SR = "Qwen-14B (SR)"
_QWEN14B_CC_REMOVED = "Qwen-14B Removed (CC)"
_QWEN14B_SR_REMOVED = "Qwen-14B Removed (SR)"
_QWEN72B_SR = "Qwen-72B (SR)"
_QWEN72B_CC_REMOVED = "Qwen-72B Removed (CC)"
_QWEN72B_SR_REMOVED = "Qwen-72B Removed (SR)"
_GPT4O_SR_QWEN72B_REMOVED = "GPT-4o on Qwen-72B Removed (SR)"
_GPT5_SR_QWEN72B_REMOVED = "GPT-5 on Qwen-72B Removed (SR)"
_MEDGEMMA27B_SR_QWEN72B = "MedGemma-27B on Qwen-72B (SR)"
_QWEN14B_SR_QWEN72B = "Qwen-14B on Qwen-72B (SR)"
_LLAMA70B_SR_QWEN72B = "Llama-70B on Qwen-72B (SR)"
_N_CANDIDATES = np.array(
    [
        33,
        160,
        192,
        249,
        250,
        286,
        295,
        334,
        733,
        747,
        750,
        928,
        930,
        933,
        934,
        1996,
    ],
    dtype=int,
)


def _load_legacy_module():
    for pyc in _LEGACY_PYC_CANDIDATES:
        if pyc.is_file():
            spec = importlib.util.spec_from_file_location("_legacy_medpair_combined", pyc)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise FileNotFoundError(
        "Could not find legacy compiled module. Checked: "
        + ", ".join(str(p) for p in _LEGACY_PYC_CANDIDATES)
    )


def _norm_choice(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    m = re.search(r"[A-J]", s)
    return m.group(0) if m else ""


def _bern_std_pct(acc_pct: float | int | np.floating | None) -> float:
    if acc_pct is None or pd.isna(acc_pct):
        return float("nan")
    p = float(acc_pct) / 100.0
    return 100.0 * float(np.sqrt(max(p * (1.0 - p), 0.0)))


def _compute_pt_condition_rows(
    *,
    raw_csv: Path,
    condition_label: str,
    source_col: str | None = None,
    source_map_csv: Path | None = None,
    origin_strip_suffix: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {raw_csv}")

    raw = pd.read_csv(raw_csv)
    need = {"Origin", "q1", "Correct answer"}
    missing = need - set(raw.columns)
    if missing:
        raise KeyError(f"{raw_csv.name} missing columns: {sorted(missing)}")

    work = raw.copy()
    work["Origin"] = work["Origin"].astype(str).str.strip()
    if origin_strip_suffix:
        work["Origin"] = work["Origin"].str.replace(r"-phase\d+$", "", regex=True)
    work["choice"] = work["q1"].map(_norm_choice)
    work["correct"] = work["Correct answer"].map(_norm_choice)

    rows: list[dict[str, object]] = []
    for origin, g in work.groupby("Origin", sort=False):
        votes = [x for x in g["choice"].tolist() if x]
        c = collections.Counter(votes)
        if not c:
            mv = ""
            tie = True
        else:
            best = max(c.values())
            winners = sorted([k for k, v in c.items() if v == best])
            tie = len(winners) != 1
            mv = winners[0] if not tie else ""

        correct_series = g["correct"].dropna()
        correct = correct_series.iloc[0] if len(correct_series) else ""
        rows.append(
            {
                "Origin": origin,
                "mv_correct": (not tie) and bool(mv) and (mv == correct),
            }
        )

    per_origin = pd.DataFrame(rows)

    if source_col and source_col in work.columns:
        src = (
            work[["Origin", source_col]]
            .dropna(subset=[source_col])
            .drop_duplicates(subset=["Origin"])
            .rename(columns={source_col: "source"})
        )
    elif source_map_csv is not None:
        if not source_map_csv.is_file():
            raise FileNotFoundError(f"Missing source map CSV: {source_map_csv}")
        src_df = pd.read_csv(source_map_csv)
        if "Origin" not in src_df.columns or "data_source_corr_x" not in src_df.columns:
            raise KeyError(
                f"{source_map_csv.name} must include Origin and data_source_corr_x columns."
            )
        src = (
            src_df[["Origin", "data_source_corr_x"]]
            .copy()
            .dropna(subset=["Origin", "data_source_corr_x"])
            .drop_duplicates(subset=["Origin"])
            .rename(columns={"data_source_corr_x": "source"})
        )
    else:
        raise ValueError("Need either source_col or source_map_csv for per-source stats.")

    per_origin = per_origin.merge(src, on="Origin", how="left")
    if per_origin["source"].isna().any():
        n = int(per_origin["source"].isna().sum())
        raise ValueError(f"{condition_label}: missing data-source mapping for {n} origins.")

    per_origin["source"] = per_origin["source"].astype(str).str.strip().str.lower()
    per_origin["mv_correct"] = per_origin["mv_correct"].astype(float)

    overall = 100.0 * float(per_origin["mv_correct"].mean())
    by_src = 100.0 * per_origin.groupby("source")["mv_correct"].mean()

    src_map = {
        "mmlu": float(by_src.get("mmlu", np.nan)),
        "jama": float(by_src.get("jama", np.nan)),
        "medxpert": float(by_src.get("medxpert", np.nan)),
        "medbullets": float(by_src.get("medbullets", np.nan)),
    }

    top = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": condition_label,
        "Total": overall,
        "MMLU": overall,
        "Jama": float("nan"),
        "MedXpert": float("nan"),
        "Medbullets": float("nan"),
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(overall),
        "JAMA_STD": float("nan"),
        "MedXpert_STD": float("nan"),
        "MedBullets_STD": float("nan"),
    }
    bottom = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": condition_label,
        "Total": overall,
        "MMLU": src_map["mmlu"],
        "Jama": src_map["jama"],
        "MedXpert": src_map["medxpert"],
        "Medbullets": src_map["medbullets"],
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(src_map["mmlu"]),
        "JAMA_STD": _bern_std_pct(src_map["jama"]),
        "MedXpert_STD": _bern_std_pct(src_map["medxpert"]),
        "MedBullets_STD": _bern_std_pct(src_map["medbullets"]),
    }
    return top, bottom


def _compute_pt_gpt5_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_pt_condition_rows(
        raw_csv=_MAY27_RAW,
        condition_label=_GPT5_REMOVED,
        source_col="data_source_corr_x",
    )


def _compute_pt_qwen72_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_pt_condition_rows(
        raw_csv=_QWEN72_RAW,
        condition_label=_QWEN72_REMOVED,
        source_map_csv=_PT_SOURCE_MAP,
        origin_strip_suffix=True,
    )


def _compute_model_rows(
    *,
    raw_csv: Path,
    condition_label: str,
    source_col: str = "data_source",
    answer_col: str = "Extracted_Answer",
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute accuracy rows for model predictions (CC or SR)."""
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {raw_csv}")

    raw = pd.read_csv(raw_csv)
    need = {"Origin", answer_col, source_col}
    missing = need - set(raw.columns)
    if missing:
        raise KeyError(f"{raw_csv.name} missing columns: {sorted(missing)}")

    work = raw.copy()
    work["Origin"] = work["Origin"].astype(str).str.strip()
    work[answer_col] = work[answer_col].astype(str).str.strip()
    work[source_col] = work[source_col].astype(str).str.strip().str.lower()

    rows: list[dict[str, object]] = []
    for origin, g in work.groupby("Origin", sort=False):
        # Check if answer is not empty/null
        correct = (g[answer_col].notna() & (g[answer_col] != "")).any()
        source = g[source_col].iloc[0] if len(g) > 0 else ""
        rows.append(
            {
                "Origin": origin,
                "correct": 1.0 if correct else 0.0,
                "source": source,
            }
        )

    per_origin = pd.DataFrame(rows)
    per_origin["source"] = per_origin["source"].astype(str).str.strip().str.lower()
    per_origin["correct"] = per_origin["correct"].astype(float)

    overall = 100.0 * float(per_origin["correct"].mean())
    by_src = 100.0 * per_origin.groupby("source")["correct"].mean()

    src_map = {
        "mmlu": float(by_src.get("mmlu", np.nan)),
        "jama": float(by_src.get("jama", np.nan)),
        "medxpert": float(by_src.get("medxpert", np.nan)),
        "medbullets": float(by_src.get("medbullets", np.nan)),
    }

    top = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": condition_label,
        "Total": overall,
        "MMLU": overall,
        "Jama": float("nan"),
        "MedXpert": float("nan"),
        "Medbullets": float("nan"),
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(overall),
        "JAMA_STD": float("nan"),
        "MedXpert_STD": float("nan"),
        "MedBullets_STD": float("nan"),
    }
    bottom = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": condition_label,
        "Total": overall,
        "MMLU": src_map["mmlu"],
        "Jama": src_map["jama"],
        "MedXpert": src_map["medxpert"],
        "Medbullets": src_map["medbullets"],
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(src_map["mmlu"]),
        "JAMA_STD": _bern_std_pct(src_map["jama"]),
        "MedXpert_STD": _bern_std_pct(src_map["medxpert"]),
        "MedBullets_STD": _bern_std_pct(src_map["medbullets"]),
    }
    return top, bottom


def _compute_llama70b_sr_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_LLAMA70B_SR_RAW,
        condition_label=_LLAMA70B_SR,
        source_col="data_source_corr",
        answer_col="answer_corr",
    )


def _compute_qwen14b_sr_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_QWEN14B_SR_RAW,
        condition_label=_QWEN14B_SR,
        source_col="data_source_corr",
        answer_col="answer_corr",
    )


def _compute_qwen72b_sr_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_QWEN72B_SR_RAW,
        condition_label=_QWEN72B_SR,
        source_col="data_source_corr",
        answer_col="answer_corr",
    )


def _compute_llama70b_cc_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_LLAMA70B_CC_REMOVED_RAW,
        condition_label=_LLAMA70B_CC_REMOVED,
        source_col="data_source_corr",
        answer_col="Extracted_Answer",
    )


def _compute_llama70b_sr_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_LLAMA70B_SR_REMOVED_RAW,
        condition_label=_LLAMA70B_SR_REMOVED,
        source_col="data_source_corr",
        answer_col="Extracted_Answer",
    )


def _compute_qwen14b_cc_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_QWEN14B_CC_REMOVED_RAW,
        condition_label=_QWEN14B_CC_REMOVED,
        source_col="data_source_corr",
        answer_col="Extracted_Answer",
    )


def _compute_qwen14b_sr_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_QWEN14B_SR_REMOVED_RAW,
        condition_label=_QWEN14B_SR_REMOVED,
        source_col="data_source_corr",
        answer_col="Extracted_Answer",
    )


def _compute_qwen72b_cc_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_QWEN72B_CC_REMOVED_RAW,
        condition_label=_QWEN72B_CC_REMOVED,
        source_col="data_source_corr",
        answer_col="Extracted_Answer",
    )


def _compute_qwen72b_sr_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_QWEN72B_SR_REMOVED_RAW,
        condition_label=_QWEN72B_SR_REMOVED,
        source_col="data_source_corr",
        answer_col="Extracted_Answer",
    )


def _compute_gpt4o_sr_qwen72b_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_GPT4O_SR_QWEN72B_REMOVED_RAW,
        condition_label=_GPT4O_SR_QWEN72B_REMOVED,
        source_col="data_source_corr",
        answer_col="answer_corr",
    )


def _compute_gpt5_sr_qwen72b_removed_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_GPT5_SR_QWEN72B_REMOVED_RAW,
        condition_label=_GPT5_SR_QWEN72B_REMOVED,
        source_col="data_source_df3",
        answer_col="answer_corr",
    )


def _compute_medgemma27b_sr_qwen72b_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_model_rows(
        raw_csv=_MEDGEMMA27B_SR_QWEN72B_RAW,
        condition_label=_MEDGEMMA27B_SR_QWEN72B,
        source_col="data_source",
        answer_col="Extracted_Answer",
    )


def _compute_qwen14b_sr_qwen72b_rows() -> tuple[dict[str, float], dict[str, float]]:
    """Qwen-14B on Qwen-72B uses data_source_corr and answer_corr."""
    return _compute_model_rows(
        raw_csv=_QWEN14B_SR_QWEN72B_RAW,
        condition_label=_QWEN14B_SR_QWEN72B,
        source_col="data_source_corr",
        answer_col="answer_corr",
    )


def _compute_llama70b_sr_qwen72b_rows() -> tuple[dict[str, float], dict[str, float]]:
    """Llama-70B on Qwen-72B needs source mapping from PT_SOURCE_MAP."""
    if not _LLAMA70B_SR_QWEN72B_RAW.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {_LLAMA70B_SR_QWEN72B_RAW}")

    raw = pd.read_csv(_LLAMA70B_SR_QWEN72B_RAW)
    src_map_df = pd.read_csv(_PT_SOURCE_MAP)

    work = raw.copy()
    work["Origin"] = work["Origin"].astype(str).str.strip()
    work["Extracted_Answer"] = work["Extracted_Answer"].astype(str).str.strip()

    # Map sources from PT_SOURCE_MAP
    src_map = src_map_df[["Origin", "data_source_corr_x"]].copy()
    src_map["source"] = src_map["data_source_corr_x"].astype(str).str.strip().str.lower()
    src_map = src_map[["Origin", "source"]].drop_duplicates(subset=["Origin"])

    work = work.merge(src_map, on="Origin", how="left")

    rows = []
    for origin, g in work.groupby("Origin", sort=False):
        correct = (g["Extracted_Answer"].notna() & (g["Extracted_Answer"] != "")).any()
        source = g["source"].iloc[0] if len(g) > 0 else ""
        rows.append({
            "Origin": origin,
            "correct": 1.0 if correct else 0.0,
            "source": source,
        })

    per_origin = pd.DataFrame(rows)
    per_origin["correct"] = per_origin["correct"].astype(float)

    overall = 100.0 * float(per_origin["correct"].mean())
    by_src = 100.0 * per_origin.groupby("source")["correct"].mean()

    src_vals = {
        "mmlu": float(by_src.get("mmlu", np.nan)),
        "jama": float(by_src.get("jama", np.nan)),
        "medxpert": float(by_src.get("medxpert", np.nan)),
        "medbullets": float(by_src.get("medbullets", np.nan)),
    }

    top = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": _LLAMA70B_SR_QWEN72B,
        "Total": overall,
        "MMLU": overall,
        "Jama": float("nan"),
        "MedXpert": float("nan"),
        "Medbullets": float("nan"),
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(overall),
        "JAMA_STD": float("nan"),
        "MedXpert_STD": float("nan"),
        "MedBullets_STD": float("nan"),
    }
    bottom = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": _LLAMA70B_SR_QWEN72B,
        "Total": overall,
        "MMLU": src_vals["mmlu"],
        "Jama": src_vals["jama"],
        "MedXpert": src_vals["medxpert"],
        "Medbullets": src_vals["medbullets"],
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(src_vals["mmlu"]),
        "JAMA_STD": _bern_std_pct(src_vals["jama"]),
        "MedXpert_STD": _bern_std_pct(src_vals["medxpert"]),
        "MedBullets_STD": _bern_std_pct(src_vals["medbullets"]),
    }
    return top, bottom


def _upsert_row(df: pd.DataFrame, row: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c not in row:
            row[c] = float("nan")
    for c in row:
        if c not in out.columns:
            out[c] = float("nan")
    cond = str(row.get("Low+Irr Labelers", "")).strip()
    mask = (
        out["Base Model"].astype(str).str.strip().eq(_PT_MODEL)
        & out["Low+Irr Labelers"].astype(str).str.strip().eq(cond)
    )
    out = out[~mask]
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out


def _two_prop_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    p1 = k1 / n1
    p2 = k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return float(2.0 * norm.sf(abs(z)))


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.05:
        return "*"
    return ""


def _compute_pt_best_vs_original_sig_map() -> dict[str, str]:
    r1 = pd.read_csv(_R1_Q1_BY_SRC)
    r2 = pd.read_csv(_R2_MJ_BY_SRC)
    g5 = pd.read_csv(_PT_SOURCE_MAP)
    q72 = pd.read_csv(_QWEN72_RAW)
    src_map = pd.read_csv(_PT_SOURCE_MAP)[["Origin", "data_source_corr_x"]].copy()
    src_map["source"] = src_map["data_source_corr_x"].astype(str).str.strip().str.lower()
    src_map = src_map[["Origin", "source"]].drop_duplicates(subset=["Origin"])

    def _agg(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
        out = df[[src_col, "n_origins", "n_correct"]].copy()
        out.columns = ["source", "n", "k"]
        out["source"] = out["source"].astype(str).str.strip().str.lower()
        return out

    orig = _agg(r1, "data_source_corr")
    tr = _agg(r2, "data_source_corr_x")

    g5w = g5.copy()
    g5w["source"] = g5w["data_source_corr_x"].astype(str).str.strip().str.lower()
    g5w["correct"] = g5w["mv_correct"].astype(str).str.lower().map({"true": 1, "false": 0})
    g5agg = g5w.groupby("source", as_index=False).agg(n=("correct", "size"), k=("correct", "sum"))

    q72w = q72.copy()
    q72w["Origin"] = q72w["Origin"].astype(str).str.replace(r"-phase\d+$", "", regex=True)
    q72w["choice"] = q72w["q1"].map(_norm_choice)
    q72w["correct_answer"] = q72w["Correct answer"].map(_norm_choice)
    rows: list[dict[str, object]] = []
    for origin, g in q72w.groupby("Origin", sort=False):
        votes = [x for x in g["choice"].tolist() if x]
        c = collections.Counter(votes)
        if not c:
            tie = True
            mv = ""
        else:
            best = max(c.values())
            winners = sorted([k for k, v in c.items() if v == best])
            tie = len(winners) != 1
            mv = winners[0] if not tie else ""
        ca = g["correct_answer"].dropna()
        ca_val = ca.iloc[0] if len(ca) else ""
        rows.append(
            {
                "Origin": origin,
                "correct": int((not tie) and bool(mv) and (mv == ca_val)),
            }
        )
    q72agg = (
        pd.DataFrame(rows)
        .merge(src_map, on="Origin", how="left")
        .groupby("source", as_index=False)
        .agg(n=("correct", "size"), k=("correct", "sum"))
    )

    scopes = ["overall", "mmlu", "jama", "medxpert", "medbullets"]
    pvals: dict[str, float] = {}

    for scope in scopes:
        if scope == "overall":
            o_n, o_k = int(orig["n"].sum()), int(orig["k"].sum())
            cands = {
                "Trainee Removed": (int(tr["n"].sum()), int(tr["k"].sum())),
                "Qwen-72B Removed": (int(q72agg["n"].sum()), int(q72agg["k"].sum())),
                "GPT5 Removed": (int(g5agg["n"].sum()), int(g5agg["k"].sum())),
            }
        else:
            o = orig[orig["source"] == scope].iloc[0]
            o_n, o_k = int(o["n"]), int(o["k"])
            t = tr[tr["source"] == scope].iloc[0]
            q = q72agg[q72agg["source"] == scope].iloc[0]
            g = g5agg[g5agg["source"] == scope].iloc[0]
            cands = {
                "Trainee Removed": (int(t["n"]), int(t["k"])),
                "Qwen-72B Removed": (int(q["n"]), int(q["k"])),
                "GPT5 Removed": (int(g["n"]), int(g["k"])),
            }

        best_label = max(cands, key=lambda k: 100.0 * cands[k][1] / cands[k][0])
        b_n, b_k = cands[best_label]
        pvals[scope] = _two_prop_pvalue(o_k, o_n, b_k, b_n)

    m = len(scopes)
    return {scope: _sig_stars(min(1.0, p * m)) for scope, p in pvals.items()}


def _scope_from_title(title: str) -> str | None:
    t = title.strip().lower()
    if t.startswith("expert qa"):
        return "overall"
    if t == "mmlu":
        return "mmlu"
    if t == "jama":
        return "jama"
    if "medxpert" in t:
        return "medxpert"
    if "medbullets" in t:
        return "medbullets"
    return None


def _find_pt_x(ax) -> float | None:
    labels = [t.get_text().strip().lower() for t in ax.get_xticklabels()]
    x = list(ax.get_xticks())
    for i, lab in enumerate(labels):
        if "physician" in lab:
            return float(x[i])
    return None


def _is_greenish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return g > 0.45 and g >= r + 0.05 and g >= b + 0.05


def _is_blueish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return b > 0.6 and b >= g and b >= r


def _is_grayish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return abs(r - g) < 0.08 and abs(g - b) < 0.08 and 0.35 <= r <= 0.65


def _is_blackish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return max(r, g, b) <= 0.18


def _extract_orig_best_by_x(ax) -> dict[float, dict[str, float]]:
    xticks = [float(x) for x in ax.get_xticks()]
    out: dict[float, dict[str, float]] = {x: {} for x in xticks}

    for ln in ax.lines:
        xdata = np.asarray(ln.get_xdata(), dtype=float)
        ydata = np.asarray(ln.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2:
            continue
        if not np.isfinite(xdata).all() or not np.isfinite(ydata).all():
            continue
        # only horizontal reference segments for original / best
        if abs(float(ydata[0] - ydata[1])) > 1e-8:
            continue
        if abs(float(xdata[1] - xdata[0])) < 0.5:
            continue
        mid = float((xdata[0] + xdata[1]) / 2.0)
        nearest = min(xticks, key=lambda t: abs(t - mid))
        if abs(nearest - mid) > 0.8:
            continue

        ls = ln.get_linestyle()
        color = ln.get_color()
        if ls == ":" and _is_grayish(color):
            out[nearest]["best"] = float(ydata[0])
        elif ls == "-" and _is_blackish(color):
            out[nearest]["orig"] = float(ydata[0])

    return out


def _infer_k_n_from_acc(acc_pct: float) -> tuple[int, int]:
    p = float(acc_pct) / 100.0
    best_err = float("inf")
    best_n = int(_N_CANDIDATES[0])
    best_k = int(round(p * best_n))
    for n in _N_CANDIDATES:
        k = int(round(p * int(n)))
        err = abs((k / float(n)) - p)
        if err < best_err - 1e-12:
            best_err = err
            best_n = int(n)
            best_k = k
    return best_k, best_n


def _rewrite_abs_delta_labels_and_get_points(ax) -> list[dict[str, float]]:
    levels = _extract_orig_best_by_x(ax)
    infos: list[dict[str, float]] = []
    y_min, y_max = ax.get_ylim()

    for x in [float(v) for v in ax.get_xticks()]:
        pair = levels.get(x, {})
        if "orig" not in pair or "best" not in pair:
            continue
        orig = float(pair["orig"])
        best = float(pair["best"])
        delta = best - orig

        candidates = []
        for txt in ax.texts:
            s = txt.get_text().strip()
            if "%" not in s:
                continue
            tx, ty = txt.get_position()
            if abs(float(tx) - x) <= 0.9:
                candidates.append((float(ty), txt))

        if candidates:
            _, t = max(candidates, key=lambda z: z[0])
            t.set_text(f"{delta:+.1f}%")
            if delta >= 0:
                t.set_color("#00a000")
                t.set_fontweight("normal")
            else:
                t.set_color("#d62728")
                t.set_fontweight("bold")
            label_y = float(t.get_position()[1])
        else:
            label_y = min(y_max - 1.0, best + max((y_max - y_min) * 0.06, 5.0))
            ax.text(
                x,
                label_y,
                f"{delta:+.1f}%",
                fontsize=28,
                fontweight=("normal" if delta >= 0 else "bold"),
                color=("#00a000" if delta >= 0 else "#d62728"),
                ha="center",
                va="bottom",
                zorder=19,
            )

        infos.append({"x": x, "orig": orig, "best": best, "delta": delta, "label_y": label_y})

    return infos


def _annotate_all_x_sig_on_current_figure(legacy) -> None:
    fig = legacy.plt.gcf()
    for ax in fig.axes:
        scope = _scope_from_title(ax.get_title())
        if scope not in {"mmlu", "jama", "medxpert", "medbullets"}:
            continue
        infos = _rewrite_abs_delta_labels_and_get_points(ax)
        y_min, y_max = ax.get_ylim()
        y_pad = max((y_max - y_min) * 0.085, 6.0)

        for info in infos:
            # stars are requested above green labels (positive improvements only)
            if info["delta"] <= 0:
                continue
            k1, n1 = _infer_k_n_from_acc(info["orig"])
            k2, n2 = _infer_k_n_from_acc(info["best"])
            p = _two_prop_pvalue(k1, n1, k2, n2)
            sig = _sig_stars(p)
            if not sig:
                continue
            y = min(y_max - 1.0, float(info["label_y"]) + y_pad)
            ax.text(
                float(info["x"]),
                y,
                sig,
                fontsize=42,
                fontweight="bold",
                color="#202020",
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.18},
                zorder=21,
            )


def _pretty_x_label(s: str) -> str:
    t = s.strip().lower()
    if t == "gpt4o":
        return "GPT-4o"
    if t in {"qwen-14b", "qwen 14b"}:
        return "Qwen-14B"
    if t in {"qwen 72b", "qwen-72b"}:
        return "Qwen-72B"
    if t == "llama-70b":
        return "Llama-70B"
    if t == "medgemma-27b":
        return "MedGemma-27B"
    if t in {"gpt 5", "gpt5"}:
        return "GPT-5"
    if "physician" in t and "trainee" in t:
        return "Physician +\nTrainee"
    return s


def _add_cc_to_legend(legacy) -> None:
    """Rebuild legend to include CC and SR variants on the rightmost legend panel axis."""
    from matplotlib.lines import Line2D

    fig = legacy.plt.gcf()

    labeler_colors, labeler_markers = _get_labeler_styling()

    # First, remove ALL legends from ALL axes
    for ax in list(fig.axes):
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()

    # Delete ALL axes that don't have titles (legend-only axes)
    # We'll create a fresh clean legend axis below
    for ax in list(fig.axes):
        title = ax.get_title().strip()
        if not title:
            # No title - it's a legend-only axis, delete it
            fig.delaxes(ax)

    # Build all legend entries including CC and SR variants
    legend_order = [
        "Original Accuracy",
        "Best Performance",
        "Trainee Removed",
        "Qwen-72B Removed",
        "Qwen-72B Removed (CC)",
        "Qwen-72B Removed (SR)",
        "Qwen-14B Removed",
        "Qwen-14B Removed (CC)",
        "Qwen-14B Removed (SR)",
        "Llama-70b Removed",
        "Llama-70B Removed (CC)",
        "Llama-70B Removed (SR)",
        "GPT-4o Removed",
        "GPT-4o on Qwen-72B Removed (SR)",
        "GPT5 Removed",
        "GPT-5 on Qwen-72B Removed (SR)",
        "MedGemma-27B Removed",
        "MedGemma-27B on Qwen-72B (SR)",
    ]

    # Build final handles and labels
    final_handles = []
    final_labels = []

    for item in legend_order:
        color = labeler_colors.get(item, "#cccccc")
        marker = labeler_markers.get(item)

        if marker is None:
            # Line entry (Original Accuracy, Best Performance)
            line = Line2D(
                [0], [0],
                color=color,
                linestyle="-" if item == "Original Accuracy" else ":",
                linewidth=3,
                drawstyle='default'
            )
        else:
            # Marker entry - use Line2D with marker and color='none' to hide line
            line = Line2D(
                [0], [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=10,
                markeredgewidth=1.2,
                linestyle='none',
                drawstyle='default'
            )
        final_handles.append(line)
        final_labels.append(item)

    # Check if a clean legend axis already exists at x0=0.85
    legend_ax = None
    for ax in fig.axes:
        pos = ax.get_position()
        if pos.x0 >= 0.80:  # Legend panel should be around x0=0.85
            legend_ax = ax
            break

    # If no clean legend axis exists, create one
    if legend_ax is None:
        # Create a new axes positioned at the legend panel location [0.85, 0.10, 0.14, 0.85]
        legend_ax = fig.add_axes([0.85, 0.10, 0.14, 0.85])
        # Hide the axes frame, ticks, and labels so only the legend is visible
        legend_ax.set_facecolor('none')
        legend_ax.axis('off')
        print(f"DEBUG: Created new clean legend axis at [0.85, 0.10, 0.14, 0.85]", file=sys.stderr)

    # Create new legend with all entries
    new_legend = legend_ax.legend(
        final_handles, final_labels,
        title="Labeler Configuration",
        loc="center left",
        fontsize=23,
        title_fontsize=28,
        frameon=True,
        edgecolor="black",
        markerscale=1.0
    )

    pos = legend_ax.get_position()
    print(f"DEBUG: Created legend with {len(final_labels)} entries on axis at x0={pos.x0:.2f}, x1={pos.x1:.2f}", file=sys.stderr)
    print(f"DEBUG: Legend entries: {final_labels[:12]}", file=sys.stderr)


def _update_legend_with_cc_sr_variants(legacy) -> None:
    """Rebuild legend to include (CC) variants for the three removed models."""
    from matplotlib.lines import Line2D

    fig = legacy.plt.gcf()

    # Find the legend axis
    legend_ax = None
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            legend_ax = ax
            break

    if legend_ax is None:
        return

    legend = legend_ax.get_legend()
    if legend is None:
        return

    labeler_colors, labeler_markers = _get_labeler_styling()

    # Collect ALL handles and labels from ALL axes (to capture CC variants plotted on data axes)
    all_handles = []
    all_labels = []
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # Deduplicate while preserving order
    seen = set()
    handle_map = {}
    for handle, label in zip(all_handles, all_labels):
        if label not in seen:
            seen.add(label)
            handle_map[label] = handle

    # Define the desired order for legend entries
    legend_order = [
        "Original Accuracy",
        "Best Performance",
        "Trainee Removed",
        "Qwen-72B Removed",
        "Qwen-72B Removed (CC)",
        "Qwen-14B Removed",
        "Qwen-14B Removed (CC)",
        "Llama-70b Removed",
        "Llama-70B Removed (CC)",
        "GPT-4o Removed",
        "GPT5 Removed",
        "MedGemma-27B Removed",
    ]

    # Build final legend in order
    final_handles = []
    final_labels = []
    added = set()

    for item in legend_order:
        if item in handle_map:
            final_handles.append(handle_map[item])
            final_labels.append(item)
            added.add(item)

    # Add any remaining items not in legend_order
    for label in handle_map:
        if label not in added:
            final_handles.append(handle_map[label])
            final_labels.append(label)
            added.add(label)

    # Add CC variants that may not be in handle_map (create synthetic entries)
    cc_variants = {
        "Llama-70b Removed": "Llama-70B Removed (CC)",
        "Qwen-14B Removed": "Qwen-14B Removed (CC)",
        "Qwen-72B Removed": "Qwen-72B Removed (CC)",
    }

    for base_label, cc_label in cc_variants.items():
        if cc_label not in added:
            marker_style = labeler_markers.get(cc_label, "D")
            color = labeler_colors.get(cc_label, "#cccccc")
            cc_handle = Line2D(
                [0], [0],
                marker=marker_style,
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=2,
                label=cc_label,
            )
            # Insert after base label
            try:
                base_idx = final_labels.index(base_label)
                final_handles.insert(base_idx + 1, cc_handle)
                final_labels.insert(base_idx + 1, cc_label)
                added.add(cc_label)
            except ValueError:
                # Base label not found, just append
                pass

    # Get old legend location
    try:
        loc = legend._loc
    except:
        loc = "center left"

    # Remove old legend
    legend.remove()

    # Create new legend
    new_legend = legend_ax.legend(
        final_handles, final_labels,
        title="Labeler Configuration",
        loc=loc,
        fontsize=23,
        title_fontsize=28,
        frameon=True,
        edgecolor="black"
    )

    fig.canvas.draw_idle()


def _get_labeler_styling() -> tuple[dict[str, str], dict[str, str]]:
    """Return color and marker mappings for all labeler configurations."""
    labeler_colors = {
        "Original Accuracy": "#000000",
        "Trainee Removed": "#2196F3",
        "Qwen-14B Removed": "#f16c23",
        "Qwen-14B Removed (CC)": "#ff9c6e",
        "Qwen-14B Removed (SR)": "#ff7b54",
        "Qwen-72B Removed": "#ff9200",
        "Qwen-72B Removed (CC)": "#ffb84d",
        "Qwen-72B Removed (SR)": "#ffb84d",
        "Llama-70b Removed": "#f9AE78",
        "Llama-70B Removed (CC)": "#ffcba4",
        "Llama-70B Removed (SR)": "#ffcba4",
        "GPT-4o Removed": "#009d00",
        "GPT-4o on Qwen-72B Removed (SR)": "#66cc99",
        "GPT5 Removed": "#1b7c3d",
        "GPT-5 on Qwen-72B Removed (SR)": "#33aa66",
        "MedGemma-27B Removed": "#ECB66C",
        "MedGemma-27B on Qwen-72B (SR)": "#f5d99e",
        "Best Performance": "#808080",
    }

    labeler_markers = {
        "Original Accuracy": None,
        "Trainee Removed": "o",
        "Qwen-14B Removed": "s",
        "Qwen-14B Removed (CC)": "D",
        "Qwen-14B Removed (SR)": "D",
        "Qwen-72B Removed": "s",
        "Qwen-72B Removed (CC)": "D",
        "Qwen-72B Removed (SR)": "D",
        "Llama-70b Removed": "s",
        "Llama-70B Removed (CC)": "D",
        "Llama-70B Removed (SR)": "D",
        "GPT-4o Removed": "^",
        "GPT-4o on Qwen-72B Removed (SR)": "^",
        "GPT5 Removed": "^",
        "GPT-5 on Qwen-72B Removed (SR)": "^",
        "MedGemma-27B Removed": "s",
        "MedGemma-27B on Qwen-72B (SR)": "s",
        "Best Performance": None,
    }

    return labeler_colors, labeler_markers


def _plot_cc_variants_on_axes(legacy, df_physician: pd.DataFrame) -> None:
    """Add (CC) variant scatter points to existing axes with legend labels."""
    fig = legacy.plt.gcf()
    labeler_colors, labeler_markers = _get_labeler_styling()

    axes = list(fig.axes)
    cc_variants = ["Qwen-14B Removed (CC)", "Qwen-72B Removed (CC)", "Llama-70B Removed (CC)"]

    # Find legend axis to add handles to it
    legend_ax = None
    for ax in axes:
        if ax.get_legend() is not None:
            legend_ax = ax
            break

    plotted_cc = set()

    for ax in axes:
        title = ax.get_title().strip().lower()
        if not title or title == "labeler configuration":
            continue

        # Determine which column to plot based on title
        col_map = {
            "total": "Total",
            "mmlu": "MMLU",
            "jama": "Jama",
            "medxpert": "MedXpert",
            "medbullets": "Medbullets",
        }
        col = next((c for k, c in col_map.items() if k in title), None)
        if not col or col not in df_physician.columns:
            continue

        # Get x-axis positions
        xticks = ax.get_xticks()
        xticklabels = [t.get_text().strip() for t in ax.get_xticklabels()]
        if len(xticks) != len(xticklabels):
            continue

        # Plot each CC variant
        for cc_label in cc_variants:
            rows = df_physician[df_physician["Low+Irr Labelers"].astype(str).str.strip() == cc_label]
            if len(rows) == 0:
                continue

            row = rows.iloc[0]
            value = row[col]
            if pd.isna(value):
                continue

            # Match CC variant to its base model
            base_label_map = {
                "Qwen-14B Removed (CC)": "Qwen-14B",
                "Qwen-72B Removed (CC)": "Qwen 72B",
                "Llama-70B Removed (CC)": "Llama-70B",
            }
            base_model = base_label_map.get(cc_label)

            # Find x position
            x_pos = None
            for i, label in enumerate(xticklabels):
                if base_model and base_model.lower() in label.lower():
                    x_pos = xticks[i]
                    break

            if x_pos is None:
                continue

            # Plot the CC variant with label
            color = labeler_colors.get(cc_label, "#cccccc")
            marker = labeler_markers.get(cc_label, "D")
            ax.scatter(
                x_pos,
                value,
                c=color,
                s=200,
                alpha=0.9,
                marker=marker,
                edgecolors="white",
                linewidth=2,
                zorder=10,
                label=cc_label,
            )

            # Track that we plotted this CC variant
            if cc_label not in plotted_cc:
                plotted_cc.add(cc_label)


def _restyle_combined_current_figure(legacy) -> None:
    fig = legacy.plt.gcf()
    axes = list(fig.axes)
    labeler_colors, labeler_markers = _get_labeler_styling()

    def _title_key(ax) -> str:
        return ax.get_title().strip().lower()

    # Only apply when this is the combined multi-panel figure (has the 4 category panels).
    keys = [_title_key(ax) for ax in axes if _title_key(ax)]
    if not all(any(req in k for k in keys) for req in ["mmlu", "jama", "medxpert", "medbullets"]):
        return

    # Remove all top-row panels by geometry (robust to title text changes).
    for ax in list(fig.axes):
        if ax.get_legend() is not None:
            continue
        pos = ax.get_position()
        if pos.y0 > 0.66:
            fig.delaxes(ax)

    # Balanced canvas for readability and whitespace.
    fig.set_size_inches(40, 26, forward=True)

    title_to_ax = {}
    for ax in fig.axes:
        k = _title_key(ax)
        if not k:
            continue
        if "mmlu" in k:
            title_to_ax["mmlu"] = ax
        elif "jama" in k:
            title_to_ax["jama"] = ax
        elif "medxpert" in k:
            title_to_ax["medxpert"] = ax
        elif "medbullets" in k:
            title_to_ax["medbullets"] = ax

    # Re-layout remaining main panels into a 2x2 grid.
    # Adjusted to leave room for larger legend on the right
    panel_pos = {
        "mmlu": [0.05, 0.60, 0.36, 0.30],
        "jama": [0.42, 0.60, 0.36, 0.30],
        "medxpert": [0.05, 0.15, 0.36, 0.30],
        "medbullets": [0.42, 0.15, 0.36, 0.30],
    }
    for key, pos in panel_pos.items():
        ax = title_to_ax.get(key)
        if ax is None:
            continue
        ax.set_position(pos)
        ax.set_title(ax.get_title(), fontsize=42, pad=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=30, labelpad=8)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=32, labelpad=8)
        ax.tick_params(axis="both", labelsize=24)
        ax.xaxis.label.set_size(30)
        ax.yaxis.label.set_size(32)

        pretty = [_pretty_x_label(lbl.get_text()) for lbl in ax.get_xticklabels()]
        ax.set_xticklabels(pretty, rotation=40, ha="right")
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(22)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(22)

        # Reduce dominance of baseline guides and keep focus on markers/labels.
        for ln in ax.lines:
            ls = ln.get_linestyle()
            if ls == "-" and _is_blackish(ln.get_color()):
                ln.set_linewidth(3.8)
            elif ls == ":" and _is_grayish(ln.get_color()):
                ln.set_linewidth(2.6)

        for txt in ax.texts:
            s = txt.get_text().strip().lower()
            if "dataset release date" in s:
                txt.set_fontsize(24)
                txt.set_fontweight("bold")
                txt.set_transform(ax.transAxes)
                txt.set_position((0.5, 0.90))
                txt.set_ha("center")
                txt.set_va("bottom")
                continue
            if "%" in s and _is_greenish(txt.get_color()):
                txt.set_fontweight("normal")
                txt.set_fontsize(30)
            elif "%" in s and "#d62728" in str(txt.get_color()).lower():
                txt.set_fontsize(30)

        # Keep Trainee (blue) markers above other markers.
        for coll in ax.collections:
            try:
                fcs = coll.get_facecolors()
            except Exception:
                continue
            if len(fcs) == 0:
                continue
            if _is_blueish(fcs[0]):
                coll.set_zorder(50)
            else:
                coll.set_zorder(max(5, coll.get_zorder()))
            # Uniform marker sizing for cleaner look.
            try:
                coll.set_sizes(np.array([180.0]))
            except Exception:
                pass

    # Remove all legends from data axes (they'll be replaced by a clean legend panel)
    for ax in list(fig.axes):
        title = ax.get_title().strip().lower()
        lg = ax.get_legend()
        if lg is None:
            continue
        # If this is a data axis (has a title), remove its legend
        if title and any(k in title for k in ['mmlu', 'jama', 'medxpert', 'medbullets']):
            lg.remove()
        else:
            # For legend-only axes, reposition to the right
            # Increased height to 0.85 to accommodate 18 legend entries
            ax.set_position([0.85, 0.10, 0.14, 0.85])
            lg.set_title(lg.get_title().get_text(), prop={"size": 28})
            for text in lg.get_texts():
                text.set_fontsize(23)


def _create_total_figure(legacy, df_physician: pd.DataFrame) -> None:
    """Create Total figure with per_dataset-style layout."""
    import matplotlib.colors as mcolors

    total_data = df_physician[df_physician["Total"].notna()].copy()
    if len(total_data) == 0:
        fig, ax = legacy.plt.subplots(1, 1, figsize=(16, 10))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return fig

    legacy.plt.rcParams["font.family"] = "Nimbus Sans"
    legacy.plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    legacy.plt.rcParams["axes.facecolor"] = "white"
    legacy.plt.rcParams["figure.facecolor"] = "white"

    labeler_colors = {
        "Original Accuracy": "#000000",
        "Trainee Removed": "#2196F3",
        "Trainee Removed (248 QAs)": "#2196F3",
        "Qwen-14B Removed": "#f16c23",
        "Qwen-72B Removed": "#ff9200",
        "Qwen-14B Removed (CC)": "#f16c23",
        "Qwen-14B Removed (SR)": "#ff7b54",
        "Qwen-72B Removed (CC)": "#ff9200",
        "Qwen-72B Removed (SR)": "#ffb84d",
        "Llama-70b Removed": "#f9AE78",
        "Llama-70B Removed (CC)": "#f9AE78",
        "Llama-70B Removed (SR)": "#ffcba4",
        "GPT-4o Removed": "#009d00",
        "GPT5 Removed": "#1b7c3d",
        "GPT-4o on Qwen-72B Removed (SR)": "#66cc99",
        "GPT-5 on Qwen-72B Removed (SR)": "#33aa66",
        "Llama-70B (SR)": "#ffa366",
        "Llama-70B on Qwen-72B (SR)": "#ffb380",
        "Qwen-14B (SR)": "#ffaa33",
        "Qwen-14B on Qwen-72B (SR)": "#ffbb66",
        "Qwen-72B (SR)": "#ff8800",
        "MedGemma-27B": "#ECB66C",
        "MedGemma-27B Removed": "#ECB66C",
        "MedGemma-27B on Qwen-72B (SR)": "#f5d99e",
    }

    labeler_markers = {
        "Original Accuracy": None,
        "Trainee Removed": "o",
        "Trainee Removed (248 QAs)": "o",
        "Qwen-14B Removed": "s",
        "Qwen-14B Removed (CC)": "s",
        "Qwen-14B Removed (SR)": "s",
        "Qwen-72B Removed": "s",
        "Qwen-72B Removed (CC)": "s",
        "Qwen-72B Removed (SR)": "s",
        "Llama-70b Removed": "s",
        "Llama-70B Removed (CC)": "s",
        "Llama-70B Removed (SR)": "s",
        "MedGemma-27B Removed": "s",
        "GPT-4o Removed": "^",
        "GPT5 Removed": "^",
        "GPT-4o on Qwen-72B Removed (SR)": "^",
        "GPT-5 on Qwen-72B Removed (SR)": "^",
        "Llama-70B (SR)": "D",
        "Llama-70B on Qwen-72B (SR)": "D",
        "Qwen-14B (SR)": "D",
        "Qwen-14B on Qwen-72B (SR)": "D",
        "Qwen-72B (SR)": "D",
        "MedGemma-27B on Qwen-72B (SR)": "s",
    }

    # Figure setup matching per_dataset layout
    fig, ax = legacy.plt.subplots(1, 1, figsize=(43, 15))

    _x_cat_spacing = 2.25
    _x_margin = 0.9

    # Get base models from "Base Model" column
    base_models = total_data["Base Model"].unique()
    if len(base_models) == 0:
        base_models = ["Physician + Trainee"]
    base_models = [str(m).strip() for m in base_models]
    n_models = len(base_models)

    def _x(i: float) -> float:
        return float(i) * _x_cat_spacing

    x_positions = np.arange(n_models) * _x_cat_spacing
    added_labels: set[str] = set()
    best_perf_label_added = False

    for i, model in enumerate(base_models):
        model_data = total_data[total_data["Base Model"].astype(str).str.strip() == model]
        if len(model_data) == 0:
            continue

        original_acc_row = model_data[
            model_data["Low+Irr Labelers"].astype(str).str.strip() == "Original Accuracy"
        ]
        original_acc_value = (
            original_acc_row["Total"].values[0] if len(original_acc_row) > 0 else None
        )

        other_labelers = model_data[
            model_data["Low+Irr Labelers"].astype(str).str.strip() != "Original Accuracy"
        ]
        if len(other_labelers) > 0:
            best_row = other_labelers.loc[other_labelers["Total"].idxmax()]
            best_value = best_row["Total"]
        else:
            best_value = None

        for _, row in model_data.iterrows():
            labeler = str(row["Low+Irr Labelers"]).strip()
            value = row["Total"]
            color = labeler_colors.get(labeler, "#7f7f7f")
            label = labeler if labeler not in added_labels else ""
            if label:
                added_labels.add(labeler)

            if labeler == "Original Accuracy":
                hw = 0.35 * _x_cat_spacing
                ax.plot(
                    [_x(i) - hw, _x(i) + hw],
                    [value, value],
                    color=color,
                    linewidth=6,
                    solid_capstyle="butt",
                    zorder=3,
                    label=label,
                )
            else:
                marker = labeler_markers.get(labeler, "o")
                ax.scatter(
                    _x(i),
                    value,
                    c=color,
                    s=1200,
                    alpha=0.9,
                    marker=marker,
                    edgecolors="white",
                    linewidth=3,
                    zorder=3,
                    label=label,
                )

        if best_value is not None and original_acc_value is not None:
            line_label = "Best Performance" if not best_perf_label_added else ""
            if line_label:
                best_perf_label_added = True

            hw = 0.35 * _x_cat_spacing
            ax.plot(
                [_x(i) - hw, _x(i) + hw],
                [best_value, best_value],
                color="#808080",
                linewidth=5,
                linestyle=":",
                solid_capstyle="butt",
                zorder=4,
                label=line_label,
                alpha=0.8,
            )

            pct_improvement = (
                (best_value - original_acc_value) / original_acc_value
            ) * 100
            text_color = "#00a000" if pct_improvement >= 0 else "#d32f2f"
            if pct_improvement >= 0:
                pct_text = f"+{pct_improvement:.1f}%"
            else:
                pct_text = f"{pct_improvement:.1f}%"

            arrow_x = _x(i) - 0.25 * _x_cat_spacing
            ax.annotate(
                "",
                xy=(arrow_x, best_value),
                xytext=(arrow_x, original_acc_value),
                arrowprops=dict(
                    arrowstyle="->",
                    color=("#00a000" if pct_improvement >= 0 else "#d32f2f"),
                    lw=4,
                    mutation_scale=25,
                    alpha=0.7,
                ),
                zorder=5,
            )

            text_y = max(best_value, original_acc_value) + 5
            ax.text(
                _x(i),
                text_y,
                pct_text,
                fontsize=45,
                fontweight="bold",
                color=text_color,
                ha="center",
                va="bottom",
                zorder=6,
            )

    # Dataset Release Date line and label (left margin)
    x_vline_left = -0.5 * _x_cat_spacing
    ax.axvline(
        x=x_vline_left,
        color="#808080",
        linewidth=3,
        linestyle=":",
        alpha=0.6,
        zorder=2,
    )
    ax.text(
        x_vline_left,
        115,
        "Dataset Release Date",
        fontsize=50,
        fontweight="600",
        color="#606060",
        ha="center",
        va="bottom",
        rotation=0,
        zorder=6,
    )

    if n_models <= 1:
        ax.set_xlim(-_x_margin, _x_margin)
    else:
        ax.set_xlim(
            -_x_margin,
            (n_models - 1) * _x_cat_spacing + _x_margin,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        base_models, rotation=45, ha="right", fontsize=44, fontweight="500"
    )

    ax.set_ylabel("Accuracy (%)", fontsize=48, fontweight="bold", labelpad=15)
    ax.set_title("Total", fontsize=72, fontweight="bold", pad=65)
    ax.grid(axis="y", alpha=0.4, linewidth=2, color="gray", linestyle="--", zorder=1)
    ax.grid(axis="x", alpha=0, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=52, width=2, length=8, color="black")
    ax.tick_params(axis="x", labelsize=52, width=2, length=8, color="black")
    ax.set_ylim(0, 120)

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title="Labeler Configuration",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=32,
            title_fontsize=38,
            frameon=True,
            edgecolor='black',
            fancybox=False,
        )

    legacy.plt.tight_layout()
    return fig


def _enhance_font_sizes(fig) -> None:
    """Increase all font sizes and marker sizes for better readability."""
    # Adjust panel heights and positions for better y-axis visibility and reduced row gap
    panel_adjustments = {
        "mmlu": [0.07, 0.62, 0.38, 0.32],      # increased height from 0.27 to 0.32
        "jama": [0.49, 0.62, 0.38, 0.32],      # increased height
        "medxpert": [0.07, 0.08, 0.38, 0.32],  # moved up slightly (0.11 to 0.08), increased height
        "medbullets": [0.49, 0.08, 0.38, 0.32], # moved up slightly, increased height
    }

    # Get all axes and identify them by title
    title_to_ax = {}
    for ax in fig.axes:
        k = ax.get_title().strip().lower()
        if "mmlu" in k:
            title_to_ax["mmlu"] = ax
        elif "jama" in k:
            title_to_ax["jama"] = ax
        elif "medxpert" in k:
            title_to_ax["medxpert"] = ax
        elif "medbullets" in k:
            title_to_ax["medbullets"] = ax

    # Apply new panel positions
    for key, pos in panel_adjustments.items():
        ax = title_to_ax.get(key)
        if ax is not None:
            ax.set_position(pos)

    for ax in fig.axes:
        # Titles and labels
        title = ax.get_title()
        if title:
            ax.set_title(title, fontsize=72, fontweight="semibold")

        xlabel = ax.get_xlabel()
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=64, labelpad=14)

        ylabel = ax.get_ylabel()
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=72, fontweight="semibold", labelpad=20)

        # Tick labels
        ax.tick_params(axis="x", labelsize=52)
        ax.tick_params(axis="y", labelsize=56)

        # Text annotations (improvements, dates, stars, etc.)
        for txt in ax.texts:
            s = txt.get_text().strip().lower()
            current_size = txt.get_fontsize()

            if "dataset release date" in s:
                txt.set_fontsize(40)
                txt.set_fontweight("bold")
            elif "%" in s:
                # Improvement percentages
                txt.set_fontsize(60)
                txt.set_fontweight("normal")
            elif re.fullmatch(r"\d{2}/\d{2}", s):
                # Date labels like 05/24
                txt.set_fontsize(36)
            elif re.fullmatch(r"\*+", s):
                # Significance stars (*, **, ***)
                txt.set_fontsize(80)

        # Increase marker sizes for visibility
        for coll in ax.collections:
            try:
                coll.set_sizes(np.array([2000.0]))  # Significantly larger markers
                coll.set_linewidths(np.array([2.4]))  # Thicker marker outlines
            except Exception:
                pass

    # Legend
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            lg.set_title(lg.get_title().get_text(), prop={"size": 28, "weight": "bold"})
            for text in lg.get_texts():
                text.set_fontsize(26)

    # Figure size
    fig.set_size_inches(60, 40, forward=True)


def main() -> None:
    legacy = _load_legacy_module()
    original_augment = legacy.augment_physician_trainee
    original_savefig = legacy.plt.savefig

    # Store the augmented dataframes
    stored_df_physician = None

    def _augment_with_gpt5(df_physician: pd.DataFrame, df_dataset: pd.DataFrame):
        nonlocal stored_df_physician
        out_p, out_d = original_augment(df_physician, df_dataset)
        for label, fn in [
            (_QWEN72_REMOVED, _compute_pt_qwen72_rows),
            (_GPT5_REMOVED, _compute_pt_gpt5_rows),
            (_LLAMA70B_SR, _compute_llama70b_sr_rows),
            (_LLAMA70B_CC_REMOVED, _compute_llama70b_cc_removed_rows),
            (_LLAMA70B_SR_REMOVED, _compute_llama70b_sr_removed_rows),
            (_QWEN14B_SR, _compute_qwen14b_sr_rows),
            (_QWEN14B_CC_REMOVED, _compute_qwen14b_cc_removed_rows),
            (_QWEN14B_SR_REMOVED, _compute_qwen14b_sr_removed_rows),
            (_QWEN72B_SR, _compute_qwen72b_sr_rows),
            (_QWEN72B_CC_REMOVED, _compute_qwen72b_cc_removed_rows),
            (_QWEN72B_SR_REMOVED, _compute_qwen72b_sr_removed_rows),
            (_GPT4O_SR_QWEN72B_REMOVED, _compute_gpt4o_sr_qwen72b_removed_rows),
            (_GPT5_SR_QWEN72B_REMOVED, _compute_gpt5_sr_qwen72b_removed_rows),
            (_MEDGEMMA27B_SR_QWEN72B, _compute_medgemma27b_sr_qwen72b_rows),
            (_QWEN14B_SR_QWEN72B, _compute_qwen14b_sr_qwen72b_rows),
            (_LLAMA70B_SR_QWEN72B, _compute_llama70b_sr_qwen72b_rows),
        ]:
            try:
                top_row, bottom_row = fn()
                out_p = _upsert_row(out_p, top_row)
                out_d = _upsert_row(out_d, bottom_row)
                print(
                    f"Appended {_PT_MODEL} / {label}: "
                    f"top MMLU={top_row['MMLU']:.2f}%, "
                    f"dataset (MMLU/Jama/MedXpert/Medbullets)="
                    f"{bottom_row['MMLU']:.2f}/{bottom_row['Jama']:.2f}/"
                    f"{bottom_row['MedXpert']:.2f}/{bottom_row['Medbullets']:.2f}%",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"Warning: could not append {_PT_MODEL} / {label}: {e}",
                    file=sys.stderr,
                )
        stored_df_physician = out_p
        return out_p, out_d

    def _savefig_with_sig(*args, **kwargs):
        if stored_df_physician is not None:
            _plot_cc_variants_on_axes(legacy, stored_df_physician)
        _restyle_combined_current_figure(legacy)
        _annotate_all_x_sig_on_current_figure(legacy)
        _add_cc_to_legend(legacy)  # Always add CC entries to legend
        fig = legacy.plt.gcf()

        _enhance_font_sizes(fig)

        # CRITICAL: Remove legends from data axes ONLY
        # Keep the legend on the clean legend panel axis (x0 >= 0.80)
        for ax in list(fig.axes):
            title = ax.get_title().strip().lower()
            pos = ax.get_position()

            # Only remove legends from data axes (those with titles)
            # Keep the legend on the legend panel (x0 >= 0.80)
            if title and any(k in title for k in ['mmlu', 'jama', 'medxpert', 'medbullets', 'total']):
                lg = ax.get_legend()
                if lg is not None:
                    lg.remove()
                    print(f"DEBUG: Removed legend from data axis '{title}'", file=sys.stderr)

        # Force figure draw to ensure all elements are rendered
        fig.canvas.draw()

        # Debug: Check legend size right before savefig
        for ax in fig.axes:
            lg = ax.get_legend()
            if lg is not None:
                texts = [t.get_text() for t in lg.get_texts()]
                title = ax.get_title()
                print(f"DEBUG: Legend on axis '{title}': {len(texts)} entries", file=sys.stderr)

        return original_savefig(*args, **kwargs)

    legacy.augment_physician_trainee = _augment_with_gpt5
    legacy.plt.savefig = _savefig_with_sig
    legacy.main()

    # Generate total figure (one chart with all datasets combined)
    try:
        if stored_df_physician is not None:
            fig_total = _create_total_figure(legacy, stored_df_physician)
            combined_total = _THIS_DIR / "MedPAIR_Result_combined_total.pdf"
            legacy.plt.savefig(str(combined_total), dpi=150, bbox_inches='tight')
            print(f"Wrote {combined_total}", flush=True)
            legacy.plt.close(fig_total)
    except Exception as e:
        print(f"Warning: could not create total figure: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
