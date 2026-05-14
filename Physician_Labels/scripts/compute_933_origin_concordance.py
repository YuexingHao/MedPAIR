#!/usr/bin/env python3
"""
Sentence-level concordance on the **933** clinician–student majority-vote cohort.

Join is **only on ``Origin``** (no ``ID_corr``): each model prediction row is matched to
``933_Clinician_Student_Majority_Vote.csv`` by ``Origin``.

Physician side uses continuous ``Sentence 1``…``Sentence 21`` scores; each is binarized with
``--phys-threshold`` (default 0.5): ≥ threshold → "high", else "low".

Model side uses ``q1``…``q20`` (and ``q21`` or ``label_21`` if present) or, alternatively,
``label_1``…``label_21`` text labels; values are mapped to the same "high" / "low" buckets.

**Precomputed match rate:** some CC exports (e.g. ``Physician_Labels/results/70B_MatchRate.csv``) have no
per-sentence ``q*`` columns. Relevancy was used upstream to build ``70B_sentence_ids`` vs
``human_sentence_ids``; the pipeline already stores the resulting overlap as ``Match_Rate`` (0–1).
For those files we set ``concordance_pct = Match_Rate * 100`` per ``Origin`` (not recomputed vs the
933 majority table). ``n_sentence_pairs`` is the count of IDs in ``human_sentence_ids`` when present.

**Qwen-14B_CC:** ``Physician_Labels/results/14B_MatchRate.csv`` — ``match_rate`` (0–100 or 0–1) per ``ID_corr``.

**Qwen-14B_SR:** ``[SR]Qwen14B_annotated_MedPAIR_relevancy.csv`` — ``Match?`` as ``hits/total`` → percent.

**Qwen-72B_CC:** ``Physician_Labels/results/72B_MatchRate.csv`` — ``match_rate`` (fraction 0–1 or percent 0–100), clamped to **0–100** for ``concordance_pct``.

**Qwen-72B_SR:** ``[SR]Qwen72B_annotated_MedPAIR_relevancy.csv`` — ``Match?`` as ``hits/total``.

**Llama-70B_CC:** ``70B_MatchRate.csv`` ``Match_Rate`` (0–1) or annotated CSV fallback.

**GPT4o:** ``GPT4o_MatchRate.csv`` — ``match_percentage`` per ``Origin`` (0–100 or 0–1, clamped).

**GPT5:** ``GPT5_MatchRate.csv`` — ``Match?`` as ``hits/total`` per ``ID_corr``.

**MedGemma / others:** per-sentence ``q*`` / ``label_*`` vs 933 majority, or precomputed columns as above.

Writes summary and long CSV under ``Figures/concordance_rate/`` by default (see ``--summary-out`` / ``--per-origin-out``).

Examples::

  python Physician_Labels/scripts/compute_933_origin_concordance.py
  python Physician_Labels/scripts/compute_933_origin_concordance.py --per-origin-out /tmp/per_origin.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
_PHYSICIAN_LABELS = _THIS.parent
_REPO_ROOT = _PHYSICIAN_LABELS.parent

_FIGURES_CONCORDANCE = _REPO_ROOT / "Figures" / "concordance_rate"
DEFAULT_MAJORITY_933 = _PHYSICIAN_LABELS / "Mar2_2026_Data" / "933_Clinician_Student_Majority_Vote.csv"
DEFAULT_SUMMARY_OUT = _FIGURES_CONCORDANCE / "concordance_933_by_origin_summary.csv"
DEFAULT_PER_ORIGIN_OUT = _FIGURES_CONCORDANCE / "concordance_933_by_origin_long.csv"


def _first_existing(*candidates: Path) -> Path:
    """Use the first path that exists on disk; else the first candidate (main() will skip if missing)."""
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0] if candidates else Path(".")


# Default inputs (dict order = concat / summary row order). See module docstring for metrics.
_SR = _PHYSICIAN_LABELS / "results"
_APT = _REPO_ROOT / "After_PT_Removal"
DEFAULT_MODEL_CSVS: dict[str, Path] = {
    "GPT4o": _SR / "GPT4o_MatchRate.csv",
    "Qwen-14B_CC": _SR / "14B_MatchRate.csv",
    "Qwen-14B_SR": _SR / "[SR]Qwen14B_annotated_MedPAIR_relevancy.csv",
    "Qwen-72B_CC": _SR / "72B_MatchRate.csv",
    "Qwen-72B_SR": _SR / "[SR]Qwen72B_annotated_MedPAIR_relevancy.csv",
    "Llama-70B_CC": _first_existing(
        _SR / "70B_MatchRate.csv",
        _APT / "Llama-70B/results/predictions/Llama70B_annotated_ORIGINAL_Accuracy.csv",
    ),
    "Llama-70B_SR": _SR / "[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv",
    "MedGemma-27B": _APT / "MedGemma-27b-text-it/data/raw/MedGemma_SR_Match_Rate.csv",
    "GPT5": _SR / "GPT5_MatchRate.csv",
}

SENTENCE_COLS_PHYS = [f"Sentence {i}" for i in range(1, 22)]


def _norm_origin(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def physician_bucket(val, threshold: float) -> float:
    """Return 'high' / 'low' / nan."""
    if pd.isna(val):
        return np.nan
    try:
        x = float(val)
    except (TypeError, ValueError):
        return np.nan
    return "high" if x >= threshold else "low"


def model_bucket(val) -> float:
    """Map model cell to 'high' / 'low' / nan."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    up = s.upper()
    if up == "REMOVED":
        return "low"
    if "HIGH" in up and "IRREL" not in up:
        return "high"
    if "LOW" in up or "IRREL" in up:
        return "low"
    try:
        v = float(s)
        return "high" if v >= 0.5 else "low"
    except ValueError:
        return np.nan


def _model_uses_precomputed_match_rate(columns: Iterable[str]) -> bool:
    """True if we should read concordance from Match_Rate instead of q*/label_* vs 933 majority."""
    cols = set(columns)
    return "Match_Rate" in cols and "Origin" in cols and "q1" not in cols


def _count_csv_int_ids(val) -> int:
    """Count comma-separated integers in human_sentence_ids-style cells."""
    if pd.isna(val):
        return 0
    parts = [p.strip() for p in str(val).split(",") if p.strip()]
    return len(parts)


def _fraction_or_percent_to_pct_clamped(val) -> float:
    """
    Map ``match_rate``-style values to concordance 0–100.
    - (0, 1]  → ×100
    - (1, 100] → treat as already percent
    Values above 100 (bad input) are clamped to 100.
    """
    if pd.isna(val):
        return np.nan
    x = float(val)
    if x > 100.0:
        return 100.0
    if x > 1.0 + 1e-9:
        return max(0.0, min(x, 100.0))
    return max(0.0, min(100.0 * x, 100.0))


def _14b_match_rate_to_pct(val) -> float:
    """14B_MatchRate ``match_rate``: 0–1 or 0–100; output clamped to [0, 100]."""
    return _fraction_or_percent_to_pct_clamped(val)


def _parse_match_question_cell(val) -> tuple[float, int]:
    """
    SR ``Match?`` cells like ``8/11`` → (100*8/11, 11). Unknown → (nan, 0).
    """
    if pd.isna(val):
        return np.nan, 0
    s = str(val).strip()
    if "/" in s:
        left, right = s.split("/", 1)
        try:
            num = float(left.strip())
            den = float(right.strip())
            if den == 0:
                return np.nan, 0
            pct = max(0.0, min(100.0 * num / den, 100.0))
            return pct, int(den)
        except ValueError:
            return np.nan, 0
    u = s.upper()
    if u in ("TRUE", "1", "YES"):
        return 100.0, 1
    if u in ("FALSE", "0", "NO"):
        return 0.0, 1
    return np.nan, 0


def _missing_row(name: str, o: str) -> dict:
    return {
        "model": name,
        "Origin": o,
        "concordance_pct": np.nan,
        "n_sentence_pairs": 0,
        "missing_model_row": True,
    }


def run_qwen_14b_cc_from_match_rate(
    phys: pd.DataFrame,
    origins_933: set[str],
    name: str,
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "ID_corr" not in df.columns or "match_rate" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns ID_corr, match_rate")
    df = df.copy()
    df["_Origin_key"] = df["ID_corr"].map(_norm_origin)
    df = df[df["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    mod_i = df.set_index("_Origin_key")
    rows_out: list[dict] = []
    for o in sorted(origins_933):
        if o not in phys.index:
            continue
        if o not in mod_i.index:
            rows_out.append(_missing_row(name, o))
            continue
        row = mod_i.loc[o]
        pct = _14b_match_rate_to_pct(row["match_rate"])
        n_pairs = 0
        if "num_human" in row.index and pd.notna(row["num_human"]):
            try:
                n_pairs = int(float(row["num_human"]))
            except (TypeError, ValueError):
                n_pairs = 0
        rows_out.append(
            {
                "model": name,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": n_pairs,
                "missing_model_row": False,
            }
        )
    return pd.DataFrame(rows_out)


def run_qwen_72b_cc_from_match_rate(
    phys: pd.DataFrame,
    origins_933: set[str],
    name: str,
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "Origin" not in df.columns or "match_rate" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns Origin, match_rate")
    df = df.copy()
    df["_Origin_key"] = df["Origin"].map(_norm_origin)
    df = df[df["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    mod_i = df.set_index("_Origin_key")
    rows_out: list[dict] = []
    for o in sorted(origins_933):
        if o not in phys.index:
            continue
        if o not in mod_i.index:
            rows_out.append(_missing_row(name, o))
            continue
        row = mod_i.loc[o]
        pct = _fraction_or_percent_to_pct_clamped(row["match_rate"])
        n_pairs = 0
        if "keep_k" in row.index and pd.notna(row["keep_k"]):
            try:
                n_pairs = int(float(row["keep_k"]))
            except (TypeError, ValueError):
                n_pairs = 0
        if n_pairs == 0 and "human_sentence_ids" in row.index:
            n_pairs = _count_csv_int_ids(row["human_sentence_ids"])
        rows_out.append(
            {
                "model": name,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": n_pairs,
                "missing_model_row": False,
            }
        )
    return pd.DataFrame(rows_out)


def run_qwen_sr_from_match_question(
    phys: pd.DataFrame,
    origins_933: set[str],
    name: str,
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "Origin" not in df.columns or "Match?" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns Origin, Match?")
    df = df.copy()
    df["_Origin_key"] = df["Origin"].map(_norm_origin)
    df = df[df["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    mod_i = df.set_index("_Origin_key")
    rows_out: list[dict] = []
    for o in sorted(origins_933):
        if o not in phys.index:
            continue
        if o not in mod_i.index:
            rows_out.append(_missing_row(name, o))
            continue
        row = mod_i.loc[o]
        pct, n_pairs = _parse_match_question_cell(row["Match?"])
        rows_out.append(
            {
                "model": name,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": n_pairs,
                "missing_model_row": False,
            }
        )
    return pd.DataFrame(rows_out)


def run_gpt4o_from_match_percentage(
    phys: pd.DataFrame,
    origins_933: set[str],
    name: str,
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "Origin" not in df.columns or "match_percentage" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns Origin, match_percentage")
    df = df.copy()
    df["_Origin_key"] = df["Origin"].map(_norm_origin)
    df = df[df["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    mod_i = df.set_index("_Origin_key")
    rows_out: list[dict] = []
    for o in sorted(origins_933):
        if o not in phys.index:
            continue
        if o not in mod_i.index:
            rows_out.append(_missing_row(name, o))
            continue
        row = mod_i.loc[o]
        pct = _fraction_or_percent_to_pct_clamped(row["match_percentage"])
        n_pairs = 0
        if "keep_k" in row.index and pd.notna(row["keep_k"]):
            try:
                n_pairs = int(float(row["keep_k"]))
            except (TypeError, ValueError):
                n_pairs = 0
        if n_pairs == 0 and "human_sentence_ids" in row.index:
            n_pairs = _count_csv_int_ids(row["human_sentence_ids"])
        rows_out.append(
            {
                "model": name,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": n_pairs,
                "missing_model_row": False,
            }
        )
    return pd.DataFrame(rows_out)


def run_gpt5_from_match_question(
    phys: pd.DataFrame,
    origins_933: set[str],
    name: str,
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "ID_corr" not in df.columns or "Match?" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns ID_corr, Match?")
    df = df.copy()
    df["_Origin_key"] = df["ID_corr"].map(_norm_origin)
    df = df[df["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    mod_i = df.set_index("_Origin_key")
    rows_out: list[dict] = []
    for o in sorted(origins_933):
        if o not in phys.index:
            continue
        if o not in mod_i.index:
            rows_out.append(_missing_row(name, o))
            continue
        row = mod_i.loc[o]
        pct, n_pairs = _parse_match_question_cell(row["Match?"])
        rows_out.append(
            {
                "model": name,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": n_pairs,
                "missing_model_row": False,
            }
        )
    return pd.DataFrame(rows_out)


def detect_sentence_label_columns(columns: Iterable[str]) -> list[str]:
    cols_set = set(columns)
    out: list[str] = []
    for i in range(1, 22):
        c = f"q{i}"
        if c in cols_set:
            out.append(c)
        else:
            break
    if len(out) >= 15:
        if len(out) == 20:
            if "q21" in cols_set:
                out.append("q21")
            elif "label_21" in cols_set:
                out.append("label_21")
        return out
    lab = [f"label_{i}" for i in range(1, 22) if f"label_{i}" in cols_set]
    if len(lab) >= 15:
        return lab
    raise ValueError(
        f"Need consecutive q1..q* or label_1..label_*; sample: {list(columns)[:30]}"
    )


def load_model_dedup(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "Origin" not in df.columns:
        raise ValueError(f"{path}: missing Origin column")
    df = df.copy()
    df["_Origin_key"] = df["Origin"].map(_norm_origin)
    df = df[df["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    return df


def concordance_for_row(
    phys_row: pd.Series,
    mod_row: pd.Series,
    model_cols: list[str],
    phys_threshold: float,
) -> tuple[float, int]:
    matches = 0
    valid = 0
    for j, mcol in enumerate(model_cols, start=1):
        scol = f"Sentence {j}"
        if scol not in phys_row.index:
            break
        pb = physician_bucket(phys_row[scol], phys_threshold)
        if mcol not in mod_row.index:
            continue
        mb = model_bucket(mod_row[mcol])
        if pd.isna(pb) or pd.isna(mb):
            continue
        valid += 1
        if pb == mb:
            matches += 1
    if valid == 0:
        return np.nan, 0
    return 100.0 * matches / valid, valid


def run_model(
    phys: pd.DataFrame,
    origins_933: set[str],
    name: str,
    csv_path: Path,
    phys_threshold: float,
) -> pd.DataFrame:
    if name == "Qwen-14B_CC":
        return run_qwen_14b_cc_from_match_rate(phys, origins_933, name, csv_path)
    if name == "Qwen-72B_CC":
        return run_qwen_72b_cc_from_match_rate(phys, origins_933, name, csv_path)
    if name in ("Qwen-14B_SR", "Qwen-72B_SR"):
        return run_qwen_sr_from_match_question(phys, origins_933, name, csv_path)
    if name == "GPT4o":
        return run_gpt4o_from_match_percentage(phys, origins_933, name, csv_path)
    if name == "GPT5":
        return run_gpt5_from_match_question(phys, origins_933, name, csv_path)

    mod = load_model_dedup(csv_path)
    mod_f = mod[mod["_Origin_key"].isin(origins_933)]

    if _model_uses_precomputed_match_rate(mod.columns):
        mod_i = mod_f.drop_duplicates(subset=["_Origin_key"], keep="first").set_index("_Origin_key")
        rows_out = []
        for o in sorted(origins_933):
            if o not in phys.index:
                continue
            if o not in mod_i.index:
                rows_out.append(
                    {
                        "model": name,
                        "Origin": o,
                        "concordance_pct": np.nan,
                        "n_sentence_pairs": 0,
                        "missing_model_row": True,
                    }
                )
                continue
            row = mod_i.loc[o]
            pct = _fraction_or_percent_to_pct_clamped(row["Match_Rate"])
            n_pairs = (
                _count_csv_int_ids(row["human_sentence_ids"])
                if "human_sentence_ids" in mod_i.columns
                else np.nan
            )
            rows_out.append(
                {
                    "model": name,
                    "Origin": o,
                    "concordance_pct": pct,
                    "n_sentence_pairs": int(n_pairs) if not pd.isna(n_pairs) else 0,
                    "missing_model_row": False,
                }
            )
        return pd.DataFrame(rows_out)

    mcols = detect_sentence_label_columns(mod.columns)
    mod_i = mod_f.set_index("_Origin_key")

    rows_out = []
    for o in sorted(origins_933):
        if o not in phys.index:
            continue
        if o not in mod_i.index:
            rows_out.append(
                {
                    "model": name,
                    "Origin": o,
                    "concordance_pct": np.nan,
                    "n_sentence_pairs": 0,
                    "missing_model_row": True,
                }
            )
            continue
        pct, n = concordance_for_row(phys.loc[o], mod_i.loc[o], mcols, phys_threshold)
        rows_out.append(
            {
                "model": name,
                "Origin": o,
                "concordance_pct": pct,
                "n_sentence_pairs": n,
                "missing_model_row": False,
            }
        )
    return pd.DataFrame(rows_out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="933-cohort sentence concordance vs model CSVs (join on Origin only)."
    )
    p.add_argument(
        "--majority-933",
        type=Path,
        default=DEFAULT_MAJORITY_933,
        help="933_Clinician_Student_Majority_Vote.csv",
    )
    p.add_argument(
        "--phys-threshold",
        type=float,
        default=0.5,
        help="Physician Sentence k score ≥ this → 'high' bucket.",
    )
    p.add_argument(
        "--models-json",
        type=Path,
        default=None,
        help='JSON object {"GPT4o": "/path/to.csv", ...} overriding default model files.',
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help="Write aggregate summary CSV here.",
    )
    p.add_argument(
        "--per-origin-out",
        type=Path,
        default=DEFAULT_PER_ORIGIN_OUT,
        help="Long per-Origin table (default: Figures/concordance_rate/concordance_933_by_origin_long.csv).",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of model keys to run (default: all).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    maj_path = args.majority_933.resolve()
    if not maj_path.is_file():
        print(f"Missing {maj_path}", file=sys.stderr)
        sys.exit(1)

    phys = pd.read_csv(maj_path, low_memory=False)
    phys["_Origin_key"] = phys["Origin"].map(_norm_origin)
    phys = phys[phys["_Origin_key"] != ""].drop_duplicates(subset=["_Origin_key"], keep="first")
    phys = phys.set_index("_Origin_key")
    origins_933 = set(phys.index)
    if len(origins_933) != 933:
        print(
            f"Warning: expected 933 unique Origins, got {len(origins_933)}",
            file=sys.stderr,
        )

    model_map = {k: Path(v) for k, v in DEFAULT_MODEL_CSVS.items()}
    if args.models_json is not None:
        custom = json.loads(args.models_json.read_text(encoding="utf-8"))
        model_map.update({k: Path(v) for k, v in custom.items()})

    if args.models:
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        model_map = {k: v for k, v in model_map.items() if k in want}

    all_long = []
    summaries = []

    for name, path in model_map.items():
        path = path.resolve()
        if not path.is_file():
            print(f"Skip {name}: missing file {path}", file=sys.stderr)
            continue
        long_df = run_model(phys, origins_933, name, path, args.phys_threshold)
        long_df["model_csv"] = str(path)
        all_long.append(long_df)

        sub = long_df[~long_df["concordance_pct"].isna()]
        n_no_overlap = int((long_df["n_sentence_pairs"] == 0).sum())
        summaries.append(
            {
                "model": name,
                "n_origins_933": len(origins_933),
                "n_origins_scored": int(sub.shape[0]),
                "n_origins_no_valid_sentence_pairs": n_no_overlap,
                "mean_concordance_pct": float(sub["concordance_pct"].mean())
                if len(sub)
                else np.nan,
                "sd_concordance_pct": float(sub["concordance_pct"].std(ddof=1))
                if len(sub) > 1
                else np.nan,
                "mean_sentence_pairs_per_origin": float(sub["n_sentence_pairs"].mean())
                if len(sub)
                else np.nan,
                "model_csv": str(path),
            }
        )

    summary_df = pd.DataFrame(summaries)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_out, index=False)
    print(f"Wrote {args.summary_out}")

    if args.per_origin_out is not None and all_long:
        per = pd.concat(all_long, ignore_index=True)
        args.per_origin_out.parent.mkdir(parents=True, exist_ok=True)
        per.to_csv(args.per_origin_out, index=False)
        print(f"Wrote {args.per_origin_out}")


if __name__ == "__main__":
    main()
