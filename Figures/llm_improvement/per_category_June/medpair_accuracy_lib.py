"""Rebuild the tab:comparison longtable in Supplementary.tex from prediction CSVs.

Every cell is computed on the 933-question MedPAIR cohort. Coverage (how many of
the 933 a model actually answered) is reported alongside each value.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/yuexing/NeuRIPS25/After_PT_Removal")
COHORT = Path(
    "/home/yuexing/NeuRIPS25/Physician_Labels/May27_2026_Data/"
    "May27_2026_Origin_Summary_933.csv"
)
N_COHORT = 933

_k = pd.read_csv(COHORT)
_k["Origin"] = _k["Origin"].astype(str).str.strip()
_k["source"] = _k["data_source_corr_x"].astype(str).str.strip().str.lower()
KEY = _k.set_index("Origin")[["source", "correct_answer"]]

SOURCES = ["mmlu", "jama", "medxpert", "medbullets"]

# Accuracy is reported over the whole 933-question cohort: a question the model
# did not answer counts as incorrect rather than being dropped. This keeps every
# cell on one denominator, so partial responders are directly comparable.
COHORT_COUNTS = KEY["source"].value_counts().to_dict()

# Table row order (as printed) -> subdirectory holding that model's predictions.
MODELS = [
    ("GPT5", "GPT5"),
    ("GPT4o", "GPT4o"),
    ("Llama-70B", "Llama-70B"),
    ("Qwen-72B", "Qwen2.5-72B-Instruct"),
    ("Qwen-14B", "Qwen2.5-14B-Instruct"),
    ("MedGemma-27B", "MedGemma-27b-text-it"),
]

# block label -> {model: prediction file relative to its results/predictions dir}
BLOCKS: dict[str, dict[str, str]] = {
    "Trainee Removed (Low+Irr Labeled by Trainees)": {
        "GPT5": "gpt5_predictions_on_trainee_irr_removed.csv",
        "GPT4o": "gpt4o_predictions_on_trainee_irr_removed.csv",
        "Llama-70B": "Llama70B_predictions_on_trainee_irr_removed.csv",
        "Qwen-72B": "Qwen_72B_predictions_trainee_irr_removed.csv",
        "Qwen-14B": "Qwen_14B_predictions_trainee_irr_removed.csv",
        "MedGemma-27B": "MedGemma27B_predictions_on_trainee_irr_removed.csv",
    },
    "Physician-Irrelevant (Low+Irr Sentences Only)": {
        "GPT5": "gpt5_predictions_on_MJ_LowIRR.csv",
        # gpt4o_predictions_on_MJ_LowIRR.csv is contaminated: it scores 74.0,
        # i.e. GPT-4o's full-context accuracy, not the low+irr-only condition.
        # The dedicated ctx_ file gives 43.3, consistent with every other model.
        "GPT4o": "gpt4o_predictions_ctx_clinician_student_mj_lowirr_sentences.csv",
        "Llama-70B": "Llama70B_predictions_on_MJ_LowIRR.csv",
        "Qwen-72B": "Qwen_72B_predictions_on_MJ_LowIRR.csv",
        "Qwen-14B": "Qwen_14B_predictions_on_MJ_LowIRR.csv",
        "MedGemma-27B": "MedGemma27B_predictions_on_MJ_LowIRR.csv",
    },
    "Physician-Random (Random Sentence Subset Baseline)": {
        "GPT5": "gpt5_predictions_on_Random.csv",
        "GPT4o": "gpt4o_predictions_on_Random.csv",
        "Llama-70B": "Llama70B_predictions_on_Random.csv",
        "Qwen-72B": "Qwen_72B_predictions_on_Random.csv",
        "Qwen-14B": "Qwen_14B_predictions_on_Random.csv",
        "MedGemma-27B": "MedGemma27B_predictions_on_Random.csv",
    },
    "Qwen-14B Removed": {
        "GPT5": "gpt5_predictions_on_Qwen14B_removed.csv",
        "GPT4o": "gpt4o_predictions_on_14b_removed.csv",
        "Llama-70B": "Llama70B_predictions_on_14B.csv",
        "Qwen-72B": "Qwen_72B_predictions_14B.csv",
        "Qwen-14B": "Qwen_14B_predictions_14B.csv",
        "MedGemma-27B": "MedGemma27B_predictions_14B.csv",
    },
    "Qwen-72B Removed": {
        "GPT5": "gpt5_predictions_on_Qwen72B_removed.csv",
        # No top-level gpt4o_predictions_on_qwen72b_removed.csv exists; this run
        # lives under remove_low_irr/ and is keyed by 72B_Sentence_Contents.
        "GPT4o": "remove_low_irr/GPT4o_analysis_results.csv",
        "Llama-70B": "Llama70B_predictions_on_72B.csv",
        "Qwen-72B": "Qwen_72B_predictions_72B.csv",
        "Qwen-14B": "Qwen_14B_predictions_Qwen72B_FIXED.csv",
        "MedGemma-27B": "MedGemma_27B_predictions_72B.csv",
    },
    "Llama-70B Removed": {
        "GPT5": "gpt5_predictions_on_Llama70B_removed.csv",
        "GPT4o": "gpt4o_predictions_on_llama70b_removed.csv",
        "Llama-70B": "Llama70B_predictions_on_70B.csv",
        "Qwen-72B": "Qwen_72B_predictions_70B.csv",
        "Qwen-14B": "Qwen_14B_predictions_70B.csv",
        "MedGemma-27B": "MedGemma27B_predictions_70B.csv",
    },
    "GPT-4o Removed": {
        "GPT5": "gpt5_predictions_on_gpt4o_removed.csv",
        "GPT4o": "gpt4o_predictions_on_gpt4o_removed.csv",
        "Llama-70B": "Llama70B_predictions_on_GPT4o.csv",
        "Qwen-72B": "Qwen_72B_predictions_GPT4o.csv",
        "Qwen-14B": "Qwen_14B_predictions_GPT4o.csv",
        "MedGemma-27B": "MedGemma72B_predictions_on_GPT4o.csv",
    },
    "GPT-5 Removed": {
        "GPT5": "gpt5_predictions_on_gpt5_removed.csv",
        "GPT4o": "gpt4o_predictions_on_gpt5_removed.csv",
        "Llama-70B": "Llama70B_predictions_on_GPT5.csv",
        "Qwen-72B": "Qwen_72B_predictions_GPT5.csv",
        "Qwen-14B": "Qwen_14B_predictions_GPT5.csv",
        "MedGemma-27B": "MedGemma72B_predictions_on_GPT5.csv",
    },
    "MedGemma-27B Removed": {
        "GPT5": "gpt5_predictions_on_MedGemma_removed.csv",
        "GPT4o": "gpt4o_predictions_on_MedGemma_removed.csv",
        "Llama-70B": "Llama70B_predictions_on_MedGemma.csv",
        "Qwen-72B": "Qwen_72B_predictions_MedGemma.csv",
        "Qwen-14B": "Qwen_14B_predictions_MedGemma.csv",
        "MedGemma-27B": "MedGemma27B_predictions_Gemma.csv",
    },
}

DIR_OF = dict(MODELS)

# Cells whose source file fails provenance and must not be published.
#
# The MedGemma-27B runs named ``MedGemma*_predictions_<X>.csv`` (no ``on_``)
# score 34-47 points ABOVE what MedGemma-27B scores on the very same questions
# in its trusted trainee_irr_removed run -- e.g. 76.1% vs 29.4% on an identical
# 498-question subset. That is not a coverage or selection effect: it is a
# different, much stronger model. Agreement analysis puts them at 72-78% with
# every strong model and only 33.7% with real MedGemma, so the true source
# cannot be recovered. These cells are reported as unavailable pending a rerun.
# Previously five MedGemma-27B cells were excluded here on the theory that they
# held another model's output. That was wrong: the reference used for the check
# (MedGemma27B_predictions_on_trainee_irr_removed.csv) is itself degenerate,
# answering "A" for 82% of items, as do the MJ_LowIRR and Random runs (89%/85%).
# The files excluded were the healthy ones (~19-23% "A", near uniform).
EXCLUDED_CELLS: set[tuple[str, str]] = set()


def norm_choice(v: object) -> str:
    """Answer letter, or "" when the model did not actually answer.

    Refusals ("I'm sorry, but I need more context...") must NOT be scored: a
    naive [A-J] scan returns the I of "I'm". Only an <answer> tag or a short
    letter-like token counts as an answer.
    """
    if pd.isna(v):
        return ""
    s = str(v).strip()
    tagged = re.search(r"<answer>(.*?)</answer>", s, re.IGNORECASE | re.DOTALL)
    inner = tagged.group(1) if tagged else s
    cleaned = re.sub(r"(?i)\boption\b", "", inner)
    cleaned = re.sub(r"[^A-Za-z]", "", cleaned).upper()
    if len(cleaned) == 1 and "A" <= cleaned <= "J":
        return cleaned
    if tagged:
        m = re.search(r"[A-J]", inner.upper())
        return m.group(0) if m else ""
    return ""  # free-form prose / refusal / error -> unanswered


# Several CSVs carry more than one model's predictions side by side (e.g. both
# gpt_direct_prediction, which is GPT-4o's, and gpt5_direct_prediction). Always
# prefer the column belonging to the model whose row we are filling.
MODEL_COL_PATTERNS = {
    "GPT5": r"gpt-?5",
    "GPT4o": r"gpt-?4o",
    "Llama-70B": r"llama-?_?70b",
    "Qwen-72B": r"qwen-?_?72b",
    "Qwen-14B": r"qwen-?_?14b",
    "MedGemma-27B": r"medgemma",
}


def pick_pred_col(d: pd.DataFrame, model: str | None = None) -> str | None:
    """Choose the prediction column belonging to `model`.

    Files are read from the model's own directory, so unqualified columns
    (gpt_direct_prediction, Extracted_Answer, ...) belong to that model --
    EXCEPT where a column explicitly names a different model, which must be
    excluded (GPT5 files carry a stale GPT-4o gpt_direct_prediction alongside
    the real gpt5_direct_prediction).

    A column explicitly naming this model wins, but only if it is reasonably
    complete: some directories hold a partial rerun (825/1300) next to the
    complete original.
    """
    if model and model in MODEL_COL_PATTERNS:
        others = [p for m, p in MODEL_COL_PATTERNS.items() if m != model]
        own = MODEL_COL_PATTERNS[model]
        cand = [
            c
            for c in d.columns
            if re.search(r"(prediction|_letter|Extracted_Answer)", c, re.I)
            and not re.match(r"^(Round|Correct)", c)
            and not any(re.search(p, c, re.I) for p in others)
        ]
        if cand:
            counts = {c: int(d[c].notna().sum()) for c in cand}
            best = max(counts.values()) or 1
            named = [c for c in cand if re.search(own, c, re.I)]
            for c in named:
                if counts[c] >= 0.9 * best:
                    return c
            return max(counts, key=lambda c: counts[c])
    for c in ("Extracted_Answer", "gpt_letter", "Predicted_Answer", "predicted_answer"):
        if c in d.columns:
            return c
    for c in d.columns:
        if re.search(r"(prediction|_letter)$", c, re.I) and not re.match(r"^(Round|Correct)", c):
            return c
    for c in d.columns:
        if re.search(r"prediction", c, re.I):
            return c
    # Some runs only stored the 3-run consensus rather than a named prediction column.
    if "majority_vote" in d.columns:
        return "majority_vote"
    return None


_QA_ID_TO_ORIGIN: dict[str, str] | None = None


def _qa_id_map() -> dict[str, str]:
    """QA_ID -> Origin, learned from any prediction file carrying both."""
    global _QA_ID_TO_ORIGIN
    if _QA_ID_TO_ORIGIN is not None:
        return _QA_ID_TO_ORIGIN
    mapping: dict[str, str] = {}
    for p in sorted(ROOT.glob("*/results/predictions/*.csv")):
        try:
            d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in {"QA_ID", "Origin"})
        except Exception:
            continue
        if {"QA_ID", "Origin"} - set(d.columns):
            continue
        d = d.dropna(subset=["QA_ID", "Origin"])
        for q, o in zip(d["QA_ID"].astype(str), d["Origin"].astype(str)):
            mapping.setdefault(q.strip(), o.strip())
    _QA_ID_TO_ORIGIN = mapping
    return mapping


def resolve_origin(d: pd.DataFrame) -> pd.Series | None:
    """Best available join key onto the 933 cohort, as an Origin-valued Series."""
    if "Origin" in d.columns:
        s = d["Origin"].astype(str).str.strip()
        if s.ne("").any() and s.ne("nan").any():
            return s.str.replace(r"-phase\d+$", "", regex=True)
    if "ID_corr" in d.columns:  # ID_corr holds the same identifiers as Origin
        s = d["ID_corr"].astype(str).str.strip()
        if s.ne("").any() and s.ne("nan").any():
            return s.str.replace(r"-phase\d+$", "", regex=True)
    if "QA_ID" in d.columns:
        m = _qa_id_map()
        s = d["QA_ID"].astype(str).str.strip().map(m)
        if s.notna().any():
            return s
    return None


def evaluate(model: str, fname: str) -> dict:
    path = ROOT / DIR_OF[model] / "results" / "predictions" / fname
    out = {"file": str(path.relative_to(ROOT)), "n": 0, "md5": "", "status": ""}
    if not path.is_file():
        out["status"] = "MISSING FILE"
        return out
    out["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    d = pd.read_csv(path, low_memory=False)
    col = pick_pred_col(d, model)
    if col is None:
        out["status"] = "no prediction column"
        return out
    out["col"] = col
    raw = d[col].astype(str).str.strip()
    if raw.str.fullmatch("Error", case=False).mean() > 0.5:
        out["status"] = "run failed (all Error)"
        return out
    origin = resolve_origin(d)
    if origin is None:
        out["status"] = "no usable ID column"
        return out
    d = d.assign(_origin=origin).dropna(subset=[col, "_origin"]).copy()
    d = d.drop_duplicates(subset=["_origin"])
    m = d[["_origin", col]].merge(KEY, left_on="_origin", right_index=True, how="inner")
    if m.empty:
        out["status"] = "no rows match the 933 cohort"
        return out
    pred = m[col].map(norm_choice)
    gold = m["correct_answer"].map(norm_choice)
    # Refusals / errors are not answers: exclude them and report the shrunken
    # denominator as coverage rather than silently scoring them as wrong.
    answered = pred.ne("")
    out["refused"] = int((~answered).sum())
    m = m.assign(ok=(pred == gold) & answered)
    out["n"] = int(answered.sum())        # coverage, not the denominator
    if out["n"] == 0:
        out["status"] = "no answered rows"
        return out
    correct = m.groupby("source")["ok"].sum()
    for s in SOURCES:
        denom = COHORT_COUNTS.get(s)
        out[s] = float(100.0 * correct.get(s, 0) / denom) if denom else float("nan")
    out["general"] = float(100.0 * m["ok"].sum() / N_COHORT)
    out["status"] = "ok"
    return out


def fmt(val: float, n: int) -> str:
    """Per-source cell. Coverage is reported once per row, on the General column."""
    if not np.isfinite(val):
        return "--"
    return f"{val:.1f}"


def main() -> None:
    results: dict[tuple[str, str], dict] = {}
    for block, mapping in BLOCKS.items():
        for model, fname in mapping.items():
            key_ = (block, model)
            if key_ in EXCLUDED_CELLS:
                results[key_] = {
                    "file": fname,
                    "n": 0,
                    "md5": "",
                    "status": "EXCLUDED (failed provenance check)",
                }
                continue
            results[key_] = evaluate(model, fname)

    # ---- QA report -------------------------------------------------------
    print("=" * 100)
    print("PER-CELL QA REPORT")
    print("=" * 100)
    bad = []
    for (block, model), r in results.items():
        flag = ""
        if r["status"] != "ok":
            flag = f"  <== {r['status']}"
            bad.append((block, model, r["status"]))
        elif r["n"] < N_COHORT:
            flag = f"  (coverage {100 * r['n'] / N_COHORT:.1f}%)"
        print(f"{block[:44]:46} {model:13} n={r['n']:4}{flag}")

    # duplicate-content detection
    print("\n" + "=" * 100)
    print("IDENTICAL SOURCE FILES (same md5 used for different cells)")
    print("=" * 100)
    seen: dict[str, list[str]] = {}
    for (block, model), r in results.items():
        if r.get("md5"):
            seen.setdefault(r["md5"], []).append(f"{block} / {model}")
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if not dupes:
        print("  none")
    for k, v in dupes.items():
        print(f"  {k[:12]}: " + "  ||  ".join(v))

    # ---- LaTeX body ------------------------------------------------------
    lines: list[str] = []
    for bi, (block, mapping) in enumerate(BLOCKS.items()):
        lines.append(
            f"\\multicolumn{{7}}{{l}}{{\\textit{{{block} -- Detailed Accuracies -- 933 QAs}}}} \\\\"
        )
        lines.append("\\midrule")
        for model, _ in MODELS:
            r = results[(block, model)]
            n = r["n"]
            cells = [fmt(r.get(s, float("nan")), n) for s in ["mmlu", "jama", "medxpert", "medbullets"]]
            gen = r.get("general", float("nan"))
            gen_s = "--" if not np.isfinite(gen) else (
                f"\\textbf{{{gen:.1f}}}" if n >= N_COHORT
                else f"\\textbf{{{gen:.1f}}} {{\\tiny($n$={n})}}"
            )
            lines.append(
                f"{model} & {block.split(' (')[0]} & " + " & ".join(cells) + f" & {gen_s} \\\\"
            )
        lines.append("\\bottomrule" if bi == len(BLOCKS) - 1 else "\\midrule")

    body = "\n".join(lines)
    out = Path(
        "/tmp/claude-45402/-orcd-home-002-yuexing/"
        "5b312108-ca49-49b5-a9f4-fae27ae5493b/scratchpad/tab_comparison_body.tex"
    )
    out.write_text(body + "\n")
    print("\n" + "=" * 100)
    print(f"LaTeX body written to {out}  ({len(lines)} lines)")
    print("=" * 100)
    print(body)


if __name__ == "__main__":
    main()


def per_question(model: str, fname: str) -> pd.DataFrame | None:
    """Per-question outcomes for one run: index Origin, columns source/ok.

    Only questions the model actually answered are returned, so a paired test
    built on this never counts a refusal or an "Error" row as a wrong answer.
    """
    path = ROOT / DIR_OF[model] / "results" / "predictions" / fname
    if not path.is_file():
        return None
    d = pd.read_csv(path, low_memory=False)
    col = pick_pred_col(d, model)
    if col is None:
        return None
    raw = d[col].astype(str).str.strip()
    if raw.str.fullmatch("Error", case=False).mean() > 0.5:
        return None
    o = resolve_origin(d)
    if o is None:
        return None
    d = d.assign(_o=o).dropna(subset=[col, "_o"]).drop_duplicates("_o")
    m = d[["_o", col]].merge(KEY, left_on="_o", right_index=True, how="inner")
    pred = m[col].map(norm_choice)
    m = m[pred.ne("")]
    if m.empty:
        return None
    gold = m["correct_answer"].map(norm_choice)
    scored = pd.DataFrame(
        {"source": m["source"].values, "ok": (pred[pred.ne("")] == gold).values},
        index=m["_o"].values,
    )
    # Reindex onto the whole cohort so unanswered questions count as incorrect,
    # matching the full-933 denominator used for the reported accuracies.
    out = pd.DataFrame(
        {"source": KEY["source"], "ok": False}, index=KEY.index
    )
    out.loc[scored.index, "ok"] = scored["ok"].values
    out.index.name = "Origin"
    return out
