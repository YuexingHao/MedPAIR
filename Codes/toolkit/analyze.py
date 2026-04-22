"""
Analysis: accuracy, refusal rate, relevance distribution, relevance-accuracy breakdown.
All functions are generic — driven by ToolkitConfig column names.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ToolkitConfig

RELEVANCE_LABELS = ["High Relevance", "Low Relevance", "Irrelevant"]
VALID_ANSWERS = set("ABCDEFGHIJ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_ci(values: List[float]) -> Tuple[float, float, float, float]:
    """Returns (mean, std, ci_low, ci_high)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, float("nan"), float("nan"), float("nan")
    std = float(np.std(values, ddof=1))
    hw = 1.96 * std / np.sqrt(n)
    return mean, std, mean - hw, mean + hw


def _fmt(label: str, values: List[float], indent: str = "    ") -> str:
    m, s, lo, hi = _mean_ci(values)
    if np.isnan(m):
        return f"{indent}{label}: N/A (no data)"
    return f"{indent}{label}: mean={m:.4f}, std={s:.4f}, 95% CI=({lo:.4f}, {hi:.4f})"


def _majority_label(row: pd.Series, sentence_count_col: str, max_sentences: int) -> Optional[str]:
    """Return the most common relevance label across the row's sentences."""
    try:
        n = min(int(row[sentence_count_col]), max_sentences)
    except (ValueError, KeyError):
        n = max_sentences
    labels = [
        str(row.get(f"label_{i + 1}", "")).strip()
        for i in range(n)
        if str(row.get(f"label_{i + 1}", "")).strip() in set(RELEVANCE_LABELS)
    ]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def accuracy_section(
    df: pd.DataFrame,
    llm_col: str,
    correct_col: str,
    groupby_col: Optional[str],
) -> str:
    lines = ["  Accuracy:"]
    valid = df[df[llm_col].astype(str).str.strip().isin(VALID_ANSWERS)].copy()
    valid["_correct"] = (
        valid[llm_col].astype(str).str.strip() == valid[correct_col].astype(str).str.strip()
    )
    n_correct = int(valid["_correct"].sum())
    n_total = len(valid)
    lines.append(f"    Overall: {n_correct}/{n_total} = {n_correct/n_total:.4f}" if n_total else "    Overall: N/A")

    if groupby_col and groupby_col in df.columns:
        for grp, gdf in valid.groupby(groupby_col):
            gc = int(gdf["_correct"].sum())
            gt = len(gdf)
            lines.append(f"    {grp}: {gc}/{gt} = {gc/gt:.4f}" if gt else f"    {grp}: N/A")

    return "\n".join(lines)


def refusal_rate_section(
    df: pd.DataFrame,
    llm_col: str,
    groupby_col: Optional[str],
) -> str:
    lines = ["  Refusal Rate (answer not a valid letter):"]
    refused = df[~df[llm_col].astype(str).str.strip().isin(VALID_ANSWERS)]
    total = len(df)
    lines.append(
        f"    Overall: {len(refused)}/{total} = {len(refused)/total:.4f}"
        if total else "    Overall: N/A"
    )
    if groupby_col and groupby_col in df.columns:
        for grp, gdf in df.groupby(groupby_col):
            gr = gdf[~gdf[llm_col].astype(str).str.strip().isin(VALID_ANSWERS)]
            lines.append(f"    {grp}: {len(gr)}/{len(gdf)} = {len(gr)/len(gdf):.4f}" if len(gdf) else f"    {grp}: N/A")
    return "\n".join(lines)


def relevance_distribution_section(
    df: pd.DataFrame,
    sentence_count_col: str,
    max_sentences: int,
    groupby_col: Optional[str],
) -> str:
    lines = ["  Relevance Distribution (% per row, mean ± 95% CI):"]

    def _compute(sub: pd.DataFrame) -> List[str]:
        per_case: Dict[str, List[float]] = {k: [] for k in RELEVANCE_LABELS}
        for _, row in sub.iterrows():
            try:
                n = min(int(row[sentence_count_col]), max_sentences)
            except (ValueError, KeyError):
                n = max_sentences
            labels = [
                str(row.get(f"label_{i + 1}", "")).strip()
                for i in range(n)
            ]
            valid = [l for l in labels if l in set(RELEVANCE_LABELS)]
            if not valid:
                continue
            total = len(valid)
            for k in RELEVANCE_LABELS:
                per_case[k].append(valid.count(k) / total)
        return [_fmt(k, per_case[k]) for k in RELEVANCE_LABELS]

    lines.append("    Overall:")
    lines.extend(_compute(df))

    if groupby_col and groupby_col in df.columns:
        for grp, gdf in df.groupby(groupby_col):
            lines.append(f"    {grp}:")
            lines.extend(_compute(gdf))

    return "\n".join(lines)


def relevance_accuracy_section(
    df: pd.DataFrame,
    llm_col: str,
    correct_col: str,
    sentence_count_col: str,
    max_sentences: int,
) -> str:
    """Accuracy segmented by each row's majority relevance label."""
    lines = ["  Accuracy by majority sentence-relevance label:"]
    counts: Dict[str, Dict[str, int]] = {
        k: {"correct": 0, "total": 0} for k in RELEVANCE_LABELS
    }

    for _, row in df.iterrows():
        llm_ans = str(row.get(llm_col, "")).strip()
        if llm_ans not in VALID_ANSWERS:
            continue
        correct_ans = str(row.get(correct_col, "")).strip()
        label = _majority_label(row, sentence_count_col, max_sentences)
        if label is None:
            continue
        counts[label]["total"] += 1
        if llm_ans == correct_ans:
            counts[label]["correct"] += 1

    for k in RELEVANCE_LABELS:
        c, t = counts[k]["correct"], counts[k]["total"]
        acc = f"{c}/{t} = {c/t:.4f}" if t else "N/A"
        lines.append(f"    {k}: {acc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_analysis(config: ToolkitConfig, input_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    lines = [
        f"=== Analysis Report ===",
        f"Input:  {input_path}",
        f"Model:  {config.model_name}",
        "",
    ]

    # Determine (llm_col, correct_col) pairs
    if config.mode == "single":
        pairs = [("LLM_answer", config.correct_answer_col or "")]
    else:
        pairs = [(q.name, q.correct_col) for q in config.questions]

    for llm_col, correct_col in pairs:
        lines.append(f"--- {llm_col} ---")

        if llm_col not in df.columns:
            lines.append(f"  Column '{llm_col}' not found in data. Skipping.")
            lines.append("")
            continue

        # Refusal rate is always available
        lines.append(refusal_rate_section(df, llm_col, config.groupby_col))

        if correct_col and correct_col in df.columns:
            lines.append(accuracy_section(df, llm_col, correct_col, config.groupby_col))
            if config.sentence_count_col and config.sentence_count_col in df.columns:
                lines.append(
                    relevance_accuracy_section(
                        df, llm_col, correct_col,
                        config.sentence_count_col, config.max_sentences,
                    )
                )
        else:
            lines.append("  (No correct-answer column configured — skipping accuracy.)")

        lines.append("")

    # Relevance distribution (independent of answer pairs)
    if config.sentence_count_col and config.sentence_count_col in df.columns:
        lines.append("--- Sentence Relevance Distribution ---")
        lines.append(
            relevance_distribution_section(
                df, config.sentence_count_col, config.max_sentences, config.groupby_col
            )
        )
        lines.append("")

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    _save_summary_csv(df, config, output_dir)


def _save_summary_csv(df: pd.DataFrame, config: ToolkitConfig, output_dir: str) -> None:
    """Save a flat per-row summary CSV with accuracy flags and relevance percentages."""
    if not config.sentence_count_col or config.sentence_count_col not in df.columns:
        return

    if config.mode == "single":
        pairs = [("LLM_answer", config.correct_answer_col or "")]
    else:
        pairs = [(q.name, q.correct_col) for q in config.questions]

    rows = []
    for _, row in df.iterrows():
        r: dict = {}
        if config.id_col and config.id_col in row.index:
            r["id"] = row[config.id_col]
        if config.groupby_col and config.groupby_col in row.index:
            r["group"] = row[config.groupby_col]

        # Majority relevance + percentages
        maj = _majority_label(row, config.sentence_count_col, config.max_sentences)
        r["majority_relevance"] = maj
        try:
            n = min(int(row[config.sentence_count_col]), config.max_sentences)
        except (ValueError, KeyError):
            n = config.max_sentences
        labels = [
            str(row.get(f"label_{i + 1}", "")).strip()
            for i in range(n)
            if str(row.get(f"label_{i + 1}", "")).strip() in set(RELEVANCE_LABELS)
        ]
        total = len(labels) or 1
        for k in RELEVANCE_LABELS:
            r[f"pct_{k.lower().replace(' ', '_')}"] = labels.count(k) / total

        # Answer correctness
        for llm_col, correct_col in pairs:
            r[f"{llm_col}_answer"] = row.get(llm_col)
            r[f"{llm_col}_refused"] = str(row.get(llm_col, "")).strip() not in VALID_ANSWERS
            if correct_col and correct_col in row.index:
                r[f"{llm_col}_is_correct"] = (
                    str(row.get(llm_col, "")).strip() == str(row.get(correct_col, "")).strip()
                )

        rows.append(r)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to: {summary_path}")
