#!/usr/bin/env python3
import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


COLS = [
    "Trainee Majority Vote",
    "GPT-4o SR",
    "GPT-5 SR",
    "Qwen-14B SR",
    "Qwen-72B SR",
    "Llama-70B SR",
    "MedGemma-27B SR",
    "Qwen-14B CC",
    "Qwen-72B CC",
    "Llama-70B CC",
]


@dataclass
class Sources:
    trainee_sr: pd.DataFrame
    gpt4o_sr: pd.DataFrame
    gpt5_sr: pd.DataFrame
    q14_sr: pd.DataFrame
    q72_sr: pd.DataFrame
    l70_sr: pd.DataFrame
    medgemma_sr: pd.DataFrame
    q14_cc_topk: pd.DataFrame
    q14_cc_fallback: pd.DataFrame
    q72_cc: pd.DataFrame
    l70_cc: pd.DataFrame


def _row_by_id(df: pd.DataFrame, qid: str) -> Optional[pd.Series]:
    for key in ["Origin", "ID_corr", "ID", "QA_ID", "QA_ID_std"]:
        if key in df.columns:
            m = df[df[key].astype(str).str.strip() == qid]
            if len(m):
                return m.iloc[0]
    return None


def _norm_sr(v: object) -> str:
    if pd.isna(v):
        return "Low/Irr"
    s = str(v).strip()
    if not s:
        return "Low/Irr"
    l = s.lower()
    if "high relevance" in l or l == "high":
        return "High"
    if "low relevance" in l or l == "low":
        return "Low"
    if "irrelevant" in l or l == "irr":
        return "Irr"
    if s in {"High", "Low", "Irr", "Low/Irr"}:
        return s
    # Table policy: empty/unselected/no-label represented as Low/Irr.
    return "Low/Irr"


def _norm_table(v: str) -> str:
    s = v.strip()
    if s.endswith("\\\\"):
        s = s[:-2].strip()
    return s


def _parse_numset(x: object) -> Set[int]:
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    # Strings like "1,2,3" or "1. 3. 5."
    out = {int(n) for n in re.findall(r"\d+", s)}
    return out


def _parse_high_sentence_list(x: object) -> Set[int]:
    if pd.isna(x):
        return set()
    s = str(x)
    # Prefer "N. <sentence>" patterns to avoid picking unrelated numbers.
    out = {int(n) for n in re.findall(r"\b(\d+)\.\s", s)}
    if out:
        return out
    # Fall back to any integer tokens.
    return {int(n) for n in re.findall(r"\d+", s)}


def parse_tables(tex_path: Path) -> Dict[str, Dict[int, Dict[str, str]]]:
    text = tex_path.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"title=Example of QA Data \(from .*? (ID\d{4})\)", text, flags=re.DOTALL
        )
    )
    out: Dict[str, Dict[int, Dict[str, str]]] = {}
    for m in matches:
        qid = m.group(1)
        sub = text[m.start() :]
        tstart = sub.find(r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}")
        if tstart < 0:
            continue
        tend = sub.find(r"\end{tabular}", tstart)
        if tend < 0:
            continue
        block = sub[tstart:tend]
        rows: Dict[int, Dict[str, str]] = {}
        for line in block.splitlines():
            if not re.match(r"\s*\d+\s*&", line):
                continue
            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 11:
                continue
            sent = int(parts[0])
            vals = [_norm_table(p) for p in parts[1:11]]
            rows[sent] = dict(zip(COLS, vals))
        if rows:
            out[qid] = rows
    return out


def expected_for_qid(qid: str, rows: List[int], src: Sources) -> Tuple[Dict[int, Dict[str, str]], Dict[str, str]]:
    r_tr = _row_by_id(src.trainee_sr, qid)
    r_g4 = _row_by_id(src.gpt4o_sr, qid)
    r_g5 = _row_by_id(src.gpt5_sr, qid)
    r_q14 = _row_by_id(src.q14_sr, qid)
    r_q72 = _row_by_id(src.q72_sr, qid)
    r_l70 = _row_by_id(src.l70_sr, qid)
    r_mg = _row_by_id(src.medgemma_sr, qid)
    r_q72cc = _row_by_id(src.q72_cc, qid)
    r_l70cc = _row_by_id(src.l70_cc, qid)
    r_q14cc_topk = _row_by_id(src.q14_cc_topk, qid)

    q14cc_source = "qwen14b_contextcite_topk_summary.csv"
    if r_q14cc_topk is not None:
        q14_cc_set = _parse_numset(r_q14cc_topk.get("qwen14b_topK_sentence_ids"))
    else:
        q14cc_source = "Qwen14B_Physician_Filtered_with_Sentences.csv (fallback top-k by Score)"
        m = src.q14_cc_fallback[src.q14_cc_fallback["ID_corr"].astype(str).str.strip() == qid].copy()
        q14_cc_set: Set[int] = set()
        if len(m):
            keep_k = int(float(m["keep_k"].dropna().iloc[0])) if m["keep_k"].dropna().size else 0
            m = m.sort_values("Score", ascending=False)
            q14_cc_set = {int(float(x)) for x in m["sentence_number"].dropna().tolist()[:keep_k]}

    q72_cc_set = _parse_numset(r_q72cc.get("72B_Sentences") if r_q72cc is not None else None)
    l70_cc_set = _parse_numset(r_l70cc.get("70B_sentence_ids") if r_l70cc is not None else None)

    trainee_high = _parse_high_sentence_list(r_tr.get("High") if r_tr is not None else None)
    l70_high = _parse_high_sentence_list(r_l70.get("High") if r_l70 is not None else None)

    expected: Dict[int, Dict[str, str]] = {}
    for s in rows:
        # SR columns
        g4 = _norm_sr(r_g4.get(f"label_{s}") if r_g4 is not None else None)
        g5 = _norm_sr(r_g5.get(f"label_{s}") if r_g5 is not None else None)
        q14 = _norm_sr(r_q14.get(f"q{s}") if r_q14 is not None else None)
        q72 = _norm_sr(r_q72.get(f"q{s}") if r_q72 is not None else None)
        mg = _norm_sr(r_mg.get(f"q{s}") if r_mg is not None else None)

        l70_raw = r_l70.get(f"q{s}") if r_l70 is not None else None
        if pd.isna(l70_raw) or str(l70_raw).strip() == "":
            l70 = "High" if s in l70_high else "Low/Irr"
        else:
            l70 = _norm_sr(l70_raw)

        exp = {
            "Trainee Majority Vote": "High" if s in trainee_high else "Low/Irr",
            "GPT-4o SR": g4,
            "GPT-5 SR": g5,
            "Qwen-14B SR": q14,
            "Qwen-72B SR": q72,
            "Llama-70B SR": l70,
            "MedGemma-27B SR": mg,
            "Qwen-14B CC": "High" if s in q14_cc_set else "Low/Irr",
            "Qwen-72B CC": "High" if s in q72_cc_set else "Low/Irr",
            "Llama-70B CC": "High" if s in l70_cc_set else "Low/Irr",
        }
        # Table policy for SR: only non-selected/no-label -> Low/Irr. Keep Low/Irr as-is.
        for k in [
            "Trainee Majority Vote",
            "GPT-4o SR",
            "GPT-5 SR",
            "Qwen-14B SR",
            "Qwen-72B SR",
            "Llama-70B SR",
            "MedGemma-27B SR",
        ]:
            if exp[k] not in {"High", "Low", "Irr", "Low/Irr"}:
                exp[k] = "Low/Irr"
        expected[s] = exp

    source_notes = {
        "Trainee Majority Vote": "[SR]Qwen14B_annotated_MedPAIR_relevancy.csv: High",
        "GPT-4o SR": "GPT4o_Self_Reported_Relevancy_Labels.csv",
        "GPT-5 SR": "gpt5-relevancy-combined-dec-12.csv",
        "Qwen-14B SR": "[SR]Qwen14B_annotated_MedPAIR_relevancy.csv",
        "Qwen-72B SR": "[SR]Qwen72B_annotated_MedPAIR_relevancy.csv",
        "Llama-70B SR": "[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv (q*; fallback to High list if q* missing)",
        "MedGemma-27B SR": "MedGemma_SR_Match_Rate.csv",
        "Qwen-14B CC": q14cc_source,
        "Qwen-72B CC": "Qwen72B_ContextCite_Match.csv: 72B_Sentences",
        "Llama-70B CC": "70B_ContextCite_Removal.csv: 70B_sentence_ids",
    }
    return expected, source_notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/yuexing/NeuRIPS25",
        help="NeuRIPS25 repo root",
    )
    parser.add_argument(
        "--tex",
        default="/home/yuexing/NeuRIPS25/medpair_paper/Appendix.tex",
        help="Path to Appendix.tex",
    )
    parser.add_argument(
        "--out-csv",
        default="/home/yuexing/NeuRIPS25/medpair_paper/audit_case_tables_report.csv",
        help="Output CSV for mismatches",
    )
    args = parser.parse_args()

    root = Path(args.repo_root)
    tex_path = Path(args.tex)

    src = Sources(
        trainee_sr=pd.read_csv(root / "Physician_Labels/results/[SR]Qwen14B_annotated_MedPAIR_relevancy.csv"),
        gpt4o_sr=pd.read_csv(root / "Dataset/GPT4o/GPT4o_Self_Reported_Relevancy_Labels.csv"),
        gpt5_sr=pd.read_csv(root / "Dataset/GPT5/gpt5-relevancy-combined-dec-12.csv"),
        q14_sr=pd.read_csv(root / "Physician_Labels/results/[SR]Qwen14B_annotated_MedPAIR_relevancy.csv"),
        q72_sr=pd.read_csv(root / "Physician_Labels/results/[SR]Qwen72B_annotated_MedPAIR_relevancy.csv"),
        l70_sr=pd.read_csv(root / "Physician_Labels/results/[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv"),
        medgemma_sr=pd.read_csv(root / "Physician_Labels/results/MedGemma_SR_Match_Rate.csv"),
        q14_cc_topk=pd.read_csv(root / "Merge/attribution/14b_contextcite_analysis/qwen14b_contextcite_topk_summary.csv"),
        q14_cc_fallback=pd.read_csv(root / "Merge/attribution/14b_contextcite_analysis/Qwen14B_Physician_Filtered_with_Sentences.csv"),
        q72_cc=pd.read_csv(root / "Dataset/Qwen72B/Qwen72B_ContextCite_Match.csv"),
        l70_cc=pd.read_csv(root / "Dataset/Llama70B/70B_ContextCite_Removal.csv"),
    )

    parsed = parse_tables(tex_path)
    print("Found case tables:", sorted(parsed.keys()))

    records = []
    for qid, table_rows in sorted(parsed.items()):
        expected, source_notes = expected_for_qid(qid, sorted(table_rows.keys()), src)
        mismatch = 0
        for sent in sorted(table_rows.keys()):
            for col in COLS:
                got = table_rows[sent][col]
                exp = expected[sent][col]
                if got != exp:
                    mismatch += 1
                    records.append(
                        {
                            "ID": qid,
                            "sentence": sent,
                            "column": col,
                            "expected": exp,
                            "got": got,
                            "source": source_notes[col],
                        }
                    )
        print(f"{qid}: rows={len(table_rows)} mismatches={mismatch}")
        if qid in {"ID0102", "ID0200"}:
            print(f"  note: Qwen-14B CC source used: {source_notes['Qwen-14B CC']}")

    out_csv = Path(args.out_csv)
    if records:
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"Mismatches written to: {out_csv}")
        print(f"Total mismatches: {len(records)}")
        return 1

    # Still write an empty report for reproducibility.
    pd.DataFrame(columns=["ID", "sentence", "column", "expected", "got", "source"]).to_csv(out_csv, index=False)
    print(f"No mismatches. Empty report written to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

