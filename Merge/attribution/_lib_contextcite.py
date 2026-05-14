"""Shared helpers for the Llama-70B / Qwen-14B / Qwen-72B ContextCite rebuilds.

All three pipelines follow the same recipe:
  1. Join attribution rows to merged_2k_with_4k_ID on QA_ID <-> 4k_ID to
     recover Origin and the centaur_question text.
  2. Restrict to Origins that appear in 933_Clinician_Student_Majority_Vote.csv.
  3. Parse the Step-1 numbered excerpt, normalize Source text, map each
     attribution row to a sentence number.
  4. Aggregate per-sentence via max Score (fill missing sentences with 0).
  5. Per-QA min-max normalize Score -> [0, 1].
  6. K = # trainee-High sentences (MV score > 0.66).
  7. is_selected_topK = rank by normalized score <= K.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT = Path("/orcd/home/002/yuexing/NeuRIPS25")
MERGE_PATH = ROOT / "Merge/data/csv/merged_2k_with_4k_ID.csv"
MV_PATH    = ROOT / "Physician_Labels/Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv"


def normalize_qa_id(s: str) -> str:
    """Normalize 'Merge_Q50' / 'Merge Q50' / 'MergeQ50' to 'Merge Q50'."""
    if not isinstance(s, str):
        return ""
    m = re.match(r"^\s*Merge[\s_]*Q(\d+)\s*$", s)
    return f"Merge Q{m.group(1)}" if m else s.strip()


def parse_numbered_sentences(centaur_question: str):
    if not isinstance(centaur_question, str):
        return {}
    m = re.search(r"### Step 1: Read excerpt\s*\n(.*?)(?:### Step 2:|### Step 3:|\Z)",
                  centaur_question, re.DOTALL)
    body = m.group(1) if m else centaur_question
    out = {}
    for line in body.split("\n"):
        line = line.strip()
        mm = re.match(r"^(\d+)\.\s*(.*)$", line)
        if mm:
            out[int(mm.group(1))] = mm.group(2).strip()
    return out


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", s)).strip().lower()


def match_source_to_sentence(src_norm: str, sentences_norm: dict):
    if not src_norm:
        return None
    for n, sent in sentences_norm.items():
        if sent == src_norm:
            return n
    best_n, best_len = None, 0
    for n, sent in sentences_norm.items():
        if sent and sent in src_norm and len(sent) > best_len:
            best_n, best_len = n, len(sent)
    if best_n is not None:
        return best_n
    for n, sent in sentences_norm.items():
        if src_norm and src_norm in sent:
            return n
    return None


def load_trainee_K(restrict_to_mv_only: bool = True):
    """Return dict Origin -> (K, list-of-trainee-High-ids)."""
    mv = pd.read_csv(MV_PATH, low_memory=False)
    sent_cols = [c for c in mv.columns if re.fullmatch(r"Sentence \d+", c)]
    lkp = {}
    for _, r in mv.iterrows():
        origin = r["Origin"]
        highs = []
        for c in sent_cols:
            v = r[c]
            if pd.notna(v):
                try:
                    if float(v) > 0.66:
                        highs.append(int(c.split()[-1]))
                except Exception:
                    pass
        lkp[origin] = (len(highs), sorted(highs))
    return lkp


def build_qa_meta(att: pd.DataFrame):
    """Return dict QA_ID -> {Origin, data_source, sentences, sentences_norm}."""
    merge = pd.read_csv(MERGE_PATH, low_memory=False)
    merge["_4k_norm"] = merge["4k_ID"].apply(normalize_qa_id)
    merge_lkp = merge.set_index("_4k_norm")

    qa_meta = {}
    seen = set()
    for qa_id, _ in att.groupby("QA_ID"):
        key = normalize_qa_id(qa_id)
        if key in seen or key not in merge_lkp.index:
            continue
        seen.add(key)
        mrow = merge_lkp.loc[key]
        if isinstance(mrow, pd.DataFrame):
            mrow = mrow.iloc[0]
        sentences = parse_numbered_sentences(mrow["centaur_question"])
        qa_meta[key] = {
            "Origin": mrow["ID"],
            "data_source": mrow["data_source"],
            "sentence_num_expected": int(mrow["sentence_number"]),
            "sentences": sentences,
            "sentences_norm": {n: normalize_text(t) for n, t in sentences.items()},
        }
    return qa_meta


def build_topk_tables(att: pd.DataFrame, model_label: str, out_dir: Path,
                     restrict_to_933: bool = True):
    """Core pipeline.  Writes per-sentence + summary + unmatched CSVs to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalize QA_ID column so joins with merged_2k work for both
    # "Merge Q1" and "Merge_Q1" variants
    att = att.copy()
    att["_QA_norm"] = att["QA_ID"].apply(normalize_qa_id)

    qa_meta = build_qa_meta(att)
    mv_K = load_trainee_K()

    rows = []
    summary_rows = []
    unmatched_rows = []

    for qa_norm, g in att.groupby("_QA_norm"):
        meta = qa_meta.get(qa_norm)
        if meta is None:
            continue
        origin = meta["Origin"]

        # Only keep Origins that are in the 933 MV file with at least one High.
        if restrict_to_933:
            if origin not in mv_K or mv_K[origin][0] == 0:
                continue

        sentences_norm = meta["sentences_norm"]
        g = g.copy().reset_index(drop=True)
        g["sentence_num"] = g["Source"].apply(
            lambda s: match_source_to_sentence(normalize_text(s), sentences_norm))
        matched = g[g["sentence_num"].notna()].copy()
        if matched.empty:
            # Record as unmatched for diagnostics but skip summary emission.
            for _, r in g.iterrows():
                unmatched_rows.append({
                    "Origin": origin, "QA_ID": qa_norm,
                    "Score": r["Score"], "Source_preview": str(r["Source"])[:120],
                })
            continue
        matched["sentence_num"] = matched["sentence_num"].astype(int)
        per_sent = (matched.groupby("sentence_num", as_index=False)["Score"]
                    .max()
                    .sort_values("sentence_num"))
        # Fill missing sentences with Score 0
        for n in sorted(meta["sentences"].keys()):
            if n not in set(per_sent["sentence_num"]):
                per_sent = pd.concat(
                    [per_sent,
                     pd.DataFrame({"sentence_num": [n], "Score": [0.0]})],
                    ignore_index=True)
        per_sent = per_sent.sort_values("sentence_num").reset_index(drop=True)

        raw = per_sent["Score"].to_numpy(dtype=float)
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo) if hi > lo else np.full_like(raw, 0.5)
        per_sent["score_normalized"] = norm
        per_sent["rank"] = per_sent["score_normalized"].rank(
            method="first", ascending=False).astype(int)

        K, trainee_highs = mv_K.get(origin, (0, []))
        per_sent["is_selected_topK"] = per_sent["rank"] <= K
        selected_ids = sorted(
            int(x) for x in per_sent.loc[per_sent["is_selected_topK"], "sentence_num"])

        for _, r in per_sent.iterrows():
            rows.append({
                "Origin": origin,
                "QA_ID": qa_norm,
                "data_source": meta["data_source"],
                "sentence_num": int(r["sentence_num"]),
                "Score_raw": float(r["Score"]),
                "score_normalized": float(r["score_normalized"]),
                "rank": int(r["rank"]),
                "trainee_High_K": K,
                "is_selected_topK": bool(r["is_selected_topK"]),
            })

        summary_rows.append({
            "Origin": origin,
            "QA_ID": qa_norm,
            "data_source": meta["data_source"],
            "K_trainee_High": K,
            f"{model_label}_topK_sentence_ids": ",".join(str(i) for i in selected_ids),
            "trainee_High_sentence_ids": ",".join(str(i) for i in trainee_highs),
            "n_sentences_in_excerpt": len(meta["sentences"]),
            "n_attribution_sources": len(g),
            "n_matched_sources": int(g["sentence_num"].notna().sum()),
        })
        for _, r in g[g["sentence_num"].isna()].iterrows():
            unmatched_rows.append({
                "Origin": origin, "QA_ID": qa_norm,
                "Score": r["Score"], "Source_preview": str(r["Source"])[:120],
            })

    pd.DataFrame(rows).sort_values(["Origin", "sentence_num"]).to_csv(
        out_dir / f"{model_label}_attributions_normalized.csv", index=False)
    pd.DataFrame(summary_rows).sort_values("Origin").to_csv(
        out_dir / f"{model_label}_contextcite_topk_summary.csv", index=False)
    pd.DataFrame(unmatched_rows).to_csv(
        out_dir / "unmatched_sources.csv", index=False)

    print(f"[{model_label}] per-sentence rows: {len(rows)}")
    print(f"[{model_label}] QA summaries:      {len(summary_rows)}")
    print(f"[{model_label}] unmatched rows:    {len(unmatched_rows)}")
