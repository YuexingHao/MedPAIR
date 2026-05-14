"""
Clinician_Student_Analysis.py
-----------------------------
Text Relevance pipeline: medpair, IRR, HardQA merge, and optional answerable export.

Rows where ``data_source_corr_x``, ``data_source_corr_y``, and ``data_source_corr`` are all empty
are filled from ``2k_sentence_seperate.csv`` (``Origin`` = ``ID``, ``data_source``) before the
14B merge; any remaining missing ``data_source_corr`` after 14B is filled from x/y columns.

Also writes **q1 majority-vote accuracy** over the **full** Text Relevance export (after 14B
``data_source_corr`` merge): same tie-breaking and letter normalization as
``Apr1_2026_Data/compute_round2_933_mj.py``. Per-source table uses ``data_source_corr`` from the
14B join (not the 933 Centaur table).

Outputs:

- ``Clinician_Student_TextRelevance_q1_majority_vote.csv``
- ``Clinician_Student_q1_MJ_accuracy_by_data_source.csv``
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from text_relevance_data_source_fill import (  # noqa: E402
    coalesce_data_source_corr_after_14b_merge,
    fill_empty_data_source_from_2k_sentence,
)


def _norm_answer_q1(x) -> str:
    """Same letter normalization as ``Apr1_2026_Data/compute_round2_933_mj.py``."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s.rstrip("'")


def majority_q1(series: pd.Series, rng: np.random.Generator):
    """Majority vote on ``q1`` per Origin; random tie-break among modes (``rng``, default 42)."""
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) == 0:
        return np.nan
    counts = Counter(vals)
    best = max(counts.values())
    tied = [k for k, v in counts.items() if v == best]
    if len(tied) == 1:
        return tied[0]
    return str(rng.choice(tied))


def krippendorff_alpha_interval(data):
    """Krippendorff's alpha for interval data. data: (n_raters, n_units), NaN = missing."""
    data = np.asarray(data, dtype=float)
    n_raters, n_units = data.shape
    valid = ~np.isnan(data)
    n_total = valid.sum()
    if n_total < 2:
        return np.nan

    # D_o: observed disagreement = mean squared difference within units
    D_o = 0.0
    total_pairs = 0
    for u in range(n_units):
        vals = data[:, u]
        vals = vals[~np.isnan(vals)]
        n_u = len(vals)
        if n_u < 2:
            continue
        for i in range(n_u):
            for j in range(i + 1, n_u):
                D_o += (vals[i] - vals[j]) ** 2
        total_pairs += n_u * (n_u - 1) / 2
    if total_pairs == 0:
        return np.nan
    D_o /= total_pairs

    # D_e: expected disagreement = 2 * variance of pooled values
    pool = data[valid]
    n = len(pool)
    if n < 2:
        return np.nan
    D_e = 2.0 * np.var(pool, ddof=0)
    if D_e == 0:
        return 1.0
    return 1.0 - (D_o / D_e)

# Read the CSV file (same folder as this script); ``_script_dir`` set above for local imports.
_workspace_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
csv_path = os.path.join(_script_dir, "Text Relevance Analysis Case View_022626.csv")
path_2k_sentence = os.path.join(_script_dir, "2k_sentence_seperate.csv")
df_full = pd.read_csv(csv_path)
df_full = fill_empty_data_source_from_2k_sentence(df_full, path_2k_sentence)

hardqa_path = os.path.join(_script_dir, "HardQA_Clinician_Student_Majority_Vote.csv")
hard_ids = (
    set(pd.read_csv(hardqa_path, usecols=["ID"])["ID"].astype(str).str.strip())
    if os.path.isfile(hardqa_path)
    else set()
)

# Add data_source_corr from 14B file: match Origin (Text Relevance) with ID_corr (14B)
_results = os.path.join(_script_dir, "..", "results")
path_14b = next(
    (
        p
        for p in (
            os.path.join(_results, "14B_Physician_Comparison_with_Stats.csv"),
            os.path.join(_results, "14B_MatchRate.csv"),
        )
        if os.path.isfile(p)
    ),
    None,
)
if path_14b is None:
    raise FileNotFoundError(
        "Missing 14B CSV for data_source_corr. Expected one of:\n"
        f"  {os.path.join(_results, '14B_Physician_Comparison_with_Stats.csv')}\n"
        f"  {os.path.join(_results, '14B_MatchRate.csv')}"
    )
df_14b = pd.read_csv(path_14b)
if "ID_corr" not in df_14b.columns or "data_source_corr" not in df_14b.columns:
    raise ValueError(f"{path_14b} must contain columns ID_corr and data_source_corr")
# One row per ID_corr (keep first if duplicates)
id_to_source = df_14b.drop_duplicates(subset=["ID_corr"])[["ID_corr", "data_source_corr"]]
# Drop existing column if present (e.g. from previous run) so merge doesn't create _x/_y
if "data_source_corr" in df_full.columns:
    df_full = df_full.drop(columns=["data_source_corr"])
df_full = df_full.merge(id_to_source, left_on="Origin", right_on="ID_corr", how="left").drop(columns=["ID_corr"])
df_full = coalesce_data_source_corr_after_14b_merge(df_full)

# ── q1 majority-vote accuracy (full cohort; data_source from 14B ``data_source_corr``) ──
q1_mj_vote_path = os.path.join(_script_dir, "Clinician_Student_TextRelevance_q1_majority_vote.csv")
q1_mj_by_src_path = os.path.join(
    _script_dir, "Clinician_Student_q1_MJ_accuracy_by_data_source.csv"
)
_need_q1 = {"Origin", "q1", "Correct answer"}
if _need_q1.issubset(df_full.columns):
    df_q1 = df_full.copy()
    df_q1["Origin"] = df_q1["Origin"].astype(str).str.strip()
    _mj_rng = np.random.default_rng(42)
    mj_rows_q1 = []
    for origin, g in df_q1.groupby("Origin", sort=False):
        mj = majority_q1(g["q1"], _mj_rng)
        correct = g["Correct answer"].iloc[0]
        if g["Correct answer"].nunique(dropna=False) > 1:
            print(
                f"  Warning (q1 MJ): Origin {origin!r} has multiple Correct answer values; "
                "using first row.",
                flush=True,
            )
        dsc_series = (
            g["data_source_corr"].dropna().astype(str).str.strip()
            if "data_source_corr" in g.columns
            else pd.Series(dtype=object)
        )
        data_src = dsc_series.iloc[0] if len(dsc_series) > 0 else np.nan
        match = (
            _norm_answer_q1(mj) == _norm_answer_q1(correct) if pd.notna(mj) else False
        )
        mj_rows_q1.append(
            {
                "Origin": origin,
                "data_source_corr": data_src,
                "q1_majority": mj,
                "Correct answer": correct,
                "matches_correct": bool(match),
            }
        )
    df_q1_out = pd.DataFrame(mj_rows_q1).sort_values("Origin").reset_index(drop=True)
    df_q1_out.to_csv(q1_mj_vote_path, index=False)
    n_q1 = len(df_q1_out)
    acc_q1 = float(df_q1_out["matches_correct"].mean()) if n_q1 else float("nan")
    print("\n--- q1 majority vote vs Correct answer (full Text Relevance cohort) ---")
    print(f"  Wrote {q1_mj_vote_path} ({n_q1} Origins)")
    print(
        f"  Overall accuracy: {acc_q1 * 100:.2f}% "
        f"({int(df_q1_out['matches_correct'].sum())}/{n_q1})"
    )
    sub_ds = df_q1_out.dropna(subset=["data_source_corr"])
    by_src_q1 = (
        sub_ds.groupby("data_source_corr", sort=True)
        .agg(
            n_origins=("matches_correct", "size"),
            n_correct=("matches_correct", "sum"),
        )
        .reset_index()
    )
    by_src_q1["accuracy"] = by_src_q1["n_correct"] / by_src_q1["n_origins"]
    by_src_q1["accuracy_pct"] = (100.0 * by_src_q1["accuracy"]).round(2)
    by_src_q1.to_csv(q1_mj_by_src_path, index=False)
    print(f"  Wrote {q1_mj_by_src_path}")
    print("  Accuracy by data_source_corr (14B):")
    for _, r in by_src_q1.iterrows():
        print(
            f"    {str(r['data_source_corr']):14s}  "
            f"{int(r['n_correct'])}/{int(r['n_origins'])}  "
            f"{r['accuracy_pct']:.2f}%"
        )
    if df_q1_out["data_source_corr"].isna().any():
        print(
            f"    (excluded {int(df_q1_out['data_source_corr'].isna().sum())} Origins "
            "with missing data_source_corr from by-source table)",
            flush=True,
        )
else:
    print(
        f"\nSkipping q1 MJ: need columns {_need_q1}; columns={list(df_full.columns)}",
        flush=True,
    )

# Write the enriched dataframe back to the CSV so the new column appears on disk
df_full.to_csv(csv_path, index=False)

# Create dataframe where "Response correct" is TRUE (handles both boolean and string "TRUE")
response_correct = df_full["Response correct"]
mask = (response_correct == True) | (response_correct.astype(str).str.upper() == "TRUE")
df_correct = df_full[mask].copy()

# Optional: reset index after filtering
df_correct = df_correct.reset_index(drop=True)

# Rename q2–q22 to Sentence 1–Sentence 21
rename_map = {f"q{i}": f"Sentence {i - 1}" for i in range(2, 23)}
df_correct = df_correct.rename(columns=rename_map)

# Recode Sentence 1–21: high relevance=1, low relevance=0.5, not relevant=0, not applicable=empty
sentence_cols = [f"Sentence {i}" for i in range(1, 22)]
relevance_map = {
    "high relevance": 1,
    "low relevance": 0.5,
    "not relevant": 0,
    "not applicable": np.nan,
}
df_correct[sentence_cols] = df_correct[sentence_cols].replace(relevance_map)

# Weight: physician = 2, non-physician = 1 (for weighted mean in medpair so averages stay in [0, 1])
df_correct["weight"] = np.where(df_correct["Respondent type"] == "physician", 2, 1)

# (Previously multiplied physician Sentence values by 2; now double weight is applied in medpair weighted mean instead.)

# HardQA: all Text-Relevance raters for HardQA Origins (quiz "Response correct" not required — HardQA items
# rarely have correct quiz rows, so restricting to correct-only would leave ~700 IDs without medpair).
if hard_ids:
    hnorm = df_full["Origin"].astype(str).str.strip()
    df_hard_tr = df_full.loc[hnorm.isin(hard_ids)].copy().reset_index(drop=True)
    df_hard_tr = df_hard_tr.rename(columns=rename_map)
    df_hard_tr[sentence_cols] = df_hard_tr[sentence_cols].replace(relevance_map)
    df_hard_tr["weight"] = np.where(df_hard_tr["Respondent type"] == "physician", 2, 1)
else:
    df_hard_tr = None

print(f"Total rows in original file: {len(df_full)}")
print(f"Rows with Response correct == TRUE: {len(df_correct)}")

# Count how many rows share the same "Origin" value
origin_counts = df_correct["Origin"].value_counts()

# Remove Origins that have exactly 1 correct row
origins_with_one_row = origin_counts[origin_counts == 1].index
df_correct = df_correct[~df_correct["Origin"].isin(origins_with_one_row)].reset_index(drop=True)
print(f"\nAfter removing Origins with 1 correct row: {len(df_correct)} rows remaining")

# Duration: convert to minutes (divide by 60), then mean, std, 95% CI for total / med student / physician
df_correct["Duration (min)"] = df_correct["Duration (s)"] / 60

def duration_stats(ser, label):
    d = ser.dropna()
    n = len(d)
    if n == 0:
        print(f"  {label}: n=0, no values")
        return
    mean_d = d.mean()
    std_d = d.std()
    se = std_d / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low, ci_high = mean_d - t_crit * se, mean_d + t_crit * se
    print(f"  {label}: n={n}, mean={mean_d:.2f} min, std={std_d:.2f}, 95% CI=[{ci_low:.2f}, {ci_high:.2f}]")

print("\nDuration (minutes) in df_correct:")
duration_stats(df_correct["Duration (min)"], "Total")
duration_stats(df_correct.loc[df_correct["Respondent type"] == "med student", "Duration (min)"], "med student")
duration_stats(df_correct.loc[df_correct["Respondent type"] == "physician", "Duration (min)"], "physician")

# Recompute counts and summary for filtered df_correct
origin_counts = df_correct["Origin"].value_counts()
print("\nOrigin value counts:")
print(origin_counts)

# Summary: how many Origins have 1, 2, 3, ... correct rows (after removal of 1-correct Origins)
# "N correct rows" = N rows with Response correct==TRUE for that Origin.
rows_per_origin = origin_counts.value_counts().sort_index(ascending=False)
print("\nSummary (CORRECT rows per Origin; 1-correct Origins excluded):")
for n_correct, n_origins in rows_per_origin.items():
    print(f"  {n_correct} correct rows: {n_origins} Origins")

# New dataframe: one row per unique Origin; Sentence 1–21 = weighted mean (physician weight 2, else 1) so result is in [0, 1]
def weighted_mean_by_origin(g):
    w = g["weight"]
    out = {}
    for col in sentence_cols:
        s = g[col]
        valid = s.notna()
        if not valid.any():
            out[col] = np.nan
        else:
            out[col] = (s[valid] * w[valid]).sum() / w[valid].sum()
    return pd.Series(out)

medpair = df_correct.groupby("Origin").apply(weighted_mean_by_origin).reset_index()

# Add data_source_corr to medpair (one value per Origin from df_correct)
origin_to_source = df_correct[["Origin", "data_source_corr"]].drop_duplicates("Origin")
medpair = medpair.merge(origin_to_source, on="Origin", how="left")

# data_source_corr table: only medpair Origins (correct rows; 1-correct excluded). Rows = correct rows per Origin.
print("\n--- data_source_corr: medpair only (CORRECT rows per Origin) ---")
table_rows = []
sources = sorted(medpair["data_source_corr"].dropna().unique())
for src in sources:
    medpair_src = medpair[medpair["data_source_corr"] == src]
    origins_src = medpair_src["Origin"]
    n_origins = len(origins_src)
    # Correct rows per Origin for these Origins (from df_correct)
    counts_src = origin_counts[origin_counts.index.isin(origins_src)]
    rows_per_origin = counts_src.value_counts().sort_index()
    row = {"data_source": src, "unique_Origins": n_origins}
    for n_rows, n_origins_with in rows_per_origin.items():
        row[f"{int(n_rows)} correct rows"] = n_origins_with
    table_rows.append(row)
if len(table_rows) > 0:
    df_source_table = pd.DataFrame(table_rows)
    row_cols = [c for c in df_source_table.columns if c not in ("data_source", "unique_Origins")]
    for c in row_cols:
        df_source_table[c] = df_source_table[c].fillna(0).astype(int)
    df_source_table = df_source_table.set_index("data_source")
    num_cols = sorted([c for c in row_cols], key=lambda x: int(x.split()[0]))
    df_source_table = df_source_table[["unique_Origins"] + num_cols]
    print(df_source_table.to_string())
if medpair["data_source_corr"].isna().any():
    n_missing = medpair["data_source_corr"].isna().sum()
    print(f"\n  (missing data_source_corr): {n_missing} Origins in medpair")

# Unique Origins in medpair that have at least one physician response (Respondent type == physician) in df_correct
physician_origins_in_correct = set(df_correct.loc[df_correct["Respondent type"] == "physician", "Origin"].unique())
medpair_origins_set = set(medpair["Origin"])
n_medpair_physician_origins = len(medpair_origins_set & physician_origins_in_correct)
print(f"\n  Unique Origins in medpair with ≥1 physician response: {n_medpair_physician_origins}")

# Inter-rater reliability (IRR) per Origin: Krippendorff's alpha across Sentence 1–21 (interval; rows=raters, cols=units)
def irr_for_origin(origin_val):
    sub = df_correct.loc[df_correct["Origin"] == origin_val, sentence_cols]
    if len(sub) < 2:
        return np.nan
    data = sub.values.astype(float)
    return krippendorff_alpha_interval(data)

medpair["IRR"] = medpair["Origin"].map(irr_for_origin)

# Summary of Krippendorff's alpha (IRR) in medpair
irr_vals = medpair["IRR"].dropna()
n_irr = len(irr_vals)
print("\n--- Krippendorff's alpha (IRR) summary in medpair ---")
print(f"  n (Origins with valid IRR): {n_irr}")
if n_irr == 0:
    print("  No valid IRR values.")
else:
    mean_irr = irr_vals.mean()
    std_irr = irr_vals.std()
    print(f"  Mean:   {mean_irr:.4f}")
    print(f"  Std:    {std_irr:.4f}")
    if n_irr >= 2:
        se_irr = std_irr / np.sqrt(n_irr)
        t_crit = stats.t.ppf(0.975, df=n_irr - 1)
        ci_low = mean_irr - t_crit * se_irr
        ci_high = mean_irr + t_crit * se_irr
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Min:    {irr_vals.min():.4f}")
    print(f"  Median: {irr_vals.median():.4f}")
    print(f"  Max:    {irr_vals.max():.4f}")
    print(f"  Q1 (25%): {irr_vals.quantile(0.25):.4f}")
    print(f"  Q3 (75%): {irr_vals.quantile(0.75):.4f}")
    # Common interpretation (Krippendorff)
    print("\n  Interpretation (typical cutoffs):")
    print("    α ≥ 0.80 : good agreement")
    print("    0.67 ≤ α < 0.80 : tentative")
    print("    α < 0.67 : poor agreement")

# Save medpair to CSV
output_path = os.path.join(_script_dir, "Clinician_Student_Majority_Vote.csv")
medpair.to_csv(output_path, index=False)
print(f"\nmedpair saved to: {output_path}")

# Build "Clinician_Student_MJ_High_Sentences": sentence text for every Sentence X where score > 0.5
# Source: Sentence_Label_Original_2k.csv has sentence_1..21 text for all 2k Origins (join on ID = Origin)
_sentence_label_candidates = [
    os.path.join(_workspace_root, "CentaurLab_Analysis", "Sentence_Label_Original_2k.csv"),
    os.path.join(
        _workspace_root,
        "After_PT_Removal",
        "CentaurLab_Analysis",
        "results",
        "tables",
        "Sentence_Label_Original_2k.csv",
    ),
]
sentence_label_path = next((p for p in _sentence_label_candidates if os.path.isfile(p)), None)
if sentence_label_path is None:
    raise FileNotFoundError(
        "Sentence_Label_Original_2k.csv not found; tried:\n  " + "\n  ".join(_sentence_label_candidates)
    )
text_cols = [f"sentence_{i}" for i in range(1, 22)]
df_texts = pd.read_csv(sentence_label_path, usecols=["ID"] + text_cols)
df_texts = df_texts.drop_duplicates(subset=["ID"]).set_index("ID")

def build_high_sentences(row):
    origin = row["Origin"]
    if origin not in df_texts.index:
        return np.nan
    texts = df_texts.loc[origin]
    high = []
    for i in range(1, 22):
        score = row.get(f"Sentence {i}")
        text = texts.get(f"sentence_{i}")
        if pd.notna(score) and score >= 0.5 and pd.notna(text) and str(text).strip():
            high.append(str(text).strip())
    return " | ".join(high) if high else np.nan

medpair["Clinician_Student_MJ_High_Sentences"] = medpair.apply(build_high_sentences, axis=1)

# Match medpair with Sentence_Label_Original_2k.csv "answerable" column
answerable_lookup = df_texts.reset_index()[["ID"]].copy()
answerable_lookup = pd.read_csv(sentence_label_path, usecols=["ID", "answerable"]) \
    if "answerable" in pd.read_csv(sentence_label_path, nrows=0).columns \
    else None

if answerable_lookup is not None:
    answerable_lookup = answerable_lookup.drop_duplicates(subset=["ID"])
    medpair = medpair.merge(answerable_lookup, left_on="Origin", right_on="ID", how="left").drop(columns=["ID"])
    print("\n--- answerable summary in medpair ---")
    total = len(medpair)
    n_yes = (medpair["answerable"] == "yes").sum()
    n_no  = (medpair["answerable"] == "no").sum()
    n_na  = medpair["answerable"].isna().sum()
    print(f"  answerable=yes : {n_yes} / {total} ({100*n_yes/total:.1f}%)")
    print(f"  answerable=no  : {n_no}  / {total} ({100*n_no/total:.1f}%)")
    print(f"  missing        : {n_na}  / {total}")
else:
    print("\n  'answerable' column not yet in Sentence_Label_Original_2k.csv — run image_necessary_check.py first.")

# Re-save full medpair with all new columns
medpair.to_csv(output_path, index=False)
print(f"\nmedpair saved to: {output_path}")

# Save answerable-only subset (933 rows where answerable == "yes")
if answerable_lookup is not None:
    medpair_answerable = medpair[medpair["answerable"] == "yes"].reset_index(drop=True)
    answerable_output_path = os.path.join(_script_dir, "933_Clinician_Student_Majority_Vote.csv")
    medpair_answerable.to_csv(answerable_output_path, index=False)
    print(f"\nAnswerable-only medpair ({len(medpair_answerable)} rows) saved to: {answerable_output_path}")
    print(medpair_answerable[["Origin", "answerable", "Clinician_Student_MJ_High_Sentences"]].head())

# --- HardQA: weighted medpair from all Text-Relevance raters for HardQA Origins; merge onto HardQA CSV ---
if hard_ids and os.path.isfile(hardqa_path) and df_hard_tr is not None and len(df_hard_tr) > 0:
    df_hard = pd.read_csv(hardqa_path)
    stale_tr = [c for c in df_hard.columns if c.endswith("_tr")]
    if stale_tr:
        df_hard = df_hard.drop(columns=stale_tr)
    _mj_meta = ["data_source_corr", "IRR", "Clinician_Student_MJ_High_Sentences", "answerable"]
    df_hard = df_hard.drop(columns=[c for c in _mj_meta if c in df_hard.columns])
    df_h = df_hard_tr
    n_hard = df_hard["ID"].nunique()
    hardqa_medpair = df_h.groupby("Origin").apply(weighted_mean_by_origin).reset_index()
    origin_to_source_h = df_h[["Origin", "data_source_corr"]].drop_duplicates("Origin")
    hardqa_medpair = hardqa_medpair.merge(origin_to_source_h, on="Origin", how="left")

    def irr_for_origin_h(origin_val):
        sub = df_h.loc[df_h["Origin"] == origin_val, sentence_cols]
        if len(sub) < 2:
            return np.nan
        return krippendorff_alpha_interval(sub.values.astype(float))

    hardqa_medpair["IRR"] = hardqa_medpair["Origin"].map(irr_for_origin_h)
    hardqa_medpair["Clinician_Student_MJ_High_Sentences"] = hardqa_medpair.apply(
        build_high_sentences, axis=1
    )
    if answerable_lookup is not None:
        hardqa_medpair = hardqa_medpair.merge(
            answerable_lookup, left_on="Origin", right_on="ID", how="left"
        ).drop(columns=["ID"])

    mj_hard = hardqa_medpair.assign(ID=hardqa_medpair["Origin"].astype(str).str.strip())
    rename_sent = {f"Sentence {i}": f"sentence_{i}" for i in range(1, 22)}
    mj_hard = mj_hard.rename(columns=rename_sent)
    mj_cols = (
        ["ID"]
        + [f"sentence_{i}" for i in range(1, 22)]
        + ["data_source_corr", "IRR", "Clinician_Student_MJ_High_Sentences", "answerable"]
    )
    mj_cols = [c for c in mj_cols if c in mj_hard.columns]
    mj_hard = mj_hard[mj_cols]
    merged_hard = df_hard.merge(mj_hard, on="ID", how="left", suffixes=("", "_tr"))
    for i in range(1, 22):
        c = f"sentence_{i}"
        ct = f"{c}_tr"
        if ct in merged_hard.columns:
            merged_hard[c] = merged_hard[ct].where(merged_hard[ct].notna(), merged_hard[c])
            merged_hard = merged_hard.drop(columns=[ct])
    merged_hard.to_csv(hardqa_path, index=False)
    n_tr = mj_hard["ID"].nunique()
    print(
        f"\nHardQA majority-vote merge: {n_tr} / {n_hard} IDs from Text Relevance "
        f"(all raters; physician weight 2); IRR NaN when only one rater; {hardqa_path}"
    )
elif hard_ids and os.path.isfile(hardqa_path):
    print(f"\nHardQA majority-vote merge: no Text Relevance rows for HardQA IDs; skipped: {hardqa_path}")
elif not os.path.isfile(hardqa_path):
    print(f"\n(skip HardQA merge) not found: {hardqa_path}")