"""
933_Clinician_Student_Analysis.py
----------------------------------
Identical pipeline to Clinician_Student_Analysis.py, but restricted to the 933
Origins where answerable == "yes" (from Sentence_Label_Original_2k.csv).
All downstream statistics, IRR, and output files reflect this filtered set.
Uses the same ``2k_sentence_seperate.csv`` backfill for empty ``data_source_corr_*`` columns
before the 14B merge.

Also writes **q1 majority-vote accuracy** from ``Text Relevance Analysis Case View_022626.csv``
for the **933 Centaur Origins** (same tie-breaking and letter normalization as
``Apr1_2026_Data/compute_round2_933_mj.py``): all label rows per ``Origin`` (after the answerable
filter) vote on ``q1``; majority is compared to that cohort’s ``Correct answer``. Source is
``data_source_corr_x`` from ``Centaur_933_Clinician_Student_Majority_Vote.csv`` (one row per Origin).
If a Centaur Origin has no Text Relevance rows after filtering, ``q1_majority`` is NaN and
``matches_correct`` is False.

Outputs:

- ``933_answerable_TextRelevance_q1_majority_vote.csv`` — 933 rows (Centaur cohort).
- ``933_answerable_q1_MJ_accuracy_by_data_source_corr_x.csv`` — ``n_origins``, ``n_correct``, ``accuracy`` per source.
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
    """Majority vote on ``q1`` per Origin; random tie-break among modes (reproducible via ``rng``)."""
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
    if valid.sum() < 2:
        return np.nan
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
    pool = data[valid]
    n = len(pool)
    if n < 2:
        return np.nan
    D_e = 2.0 * np.var(pool, ddof=0)
    if D_e == 0:
        return 1.0
    return 1.0 - (D_o / D_e)

# ── Paths ────────────────────────────────────────────────────────────────────
# ``_script_dir`` set above for local imports.
_repo_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
csv_path            = os.path.join(_script_dir, "Text Relevance Analysis Case View_022626.csv")
path_2k_sentence    = os.path.join(_script_dir, "2k_sentence_seperate.csv")
_results            = os.path.join(_script_dir, "..", "results")
path_14b            = next(
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
_sentence_label_candidates = (
    os.path.join(_repo_root, "CentaurLab_Analysis", "Sentence_Label_Original_2k.csv"),
    os.path.join(
        _repo_root,
        "After_PT_Removal",
        "CentaurLab_Analysis",
        "results",
        "tables",
        "Sentence_Label_Original_2k.csv",
    ),
)
sentence_label_path = next((p for p in _sentence_label_candidates if os.path.isfile(p)), None)
output_path         = os.path.join(_script_dir, "933_Clinician_Student_Majority_Vote.csv")
centaur_933_path    = os.path.join(_script_dir, "Centaur_933_Clinician_Student_Majority_Vote.csv")
q1_mj_out_path      = os.path.join(_script_dir, "933_answerable_TextRelevance_q1_majority_vote.csv")
q1_mj_by_src_path = os.path.join(
    _script_dir, "933_answerable_q1_MJ_accuracy_by_data_source_corr_x.csv"
)

# ── Load answerable Origins ───────────────────────────────────────────────────
if sentence_label_path is None:
    raise FileNotFoundError(
        "Missing Sentence_Label_Original_2k.csv (need ID + answerable). Tried:\n  "
        + "\n  ".join(_sentence_label_candidates)
    )
sl_cols = pd.read_csv(sentence_label_path, nrows=0).columns
if "answerable" not in sl_cols:
    raise RuntimeError("'answerable' column missing from Sentence_Label_Original_2k.csv — run image_necessary_check.py first.")

answerable_df = pd.read_csv(sentence_label_path, usecols=["ID", "answerable"]).drop_duplicates(subset=["ID"])
answerable_origins = set(answerable_df.loc[answerable_df["answerable"] == "yes", "ID"])
print(f"Answerable Origins loaded: {len(answerable_origins)}")

# ── Load and enrich raw data ──────────────────────────────────────────────────
df_full = pd.read_csv(csv_path)
df_full = fill_empty_data_source_from_2k_sentence(df_full, path_2k_sentence)

# Add data_source_corr from 14B file
if path_14b is None:
    raise FileNotFoundError(
        "Missing 14B CSV for data_source_corr. Expected one of:\n"
        f"  {os.path.join(_results, '14B_Physician_Comparison_with_Stats.csv')}\n"
        f"  {os.path.join(_results, '14B_MatchRate.csv')}"
    )
df_14b = pd.read_csv(path_14b)
if "ID_corr" not in df_14b.columns or "data_source_corr" not in df_14b.columns:
    raise ValueError(f"{path_14b} must contain columns ID_corr and data_source_corr")
id_to_source = df_14b.drop_duplicates(subset=["ID_corr"])[["ID_corr", "data_source_corr"]]
if "data_source_corr" in df_full.columns:
    df_full = df_full.drop(columns=["data_source_corr"])
df_full = df_full.merge(id_to_source, left_on="Origin", right_on="ID_corr", how="left").drop(columns=["ID_corr"])
df_full = coalesce_data_source_corr_after_14b_merge(df_full)

# ── Filter to answerable Origins ──────────────────────────────────────────────
df_full = df_full[df_full["Origin"].isin(answerable_origins)].reset_index(drop=True)
print(f"df_full after answerable filter: {len(df_full)} rows, {df_full['Origin'].nunique()} unique Origins")

# ── q1 majority-vote accuracy (same logic as compute_round2_933_mj.py) ─────────
# Restrict to the **933 Centaur cohort** (``Centaur_933_Clinician_Student_Majority_Vote.csv``):
# all raters per Origin in the Text Relevance export (after answerable filter) vote on ``q1``;
# compare majority to ``Correct answer`` from the label rows (fallback: Centaur gold if no rows).
need_q1 = {"Origin", "q1", "Correct answer"}
if need_q1.issubset(df_full.columns):
    if not os.path.isfile(centaur_933_path):
        raise FileNotFoundError(
            f"Need Centaur 933 CSV for cohort + data_source_corr_x: {centaur_933_path}"
        )
    centaur_basis = pd.read_csv(
        centaur_933_path,
        usecols=["Origin", "data_source_corr_x", "Correct answer"],
        low_memory=False,
    )
    centaur_basis = centaur_basis.drop_duplicates(subset=["Origin"], keep="first")
    centaur_basis["Origin"] = centaur_basis["Origin"].astype(str).str.strip()
    cohort = set(centaur_basis["Origin"])

    df_tr = df_full[df_full["Origin"].astype(str).str.strip().isin(cohort)].copy()
    df_tr["Origin"] = df_tr["Origin"].astype(str).str.strip()

    _mj_rng = np.random.default_rng(42)
    mj_by_origin: dict[str, tuple] = {}
    for origin, g in df_tr.groupby("Origin", sort=False):
        mj = majority_q1(g["q1"], _mj_rng)
        correct = g["Correct answer"].iloc[0]
        if g["Correct answer"].nunique(dropna=False) > 1:
            print(
                f"  Warning: Origin {origin!r} has multiple Correct answer values; using first row.",
                flush=True,
            )
        mj_by_origin[origin] = (mj, correct)

    mj_rows = []
    for _, r in centaur_basis.iterrows():
        origin = r["Origin"]
        dscx = r["data_source_corr_x"]
        ca_c = r["Correct answer"]
        if origin in mj_by_origin:
            mj, correct = mj_by_origin[origin]
            match = (
                _norm_answer_q1(mj) == _norm_answer_q1(correct)
                if pd.notna(mj)
                else False
            )
            mj_rows.append(
                {
                    "Origin": origin,
                    "data_source_corr_x": dscx,
                    "q1_majority": mj,
                    "Correct answer": correct,
                    "matches_correct": bool(match),
                }
            )
        else:
            mj_rows.append(
                {
                    "Origin": origin,
                    "data_source_corr_x": dscx,
                    "q1_majority": np.nan,
                    "Correct answer": ca_c,
                    "matches_correct": False,
                }
            )

    n_no_rows = sum(1 for row in mj_rows if pd.isna(row["q1_majority"]))
    if n_no_rows:
        print(
            f"  Note: {n_no_rows} Centaur Origins have no Text Relevance rows after answerable "
            "filter — q1_majority NaN, matches_correct False.",
            flush=True,
        )

    df_q1_mj = pd.DataFrame(mj_rows)
    df_q1_mj = df_q1_mj[
        [
            "Origin",
            "data_source_corr_x",
            "q1_majority",
            "Correct answer",
            "matches_correct",
        ]
    ]
    df_q1_mj.to_csv(q1_mj_out_path, index=False)
    n_mj = len(df_q1_mj)
    acc_mj = float(df_q1_mj["matches_correct"].mean()) if n_mj else float("nan")
    print("\n--- q1 majority vote vs Correct answer (933 Centaur cohort, Text Relevance rows) ---")
    print(f"  Wrote {q1_mj_out_path} ({n_mj} Origins)")
    print(
        f"  Overall accuracy: {acc_mj * 100:.2f}% "
        f"({int(df_q1_mj['matches_correct'].sum())}/{n_mj})"
    )
    sub_src = df_q1_mj.dropna(subset=["data_source_corr_x"])
    by_src = (
        sub_src.groupby("data_source_corr_x", sort=True)
        .agg(
            n_origins=("matches_correct", "size"),
            n_correct=("matches_correct", "sum"),
        )
        .reset_index()
    )
    by_src["accuracy"] = by_src["n_correct"] / by_src["n_origins"]
    by_src["accuracy_pct"] = (100.0 * by_src["accuracy"]).round(2)
    by_src.to_csv(q1_mj_by_src_path, index=False)
    print(f"  Wrote {q1_mj_by_src_path}")
    print("  Accuracy by data_source_corr_x (mean = fraction of Origins correct):")
    for _, r in by_src.iterrows():
        print(
            f"    {str(r['data_source_corr_x']):12s}  "
            f"{int(r['n_correct'])}/{int(r['n_origins'])}  "
            f"{r['accuracy_pct']:.2f}%"
        )
else:
    print(
        f"\nSkipping q1 MJ accuracy: need columns {need_q1}; "
        f"have {set(df_full.columns)}",
        flush=True,
    )

# ── Build df_correct ─────────────────────────────────────────────────────────
response_correct = df_full["Response correct"]
mask = (response_correct == True) | (response_correct.astype(str).str.upper() == "TRUE")
df_correct = df_full[mask].copy().reset_index(drop=True)

# Rename q2–q22 → Sentence 1–Sentence 21
rename_map = {f"q{i}": f"Sentence {i - 1}" for i in range(2, 23)}
df_correct = df_correct.rename(columns=rename_map)

# Recode relevance labels
sentence_cols = [f"Sentence {i}" for i in range(1, 22)]
relevance_map = {"high relevance": 1, "low relevance": 0.5, "not relevant": 0, "not applicable": np.nan}
df_correct[sentence_cols] = df_correct[sentence_cols].replace(relevance_map)

# Weight: physician = 2, non-physician = 1
df_correct["weight"] = np.where(df_correct["Respondent type"] == "physician", 2, 1)

print(f"\nTotal rows in answerable df_full: {len(df_full)}")
print(f"Rows with Response correct == TRUE: {len(df_correct)}")

# Remove Origins with exactly 1 correct row
origin_counts = df_correct["Origin"].value_counts()
origins_with_one_row = origin_counts[origin_counts == 1].index
df_correct = df_correct[~df_correct["Origin"].isin(origins_with_one_row)].reset_index(drop=True)
print(f"\nAfter removing Origins with 1 correct row: {len(df_correct)} rows remaining")

# ── Duration stats ────────────────────────────────────────────────────────────
df_correct["Duration (min)"] = df_correct["Duration (s)"] / 60

def duration_stats(ser, label):
    d = ser.dropna()
    n = len(d)
    if n == 0:
        print(f"  {label}: n=0, no values")
        return
    mean_d, std_d = d.mean(), d.std()
    se = std_d / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low, ci_high = mean_d - t_crit * se, mean_d + t_crit * se
    print(f"  {label}: n={n}, mean={mean_d:.2f} min, std={std_d:.2f}, 95% CI=[{ci_low:.2f}, {ci_high:.2f}]")

print("\nDuration (minutes) in df_correct (answerable only):")
duration_stats(df_correct["Duration (min)"], "Total")
duration_stats(df_correct.loc[df_correct["Respondent type"] == "med student", "Duration (min)"], "med student")
duration_stats(df_correct.loc[df_correct["Respondent type"] == "physician",   "Duration (min)"], "physician")

# ── Origin / correct-row summary ─────────────────────────────────────────────
origin_counts = df_correct["Origin"].value_counts()
rows_per_origin = origin_counts.value_counts().sort_index(ascending=False)
print("\nSummary (CORRECT rows per Origin; 1-correct Origins excluded):")
for n_correct, n_origins in rows_per_origin.items():
    print(f"  {n_correct} correct rows: {n_origins} Origins")

# ── Build medpair (weighted mean per Origin) ──────────────────────────────────
def weighted_mean_by_origin(g):
    w = g["weight"]
    out = {}
    for col in sentence_cols:
        s = g[col]
        valid = s.notna()
        out[col] = (s[valid] * w[valid]).sum() / w[valid].sum() if valid.any() else np.nan
    return pd.Series(out)

medpair = df_correct.groupby("Origin").apply(weighted_mean_by_origin).reset_index()

# Add data_source_corr
origin_to_source = df_correct[["Origin", "data_source_corr"]].drop_duplicates("Origin")
medpair = medpair.merge(origin_to_source, on="Origin", how="left")

# ── data_source_corr table ────────────────────────────────────────────────────
print("\n--- data_source_corr: medpair (answerable only, CORRECT rows per Origin) ---")
table_rows = []
for src in sorted(medpair["data_source_corr"].dropna().unique()):
    origins_src = medpair.loc[medpair["data_source_corr"] == src, "Origin"]
    n_origins = len(origins_src)
    counts_src = origin_counts[origin_counts.index.isin(origins_src)]
    rpo = counts_src.value_counts().sort_index()
    row = {"data_source": src, "unique_Origins": n_origins}
    for n_rows, n_with in rpo.items():
        row[f"{int(n_rows)} correct rows"] = n_with
    table_rows.append(row)
if table_rows:
    df_src_tbl = pd.DataFrame(table_rows)
    row_cols = [c for c in df_src_tbl.columns if c not in ("data_source", "unique_Origins")]
    for c in row_cols:
        df_src_tbl[c] = df_src_tbl[c].fillna(0).astype(int)
    df_src_tbl = df_src_tbl.set_index("data_source")
    num_cols = sorted(row_cols, key=lambda x: int(x.split()[0]))
    print(df_src_tbl[["unique_Origins"] + num_cols].to_string())
if medpair["data_source_corr"].isna().any():
    print(f"  (missing data_source_corr): {medpair['data_source_corr'].isna().sum()} Origins")

# Physician origins in medpair
physician_origins = set(df_correct.loc[df_correct["Respondent type"] == "physician", "Origin"])
n_phys = len(set(medpair["Origin"]) & physician_origins)
print(f"\n  Unique Origins in medpair with ≥1 physician response: {n_phys}")

# ── IRR (Krippendorff's alpha) ────────────────────────────────────────────────
def irr_for_origin(origin_val):
    sub = df_correct.loc[df_correct["Origin"] == origin_val, sentence_cols]
    if len(sub) < 2:
        return np.nan
    return krippendorff_alpha_interval(sub.values.astype(float))

medpair["IRR"] = medpair["Origin"].map(irr_for_origin)

irr_vals = medpair["IRR"].dropna()
n_irr = len(irr_vals)
print("\n--- Krippendorff's alpha (IRR) summary — answerable only ---")
print(f"  n (Origins with valid IRR): {n_irr}")
if n_irr > 0:
    mean_irr, std_irr = irr_vals.mean(), irr_vals.std()
    print(f"  Mean:   {mean_irr:.4f}")
    print(f"  Std:    {std_irr:.4f}")
    if n_irr >= 2:
        se_irr = std_irr / np.sqrt(n_irr)
        t_crit = stats.t.ppf(0.975, df=n_irr - 1)
        print(f"  95% CI: [{mean_irr - t_crit*se_irr:.4f}, {mean_irr + t_crit*se_irr:.4f}]")
    print(f"  Min:    {irr_vals.min():.4f}")
    print(f"  Median: {irr_vals.median():.4f}")
    print(f"  Max:    {irr_vals.max():.4f}")
    print(f"  Q1 (25%): {irr_vals.quantile(0.25):.4f}")
    print(f"  Q3 (75%): {irr_vals.quantile(0.75):.4f}")
    print("\n  Interpretation (typical cutoffs):")
    print("    α ≥ 0.80 : good agreement")
    print("    0.67 ≤ α < 0.80 : tentative")
    print("    α < 0.67 : poor agreement")

# ── Clinician_Student_MJ_High_Sentences ──────────────────────────────────────
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
        text  = texts.get(f"sentence_{i}")
        if pd.notna(score) and score > 0.5 and pd.notna(text) and str(text).strip():
            high.append(str(text).strip())
    return " | ".join(high) if high else np.nan

medpair["Clinician_Student_MJ_High_Sentences"] = medpair.apply(build_high_sentences, axis=1)

# Add answerable column (all "yes" by construction)
medpair["answerable"] = "yes"

# ── Save ──────────────────────────────────────────────────────────────────────
medpair.to_csv(output_path, index=False)
print(f"\n933 answerable medpair ({len(medpair)} rows) saved to: {output_path}")
print(medpair[["Origin", "data_source_corr", "IRR", "answerable", "Clinician_Student_MJ_High_Sentences"]].head())
