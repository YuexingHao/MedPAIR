"""
Backfill empty ``data_source_corr_*`` columns in Text Relevance exports from ``2k_sentence_seperate.csv``.
Used by ``Clinician_Student_Analysis.py`` and ``933_Clinician_Student_Analysis.py``.
"""
from __future__ import annotations

import os

import pandas as pd


def _series_cell_empty(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    return s.isna() | (t == "") | (t.str.lower() == "nan")


def fill_empty_data_source_from_2k_sentence(
    df: pd.DataFrame,
    path_2k: str,
    *,
    origin_col: str = "Origin",
) -> pd.DataFrame:
    """
    Where ``data_source_corr_x``, ``data_source_corr_y``, and ``data_source_corr`` are all empty,
    set all three from ``2k_sentence_seperate.csv`` via ``Origin`` == ``ID`` (``data_source``,
    lowercased to match Centaur / 14B style).
    """
    cols = ("data_source_corr_x", "data_source_corr_y", "data_source_corr")
    if not all(c in df.columns for c in cols):
        return df
    if not os.path.isfile(path_2k):
        print(f"Note: skip 2k data_source backfill (missing file): {path_2k}", flush=True)
        return df

    empty_mask = (
        _series_cell_empty(df[cols[0]])
        & _series_cell_empty(df[cols[1]])
        & _series_cell_empty(df[cols[2]])
    )
    n_empty = int(empty_mask.sum())
    if n_empty == 0:
        return df

    df_2k = pd.read_csv(path_2k, usecols=["ID", "data_source"], low_memory=False)
    df_2k["ID"] = df_2k["ID"].astype(str).str.strip()
    df_2k["data_source"] = (
        df_2k["data_source"].astype(str).str.strip().str.lower()
    )
    id_to_ds = df_2k.drop_duplicates(subset=["ID"], keep="first").set_index("ID")["data_source"]

    mapped = df[origin_col].astype(str).str.strip().map(id_to_ds)
    fill_mask = empty_mask & mapped.notna()
    n_fill = int(fill_mask.sum())
    if n_fill:
        vals = mapped.loc[fill_mask].values
        for c in cols:
            df.loc[fill_mask, c] = vals
        print(
            f"Filled data_source from 2k_sentence_seperate.csv: {n_fill} rows "
            f"(of {n_empty} with all three source columns empty)",
            flush=True,
        )
    elif n_empty:
        print(
            f"Note: {n_empty} rows had empty source columns but no ID match in 2k_sentence_seperate.csv",
            flush=True,
        )
    return df


def coalesce_data_source_corr_after_14b_merge(df: pd.DataFrame) -> pd.DataFrame:
    """After merging 14B ``data_source_corr``, fill remaining NaNs from x / y columns."""
    if "data_source_corr" not in df.columns:
        return df
    out = df.copy()
    if "data_source_corr_x" in out.columns:
        out["data_source_corr"] = out["data_source_corr"].fillna(out["data_source_corr_x"])
    if "data_source_corr_y" in out.columns:
        out["data_source_corr"] = out["data_source_corr"].fillna(out["data_source_corr_y"])
    return out
