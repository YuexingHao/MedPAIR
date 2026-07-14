#!/usr/bin/env python3
"""
Build slim round-1/2 exports (``Llama70B_ORIGINAL_predictions.csv``) and
``Llama70B_predictions_on_Trainee.csv`` from:

- ``Llama_70b_FIRST_ROUND.csv`` (Round 1 / original-stem run)
- ``Llama_70b_SECOND_ROUND.csv`` (Round 2 / trainee-stem run)
- ``llama_70b_both_rounds.csv`` for ``Origin`` and ``QA_ID`` alignment (``SECOND`` may omit
  ``Origin`` or duplicate it; we take ``Origin`` from ``both_rounds``).

Exports match the usual prediction layout (**no** ``answer_corr`` column). Sankey / pooled eval
use ``Llama70B_annotated_ORIGINAL_Accuracy.csv`` (``Llama70B_answer``) when present; this script
only rebuilds the slim CSV from ``Llama_70b_FIRST_ROUND.csv``.

Run::

  python After_PT_Removal/Llama-70B/results/scripts/build_llama_round_prediction_exports.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LLAMA_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = LLAMA_ROOT / "results" / "predictions"
REF_PATH = LLAMA_ROOT / "data" / "raw" / "14B_High_Low_Irr_Data_Source_.csv"
FIRST_PATH = LLAMA_ROOT / "Llama_70b_FIRST_ROUND.csv"
SECOND_PATH = LLAMA_ROOT / "Llama_70b_SECOND_ROUND.csv"
BOTH_PATH = LLAMA_ROOT / "llama_70b_both_rounds.csv"


def main() -> None:
    br = pd.read_csv(BOTH_PATH, usecols=["QA_ID", "Origin"])
    br["Origin"] = br["Origin"].astype(str).str.strip()

    ref = pd.read_csv(REF_PATH, usecols=["Origin", "data_source_df3"]).drop_duplicates(
        subset=["Origin"]
    )
    ref["Origin"] = ref["Origin"].astype(str).str.strip()

    f1 = pd.read_csv(FIRST_PATH)
    f2 = pd.read_csv(SECOND_PATH).drop(columns=["Origin"], errors="ignore")

    r1 = f1.merge(br, on="QA_ID", how="inner").merge(ref, on="Origin", how="left")
    r1["data_source"] = r1["data_source_df3"].fillna("").astype(str)

    r2 = f2.merge(br, on="QA_ID", how="inner").merge(ref, on="Origin", how="left")
    r2["data_source"] = r2["data_source_df3"].fillna("").astype(str)

    cols = ["QA_ID", "Origin", "data_source", "Raw_Response", "Extracted_Answer"]
    out_orig = PRED_DIR / "Llama70B_ORIGINAL_predictions.csv"
    out_tr = PRED_DIR / "Llama70B_predictions_on_Trainee.csv"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    r1[cols].to_csv(out_orig, index=False)
    r2[cols].to_csv(out_tr, index=False)
    print(f"Wrote {out_orig} ({len(r1)} rows) ← Llama_70b_FIRST_ROUND.csv")
    print(f"Wrote {out_tr} ({len(r2)} rows) ← Llama_70b_SECOND_ROUND.csv")


if __name__ == "__main__":
    main()
