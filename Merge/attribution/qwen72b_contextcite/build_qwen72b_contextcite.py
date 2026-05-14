"""Rebuild Qwen-72B ContextCite top-K labels.

Reads:  /orcd/home/002/yuexing/NeuRIPS25/Merge/attribution/qwen72b_contextcite/attributions_summary_combined.csv
Writes: qwen72b_attributions_normalized.csv, qwen72b_contextcite_topk_summary.csv,
        unmatched_sources.csv
All in this directory.

Restricts to QAs whose Origin appears in 933_Clinician_Student_Majority_Vote.csv
with at least one trainee-High sentence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from _lib_contextcite import build_topk_tables

HERE = Path(__file__).resolve().parent
RAW  = HERE / "attributions_summary_combined.csv"


def main():
    att = pd.read_csv(RAW, low_memory=False)
    build_topk_tables(att, model_label="qwen72b", out_dir=HERE,
                      restrict_to_933=True)


if __name__ == "__main__":
    main()
