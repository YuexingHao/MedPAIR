"""Append the 298 MedBullets ContextCite rows produced by the SLURM job to
`Merge/data/csv/llama70b_attributions_summary.csv`, then refresh the top-K
summary, the concordance bar chart, and the heatmap.

Run *after* run_medbullets.sbatch has finished:
    python append_medbullets_to_summary.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path("/orcd/home/002/yuexing/NeuRIPS25")
MAIN_CSV = ROOT / "Merge/data/csv/llama70b_attributions_summary.csv"
NEW_CSV = HERE / "llama70b_attribution_scores_medbullets/attributions_summary_1035_1332.csv"


def main() -> None:
    if not NEW_CSV.exists():
        print(f"[!] missing {NEW_CSV} — did the SLURM job finish?")
        sys.exit(1)

    main_df = pd.read_csv(MAIN_CSV, low_memory=False)
    new_df = pd.read_csv(NEW_CSV, low_memory=False)

    # Align columns: the per-QA frames carry Score, Source, QA_ID, Extracted_Answer
    # (and sometimes Raw_Response). The main summary file does NOT carry
    # Raw_Response, so drop it before concat to keep the schema stable.
    keep = [c for c in main_df.columns if c in new_df.columns]
    new_aligned = new_df[keep].copy()

    # Drop any QA_IDs that already exist in main_df, so re-running is idempotent.
    overlap = set(main_df["QA_ID"].astype(str)) & set(new_aligned["QA_ID"].astype(str))
    if overlap:
        print(f"[i] {len(overlap)} QA_IDs already in main file — keeping main rows.")
        new_aligned = new_aligned[~new_aligned["QA_ID"].astype(str).isin(overlap)]

    backup = MAIN_CSV.with_suffix(".csv.bak_pre_medbullets")
    if not backup.exists():
        shutil.copy2(MAIN_CSV, backup)
        print(f"[i] backup -> {backup}")

    appended = pd.concat([main_df, new_aligned], ignore_index=True)
    appended.to_csv(MAIN_CSV, index=False)
    print(f"[ok] appended {len(new_aligned)} rows -> {MAIN_CSV}")
    print(f"     unique QAs: {appended['QA_ID'].nunique()}  (was {main_df['QA_ID'].nunique()})")

    # ---- refresh downstream: top-K summary, bar, heatmap, appendix CC cols ----
    print("[..] rebuilding top-K summary")
    subprocess.check_call([sys.executable, str(HERE / "build_llama70b_contextcite.py")])

    print("[..] regenerating concordance bar + heatmap CSVs + figures")
    subprocess.check_call([sys.executable, str(ROOT / "Figures/concordance_rate/update_cc_concordance_and_heatmap.py")])

    print("[..] refreshing Appendix.tex CC labels")
    subprocess.check_call([sys.executable, "/tmp/update_appendix_cc_labels.py"])


if __name__ == "__main__":
    main()
