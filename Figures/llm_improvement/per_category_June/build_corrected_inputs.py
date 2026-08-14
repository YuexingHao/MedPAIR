"""Generate corrected inputs for the combined MedPAIR figure.

The legacy module wants ``ExpertQA_933_by_data_source.csv`` and warns when it
falls back to ``PhysicianEval_Result_Report.csv``. That file has never existed
in this repo, so the fallback has always been used -- and its per-source columns
are not per-source accuracies: its ``MMLU`` column tracks overall accuracy
(0/35 rows match true MMLU) and its ``Jama`` column spans 1.5-36.4 where true
JAMA accuracy spans 66-89. This script rebuilds both inputs from the raw
prediction files so the per-dataset panels plot real per-dataset numbers.

Outputs (written next to this script):
  ExpertQA_933_by_data_source.csv   per-source accuracies, for the 4 panels
  PhysicianEval_933_corrected.csv   Expert QA (933) overall, for the Total panel

Caveats carried through deliberately:
  * Llama-70B "Original" only ever emits options A-D, even on 10-option
    MedXpert, so its baseline reflects a truncated-option run.
  * MedGemma-27B "Original" covers 375/933 and fails the provenance check.
  Both are retained on request; their baselines and deltas are not reliable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medpair_accuracy_lib as B  # noqa: E402

THIS = Path(__file__).resolve().parent

# figure label -> (model key in build_tab_comparison, prediction file)
ORIGINAL_FILES = {
    "GPT5": "gpt5_predictions_Original_Accuracy.csv",
    "GPT4o": "gpt4o_predictions_Original_Accuracy.csv",
    "Llama-70B": "Llama70B_ORIGINAL_predictions.csv",
    "Qwen-72B": "Qwen_72B_predictions_ORIGINAL.csv",
    "Qwen-14B": "Qwen_14B_predictions_ORIGINAL.csv",
    "MedGemma-27B": "MedGemma27B_predictions_ORIGINAL.csv",
}

# table block -> legend label the legacy module knows (spelling matters: it
# hard-codes "Llama-70b Removed" and "GPT5 Removed").
BLOCK_TO_LABEL = {
    "Trainee Removed (Low+Irr Labeled by Trainees)": "Trainee Removed",
    "Qwen-14B Removed": "Qwen-14B Removed",
    "Qwen-72B Removed": "Qwen-72B Removed",
    "Llama-70B Removed": "Llama-70b Removed",
    "GPT-4o Removed": "GPT-4o Removed",
    "GPT-5 Removed": "GPT5 Removed",
    "MedGemma-27B Removed": "MedGemma-27B Removed",
}

# build_tab_comparison model name -> name the legacy module expects on the x axis
MODEL_TO_FIGURE = {
    "GPT5": "GPT 5",
    "GPT4o": "GPT4o",
    "Llama-70B": "Llama-70B",
    "Qwen-72B": "Qwen 72B",
    "Qwen-14B": "Qwen-14B",
    "MedGemma-27B": "MedGemma-27B",
}


def _row(model: str, label: str, r: dict) -> dict:
    def g(k):
        v = r.get(k, float("nan"))
        return float(v) if v is not None else float("nan")

    return {
        "Base Model": MODEL_TO_FIGURE[model],
        "Low+Irr Labelers": label,
        "Total": g("general"),
        "MMLU": g("mmlu"),
        "Jama": g("jama"),
        "MedXpert": g("medxpert"),
        "Medbullets": g("medbullets"),
        "Total_STD": _std(g("general")),
        "MMLU_STD": _std(g("mmlu")),
        "JAMA_STD": _std(g("jama")),
        "MedXpert_STD": _std(g("medxpert")),
        "MedBullets_STD": _std(g("medbullets")),
        "n_scored": r.get("n", 0),
    }


def _std(acc_pct: float) -> float:
    if acc_pct is None or not np.isfinite(acc_pct):
        return float("nan")
    p = acc_pct / 100.0
    return 100.0 * float(np.sqrt(max(p * (1.0 - p), 0.0)))


def main() -> None:
    rows: list[dict] = []
    notes: list[str] = []

    for model, fname in ORIGINAL_FILES.items():
        r = B.evaluate(model, fname)
        if r["status"] != "ok":
            notes.append(f"Original / {model}: {r['status']}")
            continue
        rows.append(_row(model, "Original Accuracy", r))
        if r["n"] < 900:
            notes.append(f"Original / {model}: coverage {100 * r['n'] / 933:.1f}%")

    for block, label in BLOCK_TO_LABEL.items():
        for model, fname in B.BLOCKS[block].items():
            if (block, model) in B.EXCLUDED_CELLS:
                notes.append(f"{label} / {model}: excluded (failed provenance)")
                continue
            r = B.evaluate(model, fname)
            if r["status"] != "ok":
                notes.append(f"{label} / {model}: {r['status']}")
                continue
            rows.append(_row(model, label, r))

    df = pd.DataFrame(rows)

    ds = THIS / "ExpertQA_933_by_data_source.csv"
    df.to_csv(ds, index=False)
    print(f"wrote {ds}  ({len(df)} rows)")

    # Physician/category table: its "MMLU" column is read as Expert QA (933),
    # i.e. overall accuracy -- not the MMLU subset. Hard/Impossible cohorts are
    # not recomputed here; the Total panel only reads Expert QA (933).
    phys = df.copy()
    phys["MMLU"] = phys["Total"]
    phys["MMLU_STD"] = phys["Total_STD"]
    for c in ["Jama", "MedXpert", "Medbullets", "JAMA_STD", "MedXpert_STD", "MedBullets_STD"]:
        phys[c] = float("nan")
    pf = THIS / "PhysicianEval_933_corrected.csv"
    phys.to_csv(pf, index=False)
    print(f"wrote {pf}  ({len(phys)} rows)")

    if notes:
        print("\nnotes:")
        for n in notes:
            print("  -", n)


if __name__ == "__main__":
    main()
