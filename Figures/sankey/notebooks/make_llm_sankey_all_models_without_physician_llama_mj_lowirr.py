#!/usr/bin/env python3
"""Regenerate no-physician Sankey with Llama-70B R1 from MJ_LowIRR.

This wraps the existing compiled generator used for
`llm_sankey_all_models_without_physician_relevant.*`, overriding only
Llama-70B round-1 inputs to:
  After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_MJ_LowIRR.csv

Round-2 behavior remains unchanged (trainee_irr_removed mapping).
Outputs are copied to:
  - figures/llm_sankey_all_models_without_physician.html
  - figures/llm_sankey_all_models_without_physician.png
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import sys
from pathlib import Path


def _load_base_module():
    pyc = (
        Path(__file__).resolve().parent
        / "__pycache__"
        / "make_llm_sankey_all_models_without_physician_relevant.cpython-313.pyc"
    )
    if not pyc.exists():
        raise FileNotFoundError(f"Missing compiled generator: {pyc}")
    loader = importlib.machinery.SourcelessFileLoader("sankey_no_phys_patch", str(pyc))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_base_module()
    orig_load_round_letters = mod.load_round_letters

    def load_round_letters_patched(repo: Path, model_name: str, round_kind: str):
        # Keep all original behavior except Llama-70B R1 source.
        if model_name == "Llama-70B" and round_kind == "r1":
            p = (
                repo
                / "After_PT_Removal"
                / "Llama-70B"
                / "results"
                / "predictions"
                / "Llama70B_predictions_on_MJ_LowIRR.csv"
            )
            df = mod.pd.read_csv(
                p,
                usecols=["Origin", "llama70b_direct_prediction", "llama70b_extracted_answer"],
            ).copy()
            df["letter"] = df["llama70b_direct_prediction"].apply(mod.extract_letter)
            fb = df["llama70b_extracted_answer"].apply(mod.extract_letter)
            df["letter"] = df["letter"].where(df["letter"].notna(), fb)
            df["Origin"] = df["Origin"].astype(str).str.strip()
            return df[["Origin", "letter"]].drop_duplicates(subset=["Origin"])
        return orig_load_round_letters(repo, model_name=model_name, round_kind=round_kind)

    mod.load_round_letters = load_round_letters_patched

    # Ensure Llama is included.
    if "--include-llama" not in sys.argv:
        sys.argv.append("--include-llama")

    mod.main()

    # Base module writes *_without_physician_relevant.* ; copy to requested names.
    repo = mod.find_repo_root()
    fig_dir = repo / "Figures" / "sankey" / "figures"
    src_html = fig_dir / "llm_sankey_all_models_without_physician_relevant.html"
    src_png = fig_dir / "llm_sankey_all_models_without_physician_relevant.png"
    dst_html = fig_dir / "llm_sankey_all_models_without_physician.html"
    dst_png = fig_dir / "llm_sankey_all_models_without_physician.png"
    shutil.copy2(src_html, dst_html)
    shutil.copy2(src_png, dst_png)
    print(f"Copied {src_html} -> {dst_html}")
    print(f"Copied {src_png} -> {dst_png}")


if __name__ == "__main__":
    main()
