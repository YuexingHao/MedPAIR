#!/usr/bin/env python3
"""Rebuild llm_sankey_all_models_without_physician with explicit remapping.

Baseline values for all models are read from:
  Figures/sankey/data/sankey_spurious_expert933_results.csv

Then selected model summaries are recomputed using:
  - R1: Llama70B_predictions_on_Trainee.csv
  - R2: Llama70B_predictions_on_MJ_LowIRR.csv
  - R1: MedGemma27B_predictions_ORIGINAL.csv
  - R2: MedGemma27B_predictions_on_MJ_LowIRR.csv

This preserves all other model numbers while fixing mapped columns.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["Qwen-72B", "Llama-70B", "Qwen-14B", "MedGemma-27B", "GPT4o", "GPT 5"]
BASELINE_MODEL_NAME = {
    "Qwen-72B": "Qwen 72B",
    "Llama-70B": "Llama-70B",
    "Qwen-14B": "Qwen-14B",
    "MedGemma-27B": "MedGemma-27B",
    "GPT4o": "GPT4o",
    "GPT 5": "GPT 5",
}


def _load_base_module():
    pyc = (
        Path(__file__).resolve().parent
        / "__pycache__"
        / "make_llm_sankey_all_models_without_physician_relevant.cpython-313.pyc"
    )
    if not pyc.exists():
        raise FileNotFoundError(f"Missing compiled generator: {pyc}")
    loader = importlib.machinery.SourcelessFileLoader("sankey_no_phys_patch2", str(pyc))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _baseline_summary_for_model(df: pd.DataFrame, model: str) -> dict:
    bname = BASELINE_MODEL_NAME.get(model, model)
    by_src = df[(df["row_type"] == "by_medpair_source") & (df["model"] == bname)].copy()
    pooled = df[(df["row_type"] == "model_pooled_over_sources") & (df["model"] == bname)].copy()
    if by_src.empty or pooled.empty:
        raise ValueError(f"Missing baseline rows for model: {model}")

    src_map = {}
    for _, r in by_src.iterrows():
        src_map[str(r["medpair_source"])] = {
            "r1_correct": int(r["round1_correct"]),
            "spurious": int(r["round2_incorrect_spurious"]),
            "r2_correct": int(r["round2_still_correct"]),
        }

    p = pooled.iloc[0]
    r1 = int(p["round1_correct"])
    sp = int(p["round2_incorrect_spurious"])
    r2 = int(p["round2_still_correct"])
    pct = float(p["pct_spurious_among_round1_correct"])
    return {
        "model": model,
        "by_source": src_map,
        "round1_correct": r1,
        "spurious": sp,
        "round2_correct": r2,
        "pct_spurious": pct,
    }


def main() -> None:
    mod = _load_base_module()
    repo = mod.find_repo_root()

    baseline_csv = repo / "Figures" / "sankey" / "data" / "sankey_spurious_expert933_results.csv"
    baseline = pd.read_csv(baseline_csv)

    summaries = [_baseline_summary_for_model(baseline, m) for m in MODELS]

    # Recompute mapped models with explicit file selection.
    orig_load_round_letters = mod.load_round_letters
    canonical = mod.pd.read_csv(
        repo
        / "After_PT_Removal"
        / "shared"
        / "data"
        / "Centaur_Lab_First_Round_933_MJ_LowIRR_as_NewSentences_for_rerun.csv",
        usecols=["Origin"],
    ).copy()
    canonical["Origin"] = canonical["Origin"].astype(str).str.strip()
    canonical = canonical[["Origin"]].drop_duplicates(subset=["Origin"])

    def patched_load_round_letters(repo_path: Path, *, model_name: str, round_kind: str):
        if model_name == "Llama-70B" and round_kind == "r1":
            p = (
                repo_path
                / "After_PT_Removal"
                / "Llama-70B"
                / "results"
                / "predictions"
                / "Llama70B_predictions_on_Trainee.csv"
            )
            df = mod.pd.read_csv(
                p,
                usecols=["Origin", "Extracted_Answer"],
            ).copy()
            df["letter"] = df["Extracted_Answer"].apply(mod.extract_letter)
            df["Origin"] = df["Origin"].astype(str).str.strip()
            return df[["Origin", "letter"]].drop_duplicates(subset=["Origin"])
        if model_name == "Llama-70B" and round_kind == "r2":
            p = (
                repo_path
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
            df = df[["Origin", "letter"]].drop_duplicates(subset=["Origin"])
            # Count unfinished reruns as wrong instead of failing generation.
            df = canonical.merge(df, on="Origin", how="left")
            df["letter"] = df["letter"].fillna("__MISSING__")
            return df
        if model_name == "MedGemma-27B" and round_kind == "r1":
            p = (
                repo_path
                / "After_PT_Removal"
                / "MedGemma-27b-text-it"
                / "results"
                / "predictions"
                / "MedGemma27B_predictions_ORIGINAL.csv"
            )
            df = mod.pd.read_csv(
                p,
                usecols=["Origin", "Extracted_Answer"],
            ).copy()
            df["letter"] = df["Extracted_Answer"].apply(mod.extract_letter)
            df["Origin"] = df["Origin"].astype(str).str.strip()
            return df[["Origin", "letter"]].drop_duplicates(subset=["Origin"])
        if model_name == "MedGemma-27B" and round_kind == "r2":
            p = (
                repo_path
                / "After_PT_Removal"
                / "MedGemma-27b-text-it"
                / "results"
                / "predictions"
                / "MedGemma27B_predictions_on_MJ_LowIRR.csv"
            )
            df = mod.pd.read_csv(
                p,
                usecols=["Origin", "medgemma_direct_prediction", "medgemma_extracted_answer"],
            ).copy()
            df["letter"] = df["medgemma_direct_prediction"].apply(mod.extract_letter)
            fb = df["medgemma_extracted_answer"].apply(mod.extract_letter)
            df["letter"] = df["letter"].where(df["letter"].notna(), fb)
            df["Origin"] = df["Origin"].astype(str).str.strip()
            df = df[["Origin", "letter"]].drop_duplicates(subset=["Origin"])
            # Count unfinished reruns as wrong instead of failing generation.
            df = canonical.merge(df, on="Origin", how="left")
            df["letter"] = df["letter"].fillna("__MISSING__")
            return df
        return orig_load_round_letters(repo_path, model_name=model_name, round_kind=round_kind)

    mod.load_round_letters = patched_load_round_letters
    ref_933 = mod.load_ref_933(repo)
    gold = mod.load_gold(repo)
    recomputed = {
        "Llama-70B": mod.summarize_model(repo, "Llama-70B", ref_933, gold)[0],
        "MedGemma-27B": mod.summarize_model(repo, "MedGemma-27B", ref_933, gold)[0],
    }

    for i, s in enumerate(summaries):
        model = s["model"]
        if model in recomputed:
            summaries[i] = recomputed[model]

    specs = [[{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}], [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}]]
    slots = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=specs,
        subplot_titles=[
            mod.panel_title(s["model"], int(s["round1_correct"]), int(s["round2_correct"]))
            for s in summaries
        ],
        horizontal_spacing=0.07,
        vertical_spacing=0.32,
    )

    for s, (row, col) in zip(summaries, slots):
        data = mod.sankey_for_summary(s)
        fig.add_trace(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=28,
                    thickness=18,
                    line=dict(color="white", width=2),
                    label=data["nodes"],
                    color=data["node_colors"],
                    x=data["x"],
                    y=data["y"],
                ),
                link=dict(
                    source=data["sources"],
                    target=data["targets"],
                    value=data["values"],
                    color=data["link_colors"],
                ),
                textfont=dict(size=34, family="Arial", color="#111827"),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        font=dict(size=36, family="Arial"),
        width=2560,
        height=1440,
        margin=dict(l=32, r=32, t=260, b=36),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for ann in fig.layout.annotations or []:
        ann.font = dict(size=56, family="Arial", color="#1f2937")
        ann.yshift = (ann.yshift or 0) + 90

    out_dir = repo / "Figures" / "sankey" / "figures"
    out_html = out_dir / "llm_sankey_all_models_without_physician.html"
    out_png = out_dir / "llm_sankey_all_models_without_physician.png"
    fig.write_html(str(out_html))
    fig.write_image(str(out_png), width=2560, height=1440, scale=3)
    print(f"Wrote {out_html}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
