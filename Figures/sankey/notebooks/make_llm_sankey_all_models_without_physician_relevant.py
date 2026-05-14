#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATASET_KEYS = ["MMLU", "JAMA", "MedXpert", "MedBullets"]
SOURCE_MAP = {
    "mmlu": "MMLU",
    "jama": "JAMA",
    "medxpert": "MedXpert",
    "medbullets": "MedBullets",
}

MODEL_ORDER_DEFAULT = [
    "Qwen-72B",
    "Qwen-14B",
    "MedGemma-27B",
    "GPT4o",
    "GPT 5",
]


def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in [cur.parent, *cur.parents]:
        marker = p / "Physician_Labels" / "Mar2_2026_Data" / "933_Clinician_Student_Majority_Vote.csv"
        if marker.is_file():
            return p
    raise FileNotFoundError("Could not locate NeuRIPS25 root from script path.")


def extract_letter(v: object) -> str | None:
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return None
    m = re.search(r"Option\s*([A-J])", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"<answer>\s*([A-J])\s*</answer>", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-J])\b", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_ref_933(repo: Path) -> pd.DataFrame:
    p = repo / "Physician_Labels" / "Mar2_2026_Data" / "933_Clinician_Student_Majority_Vote.csv"
    df = pd.read_csv(p, usecols=["Origin", "data_source_corr"])
    df = df.drop_duplicates(subset=["Origin"]).copy()
    df["dataset"] = df["data_source_corr"].astype(str).str.strip().str.lower().map(SOURCE_MAP)
    missing = sorted(df[df["dataset"].isna()]["data_source_corr"].astype(str).unique().tolist())
    if missing:
        raise ValueError(f"Unknown data_source_corr values in 933 reference: {missing}")
    return df[["Origin", "dataset"]]


def load_gold(repo: Path) -> pd.DataFrame:
    p = repo / "After_PT_Removal" / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_Original_Accuracy.csv"
    df = pd.read_csv(p, usecols=["Origin", "answer_corr"]).drop_duplicates(subset=["Origin"]).copy()
    df["gold"] = df["answer_corr"].astype(str).str.strip().str.upper()
    return df[["Origin", "gold"]]


def load_round_letters(repo: Path, *, model_name: str, round_kind: str) -> pd.DataFrame:
    if model_name == "GPT4o" and round_kind == "r1":
        p = repo / "After_PT_Removal" / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_Original_Accuracy.csv"
        df = pd.read_csv(p, usecols=["Origin", "gpt_direct_prediction"]).copy()
        df["letter"] = df["gpt_direct_prediction"].apply(extract_letter)
    elif model_name == "GPT4o" and round_kind == "r2":
        p = repo / "After_PT_Removal" / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_on_trainee_irr_removed.csv"
        df = pd.read_csv(p, usecols=["Origin", "gpt_direct_prediction"]).copy()
        df["letter"] = df["gpt_direct_prediction"].apply(extract_letter)
    elif model_name == "GPT 5" and round_kind == "r1":
        p = repo / "After_PT_Removal" / "GPT5" / "results" / "predictions" / "gpt5_predictions_Original_Accuracy.csv"
        df = pd.read_csv(p, usecols=["ID_corr", "majority_vote"]).copy()
        df = df.rename(columns={"ID_corr": "Origin"})
        df["letter"] = df["majority_vote"].apply(extract_letter)
    elif model_name == "GPT 5" and round_kind == "r2":
        p = repo / "After_PT_Removal" / "GPT5" / "results" / "predictions" / "gpt5_predictions_on_trainee_irr_removed.csv"
        df = pd.read_csv(p, usecols=["Origin", "gpt5_direct_prediction"]).copy()
        df["letter"] = df["gpt5_direct_prediction"].apply(extract_letter)
    elif model_name == "Qwen-72B" and round_kind == "r1":
        p = repo / "After_PT_Removal" / "Qwen2.5-72B-Instruct" / "results" / "predictions" / "Qwen_72B_predictions_ORIGINAL.csv"
        df = pd.read_csv(p, usecols=["Origin", "Extracted_Answer"]).copy()
        df["letter"] = df["Extracted_Answer"].apply(extract_letter)
    elif model_name == "Qwen-72B" and round_kind == "r2":
        p = repo / "After_PT_Removal" / "Qwen2.5-72B-Instruct" / "results" / "predictions" / "Qwen_72B_predictions_trainee_irr_removed.csv"
        df = pd.read_csv(p, usecols=["Origin", "qwen72b_direct_prediction"]).copy()
        df["letter"] = df["qwen72b_direct_prediction"].apply(extract_letter)
    elif model_name == "Qwen-14B" and round_kind == "r1":
        p = repo / "After_PT_Removal" / "Qwen2.5-14B-Instruct" / "results" / "predictions" / "Qwen_14B_predictions_ORIGINAL.csv"
        df = pd.read_csv(p, usecols=["Origin", "Extracted_Answer"]).copy()
        df["letter"] = df["Extracted_Answer"].apply(extract_letter)
    elif model_name == "Qwen-14B" and round_kind == "r2":
        p = repo / "After_PT_Removal" / "Qwen2.5-14B-Instruct" / "results" / "predictions" / "Qwen_14B_predictions_trainee_irr_removed.csv"
        df = pd.read_csv(p, usecols=["Origin", "qwen14b_direct_prediction"]).copy()
        df["letter"] = df["qwen14b_direct_prediction"].apply(extract_letter)
    elif model_name == "MedGemma-27B" and round_kind == "r1":
        p = repo / "After_PT_Removal" / "MedGemma-27b-text-it" / "results" / "predictions" / "MedGemma27B_predictions_ORIGINAL.csv"
        df = pd.read_csv(p, usecols=["Origin", "Extracted_Answer"]).copy()
        df["letter"] = df["Extracted_Answer"].apply(extract_letter)
    elif model_name == "MedGemma-27B" and round_kind == "r2":
        p = repo / "After_PT_Removal" / "MedGemma-27b-text-it" / "results" / "predictions" / "MedGemma27B_predictions_on_trainee_irr_removed.csv"
        df = pd.read_csv(p, usecols=["Origin", "medgemma_direct_prediction"]).copy()
        df["letter"] = df["medgemma_direct_prediction"].apply(extract_letter)
    elif model_name == "Llama-70B" and round_kind == "r1":
        p = repo / "After_PT_Removal" / "Llama-70B" / "results" / "predictions" / "Llama70B_ORIGINAL_predictions.csv"
        df = pd.read_csv(p, usecols=["Origin", "Extracted_Answer"]).copy()
        df["letter"] = df["Extracted_Answer"].apply(extract_letter)
    elif model_name == "Llama-70B" and round_kind == "r2":
        p = repo / "After_PT_Removal" / "Llama-70B" / "results" / "predictions" / "Llama70B_predictions_on_trainee_irr_removed.csv"
        df = pd.read_csv(p, usecols=["Origin", "llama70b_direct_prediction"]).copy()
        df["letter"] = df["llama70b_direct_prediction"].apply(extract_letter)
    else:
        raise ValueError(f"Unsupported model/round: {model_name}/{round_kind}")

    return df[["Origin", "letter"]].drop_duplicates(subset=["Origin"])


def summarize_model(repo: Path, model_name: str, ref_933: pd.DataFrame, gold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    r1 = load_round_letters(repo, model_name=model_name, round_kind="r1").rename(columns={"letter": "r1"})
    r2 = load_round_letters(repo, model_name=model_name, round_kind="r2").rename(columns={"letter": "r2"})
    merged = ref_933.merge(gold, on="Origin", how="left").merge(r1, on="Origin", how="left").merge(r2, on="Origin", how="left")

    missing_r2 = int(merged["r2"].isna().sum())
    if missing_r2:
        raise ValueError(
            f"{model_name}: {missing_r2}/933 Origins have no round-2 prediction yet. "
            "Finish rerun before generating final figure."
        )

    by_source = {}
    total_r1 = total_sp = total_r2c = 0
    for ds in DATASET_KEYS:
        sub = merged[merged["dataset"] == ds].copy()
        r1c = int((sub["r1"] == sub["gold"]).sum())
        sp = int(((sub["r1"] == sub["gold"]) & (sub["r2"] != sub["gold"])).sum())
        r2c = r1c - sp
        by_source[ds] = {"r1_correct": r1c, "spurious": sp, "r2_correct": r2c}
        total_r1 += r1c
        total_sp += sp
        total_r2c += r2c

    summary = {
        "model": model_name,
        "by_source": by_source,
        "round1_correct": total_r1,
        "spurious": total_sp,
        "round2_correct": total_r2c,
        "pct_spurious": (100.0 * total_sp / total_r1) if total_r1 else 0.0,
    }
    return summary, merged


def sankey_for_summary(summary: dict) -> dict:
    r1 = {k: int(v["r1_correct"]) for k, v in summary["by_source"].items()}
    sp = {k: int(v["spurious"]) for k, v in summary["by_source"].items()}
    r2 = {k: int(v["r2_correct"]) for k, v in summary["by_source"].items()}

    total_r1 = int(summary["round1_correct"])
    total_sp = int(summary["spurious"])
    pct = (100.0 * total_sp / total_r1) if total_r1 else 0.0

    n_src = len(DATASET_KEYS)
    base_colors = ["#3b82f6", "#0ea5e9", "#06b6d4", "#14b8a6"]
    sources, targets, values, colors = [], [], [], []

    if total_sp == 0:
        nodes = DATASET_KEYS + ["R2 correct"]
        node_colors = base_colors + ["#10b981"]
        green_idx = n_src
        for i, ds in enumerate(DATASET_KEYS):
            if r2[ds] > 0:
                sources.append(i); targets.append(green_idx); values.append(r2[ds]); colors.append("rgba(16,185,129,0.4)")
    else:
        nodes = DATASET_KEYS + ["R2 correct", f"Spurious {pct:.1f}%"]
        node_colors = base_colors + ["#10b981", "#ef4444"]
        green_idx = n_src
        red_idx = n_src + 1
        for i, ds in enumerate(DATASET_KEYS):
            if r2[ds] > 0:
                sources.append(i); targets.append(green_idx); values.append(r2[ds]); colors.append("rgba(16,185,129,0.4)")
            if sp[ds] > 0:
                sources.append(i); targets.append(red_idx); values.append(sp[ds]); colors.append("rgba(239,68,68,0.4)")

    if total_sp == 0:
        xs = [0.04] * n_src + [0.988]
        ys = [0.1 + i * (0.8 / (n_src - 1)) for i in range(n_src)] + [0.5]
    else:
        xs = [0.04] * n_src + [0.988, 0.988]
        ys = [0.1 + i * (0.8 / (n_src - 1)) for i in range(n_src)] + [0.2, 0.8]

    return {
        "nodes": nodes,
        "node_colors": node_colors,
        "sources": sources,
        "targets": targets,
        "values": values,
        "link_colors": colors,
        "x": xs,
        "y": ys,
    }


def panel_title(name: str, r1: int, r2: int) -> str:
    return (
        f"<span style='font-size:56px;font-weight:600;font-family:Arial;color:#1f2937'>{name} ({r1} QAs)</span><br>"
        f"<span style='font-size:42px;font-family:Arial;color:#4b5563'>R1 Correct: {r1} | R2 Correct: {r2}</span>"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build Expert-933 no-physician relevant-sentence Sankey panels. "
            "Default excludes Llama-70B while its rerun is pending."
        )
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--include-llama", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = find_repo_root()
    out_dir = (args.output_dir or (repo / "Figures" / "sankey" / "figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "llm_sankey_all_models_without_physician_relevant.html"
    png_path = out_dir / "llm_sankey_all_models_without_physician_relevant.png"
    summary_csv = repo / "Figures" / "sankey" / "data" / "sankey_spurious_expert933_results_relevant_no_physician.csv"

    ref_933 = load_ref_933(repo)
    gold = load_gold(repo)
    model_order = list(MODEL_ORDER_DEFAULT)
    if args.include_llama:
        model_order.insert(1, "Llama-70B")

    summaries = []
    for model_name in model_order:
        summaries.append(summarize_model(repo, model_name, ref_933, gold)[0])

    if len(summaries) == 5:
        specs = [
            [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
            [{"type": "sankey"}, {"type": "sankey"}, None],
        ]
        slots = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    else:
        specs = [
            [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
            [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
        ]
        slots = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=specs,
        subplot_titles=[
            panel_title(s["model"], int(s["round1_correct"]), int(s["round2_correct"]))
            for s in summaries
        ],
        horizontal_spacing=0.07,
        vertical_spacing=0.32,
    )
    for s, (row, col) in zip(summaries, slots):
        data = sankey_for_summary(s)
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

    fig.write_html(str(html_path))
    print(f"Wrote {html_path}")
    if not args.no_png:
        fig.write_image(str(png_path), width=2560, height=1440, scale=3)
        print(f"Wrote {png_path}")

    rows = []
    for s in summaries:
        rows.append(
            {
                "model": s["model"],
                "n_expert_qa": 933,
                "round1_correct": int(s["round1_correct"]),
                "round2_incorrect_spurious": int(s["spurious"]),
                "round2_still_correct": int(s["round2_correct"]),
                "pct_spurious_among_round1_correct": round(float(s["pct_spurious"]), 4),
            }
        )
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    baseline_csv = repo / "Figures" / "sankey" / "data" / "sankey_spurious_expert933_results.csv"
    if baseline_csv.is_file():
        base = pd.read_csv(baseline_csv)
        base = base[(base["row_type"] == "model_pooled_over_sources") & (base["model"].isin(model_order))]
        base = base[["model", "round1_correct", "round2_incorrect_spurious", "round2_still_correct", "pct_spurious_among_round1_correct"]]
        cur = pd.DataFrame(rows)
        merged = base.merge(cur, on="model", suffixes=("_baseline", "_relevant"))
        print("\nBaseline vs new relevant (933 expert QAs):")
        print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
