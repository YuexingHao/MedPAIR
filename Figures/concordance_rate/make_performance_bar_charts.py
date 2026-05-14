#!/usr/bin/env python3
"""
Concordance bar charts.

1. **Default (933 cohort):** same layout as ``After_PT_Removal/GPT4o/performance_bar_charts.png`` —
   2×3 grid: **MMLU**, **JAMA**, **MedBullets**, **MedXpertQA** (left 2×2) and **Overall** (right,
   full height). Built from ``concordance_933_by_origin_long.csv`` merged with
   ``933_Clinician_Student_Majority_Vote.csv`` ``data_source_corr`` (run
   ``compute_933_origin_concordance.py --per-origin-out …`` first).

2. **Transposed table:** ``model_performance_transposed.csv`` (CC/SR × dataset + physicians) —
   pass ``--csv`` to that file (columns ``Model``, ``*_Mean``, ``*_SD``).

Run from repo root::

  python Figures/concordance_rate/make_performance_bar_charts.py

  python Figures/concordance_rate/make_performance_bar_charts.py \\
      --csv After_PT_Removal/GPT4o/results/tables/model_performance_transposed.csv \\
      --output Figures/concordance_rate/performance_bar_charts.png

933 outputs live in ``Figures/concordance_rate/`` (``concordance_933_summary_bar.png``, ``concordance_933_by_origin_*.csv``).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent


def _default_933_long_csv() -> Path:
    return _THIS_DIR / "concordance_933_by_origin_long.csv"


def _default_maj_933_csv() -> Path:
    return (
        _REPO_ROOT
        / "Physician_Labels"
        / "Mar2_2026_Data"
        / "933_Clinician_Student_Majority_Vote.csv"
    )


def _default_transposed_csv() -> Path:
    return (
        _REPO_ROOT
        / "After_PT_Removal"
        / "GPT4o"
        / "results"
        / "tables"
        / "model_performance_transposed.csv"
    )


def _default_output_png() -> Path:
    return _THIS_DIR / "concordance_933_summary_bar.png"


def _default_transposed_output_png() -> Path:
    return _THIS_DIR / "performance_bar_charts.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bar charts: 933 multipanel (like GPT4o figure) or transposed CC/SR table."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Transposed input: model_performance_transposed.csv (Model, *_Mean, *_SD). "
        "If omitted, builds 933 figure from --933-long + majority vote.",
    )
    p.add_argument(
        "--933-long",
        type=Path,
        dest="csv_933_long",
        default=None,
        help="Per-Origin concordance long CSV (default: Figures/concordance_rate/concordance_933_by_origin_long.csv).",
    )
    p.add_argument(
        "--majority-933",
        type=Path,
        default=None,
        help="933 majority-vote table for data_source_corr (default: Mar2_2026_Data/933_…csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: concordance_933_summary_bar.png or performance_bar_charts.png)",
    )
    p.add_argument("--dpi", type=int, default=600, help="Figure DPI for savefig.")
    p.add_argument(
        "--transposed",
        action="store_true",
        help="Force transposed multi-panel mode using --csv (expects Model, *_Mean, *_SD).",
    )
    return p.parse_args()


# --- Shared style -----------------------------------------------------------

def _style_rc() -> None:
    plt.rcParams["font.family"] = "Nimbus Sans"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


COLORS = {
    "GPT-4o (SR)": "#009d00",
    "MedGemma-27B (SR)": "#E64825",
    "GPT-5 (SR)": "#1b7c3d",
    "Qwen-14B (CC)": "#f16c23",
    "Qwen-72B (CC)": "#ff9200",
    "Llama-70B (CC)": "#f9AE78",
    "Qwen-14B (SR)": "#f16c23",
    "Qwen-72B (SR)": "#ff9200",
    "Llama-70B (SR)": "#f9AE78",
    "Physician + Trainee": "#2196F3",
}

# Keys in ``concordance_933_by_origin_long.csv`` ``model`` column (from compute_933_origin_concordance.py).
SUMMARY_MODEL_TO_LABEL = {
    "GPT4o": "GPT-4o (SR)",
    "Qwen-14B_CC": "Qwen-14B (CC)",
    "Qwen-14B_SR": "Qwen-14B (SR)",
    "Qwen-72B_CC": "Qwen-72B (CC)",
    "Qwen-72B_SR": "Qwen-72B (SR)",
    "Llama-70B_CC": "Llama-70B (CC)",
    "Llama-70B_SR": "Llama-70B (SR)",
    "MedGemma-27B": "MedGemma-27B (SR)",
    "GPT5": "GPT-5 (SR)",
}

SUMMARY_MODEL_ORDER = [
    "GPT4o",
    "Qwen-14B_CC",
    "Qwen-14B_SR",
    "Qwen-72B_CC",
    "Qwen-72B_SR",
    "Llama-70B_CC",
    "Llama-70B_SR",
    "MedGemma-27B",
    "GPT5",
]

# data_source_corr in 933 majority vote → panel title (same names as transposed CSV)
SOURCE_KEY_TO_PANEL = {
    "mmlu": "MMLU",
    "jama": "JAMA",
    "medbullets": "MedBullets",
    "medxpert": "MedXpertQA",
}


def build_933_transposed_from_long(long_path: Path, maj_path: Path) -> pd.DataFrame:
    """One row per model: Model, {MMLU,JAMA,...}_Mean/_SD, Overall_* from per-Origin concordance."""
    long = pd.read_csv(long_path, low_memory=False)
    need = {"model", "Origin", "concordance_pct"}
    if not need.issubset(long.columns):
        raise ValueError(f"Long CSV must have {need}; got {list(long.columns)}")
    maj = pd.read_csv(maj_path, usecols=["Origin", "data_source_corr"], low_memory=False)
    maj["Origin"] = maj["Origin"].astype(str).str.strip()
    long["Origin"] = long["Origin"].astype(str).str.strip()
    m = long.merge(maj, on="Origin", how="left")
    m = m[m["concordance_pct"].notna()].copy()

    rows: list[dict] = []
    for raw_model in SUMMARY_MODEL_ORDER:
        mdf = m[m["model"] == raw_model]
        if mdf.empty:
            continue
        label = SUMMARY_MODEL_TO_LABEL.get(raw_model, raw_model)
        row: dict = {"Model": label}
        sl = mdf["data_source_corr"].astype(str).str.strip().str.lower()
        for key, panel in SOURCE_KEY_TO_PANEL.items():
            sub = mdf.loc[sl == key, "concordance_pct"].astype(float)
            row[f"{panel}_Mean"] = float(sub.mean()) if len(sub) else np.nan
            row[f"{panel}_SD"] = float(sub.std(ddof=1)) if len(sub) > 1 else np.nan
        ov = mdf["concordance_pct"].astype(float)
        row["Overall_Mean"] = float(ov.mean())
        row["Overall_SD"] = float(ov.std(ddof=1)) if len(ov) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _is_transposed_performance_df(df: pd.DataFrame) -> bool:
    return "Model" in df.columns and "MMLU_Mean" in df.columns


def plot_transposed(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    *,
    suptitle: str | None = None,
) -> None:
    """Original 2×3 layout: MMLU, JAMA, MedBullets, MedXpertQA, Overall."""
    datasets = ["MMLU", "JAMA", "MedBullets", "MedXpertQA", "Overall"]

    fig = plt.figure(figsize=(28, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.8, wspace=0.2)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[:, 2]),
    ]

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.01, color="#374151")

    label_alias = {"Physicians (SR)": "Physician + Trainee"}

    base_order = [
        "GPT-4o",
        "Qwen-14B",
        "Qwen-72B",
        "Llama-70B",
        "MedGemma-27B",
        "GPT-5",
        "Physician + Trainee",
    ]
    base_order_index = {name: i for i, name in enumerate(base_order)}

    def model_sort_key(model_name: str) -> tuple:
        normalized = label_alias.get(str(model_name), str(model_name)).strip()
        if normalized == "Physician + Trainee":
            return (base_order_index["Physician + Trainee"], 0, normalized)

        tag_match = re.search(r"\((CC|SR)\)$", normalized)
        tag = tag_match.group(1) if tag_match else None
        base_name = re.sub(r"\s*\((CC|SR)\)$", "", normalized).strip()

        variant_rank = 0 if tag == "CC" else 1 if tag == "SR" else 0
        return (base_order_index.get(base_name, len(base_order)), variant_rank, normalized)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        mean_col = f"{dataset}_Mean"
        sd_col = f"{dataset}_SD"

        models_raw = df["Model"].astype(str).values
        models = np.array([label_alias.get(m, m) for m in models_raw])
        means = df[mean_col].values
        sds = df[sd_col].values

        order_idx = sorted(range(len(models)), key=lambda i: model_sort_key(models[i]))
        models = models[order_idx]
        means = means[order_idx]
        sds = sds[order_idx]

        bar_colors = [COLORS.get(model, "#808080") for model in models]

        x_pos = np.arange(len(models))

        ax.bar(
            x_pos,
            means,
            yerr=sds,
            color=bar_colors,
            capsize=0,
            alpha=0.8,
            edgecolor="none",
            linewidth=0,
            error_kw={"linewidth": 3, "ecolor": "black", "capthick": 3},
        )

        if idx not in (1, 3):
            ax.set_ylabel("Concordance Rate (%)", fontsize=20, fontweight="bold")
        ax.set_title(dataset, fontsize=25, fontweight="bold", pad=15)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=14)

        ax.set_ylim(0, 100)

        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1)
        ax.set_axisbelow(True)

        ax.tick_params(axis="both", labelsize=15)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    _style_rc()

    csv_in = args.csv.resolve() if args.csv else None

    if args.transposed:
        tpath = csv_in or _default_transposed_csv().resolve()
        if not tpath.is_file():
            raise FileNotFoundError(f"Missing transposed CSV: {tpath}")
        df = pd.read_csv(tpath)
        if not _is_transposed_performance_df(df):
            raise ValueError(f"Expected Model + MMLU_Mean columns in {tpath}")
        out_path = (args.output or _default_transposed_output_png()).resolve()
        plot_transposed(df, out_path, args.dpi)
        return

    if csv_in is not None and csv_in.is_file():
        df_probe = pd.read_csv(csv_in, nrows=2)
        if _is_transposed_performance_df(df_probe):
            df = pd.read_csv(csv_in)
            out_path = (args.output or _default_transposed_output_png()).resolve()
            plot_transposed(df, out_path, args.dpi)
            return

    long_path = (args.csv_933_long or _default_933_long_csv()).resolve()
    maj_path = (args.majority_933 or _default_maj_933_csv()).resolve()
    if not long_path.is_file():
        raise FileNotFoundError(
            f"Missing long CSV: {long_path}\n"
            "Run: python Physician_Labels/scripts/compute_933_origin_concordance.py"
        )
    if not maj_path.is_file():
        raise FileNotFoundError(f"Missing 933 majority vote: {maj_path}")

    tw = build_933_transposed_from_long(long_path, maj_path)
    out_path = (args.output or _default_output_png()).resolve()
    plot_transposed(tw, out_path, args.dpi)


if __name__ == "__main__":
    main()
