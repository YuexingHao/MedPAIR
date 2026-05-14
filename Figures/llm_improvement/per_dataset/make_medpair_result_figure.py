#!/usr/bin/env python3
"""
Five-panel figure for the **933 Expert QA** cohort, stratified by **MedPAIR data source**.

Layout (3×2 grid): **Total** spans the top row; **MMLU** and **JAMA** on the second row;
**MedXpert** and **MedBullets** on the third.

**Input CSV** must contain ``Base Model``, ``Low+Irr Labelers``, and numeric columns
``Total``, ``MMLU``, ``Jama``, ``MedXpert``, ``Medbullets`` where every value is computed
**only on the 933 Expert QA items** (``Total`` = all 933; the other four = per-source
subsets of that cohort). Prefer ``ExpertQA_933_by_data_source.csv`` in this folder or
``Figures/llm_improvement/per_category/per_category/``.

If only ``PhysicianEval_Result_Report.csv`` is available, the script still runs, but that
table uses different semantics for ``Jama`` / ``MedXpert`` / ``Medbullets`` (physician
subsets, not data sources)—generate the 933-by-source report for the intended figure.

Default output: ``Figures/llm_improvement/per_dataset/MedPAIR_Result_per_dataset.pdf``.

Run from repo root (``NeuRIPS25``)::

  python Figures/llm_improvement/per_dataset/make_medpair_result_figure.py
  python Figures/llm_improvement/per_dataset/make_medpair_result_figure.py --input path/to.csv --output out.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent.parent
_GPT4O_ROOT = _WORKSPACE_ROOT / "After_PT_Removal" / "GPT4o"
DEFAULT_FIGURES_DIR = _THIS_DIR
_PER_CATEGORY_DATA_DIR = (
    _WORKSPACE_ROOT / "Figures" / "llm_improvement" / "per_category" / "per_category"
)

DEFAULT_INPUT_CSV = _PER_CATEGORY_DATA_DIR / "PhysicianEval_Result_Report.csv"

_METRIC_COLS = ("Total", "MMLU", "Jama", "MedXpert", "Medbullets")

if not (_GPT4O_ROOT / "paths.py").is_file():
    print(
        f"Expected GPT4o layout at {_GPT4O_ROOT} (paths.py missing).",
        file=sys.stderr,
    )
    sys.exit(1)
if str(_GPT4O_ROOT) not in sys.path:
    sys.path.insert(0, str(_GPT4O_ROOT))
import paths  # noqa: E402


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common header variants onto ``Total`` / ``MMLU`` / ``Jama`` / ``MedXpert`` / ``Medbullets``."""
    out = df.copy()
    rename_map: dict[str, str] = {}
    for c in out.columns:
        key = str(c).strip().lower()
        if key == "total":
            rename_map[c] = "Total"
        elif key == "mmlu":
            rename_map[c] = "MMLU"
        elif key == "jama":
            rename_map[c] = "Jama"
        elif key == "medxpert":
            rename_map[c] = "MedXpert"
        elif key in ("medbullets", "med_bullets"):
            rename_map[c] = "Medbullets"
    return out.rename(columns=rename_map)


def _resolve_labeler_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    labeler_candidates = [
        "Low+Irr Labelers",
        "Low + Irr Labelers",
        "Low+IRR Labelers",
        "Low + IRR Labelers",
    ]
    labeler_col = next((c for c in labeler_candidates if c in df.columns), None)
    if labeler_col is None:
        raise KeyError(
            "Could not find labeler column. Available columns: "
            f"{df.columns.tolist()}"
        )
    if labeler_col != "Low+Irr Labelers":
        df = df.rename(columns={labeler_col: "Low+Irr Labelers"})
    df["Low+Irr Labelers"] = df["Low+Irr Labelers"].astype(str).str.strip()
    if "Base Model" in df.columns:
        df["Base Model"] = df["Base Model"].astype(str).str.strip()
    return df


def make_figure(
    df: pd.DataFrame,
    output_path: Path,
    show: bool = False,
) -> None:
    df = _normalize_metric_columns(df)
    df = _resolve_labeler_column(df)
    df = df[df["Low+Irr Labelers"].astype(str).str.strip() != "Trainee Removed (IRR)"]

    columns = list(_METRIC_COLS)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            "per_dataset figure needs 933-by-source columns "
            f"{list(_METRIC_COLS)}; missing: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    plt.rcParams["font.family"] = "Nimbus Sans"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    labeler_colors = {
        "Original Accuracy": "#000000",
        "Trainee Removed": "#2196F3",
        "Trainee Removed (248 QAs)": "#2196F3",
        "Qwen-14B Removed": "#f16c23",
        "Qwen-72B Removed": "#ff9200",
        "Llama-70b Removed": "#f9AE78",
        "GPT-4o Removed": "#009d00",
        "GPT5 Removed": "#1b7c3d",
        "MedGemma-27B": "#ECB66C",
        "MedGemma-27B Removed": "#ECB66C",
    }

    labeler_markers = {
        "Original Accuracy": None,
        "Trainee Removed": "o",
        "Trainee Removed (248 QAs)": "o",
        "Qwen-14B Removed": "s",
        "Qwen-72B Removed": "s",
        "Llama-70b Removed": "s",
        "MedGemma-27B Removed": "s",
        "GPT-4o Removed": "^",
        "GPT5 Removed": "^",
    }

    release_dates = {
        "GPT4o": "05/24",
        "Qwen-14B": "09/24",
        "Qwen 72B": "09/24",
        "Llama-70B": "12/23",
        "MedGemma-27B": "05/25",
        "GPT 5": "08/25",
        "Trainee": "",
        "Physician": "",
    }

    # Notebook: figsize=(43, 55), 3x2 grid, Total spanning top row
    fig = plt.figure(figsize=(43, 55))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.2, height_ratios=[1, 1, 1])
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    titles = ["Total", "MMLU", "JAMA", "MedXpert", "MedBullets"]

    _x_cat_spacing = 2.25
    _x_margin = 0.9
    _canonical_order = [
        "GPT4o",
        "Qwen-14B",
        "Qwen 72B",
        "Llama-70B",
        "MedGemma-27B",
        "GPT 5",
        "Trainee",
    ]
    _present = set(df["Base Model"].astype(str).str.strip())
    base_models = [m for m in _canonical_order if m in _present]
    if not base_models:
        raise ValueError("No Base Model values in dataframe match the expected labels.")

    def _x(i: float) -> float:
        return float(i) * _x_cat_spacing

    # Vertical “Dataset Release Date” lines (scaled like integer positions in the notebook)
    x_vline_left = -0.5 * _x_cat_spacing
    x_vline_mid: float | None = None
    if "Llama-70B" in base_models and "MedGemma-27B" in base_models:
        i_l = base_models.index("Llama-70B")
        i_m = base_models.index("MedGemma-27B")
        x_vline_mid = 0.5 * (_x(i_l) + _x(i_m))

    for ax_idx, (ax, col, title) in enumerate(zip(axes, columns, titles)):
        n_models = len(base_models)
        x_positions = np.arange(n_models) * _x_cat_spacing
        added_labels: set[str] = set()
        best_perf_label_added = False

        for i, model in enumerate(base_models):
            model_data = df[df["Base Model"] == model]
            if len(model_data) == 0:
                continue

            original_acc_row = model_data[
                model_data["Low+Irr Labelers"] == "Original Accuracy"
            ]
            original_acc_value = (
                original_acc_row[col].values[0] if len(original_acc_row) > 0 else None
            )

            other_labelers = model_data[
                model_data["Low+Irr Labelers"] != "Original Accuracy"
            ]
            if len(other_labelers) > 0:
                best_row = other_labelers.loc[other_labelers[col].idxmax()]
                best_value = best_row[col]
            else:
                best_value = None

            for _, row in model_data.iterrows():
                labeler = row["Low+Irr Labelers"]
                value = row[col]
                color = labeler_colors.get(labeler, "#7f7f7f")
                label = labeler if labeler not in added_labels else ""
                if label:
                    added_labels.add(labeler)

                if labeler == "Original Accuracy":
                    hw = 0.35 * _x_cat_spacing
                    ax.plot(
                        [_x(i) - hw, _x(i) + hw],
                        [value, value],
                        color=color,
                        linewidth=6,
                        solid_capstyle="butt",
                        zorder=3,
                        label=label,
                    )
                else:
                    marker = labeler_markers.get(labeler, "o")
                    ax.scatter(
                        _x(i),
                        value,
                        c=color,
                        s=1200,
                        alpha=0.9,
                        marker=marker,
                        edgecolors="white",
                        linewidth=3,
                        zorder=3,
                        label=label,
                    )

            if best_value is not None and original_acc_value is not None:
                line_label = (
                    "Best Performance" if (ax_idx == 0 and not best_perf_label_added) else ""
                )
                if line_label:
                    best_perf_label_added = True

                hw = 0.35 * _x_cat_spacing
                ax.plot(
                    [_x(i) - hw, _x(i) + hw],
                    [best_value, best_value],
                    color="#808080",
                    linewidth=5,
                    linestyle=":",
                    solid_capstyle="butt",
                    zorder=4,
                    label=line_label,
                    alpha=0.8,
                )

                pct_improvement = (
                    (best_value - original_acc_value) / original_acc_value
                ) * 100
                text_color = "#00a000" if pct_improvement >= 0 else "#d32f2f"
                arrow_color = "#00a000" if pct_improvement >= 0 else "#d32f2f"
                if pct_improvement >= 0:
                    pct_text = f"+{pct_improvement:.1f}%"
                else:
                    pct_text = f"{pct_improvement:.1f}%"

                arrow_x = _x(i) - 0.25 * _x_cat_spacing
                ax.annotate(
                    "",
                    xy=(arrow_x, best_value),
                    xytext=(arrow_x, original_acc_value),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=arrow_color,
                        lw=4,
                        mutation_scale=25,
                        alpha=0.7,
                    ),
                    zorder=5,
                )

                text_y = max(best_value, original_acc_value) + 5
                ax.text(
                    _x(i),
                    text_y,
                    pct_text,
                    fontsize=45,
                    fontweight="bold",
                    color=text_color,
                    ha="center",
                    va="bottom",
                    zorder=6,
                )

            if release_dates.get(model, ""):
                ax.text(
                    _x(i),
                    -10,
                    release_dates[model],
                    fontsize=45,
                    fontweight="500",
                    color="#606060",
                    ha="center",
                    va="top",
                    zorder=6,
                )

        # Vertical reference lines + labels (notebook logic, x scaled by _x_cat_spacing)
        first_panel = title == "Total"
        rest_panels = title in ("MMLU", "JAMA", "MedXpert", "MedBullets")

        if first_panel:
            ax.axvline(
                x=x_vline_left,
                color="#808080",
                linewidth=3,
                linestyle=":",
                alpha=0.6,
                zorder=2,
            )
            ax.text(
                x_vline_left,
                115,
                "Dataset Release Date",
                fontsize=50,
                fontweight="600",
                color="#606060",
                ha="center",
                va="bottom",
                rotation=0,
                zorder=6,
            )
        elif rest_panels and x_vline_mid is not None:
            ax.axvline(
                x=x_vline_mid,
                color="#808080",
                linewidth=3,
                linestyle=":",
                alpha=0.6,
                zorder=2,
            )
            ax.text(
                x_vline_mid,
                115,
                "Dataset Release Date",
                fontsize=50,
                fontweight="600",
                color="#606060",
                ha="center",
                va="bottom",
                rotation=0,
                zorder=6,
            )

        if n_models <= 1:
            ax.set_xlim(-_x_margin, _x_margin)
        else:
            ax.set_xlim(
                -_x_margin,
                (n_models - 1) * _x_cat_spacing + _x_margin,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            base_models, rotation=45, ha="right", fontsize=44, fontweight="500"
        )

        if ax_idx in [0, 1, 3]:
            ax.set_ylabel(
                "Accuracy (%)", fontsize=48, fontweight="bold", labelpad=15
            )

        ax.set_title(title, fontsize=72, fontweight="bold", pad=65)
        ax.grid(axis="y", alpha=0.4, linewidth=2, color="gray", linestyle="--", zorder=1)
        ax.grid(axis="x", alpha=0, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=52, width=2, length=8, color="black")
        ax.tick_params(axis="x", labelsize=52, width=2, length=8, color="black")
        ax.set_ylim(-20, 125)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["left"].set_edgecolor("black")
        ax.spines["bottom"].set_linewidth(2.5)
        ax.spines["bottom"].set_edgecolor("black")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    legend_order = [
        "Original Accuracy",
        "Best Performance",
        "Trainee Removed",
        "GPT-4o Removed",
        "GPT5 Removed",
        "Qwen-72B Removed",
        "Qwen-14B Removed",
        "Llama-70b Removed",
        "MedGemma-27B Removed",
    ]
    ordered_handles = []
    ordered_labels = []
    for lab in legend_order:
        if lab in by_label:
            ordered_labels.append(lab)
            ordered_handles.append(by_label[lab])

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        fontsize=45,
        framealpha=1,
        markerscale=1.8,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Labeler Configuration",
        title_fontsize=52,
        labelspacing=1.5,
        borderpad=1.5,
        handletextpad=1.2,
    )

    plt.subplots_adjust(left=0.01, right=0.86, top=0.96, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "CSV path (default: ExpertQA_933_by_data_source.csv if present, else "
            "PhysicianEval_Result_Report.csv under Figures/llm_improvement/per_category/per_category/ or GPT4o tables)"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output PDF (default: {DEFAULT_FIGURES_DIR / 'MedPAIR_Result_per_dataset.pdf'})",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window after saving.",
    )
    return p.parse_args()


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def main() -> None:
    args = parse_args()
    phys_gpt4o = (paths.TABLES / "PhysicianEval_Result_Report.csv").resolve()
    classic = (paths.TABLES / "Result_Report.csv").resolve()

    expert_candidates = [
        _THIS_DIR / "ExpertQA_933_by_data_source.csv",
        _PER_CATEGORY_DATA_DIR / "ExpertQA_933_by_data_source.csv",
    ]
    phys_candidates = [
        _THIS_DIR / "PhysicianEval_Result_Report.csv",
        _PER_CATEGORY_DATA_DIR / "PhysicianEval_Result_Report.csv",
        DEFAULT_INPUT_CSV,
        phys_gpt4o,
    ]

    if args.input is not None:
        input_csv = args.input.resolve()
    elif (found := _first_existing(expert_candidates)) is not None:
        input_csv = found
    elif (found := _first_existing(phys_candidates)) is not None:
        input_csv = found
        if input_csv.name == "PhysicianEval_Result_Report.csv":
            print(
                "Warning: PhysicianEval_Result_Report.csv is not a 933-by-data-source table: "
                "'Total' is pooled accuracy, 'MMLU' is the 933 Expert QA column, and "
                "Jama / MedXpert / Medbullets are physician subsets—not MedPAIR sources. "
                "Use ExpertQA_933_by_data_source.csv for a consistent 933-only figure.",
                file=sys.stderr,
            )
    else:
        input_csv = classic

    output_pdf = (args.output or (DEFAULT_FIGURES_DIR / "MedPAIR_Result_per_dataset.pdf")).resolve()

    if not input_csv.is_file():
        print(f"Input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_csv)
    make_figure(
        df,
        output_pdf,
        show=args.show,
    )
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
