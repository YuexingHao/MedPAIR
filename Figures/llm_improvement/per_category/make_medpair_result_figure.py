#!/usr/bin/env python3
"""
Build the MedPAIR **per-category** figure (``MedPAIR_Result_per_category.pdf``): **three panels** for
letter accuracy on physician-defined cohorts (see ``physician_eval_to_result_report.py``).

**Panels (CSV columns):**

- **Expert QA (933)** ← ``MMLU``
- **Hard QA (733)** ← ``Jama``
- **Impossible QA (334)** ← ``MedXpert``

``Total`` and ``Medbullets`` in the wide table are not shown here (pooled / macro mean across the
three cohorts). For MedPAIR **data source** breakdown within the 933 cohort, use
``Figures/llm_improvement/per_dataset/make_medpair_result_figure.py`` with ``ExpertQA_933_by_data_source.csv``.

Default CSV search order: ``PhysicianEval_Result_Report.csv`` (this folder or ``per_category/`` subfolder),
then GPT4o ``tables/``, then ``Result_Report.csv``.

Run from repo root (``NeuRIPS25``)::

  python Figures/llm_improvement/per_category/make_medpair_result_figure.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical location: ``Figures/llm_improvement/per_category/``; repo root is three levels up.
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent.parent
_GPT4O_ROOT = _WORKSPACE_ROOT / "After_PT_Removal" / "GPT4o"
DEFAULT_FIGURES_DIR = _THIS_DIR
_LEGACY_NESTED_DIR = _THIS_DIR / "per_category"

if not (_GPT4O_ROOT / "paths.py").is_file():
    print(
        f"Expected GPT4o layout at {_GPT4O_ROOT} (paths.py missing).",
        file=sys.stderr,
    )
    sys.exit(1)
if str(_GPT4O_ROOT) not in sys.path:
    sys.path.insert(0, str(_GPT4O_ROOT))
import paths  # noqa: E402

# Wide report columns for the three physician subsets (933 / HardQA / Impossible).
_CATEGORY_METRIC_COLS = ("MMLU", "Jama", "MedXpert")
_CATEGORY_TITLES = (
    "Expert QA (933)",
    "Hard QA (733)",
    "Impossible QA (334)",
)


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
    df = _normalize_metric_columns(_resolve_labeler_column(df))
    df = df[df["Low+Irr Labelers"].astype(str).str.strip() != "Trainee Removed (IRR)"]
    missing = [c for c in _CATEGORY_METRIC_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"CSV must include columns {list(_CATEGORY_METRIC_COLS)}; missing: {missing}. "
            "Use _normalize_metric_columns-compatible headers (e.g. mmlu, jama, medxpert)."
        )

    plt.rcParams["font.family"] = "Nimbus Sans"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    labeler_colors = {
        "Original Accuracy": "#000000",
        "Trainee Removed": "#2196F3",
        "Trainee Removed (IRR)": "#2196F3",
        "Trainee Removed (248 QAs)": "#2196F3",
        "Qwen-14B Removed": "#f16c23",
        "Qwen-14B Removed (GPT4o CSV)": "#f16c23",
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
        "Trainee Removed (IRR)": "o",
        "Trainee Removed (248 QAs)": "o",
        "Qwen-14B Removed": "s",
        "Qwen-14B Removed (GPT4o CSV)": "s",
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
        "Physicians": "",
    }

    # One row × three columns: Expert QA / Hard QA / Impossible
    fig = plt.figure(figsize=(66, 22))
    gs = fig.add_gridspec(1, 3, wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    columns = list(_CATEGORY_METRIC_COLS)
    titles = list(_CATEGORY_TITLES)

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

    for ax_idx, (ax, col, title) in enumerate(zip(axes, columns, titles)):
        n_models = len(base_models)
        x_positions = np.arange(n_models) * _x_cat_spacing

        def _x(i: int) -> float:
            return float(i) * _x_cat_spacing

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

        if ax_idx == 0:
            ax.set_ylabel(
                "Accuracy (%)", fontsize=48, fontweight="bold", labelpad=15
            )
        ax.set_title(title, fontsize=56, fontweight="bold", pad=50)
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
        "Trainee Removed (IRR)",
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
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
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
            "CSV with Base Model, Low+Irr Labelers, MMLU, Jama, MedXpert "
            "(933 / HardQA / Impossible; default: see module docstring)"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output PDF path (default: {DEFAULT_FIGURES_DIR / 'MedPAIR_Result_per_category.pdf'})",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window after saving (default: save only).",
    )
    return p.parse_args()


def _resolve_default_physician_csv() -> Path | None:
    for path in (
        _THIS_DIR / "PhysicianEval_Result_Report.csv",
        _LEGACY_NESTED_DIR / "PhysicianEval_Result_Report.csv",
    ):
        if path.is_file():
            return path.resolve()
    return None


def main() -> None:
    args = parse_args()
    phys_gpt4o = (paths.TABLES / "PhysicianEval_Result_Report.csv").resolve()
    classic = (paths.TABLES / "Result_Report.csv").resolve()
    if args.input is not None:
        input_csv = args.input.resolve()
    elif (found := _resolve_default_physician_csv()) is not None:
        input_csv = found
    elif phys_gpt4o.is_file():
        input_csv = phys_gpt4o
    else:
        input_csv = classic
    output_pdf = (args.output or (DEFAULT_FIGURES_DIR / "MedPAIR_Result_per_category.pdf")).resolve()

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

    legacy_pdf = _LEGACY_NESTED_DIR / "MedPAIR_Result_per_category.pdf"
    if legacy_pdf.resolve() != output_pdf.resolve() and _LEGACY_NESTED_DIR.is_dir():
        shutil.copy2(output_pdf, legacy_pdf)
        print(f"Wrote {legacy_pdf}")


if __name__ == "__main__":
    main()
