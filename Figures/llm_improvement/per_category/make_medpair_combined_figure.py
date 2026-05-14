#!/usr/bin/env python3
"""
Combined MedPAIR figure: **row 1** matches ``MedPAIR_Result_per_category.pdf`` (three physician
cohorts); **rows 2–3** match the lower half of ``MedPAIR_Result_per_dataset.pdf`` (MMLU / JAMA /
MedXpert / MedBullets on the 933 Expert QA cohort by MedPAIR data source—no spanning ``Total`` row).

**Inputs**

- **Physician / category CSV** (default: same search as ``make_medpair_result_figure.py``): columns
  ``MMLU``, ``Jama``, ``MedXpert`` for Expert QA (933), Hard QA (733), Impossible QA (334).
- **933-by-source CSV** (default: same search as ``Figures/llm_improvement/per_dataset/make_medpair_result_figure.py``):
  ``Total``, ``MMLU``, ``Jama``, ``MedXpert``, ``Medbullets`` for the bottom four panels (the
  ``Total`` column is read for consistency but only the four source panels are plotted).

Default output: ``Figures/llm_improvement/per_category/MedPAIR_Result_combined.pdf`` (and a copy under
``Figures/llm_improvement/per_category/per_category/`` when that folder exists).

**Physician + Trainee** (optional): if both exist, appends one x-axis category using
``Clinician_Student_q1_MJ_accuracy_by_data_source.csv`` (from ``Clinician_Student_Analysis.py``,
full Text Relevance cohort) as **Round 1 / Original Accuracy**, and
``Round2_933_MJ_accuracy_by_data_source.csv`` (from ``compute_round2_933_mj.py``) as **Round 2 /
Trainee Removed**. Row 1 **Expert QA (933)** shows pooled R1/R2; Hard / Impossible panels do not plot
this series (Expert-only). Rows 2–3 use per-source accuracies (MMLU / JAMA / MedXpert / MedBullets).

Run from repo root::

  python Figures/llm_improvement/per_category/make_medpair_combined_figure.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent.parent
_GPT4O_ROOT = _WORKSPACE_ROOT / "After_PT_Removal" / "GPT4o"
_PER_DATASET_DIR = _WORKSPACE_ROOT / "Figures" / "llm_improvement" / "per_dataset"
_LEGACY_NESTED_DIR = _THIS_DIR / "per_category"
_PER_CATEGORY_NESTED = _LEGACY_NESTED_DIR

DEFAULT_OUTPUT = _THIS_DIR / "MedPAIR_Result_combined.pdf"
DEFAULT_TOTAL_OUTPUT = _THIS_DIR / "MedPAIR_Result_combined_total.pdf"
_MCNEMAR_CSV = _THIS_DIR / "medpair_mcnemar_bonferroni.csv"


def _load_mcnemar_stars() -> dict[tuple[str, str], str]:
    if not _MCNEMAR_CSV.is_file():
        return {}
    df = pd.read_csv(_MCNEMAR_CSV)
    out: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        stars = str(row.get("signif", "") or "")
        if stars and stars != "ns":
            out[(str(row["base_model"]), str(row["data_source"]))] = stars
    return out

if not (_GPT4O_ROOT / "paths.py").is_file():
    print(
        f"Expected GPT4o layout at {_GPT4O_ROOT} (paths.py missing).",
        file=sys.stderr,
    )
    sys.exit(1)
if str(_GPT4O_ROOT) not in sys.path:
    sys.path.insert(0, str(_GPT4O_ROOT))
import paths  # noqa: E402

_CATEGORY_COLS_TITLES = (
    ("MMLU", "Expert QA (933)"),
    ("Jama", "Hard QA (733)"),
    ("MedXpert", "Impossible QA (334)"),
)
_DATASET_COLS_TITLES = (
    ("MMLU", "MMLU"),
    ("Jama", "JAMA"),
    ("MedXpert", "MedXpert"),
    ("Medbullets", "MedBullets"),
)

_PHYSICIAN_TRAINEE_MODEL = "Physician + Trainee"
_R1_Q1_BY_SRC = (
    _WORKSPACE_ROOT
    / "Physician_Labels"
    / "Mar2_2026_Data"
    / "Clinician_Student_q1_MJ_accuracy_by_data_source.csv"
)
_R2_MJ_BY_SRC = (
    _WORKSPACE_ROOT
    / "Physician_Labels"
    / "Apr1_2026_Data"
    / "Round2_933_MJ_accuracy_by_data_source.csv"
)
_SOURCE_KEY_TO_DATASET_COL = {
    "mmlu": "MMLU",
    "jama": "Jama",
    "medxpert": "MedXpert",
    "medbullets": "Medbullets",
}

# Draw after other removal scatters: large markers (s=1200) overlap when accuracies are close;
# Qwen-72B / GPT-4o Removed points can sit on top of Trainee Removed if drawn later at same zorder.
_LABELERS_DRAW_SCATTER_LAST = frozenset(
    {
        "Trainee Removed",
        "Trainee Removed (248 QAs)",
    }
)


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_metric_columns(_resolve_labeler_column(df))
    return out[out["Low+Irr Labelers"].astype(str).str.strip() != "Trainee Removed (IRR)"]


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _default_physician_csv() -> Path | None:
    return _first_existing(
        [
            _THIS_DIR / "PhysicianEval_Result_Report.csv",
            _LEGACY_NESTED_DIR / "PhysicianEval_Result_Report.csv",
            (paths.TABLES / "PhysicianEval_Result_Report.csv").resolve(),
        ]
    )


def _default_dataset_csv() -> Path | None:
    expert_candidates = [
        _PER_DATASET_DIR / "ExpertQA_933_by_data_source.csv",
        _PER_CATEGORY_NESTED / "ExpertQA_933_by_data_source.csv",
    ]
    phys_candidates = [
        _PER_DATASET_DIR / "PhysicianEval_Result_Report.csv",
        _PER_CATEGORY_NESTED / "PhysicianEval_Result_Report.csv",
        _THIS_DIR / "PhysicianEval_Result_Report.csv",
        (paths.TABLES / "PhysicianEval_Result_Report.csv").resolve(),
    ]
    if (found := _first_existing(expert_candidates)) is not None:
        return found
    return _first_existing(phys_candidates)


def _pooled_accuracy_pct(df: pd.DataFrame) -> float:
    return 100.0 * float(df["n_correct"].sum()) / float(df["n_origins"].sum())


def _by_source_accuracy_pct(df: pd.DataFrame) -> dict[str, float]:
    """Accept Round 1 (``data_source_corr``) or Round 2 (``data_source_corr_x``) by-source tables."""
    out: dict[str, float] = {}
    if "data_source_corr_x" in df.columns:
        src_col = "data_source_corr_x"
    elif "data_source_corr" in df.columns:
        src_col = "data_source_corr"
    else:
        return out
    for _, row in df.iterrows():
        key = str(row[src_col]).strip().lower()
        col = _SOURCE_KEY_TO_DATASET_COL.get(key)
        if col is not None:
            out[col] = float(row["accuracy_pct"])
    return out


def augment_physician_trainee(
    df_physician: pd.DataFrame,
    df_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append ``Physician + Trainee`` rows: Round 1 from ``Clinician_Student_Analysis`` by-source CSV,
    Round 2 (trainee removed) from ``compute_round2_933_mj.py`` → ``Round2_933_MJ_accuracy_by_data_source.csv``.
    """
    if not _R1_Q1_BY_SRC.is_file() or not _R2_MJ_BY_SRC.is_file():
        print(
            f"Skipping {_PHYSICIAN_TRAINEE_MODEL}: need\n  {_R1_Q1_BY_SRC}\n  {_R2_MJ_BY_SRC}",
            file=sys.stderr,
        )
        return df_physician, df_dataset

    r1 = pd.read_csv(_R1_Q1_BY_SRC)
    r2 = pd.read_csv(_R2_MJ_BY_SRC)
    t1 = _pooled_accuracy_pct(r1)
    t2 = _pooled_accuracy_pct(r2)
    m1 = _by_source_accuracy_pct(r1)
    m2 = _by_source_accuracy_pct(r2)

    p_orig = {
        "Base Model": _PHYSICIAN_TRAINEE_MODEL,
        "Low+Irr Labelers": "Original Accuracy",
        "Total": t1,
        "MMLU": t1,
        "Jama": np.nan,
        "MedXpert": np.nan,
    }
    p_tr = {
        "Base Model": _PHYSICIAN_TRAINEE_MODEL,
        "Low+Irr Labelers": "Trainee Removed",
        "Total": t2,
        "MMLU": t2,
        "Jama": np.nan,
        "MedXpert": np.nan,
    }
    for d in (p_orig, p_tr):
        for c in df_physician.columns:
            if c not in d:
                d[c] = np.nan

    ds_orig = {
        "Base Model": _PHYSICIAN_TRAINEE_MODEL,
        "Low+Irr Labelers": "Original Accuracy",
        "Total": t1,
        "MMLU": m1.get("MMLU", np.nan),
        "Jama": m1.get("Jama", np.nan),
        "MedXpert": m1.get("MedXpert", np.nan),
        "Medbullets": m1.get("Medbullets", np.nan),
    }
    ds_tr = {
        "Base Model": _PHYSICIAN_TRAINEE_MODEL,
        "Low+Irr Labelers": "Trainee Removed",
        "Total": t2,
        "MMLU": m2.get("MMLU", np.nan),
        "Jama": m2.get("Jama", np.nan),
        "MedXpert": m2.get("MedXpert", np.nan),
        "Medbullets": m2.get("Medbullets", np.nan),
    }
    for d in (ds_orig, ds_tr):
        for c in df_dataset.columns:
            if c not in d:
                d[c] = np.nan

    extra_p = pd.DataFrame([p_orig, p_tr])
    extra_d = pd.DataFrame([ds_orig, ds_tr])
    print(
        f"Appended {_PHYSICIAN_TRAINEE_MODEL}: Original {t1:.2f}%  Trainee Removed {t2:.2f}%",
        flush=True,
    )
    return (
        pd.concat([df_physician, extra_p], ignore_index=True),
        pd.concat([df_dataset, extra_d], ignore_index=True),
    )


def _canonical_base_models(df: pd.DataFrame) -> list[str]:
    order = [
        _PHYSICIAN_TRAINEE_MODEL,
        "GPT4o",
        "Qwen-14B",
        "Qwen 72B",
        "Llama-70B",
        "MedGemma-27B",
        "GPT 5",
        "Trainee",
    ]
    present = set(df["Base Model"].astype(str).str.strip())
    return [m for m in order if m in present]


def make_combined_figure(
    df_physician: pd.DataFrame,
    df_dataset: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    section: Literal["all", "total", "datasets"] = "all",
) -> None:
    dfp = _prep_df(df_physician)
    dfd = _prep_df(df_dataset)
    dfp, dfd = augment_physician_trainee(dfp, dfd)
    stars_lookup = _load_mcnemar_stars()

    for col, _t in _CATEGORY_COLS_TITLES:
        if col not in dfp.columns:
            raise KeyError(f"Physician CSV missing column {col!r}. Have: {dfp.columns.tolist()}")
    for col, _t in _DATASET_COLS_TITLES:
        if col not in dfd.columns:
            raise KeyError(f"Dataset CSV missing column {col!r}. Have: {dfd.columns.tolist()}")

    bp = set(dfp["Base Model"].astype(str).str.strip())
    bd = set(dfd["Base Model"].astype(str).str.strip())
    order = [
        _PHYSICIAN_TRAINEE_MODEL,
        "GPT4o",
        "Qwen-14B",
        "Qwen 72B",
        "Llama-70B",
        "MedGemma-27B",
        "GPT 5",
        "Trainee",
    ]
    base_models = [m for m in order if m in bp and m in bd]
    if not base_models:
        raise ValueError(
            "No shared Base Model labels between physician and dataset tables after filtering."
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
        _PHYSICIAN_TRAINEE_MODEL: "",
    }

    _x_cat_spacing = 2.25
    _x_margin = 0.9
    _first_dated_idx = next(
        (i for i, m in enumerate(base_models) if release_dates.get(m, "")),
        0,
    )
    x_vline_left = (float(_first_dated_idx) - 0.5) * _x_cat_spacing
    x_vline_mid: float | None = None
    if "Llama-70B" in base_models and "MedGemma-27B" in base_models:
        i_l = base_models.index("Llama-70B")
        i_m = base_models.index("MedGemma-27B")
        x_vline_mid = 0.5 * (float(i_l) * _x_cat_spacing + float(i_m) * _x_cat_spacing)

    if section == "total":
        fig = plt.figure(figsize=(28, 18))
        gs = fig.add_gridspec(1, 1)
        ax_total = fig.add_subplot(gs[0, 0])
        axes_bottom = []
    elif section == "datasets":
        fig = plt.figure(figsize=(43, 37))
        gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.2)
        ax_total = None
        axes_bottom = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
    else:
        fig = plt.figure(figsize=(43, 55))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.6, wspace=0.2)
        ax_total = fig.add_subplot(gs[0, :])
        axes_bottom = [
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
        ]
    all_axes = ([ax_total] if ax_total is not None else []) + axes_bottom

    Vline = Literal["none", "first", "rest"]

    def _plot_axis(
        ax: plt.Axes,
        df: pd.DataFrame,
        col: str,
        title: str,
        *,
        global_idx: int,
        vline: Vline,
        title_fs: float,
        show_ylabel: bool,
        add_best_legend: bool,
        x_axis_models: list[str] | None = None,
        show_release_dates: bool = True,
        stats_key: str | None = None,
    ) -> None:
        """
        ``x_axis_models``: if set, only these models are plotted and labeled on the x-axis
        (same order as ``base_models`` but may omit entries, e.g. Physician + Trainee on Hard/Impossible).
        """
        def _x(i: float) -> float:
            return float(i) * _x_cat_spacing

        models_axis = base_models if x_axis_models is None else x_axis_models
        n_models = len(models_axis)
        x_positions = np.arange(n_models) * _x_cat_spacing
        added_labels: set[str] = set()
        best_perf_label_added = False

        for i, model in enumerate(models_axis):
            model_data = df[df["Base Model"] == model]
            if len(model_data) == 0:
                continue

            original_acc_row = model_data[
                model_data["Low+Irr Labelers"] == "Original Accuracy"
            ]
            original_acc_value = None
            if len(original_acc_row) > 0:
                v0 = original_acc_row[col].iloc[0]
                if pd.notna(v0):
                    original_acc_value = float(v0)

            other_labelers = model_data[
                model_data["Low+Irr Labelers"] != "Original Accuracy"
            ]
            ol_valid = other_labelers.dropna(subset=[col])
            if len(ol_valid) > 0:
                best_row = ol_valid.loc[ol_valid[col].idxmax()]
                best_value = best_row[col]
                best_labeler = str(best_row["Low+Irr Labelers"])
                if pd.isna(best_value):
                    best_value = None
                else:
                    best_value = float(best_value)
            else:
                best_value = None
                best_labeler = ""

            # Draw Original Accuracy first (thick lines); non-original scatters last with high
            # zorder so Trainee Removed blue dots are not covered by black lines or dotted best.
            for _, row in model_data.iterrows():
                if row["Low+Irr Labelers"] != "Original Accuracy":
                    continue
                value = row[col]
                if pd.isna(value):
                    continue
                value = float(value)
                color = labeler_colors.get("Original Accuracy", "#000000")
                label = (
                    "Original Accuracy" if "Original Accuracy" not in added_labels else ""
                )
                if label:
                    added_labels.add("Original Accuracy")
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

            if (
                best_value is not None
                and original_acc_value is not None
                and original_acc_value != 0
            ):
                line_label = ""
                if add_best_legend and not best_perf_label_added:
                    line_label = "Best Performance"
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

                pct_improvement = best_value - original_acc_value
                text_color = "#00a000" if pct_improvement >= 0 else "#d32f2f"
                arrow_color = "#00a000" if pct_improvement >= 0 else "#d32f2f"
                pct_text = (
                    f"+{pct_improvement:.1f}%"
                    if pct_improvement >= 0
                    else f"{pct_improvement:.1f}%"
                )

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

                if stats_key is not None and best_labeler == "Trainee Removed":
                    stars = stars_lookup.get((model, stats_key), "")
                    if stars:
                        ax.text(
                            _x(i),
                            text_y + 11,
                            stars,
                            fontsize=80,
                            fontweight="bold",
                            color="#d62728",
                            ha="center",
                            va="bottom",
                            zorder=7,
                        )

            if show_release_dates and release_dates.get(model, ""):
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

            _scatter_other_z = 15
            _scatter_trainee_removed_z = 25

            def _draw_removal_scatter(row: pd.Series, *, z: int) -> None:
                labeler = row["Low+Irr Labelers"]
                value = row[col]
                if pd.isna(value):
                    return
                value_f = float(value)
                color = labeler_colors.get(labeler, "#7f7f7f")
                label = labeler if labeler not in added_labels else ""
                if label:
                    added_labels.add(labeler)
                marker = labeler_markers.get(labeler, "o")
                ax.scatter(
                    _x(i),
                    value_f,
                    c=color,
                    s=1200,
                    alpha=0.9,
                    marker=marker,
                    edgecolors="white",
                    linewidth=3,
                    zorder=z,
                    label=label,
                )

            last_pass: list[pd.Series] = []
            for _, row in model_data.iterrows():
                labeler = row["Low+Irr Labelers"]
                if labeler == "Original Accuracy":
                    continue
                if pd.isna(row[col]):
                    continue
                if labeler in _LABELERS_DRAW_SCATTER_LAST:
                    last_pass.append(row)
                else:
                    _draw_removal_scatter(row, z=_scatter_other_z)
            for row in last_pass:
                _draw_removal_scatter(row, z=_scatter_trainee_removed_z)

        if vline == "first" and show_release_dates:
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
        elif vline == "rest" and x_vline_mid is not None and show_release_dates:
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
            models_axis, rotation=45, ha="right", fontsize=44, fontweight="500"
        )

        if show_ylabel:
            ax.set_ylabel(
                "Accuracy (%)", fontsize=48, fontweight="bold", labelpad=15
            )
        ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=55)
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

    if ax_total is not None:
        _plot_axis(
            ax_total,
            dfp,
            "MMLU",
            "Total",
            global_idx=0,
            vline="first",
            title_fs=72,
            show_ylabel=True,
            add_best_legend=True,
            show_release_dates=False,
            stats_key="Total",
        )

    if axes_bottom:
        for j, (ax, (col, title)) in enumerate(zip(axes_bottom, _DATASET_COLS_TITLES)):
            ds_first = title == "MMLU"
            vline: Vline = "first" if ds_first else "rest"
            _plot_axis(
                ax,
                dfd,
                col,
                title,
                global_idx=1 + j,
                vline=vline,
                title_fs=72,
                show_ylabel=(j in (0, 2)),
                add_best_legend=(section == "datasets" and j == 0),
                stats_key=col,
            )

    handles, labels = all_axes[0].get_legend_handles_labels()
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

    if section == "total":
        legend_fontsize = 26
        legend_title_fontsize = 30
        legend_markerscale = 1.2
        legend_labelspacing = 1.0
        legend_borderpad = 1.0
        legend_handletextpad = 0.9
        adjust_right = 0.78
    elif section == "datasets":
        legend_fontsize = 38
        legend_title_fontsize = 44
        legend_markerscale = 1.5
        legend_labelspacing = 1.3
        legend_borderpad = 1.3
        legend_handletextpad = 1.1
        adjust_right = 0.85
    else:
        legend_fontsize = 45
        legend_title_fontsize = 52
        legend_markerscale = 1.8
        legend_labelspacing = 1.5
        legend_borderpad = 1.5
        legend_handletextpad = 1.2
        adjust_right = 0.86

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="center left",
        bbox_to_anchor=(adjust_right + 0.02, 0.5),
        fontsize=legend_fontsize,
        framealpha=1,
        markerscale=legend_markerscale,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Labeler Configuration",
        title_fontsize=legend_title_fontsize,
        labelspacing=legend_labelspacing,
        borderpad=legend_borderpad,
        handletextpad=legend_handletextpad,
    )

    plt.subplots_adjust(left=0.01, right=adjust_right, top=0.96, bottom=0.08)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--physician-input",
        type=Path,
        default=None,
        help="Physician wide CSV (933 / Hard / Impossible columns).",
    )
    p.add_argument(
        "--dataset-input",
        type=Path,
        default=None,
        help="933-by-data-source wide CSV (Total + four source columns).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output PDF (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument("--show", action="store_true", help="Show interactive window after saving.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phys_path = (
        args.physician_input.resolve()
        if args.physician_input is not None
        else _default_physician_csv()
    )
    data_path = (
        args.dataset_input.resolve()
        if args.dataset_input is not None
        else _default_dataset_csv()
    )
    out = (args.output or DEFAULT_OUTPUT).resolve()

    if phys_path is None or not phys_path.is_file():
        print(f"Physician CSV not found: {phys_path}", file=sys.stderr)
        sys.exit(1)
    if data_path is None or not data_path.is_file():
        print(f"Dataset CSV not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    if data_path.name == "PhysicianEval_Result_Report.csv":
        print(
            "Warning: dataset table is PhysicianEval_Result_Report.csv; bottom panels are not "
            "true MedPAIR data-source splits within 933. Prefer ExpertQA_933_by_data_source.csv.",
            file=sys.stderr,
        )

    df_p = pd.read_csv(phys_path)
    df_d = pd.read_csv(data_path)

    total_out = out.with_name("MedPAIR_Result_combined_total.pdf")
    make_combined_figure(df_p, df_d, total_out, show=False, section="total")
    print(f"Wrote {total_out}")

    make_combined_figure(df_p, df_d, out, show=args.show, section="datasets")
    print(f"Wrote {out}")

    legacy_pdf = _LEGACY_NESTED_DIR / "MedPAIR_Result_combined.pdf"
    legacy_total_pdf = _LEGACY_NESTED_DIR / "MedPAIR_Result_combined_total.pdf"
    if _LEGACY_NESTED_DIR.is_dir():
        if legacy_pdf.resolve() != out.resolve():
            shutil.copy2(out, legacy_pdf)
            print(f"Wrote {legacy_pdf}")
        if legacy_total_pdf.resolve() != total_out.resolve():
            shutil.copy2(total_out, legacy_total_pdf)
            print(f"Wrote {legacy_total_pdf}")


if __name__ == "__main__":
    main()
