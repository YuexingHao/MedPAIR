from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PER_CATEGORY_DIR = THIS_DIR.parent / "per_category"
PER_CATEGORY_SCRIPT = PER_CATEGORY_DIR / "make_medpair_combined_figure.py"

OUT_PDF = THIS_DIR / "MedPAIR_Result_original_baseline.pdf"
OUT_TOTAL_PDF = THIS_DIR / "MedPAIR_Result_original_baseline_total.pdf"

MODEL_ORDER = [
    "GPT4o",
    "Qwen-14B",
    "Qwen-72B",
    "Llama-70B",
    "MedGemma-27B",
    "GPT-5",
    "Physician + Trainee",
]

METRICS = [
    ("MMLU", "MMLU"),
    ("Jama", "JAMA"),
    ("MedXpert", "MedXpert"),
    ("Medbullets", "MedBullets"),
]

COND_STYLES = {
    "Trainee Removed": {"color": "#2ca0ff", "marker": "o"},
    "GPT-4o Removed": {"color": "#16c60c", "marker": "^"},
    "GPT5 Removed": {"color": "#2b9348", "marker": "^"},
    "Qwen-72B Removed": {"color": "#ff9f1c", "marker": "s"},
    "Qwen-14B Removed": {"color": "#f27a32", "marker": "s"},
    "Llama-70b Removed": {"color": "#f8b37f", "marker": "s"},
    "MedGemma-27B Removed": {"color": "#e8bd77", "marker": "s"},
}


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _build_augmented_dataset_table() -> pd.DataFrame:
    if not PER_CATEGORY_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing script: {PER_CATEGORY_SCRIPT}")

    cat_mod = _load_module(PER_CATEGORY_SCRIPT, "medpair_per_category_current")
    legacy = cat_mod._load_legacy_module()

    in_csv = legacy._default_dataset_csv()
    df = pd.read_csv(in_csv)

    _, df_data = legacy.augment_physician_trainee(df, df)

    for fn in [cat_mod._compute_pt_qwen72_rows, cat_mod._compute_pt_gpt5_rows]:
        _, bottom_row = fn()
        df_data = cat_mod._upsert_row(df_data, bottom_row)

    return df_data


def _build_points(df_data: pd.DataFrame) -> pd.DataFrame:
    work = df_data.copy()
    work["Base Model"] = work["Base Model"].astype(str).str.strip()
    work["Low+Irr Labelers"] = work["Low+Irr Labelers"].astype(str).str.strip()

    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        d_model = work[work["Base Model"] == model]
        if d_model.empty:
            continue

        d_orig = d_model[d_model["Low+Irr Labelers"] == "Original Accuracy"]
        if d_orig.empty:
            continue
        orig_row = d_orig.iloc[0]

        for cond, style in COND_STYLES.items():
            d_cond = d_model[d_model["Low+Irr Labelers"] == cond]
            if d_cond.empty:
                continue
            cond_row = d_cond.iloc[0]

            for metric_col, metric_title in METRICS:
                if metric_col not in orig_row or metric_col not in cond_row:
                    continue
                orig = pd.to_numeric(orig_row[metric_col], errors="coerce")
                val = pd.to_numeric(cond_row[metric_col], errors="coerce")
                if pd.isna(orig) or pd.isna(val):
                    continue
                rows.append(
                    {
                        "metric_col": metric_col,
                        "metric_title": metric_title,
                        "base_model": model,
                        "condition": cond,
                        "delta": float(val - orig),
                        "orig": float(orig),
                        "color": style["color"],
                        "marker": style["marker"],
                    }
                )
    return pd.DataFrame(rows)


def _plot(points: pd.DataFrame, out_pdf: Path) -> None:
    if points.empty:
        raise RuntimeError("No points were generated for plotting.")

    max_abs_delta = float(np.nanmax(np.abs(points["delta"].to_numpy())))
    x_lim = max(5.0, max_abs_delta + 2.0)

    fig = plt.figure(figsize=(44, 28))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.62], wspace=0.18, hspace=0.28)

    axes = {
        "MMLU": fig.add_subplot(gs[0, 0]),
        "JAMA": fig.add_subplot(gs[0, 1]),
        "MedXpert": fig.add_subplot(gs[1, 0]),
        "MedBullets": fig.add_subplot(gs[1, 1]),
    }

    for metric_col, metric_title in METRICS:
        ax = axes[metric_title]
        sub_m = points[points["metric_col"] == metric_col]

        for cond, style in COND_STYLES.items():
            sub = sub_m[sub_m["condition"] == cond]
            if sub.empty:
                continue
            x = sub["delta"].to_numpy()
            y = sub["orig"].to_numpy()
            # Raise the Trainee (blue) markers above the rest.
            zorder = 6 if cond == "Trainee Removed" else 3
            ax.scatter(
                x,
                y,
                s=800,
                marker=style["marker"],
                c=style["color"],
                edgecolors="white",
                linewidths=0.9,
                alpha=0.96,
                zorder=zorder,
            )

        ax.axvline(0.0, color="#8a8a8a", linestyle="--", linewidth=2.2, alpha=0.85, zorder=1)
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.32, zorder=0)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(0, 100)

        ax.set_title(metric_title, fontsize=46, fontweight="normal", pad=12)
        ax.set_xlabel("Accuracy Change vs Original (%)", fontsize=40, labelpad=10)
        if metric_title in {"MMLU", "MedXpert"}:
            ax.set_ylabel("Original Accuracy (%)", fontsize=42, labelpad=12)

        ax.tick_params(axis="both", labelsize=30)

    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")

    handles = []
    for cond, style in COND_STYLES.items():
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=style["marker"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.9,
                markersize=22,
                label=cond,
            )
        )

    legend = legend_ax.legend(
        handles=handles,
        title="Labeler Configuration",
        loc="center",
        frameon=True,
        fancybox=True,
        framealpha=0.98,
        edgecolor="#b8b8b8",
        borderpad=1.1,
        labelspacing=1.05,
        handletextpad=0.8,
    )
    legend.get_title().set_fontsize(34)
    legend.get_title().set_fontweight("normal")
    for text in legend.get_texts():
        text.set_fontsize(29)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_data = _build_augmented_dataset_table()
    points = _build_points(df_data)
    _plot(points, OUT_PDF)
    shutil.copyfile(OUT_PDF, OUT_TOTAL_PDF)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_TOTAL_PDF}")


if __name__ == "__main__":
    main()
