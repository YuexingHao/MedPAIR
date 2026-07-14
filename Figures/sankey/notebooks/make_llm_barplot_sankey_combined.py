#!/usr/bin/env python3
"""Create per-model stacked bar charts from raw CSV-derived Sankey counts.

Output:
  Figures/sankey/figures/llm_barplot_sankey_combined.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from sankey_counts_from_raw import MODELS, SETTINGS, compute_all_summaries, counts_tuple_map


TOTAL_QAS = 933
TEXT_COLOR = "#000000"
FONT_FAMILY = "sans-serif"
TITLE_SIZE = 34
AXIS_LABEL_SIZE = 34
TICK_SIZE = 28
VALUE_SIZE = 22
LEGEND_SIZE = 26
BAR_HEIGHT = 0.58


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    out_path = repo / "Figures" / "sankey" / "figures" / "llm_barplot_sankey_combined.png"

    summaries = compute_all_summaries(repo)
    counts = counts_tuple_map(summaries)

    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    # green = stayed correct in R2, red = spurious (R1 correct -> R2 wrong), pink = R1 incorrect.
    color_r2 = "#34C99A"
    color_spurious = "#F16667"
    color_r1_incorrect = "#FFB7B7"

    fig, axes = plt.subplots(2, 3, figsize=(26, 14), sharex=True, sharey=True)
    axes = axes.ravel()
    y = np.arange(len(SETTINGS))

    for ax, model in zip(axes, MODELS):
        r1 = np.array([counts[model][k][0] for k in SETTINGS], dtype=float)
        r2 = np.array([counts[model][k][1] for k in SETTINGS], dtype=float)
        spurious = np.maximum(r1 - r2, 0)
        r1_incorrect = np.maximum(TOTAL_QAS - r1, 0)

        ax.barh(y, r2, color=color_r2, height=BAR_HEIGHT, label="R2 Correct")
        ax.barh(y, spurious, left=r2, color=color_spurious, height=BAR_HEIGHT, label="R1 Correct -> R2 Incorrect")
        ax.barh(
            y,
            r1_incorrect,
            left=r2 + spurious,
            color=color_r1_incorrect,
            height=BAR_HEIGHT,
            label="R1 Incorrect",
        )

        ax.set_title(model, fontsize=TITLE_SIZE, fontweight="normal", color=TEXT_COLOR, pad=8)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{s}-Only" for s in SETTINGS], fontsize=TICK_SIZE, color=TEXT_COLOR
        )
        ax.set_xlim(0, TOTAL_QAS)
        ax.grid(axis="x", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=TICK_SIZE, colors=TEXT_COLOR)
        ax.tick_params(axis="y", labelsize=TICK_SIZE, colors=TEXT_COLOR)

        for i in range(len(SETTINGS)):
            pct = (spurious[i] / r1[i] * 100.0) if r1[i] > 0 else 0.0
            label = f"{pct:.1f}%"
            ax.text(
                r2[i] + spurious[i] / 2,
                y[i],
                label,
                ha="center",
                va="center",
                fontsize=VALUE_SIZE + 6,
                fontweight="normal",
                color="#000000",
            )

    axes[0].invert_yaxis()

    fig.supxlabel("Question Count (Total = 933)", fontsize=AXIS_LABEL_SIZE, color=TEXT_COLOR, y=0.12)
    fig.supylabel("Sentence Selection Setting", fontsize=AXIS_LABEL_SIZE, color=TEXT_COLOR, x=0.015)
    handles = [
        Patch(facecolor=color_r2, label="R2 Correct"),
        Patch(facecolor=color_spurious, label="R1 Correct -> R2 Incorrect"),
        Patch(facecolor=color_r1_incorrect, label="R1 Incorrect"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=LEGEND_SIZE,
        borderpad=0.6,
        handlelength=1.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.2)
    for txt in legend.get_texts():
        txt.set_color(TEXT_COLOR)

    fig.subplots_adjust(left=0.18, right=0.995, top=0.95, bottom=0.21, wspace=0.03, hspace=0.12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
