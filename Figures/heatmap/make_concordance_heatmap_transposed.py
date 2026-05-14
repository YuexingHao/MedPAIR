#!/usr/bin/env python3
"""Transposed concordance heatmap (upper-triangular).

Reads concordance_10x10_SR_CC_jaccard_high.csv and produces a single
heatmap whose row/column axes are swapped relative to
concordance_heatmap_plus_overall.pdf. The bar-chart panel is dropped.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_CSV = _THIS_DIR / "concordance_10x10_SR_CC_jaccard_high.csv"
_OUT_PDF = _THIS_DIR / "concordance_heatmap_transposed.pdf"
_OUT_PNG = _THIS_DIR / "concordance_heatmap_transposed.png"

ROW_ORDER_ORIG = [
    "GPT-4o (SR)",
    "Qwen-14B (CC)",
    "Qwen-14B (SR)",
    "Qwen-72B (CC)",
    "Qwen-72B (SR)",
    "Llama-70B (CC)",
    "Llama-70B (SR)",
    "MedGemma-27B (SR)",
    "GPT-5 (SR)",
]
COL_ORDER_ORIG = [
    "Human",
    "GPT-4o (SR)",
    "Qwen-14B (CC)",
    "Qwen-14B (SR)",
    "Qwen-72B (CC)",
    "Qwen-72B (SR)",
    "Llama-70B (CC)",
    "Llama-70B (SR)",
    "MedGemma-27B (SR)",
]


def main() -> None:
    plt.rcParams["font.family"] = "Nimbus Sans"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    df = pd.read_csv(_CSV, index_col=0)

    # Original layout: rows = ROW_ORDER_ORIG (9), cols = COL_ORDER_ORIG (9).
    # Lower-triangular — cell (i, j) filled iff j <= i.
    orig = df.loc[ROW_ORDER_ORIG, COL_ORDER_ORIG].to_numpy(dtype=float)

    # Transpose: what was a row becomes a column, what was a column becomes a row.
    data = orig.T
    row_labels = COL_ORDER_ORIG  # new y-axis
    col_labels = list(ROW_ORDER_ORIG)  # new x-axis

    # Flip columns so the filled triangle sits on the left (anti-diagonal mask).
    data = data[:, ::-1]
    col_labels = col_labels[::-1]

    n_rows = len(row_labels)
    n_cols = len(col_labels)

    # After the column flip, filled cells are those with i + j <= n_cols - 1.
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            if i + j > n_cols - 1:
                mask[i, j] = True

    # Split into the "Human" row (top) and the model-vs-model triangle (bottom).
    data_top, mask_top = data[:1], mask[:1]
    data_bot, mask_bot = data[1:], mask[1:]
    row_labels_top = [row_labels[0]]
    row_labels_bot = row_labels[1:]

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")

    # Color scale from visible cells only (matches the original colorbar range ~30-85).
    visible = data[~mask]
    vmin = float(np.floor(np.nanmin(visible)))
    vmax = float(np.ceil(np.nanmax(visible)))

    cell_fs = 24  # cell-number font size
    tick_fs = 20

    fig = plt.figure(figsize=(15, 10), dpi=300)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, len(data_bot)],
        hspace=0.08,  # small gap between Human row and triangle
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    im_top = ax_top.imshow(
        np.ma.array(data_top, mask=mask_top),
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    im_bot = ax_bot.imshow(
        np.ma.array(data_bot, mask=mask_bot),
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )

    def annotate(ax, data_arr, mask_arr):
        nr, nc = data_arr.shape
        for i in range(nr):
            for j in range(nc):
                if mask_arr[i, j] or np.isnan(data_arr[i, j]):
                    continue
                val = data_arr[i, j]
                norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if norm_val > 0.55 else "#1f2937"
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=cell_fs, color=color,
                )

    annotate(ax_top, data_top, mask_top)
    annotate(ax_bot, data_bot, mask_bot)

    # X-axis labels on top of the Human row.
    ax_top.set_xticks(np.arange(n_cols))
    ax_top.set_xticklabels(col_labels, rotation=45, ha="left", fontsize=tick_fs)
    ax_top.xaxis.tick_top()
    ax_top.xaxis.set_label_position("top")
    ax_top.set_yticks([0])
    ax_top.set_yticklabels(row_labels_top, fontsize=tick_fs)

    # Bottom triangle: hide x-ticks (already shown on top), keep row labels.
    ax_bot.set_xticks([])
    ax_bot.set_yticks(np.arange(len(row_labels_bot)))
    ax_bot.set_yticklabels(row_labels_bot, fontsize=tick_fs)

    for ax in (ax_top, ax_bot):
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(
        im_bot, ax=[ax_top, ax_bot], shrink=0.9, pad=0.02, fraction=0.06, aspect=18,
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=22, length=0)

    fig.savefig(_OUT_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(_OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {_OUT_PDF}")
    print(f"Wrote {_OUT_PNG}")


if __name__ == "__main__":
    main()
