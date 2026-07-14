#!/usr/bin/env python3
"""Regenerate Sankey panel PNGs directly from raw prediction CSV counts.

Outputs:
  Figures/sankey/figures/llm_sankey_all_models_without_physician_relevant.{html,png}
  Figures/sankey/figures/llm_sankey_all_models_without_physician.{html,png}
  Figures/sankey/figures/_tmp_random/llm_sankey_all_models_without_physician_relevant.{html,png}
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sankey_counts_from_raw import MODELS, SETTINGS, compute_all_summaries


def _load_base_module():
    pyc = (
        Path(__file__).resolve().parent
        / "__pycache__"
        / "make_llm_sankey_all_models_without_physician_relevant.cpython-313.pyc"
    )
    if not pyc.exists():
        raise FileNotFoundError(f"Missing compiled generator: {pyc}")
    loader = importlib.machinery.SourcelessFileLoader("sankey_base_r2total", str(pyc))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def render_panel_set(mod, summaries: list[dict], out_html: Path, out_png: Path) -> None:
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

    for idx, (s, (row, col)) in enumerate(zip(summaries, slots)):
        d = mod.sankey_for_summary(s)
        # Declutter: the four source labels (MMLU/JAMA/MedXpert/MedBullets) repeat in
        # every panel, so show them only on the leftmost grid column (keeps both rows
        # of the 2x3 standalone labeled). Keep the right-side sink labels ("R2 correct"
        # and the per-panel "Spurious X%") on every panel.
        labels = list(d["nodes"])
        if col != 1:
            labels = ["" if xpos < 0.5 else lab for lab, xpos in zip(labels, d["x"])]
        fig.add_trace(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=34,
                    thickness=20,
                    line=dict(color="white", width=2),
                    label=labels,
                    color=d["node_colors"],
                    x=d["x"],
                    y=d["y"],
                ),
                link=dict(
                    source=d["sources"],
                    target=d["targets"],
                    value=d["values"],
                    color=d["link_colors"],
                ),
                textfont=dict(size=50, family="Arial", color="#111827"),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        font=dict(size=40, family="Arial"),
        width=2560,
        height=1840,
        margin=dict(l=32, r=32, t=260, b=36),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for ann in fig.layout.annotations or []:
        ann.font = dict(size=56, family="Arial", color="#1f2937")
        ann.yshift = (ann.yshift or 0) + 90

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    fig.write_image(str(out_png), width=2560, height=1840, scale=3)
    print(f"Wrote {out_html}")
    print(f"Wrote {out_png}")


def main() -> None:
    mod = _load_base_module()
    repo = mod.find_repo_root()
    fig_dir = repo / "Figures" / "sankey" / "figures"

    summaries_by_setting = compute_all_summaries(repo)

    for setting in SETTINGS:
        summaries = [summaries_by_setting[setting][m] for m in MODELS]
        if setting == "Relevant":
            out_html = fig_dir / "llm_sankey_all_models_without_physician_relevant.html"
            out_png = fig_dir / "llm_sankey_all_models_without_physician_relevant.png"
        elif setting == "Irrelevant":
            out_html = fig_dir / "llm_sankey_all_models_without_physician.html"
            out_png = fig_dir / "llm_sankey_all_models_without_physician.png"
        else:
            out_html = fig_dir / "_tmp_random" / "llm_sankey_all_models_without_physician_relevant.html"
            out_png = fig_dir / "_tmp_random" / "llm_sankey_all_models_without_physician_relevant.png"

        render_panel_set(mod, summaries, out_html=out_html, out_png=out_png)


if __name__ == "__main__":
    main()
