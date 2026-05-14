#!/usr/bin/env python3
"""
Build the combined LLM Sankey HTML (and optional PNG) from Expert-933 spurious counts.

Reads ``Figures/sankey/data/sankey_spurious_expert933.json`` produced by
``Figures/sankey/compute_sankey_spurious_data.py`` (the same run also writes
``sankey_spurious_expert933_results.csv``; this script does **not** read that tidy CSV).

**Physicians (933 QAs)** — four MedPAIR sources on the left like the LLM panels: Round 1 mass
per source is cohort size (``n_origins``); Round 2 uses clinician majority vote vs gold from
``physicians_round2_mj`` in the JSON (or
``Physician_Labels/Apr1_2026_Data/Round2_933_MJ_accuracy_by_data_source.csv`` if the key is absent).

Same layout as the manuscript notebook: Physicians top-center; six models in a 3×2 grid below.

**Subplot grid (1-based row, col as in Plotly):**

- Row 1: empty corners | **Physicians** (center) | empty corners — ``subplot_titles`` must list
  **only the seven** panels that exist (Plotly drops ``None`` cells but only consumes the first
  *N* title strings for *N* real subplots; passing nine titles mislabels every panel).
- Row 2: **Qwen 72B** | **Llama-70B** | **Qwen-14B**
- Row 3: **MedGemma-27B** | **GPT4o** | **GPT 5**

**Title lines** “R1 Correct” / “R2 Correct” are **sums over the four MedPAIR sources**. For LLM
panels these are model correct counts (Original vs Trainee Removed). For **Physicians**, R1 is
the sum of per-source cohort sizes (933 total) and R2 Correct is the sum of majority-vote
correct QAs vs gold (see ``sankey_spurious_expert933_results.csv`` physician rows).

**Percentages:** pooled spurious % appears on the **red sink node**; per-source breakdown is
in the data CSV / caption (link labels are omitted so text does not print on colored flows).
When spurious count is positive, the red node shows *pooled* spurious %
= sum(spurious) / sum(R1 correct). In-panel sink labels are short (“Spurious X%”).
If spurious is **zero** (e.g. Llama-70B with current inputs), the red sink node is **omitted**
so only the green sink appears—matching R1 Correct = R2 Correct in the title.

Run from repo root (or any cwd under the repo)::

  python Figures/sankey/notebooks/make_llm_sankey_all_models.py
  python Figures/sankey/notebooks/make_llm_sankey_all_models.py --no-png

**Figure size:** layout is **16:9** (``FIG_WIDTH`` × ``FIG_WIDTH×9/16``). ``vertical_spacing``
between subplot rows is set high so titles do not overlap Sankeys; raise ``FIG_WIDTH`` if needed.

**PNG export:** Plotly’s default static export (Kaleido + bundled Chromium). On many headless
servers that browser exits immediately; the script then tries **system** Chrome/Chromium via
``--headless=new`` and ``--screenshot=`` on the written HTML (set ``CHROMIUM_PATH`` or
``--chromium-binary`` if the binary is not on ``PATH``). Use ``--no-chromium-fallback`` to skip
that step, or ``--no-png`` to write HTML only. Legacy ``--png-engine orca`` exists if Orca is
installed (Plotly deprecates Orca in favor of Kaleido).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_THIS_DIR = Path(__file__).resolve().parent

# Figure geometry: **16:9** (``FIG_HEIGHT = FIG_WIDTH * 9/16``). Use a wide canvas so seven
# Sankeys + subplot titles do not overlap; increase ``FIG_WIDTH`` if text still feels tight.
FIG_WIDTH = 2560
FIG_ASPECT = (16, 9)  # w:h
FIG_HEIGHT = int(round(FIG_WIDTH * FIG_ASPECT[1] / FIG_ASPECT[0]))

# In-panel Sankey labels: ``go.Sankey`` uses trace ``textfont`` for node labels (there is no
# ``node.font`` in Plotly’s Sankey schema). Use a clearly large ``size`` so PNG export matches.
SANKEY_NODE_FONT = dict(size=34, family="Arial", color="#111827")
# Figure-wide default (axes/legend — minimal effect here but keeps exports consistent).
LAYOUT_FONT_SIZE = 36
# Subplot titles: ``format_panel_title`` uses explicit **px** in HTML so Kaleido/Chromium match
# on-screen size (annotation ``font.size`` alone does not reliably style ``<span>`` children).
SUBPLOT_TITLE_FONT_PX = 36
SUBPLOT_SUBTITLE_FONT_PX = 26
# Nudge each subplot title upward (pixels) so it sits clearer of the Sankey below.
SUBPLOT_TITLE_YSHIFT = 34

# MedPAIR sources (same order as ``compute_sankey_spurious_data.DATASET_KEYS`` / JSON).
DATASET_KEYS = ["MMLU", "JAMA", "MedXpert", "MedBullets"]
_PHYS_MJ_SOURCE_NORMALIZE = {
    "mmlu": "MMLU",
    "jama": "JAMA",
    "medxpert": "MedXpert",
    "medbullets": "MedBullets",
}


def _find_repo_root() -> Path:
    for p in [_THIS_DIR, *_THIS_DIR.parents]:
        if (
            p / "Physician_Labels" / "Mar2_2026_Data" / "933_Clinician_Student_Majority_Vote.csv"
        ).is_file():
            return p
    raise FileNotFoundError(
        "Could not find NeuRIPS25 repo root (933_Clinician_Student_Majority_Vote.csv). "
        f"Searched upward from {_THIS_DIR}"
    )


def _load_payload(repo: Path) -> dict:
    path = repo / "Figures" / "sankey" / "data" / "sankey_spurious_expert933.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Run: python Figures/sankey/compute_sankey_spurious_data.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _datasets_from_payload(payload: dict) -> dict:
    """Strip JSON to keys expected by Sankey helpers (round1_correct, round2_incorrect lists)."""
    out = {}
    for ds, block in payload["datasets"].items():
        out[ds] = {
            "round1_correct": block["round1_correct"],
            "round2_incorrect": block["round2_incorrect"],
        }
        if "max_length" in block:
            out[ds]["max_length"] = block["max_length"]
    return out


def create_overall_sankey_with_percentages(
    datasets: dict,
    dataset_list: list[str],
    model_idx: int,
) -> dict:
    """Overall Sankey diagram with percentages on flows.

    If there is no spurious flow (R1-correct → R2-wrong), the red sink node is omitted
    so the diagram matches the title (R1 Correct = R2 Correct).
    """
    round1_totals = {ds: datasets[ds]["round1_correct"][model_idx] for ds in dataset_list}
    round2_incorrect_totals = {
        ds: datasets[ds]["round2_incorrect"][model_idx] for ds in dataset_list
    }
    round2_correct_totals = {
        ds: round1_totals[ds] - round2_incorrect_totals[ds] for ds in dataset_list
    }

    total_round1 = sum(round1_totals.values())
    total_spurious = sum(round2_incorrect_totals.values())
    overall_spurious_pct = (total_spurious / total_round1 * 100) if total_round1 > 0 else 0

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    link_labels: list[str] = []

    n_src = len(dataset_list)
    base_colors = ["#3b82f6", "#0ea5e9", "#06b6d4", "#14b8a6"]

    if total_spurious == 0:
        green_idx = n_src
        nodes = dataset_list + ["R2 correct"]
        node_colors = base_colors + ["#10b981"]
        for i, ds in enumerate(dataset_list):
            if round2_correct_totals[ds] > 0:
                sources.append(i)
                targets.append(green_idx)
                values.append(round2_correct_totals[ds])
                link_colors.append("rgba(16, 185, 129, 0.4)")
                link_labels.append("")
    else:
        green_idx = n_src
        red_idx = n_src + 1
        nodes = dataset_list + [
            "R2 correct",
            f"Spurious {overall_spurious_pct:.1f}%",
        ]
        node_colors = base_colors + ["#10b981", "#ef4444"]
        for i, ds in enumerate(dataset_list):
            if round2_correct_totals[ds] > 0:
                sources.append(i)
                targets.append(green_idx)
                values.append(round2_correct_totals[ds])
                link_colors.append("rgba(16, 185, 129, 0.4)")
                link_labels.append("")

            if round2_incorrect_totals[ds] > 0:
                sources.append(i)
                targets.append(red_idx)
                values.append(round2_incorrect_totals[ds])
                link_colors.append("rgba(239, 68, 68, 0.4)")
                pct = (
                    (round2_incorrect_totals[ds] / round1_totals[ds] * 100)
                    if round1_totals[ds] > 0
                    else 0
                )
                # Omit link labels — they draw on top of colored flows and clash with node text.
                link_labels.append("")

    return {
        "nodes": nodes,
        "node_colors": node_colors,
        "sources": sources,
        "targets": targets,
        "values": values,
        "link_colors": link_colors,
        "link_labels": link_labels,
    }


def _load_physicians_round2_mj_from_csv(path: Path) -> dict[str, dict[str, int]]:
    """Per source: R1 mass = n_origins; red flow = MJ answer wrong vs gold in Round 2."""
    out: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["data_source_corr_x"].strip().lower()
            ds = _PHYS_MJ_SOURCE_NORMALIZE[key]
            n_o = int(row["n_origins"])
            n_c = int(row["n_correct"])
            out[ds] = {"round1_correct": n_o, "round2_incorrect": n_o - n_c}
    missing = [ds for ds in DATASET_KEYS if ds not in out]
    if missing:
        raise ValueError(f"{path}: missing sources {missing}")
    return out


def _physicians_datasets_list_shape(block: dict[str, dict[str, int]]) -> dict:
    """One pseudo-model index 0 for ``create_overall_sankey_with_percentages``."""
    shaped: dict = {}
    for ds in DATASET_KEYS:
        b = block[ds]
        r1 = int(b["round1_correct"])
        r2i = int(b["round2_incorrect"])
        shaped[ds] = {
            "round1_correct": [r1],
            "round2_incorrect": [r2i],
            "max_length": r1,
        }
    return shaped


def _load_physicians_block(repo: Path, payload: dict) -> dict[str, dict[str, int]]:
    b = payload.get("physicians_round2_mj")
    if isinstance(b, dict) and b:
        return b
    path = (
        repo
        / "Physician_Labels"
        / "Apr1_2026_Data"
        / "Round2_933_MJ_accuracy_by_data_source.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Physicians Sankey: expected key physicians_round2_mj in JSON or {path}"
        )
    return _load_physicians_round2_mj_from_csv(path)


def get_model_totals(datasets: dict, models: list[str], model_name: str) -> tuple[int, int]:
    model_idx = models.index(model_name)
    keys = list(datasets.keys())
    total_round1 = sum(datasets[ds]["round1_correct"][model_idx] for ds in keys)
    total_round2_correct = sum(
        datasets[ds]["round1_correct"][model_idx] - datasets[ds]["round2_incorrect"][model_idx]
        for ds in keys
    )
    return total_round1, total_round2_correct


def format_panel_title(
    name: str,
    total_round1: int,
    total_round2_correct: int,
    *,
    title_px: int | None = None,
    subtitle_px: int | None = None,
) -> str:
    tp = title_px if title_px is not None else SUBPLOT_TITLE_FONT_PX
    sp = subtitle_px if subtitle_px is not None else SUBPLOT_SUBTITLE_FONT_PX
    return (
        f"<span style='font-size:{tp}px;font-weight:600;font-family:Arial;color:#1f2937'>"
        f"{name} ({total_round1} QAs)</span><br>"
        f"<span style='font-size:{sp}px;font-family:Arial;color:#4b5563'>"
        f"R1 Correct: {total_round1} | R2 Correct: {total_round2_correct}</span>"
    )


def _sankey_fixed_xy_llm(n_src: int, *, has_red: bool) -> tuple[list[float], list[float]]:
    """Sources left, sinks right; y increases downward. Sinks slightly inset vertically."""
    x_left, x_right = 0.04, 0.988
    if n_src <= 1:
        ys_left = [0.5]
    else:
        ys_left = [0.1 + i * (0.8 / (n_src - 1)) for i in range(n_src)]
    xs = [x_left] * n_src
    ys = list(ys_left)
    if not has_red:
        xs.append(x_right)
        ys.append(0.5)
    else:
        xs.extend([x_right, x_right])
        ys.extend([0.2, 0.8])
    return xs, ys


def build_figure(
    datasets: dict,
    models: list[str],
    physicians_datasets: dict,
    *,
    dataset_list: list[str] | None = None,
    model_order: list[str] | None = None,
    include_physicians: bool = True,
) -> go.Figure:
    dataset_list = dataset_list or list(DATASET_KEYS)
    model_order = model_order or [
        "Qwen 72B",
        "Llama-70B",
        "Qwen-14B",
        "MedGemma-27B",
        "GPT4o",
        "GPT 5",
    ]

    _title_px = SUBPLOT_TITLE_FONT_PX if include_physicians else 56
    _subtitle_px = SUBPLOT_SUBTITLE_FONT_PX if include_physicians else 42
    subplot_titles: list[str] = []
    if include_physicians:
        phy_r1 = sum(int(physicians_datasets[ds]["round1_correct"][0]) for ds in dataset_list)
        phy_r2c = sum(
            int(physicians_datasets[ds]["round1_correct"][0])
            - int(physicians_datasets[ds]["round2_incorrect"][0])
            for ds in dataset_list
        )
        subplot_titles.append(format_panel_title("Physicians", phy_r1, phy_r2c))
    for model in model_order:
        tr1, tr2c = get_model_totals(datasets, models, model)
        subplot_titles.append(
            format_panel_title(model, tr1, tr2c, title_px=_title_px, subtitle_px=_subtitle_px)
        )

    if include_physicians:
        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[
                [None, {"type": "sankey"}, None],
                [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
                [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
            ],
            subplot_titles=subplot_titles,
            vertical_spacing=0.18,
            horizontal_spacing=0.07,
            row_heights=[1 / 3, 1 / 3, 1 / 3],
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
                [{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}],
            ],
            subplot_titles=subplot_titles,
            vertical_spacing=0.32,
            horizontal_spacing=0.07,
            row_heights=[1 / 2, 1 / 2],
        )

    if include_physicians:
        phys_data = create_overall_sankey_with_percentages(
            physicians_datasets, dataset_list, model_idx=0
        )
        has_red_phys = len(phys_data["nodes"]) == len(dataset_list) + 2
        pxs, pys = _sankey_fixed_xy_llm(len(dataset_list), has_red=has_red_phys)
        fig.add_trace(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=28,
                    thickness=18,
                    line=dict(color="white", width=2),
                    label=phys_data["nodes"],
                    color=phys_data["node_colors"],
                    x=pxs,
                    y=pys,
                ),
                link=dict(
                    source=phys_data["sources"],
                    target=phys_data["targets"],
                    value=phys_data["values"],
                    color=phys_data["link_colors"],
                    label=phys_data["link_labels"],
                ),
                textfont=SANKEY_NODE_FONT,
            ),
            row=1,
            col=2,
        )

    _row_offset = 2 if include_physicians else 1
    for idx, model_name in enumerate(model_order):
        model_idx = models.index(model_name)
        row = (idx // 3) + _row_offset
        col = (idx % 3) + 1
        sankey_data = create_overall_sankey_with_percentages(
            datasets, dataset_list, model_idx
        )
        has_red = len(sankey_data["nodes"]) == len(dataset_list) + 2
        lxs, lys = _sankey_fixed_xy_llm(len(dataset_list), has_red=has_red)
        fig.add_trace(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=28,
                    thickness=18,
                    line=dict(color="white", width=2),
                    label=sankey_data["nodes"],
                    color=sankey_data["node_colors"],
                    x=lxs,
                    y=lys,
                ),
                link=dict(
                    source=sankey_data["sources"],
                    target=sankey_data["targets"],
                    value=sankey_data["values"],
                    color=sankey_data["link_colors"],
                    label=sankey_data["link_labels"],
                ),
                textfont=SANKEY_NODE_FONT,
            ),
            row=row,
            col=col,
        )

    # Ensure every Sankey trace picks up ``textfont`` (some exporters only respect a late update).
    fig.update_traces(selector=dict(type="sankey"), textfont=SANKEY_NODE_FONT)

    fig.update_layout(
        font=dict(size=LAYOUT_FONT_SIZE, family="Arial"),
        height=FIG_HEIGHT,
        width=FIG_WIDTH,
        margin=dict(l=32, r=32, t=260 if not include_physicians else 118, b=36),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    _extra_yshift = SUBPLOT_TITLE_YSHIFT if include_physicians else 90
    _ann_base_px = SUBPLOT_TITLE_FONT_PX if include_physicians else 56
    for annotation in fig.layout.annotations or []:
        annotation.font = dict(size=_ann_base_px, family="Arial", color="#1f2937")
        prev = annotation.yshift
        annotation.yshift = (prev if prev is not None else 0) + _extra_yshift

    return fig


def _chromium_executable_candidates(explicit: str | None) -> list[str]:
    """Ordered Chrome/Chromium paths for headless PNG fallback."""
    out: list[str] = []
    for x in (explicit, os.environ.get("CHROMIUM_PATH"), os.environ.get("GOOGLE_CHROME_BIN")):
        if x and str(x).strip():
            out.append(str(x).strip())
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "brave-browser",
    ):
        p = shutil.which(name)
        if p:
            out.append(p)
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def export_png_via_headless_chromium(
    html_path: Path,
    png_path: Path,
    *,
    width_px: int,
    height_px: int,
    chrome_binary: str | None,
    virtual_time_budget_ms: int = 30_000,
) -> tuple[bool, str]:
    """
    Render ``html_path`` with headless Chromium and write ``png_path``.

    Returns (success, last_error_or_exe). Matches Kaleido ``scale=3`` when
    width_px = FIG_WIDTH * 3 and height_px = FIG_HEIGHT * 3.
    """
    uri = html_path.resolve().as_uri()
    last_err = "no Chromium/Chrome candidate"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path.is_file():
        png_path.unlink()

    for exe in _chromium_executable_candidates(chrome_binary):
        cmd = [
            exe,
            "--headless=new",
            "--hide-scrollbars",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            f"--window-size={width_px},{height_px}",
            f"--virtual-time-budget={virtual_time_budget_ms}",
            f"--screenshot={png_path}",
            uri,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            last_err = f"{exe}: {e}"
            continue
        if proc.returncode == 0 and png_path.is_file() and png_path.stat().st_size > 0:
            return True, exe
        tail = (proc.stderr or proc.stdout or "").strip()
        last_err = f"{exe} rc={proc.returncode}" + (f" {tail[:200]}" if tail else "")
    return False, last_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined LLM Sankey from Expert-933 JSON.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for llm_sankey_all_models.html (default: Figures/sankey/figures)",
    )
    p.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG export (avoids kaleido/orca dependency).",
    )
    p.add_argument(
        "--png-engine",
        choices=("kaleido", "orca"),
        default="kaleido",
        help="kaleido: default Plotly export (omit engine= to reduce deprecation noise). "
        "orca: legacy static export if `orca` is on PATH (deprecated by Plotly).",
    )
    p.add_argument(
        "--chromium-binary",
        type=str,
        default=None,
        help="Path to Chrome/Chromium for PNG fallback after Kaleido fails "
        "(or set CHROMIUM_PATH / GOOGLE_CHROME_BIN).",
    )
    p.add_argument(
        "--no-physicians",
        action="store_true",
        help="Omit the Physicians row; write llm_sankey_all_models_without_physician.{html,png}.",
    )
    p.add_argument(
        "--no-chromium-fallback",
        action="store_true",
        help="Do not try headless system Chrome/Chromium when Kaleido fails.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = _find_repo_root()
    payload = _load_payload(repo)
    datasets = _datasets_from_payload(payload)
    models = payload["models"]
    phy_block = _load_physicians_block(repo, payload)
    physicians_datasets = _physicians_datasets_list_shape(phy_block)

    out_dir = (args.output_dir or (repo / "Figures" / "sankey" / "figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "llm_sankey_all_models_without_physician" if args.no_physicians else "llm_sankey_all_models"
    html_path = out_dir / f"{stem}.html"
    png_path = out_dir / f"{stem}.png"

    fig = build_figure(
        datasets,
        models,
        physicians_datasets,
        include_physicians=not args.no_physicians,
    )
    fig.write_html(str(html_path))
    print(f"Wrote {html_path}")
    print(
        "Subplot layout (row, col) → model: "
        "(1,2) Physicians; "
        "(2,1) Qwen 72B, (2,2) Llama-70B, (2,3) Qwen-14B; "
        "(3,1) MedGemma-27B, (3,2) GPT4o, (3,3) GPT 5"
    )

    if not args.no_png:

        def _err_summary(exc: BaseException) -> str:
            line = str(exc).strip().split("\n", 1)[0]
            return line if len(line) <= 240 else line[:237] + "…"

        try:
            if args.png_engine == "orca":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    fig.write_image(
                        str(png_path),
                        width=FIG_WIDTH,
                        height=FIG_HEIGHT,
                        scale=3,
                        engine="orca",
                    )
                print(f"Wrote {png_path} (engine=orca)")
            else:
                # Avoid passing engine= so Plotly uses Kaleido without engine-deprecation spam.
                fig.write_image(str(png_path), width=FIG_WIDTH, height=FIG_HEIGHT, scale=3)
                print(f"Wrote {png_path}")
        except Exception as e:
            print(
                f"PNG not written — {png_path.name} unchanged (HTML was updated). "
                f"{_err_summary(e)}",
                file=sys.stderr,
            )
            if not args.no_chromium_fallback:
                ok, info = export_png_via_headless_chromium(
                    html_path,
                    png_path,
                    width_px=FIG_WIDTH * 3,
                    height_px=FIG_HEIGHT * 3,
                    chrome_binary=args.chromium_binary,
                )
                if ok:
                    print(f"Wrote {png_path} (headless Chromium: {info})")
                else:
                    print(
                        f"Headless Chromium fallback failed: {info}",
                        file=sys.stderr,
                    )
                    print(
                        "Hint: install Chrome/Chromium on PATH, set CHROMIUM_PATH, or run "
                        "`kaleido_get_chrome` (or `kaleido.get_chrome_sync()` in Python); otherwise "
                        "use --no-png and open the .html locally to export PNG.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "Hint: on headless nodes Kaleido often fails — omit --no-chromium-fallback "
                    "to try system Chrome, or use --no-png and export from a desktop browser.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
