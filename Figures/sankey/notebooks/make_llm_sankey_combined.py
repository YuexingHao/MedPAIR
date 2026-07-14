#!/usr/bin/env python3
"""Build a style-aligned combined Sankey figure from two 2x3 source PNGs.

Top row source: llm_sankey_all_models_without_physician_relevant.png
Bottom row source: llm_sankey_all_models_without_physician.png

This compositor:
- normalizes typography across rows,
- adds larger row-level titles,
- uses per-column model headers once,
- shows only R1/R2 lines above each panel (no repeated model names in row 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd
import time
from sankey_counts_from_raw import compute_all_summaries, counts_tuple_map


MODELS = ["Qwen-72B", "Llama-70B", "Qwen-14B", "MedGemma-27B", "GPT4o", "GPT 5"]

# Top row in the combined figure
TOP_COUNTS = {
    "Qwen-72B": (527, 283),
    "Llama-70B": (605, 515),
    "Qwen-14B": (534, 272),
    "MedGemma-27B": (250, 86),
    "GPT4o": (690, 359),
    "GPT 5": (805, 758),
}

# Bottom row in the combined figure
BOTTOM_COUNTS = {
    "Qwen-72B": (527, 418),
    "Llama-70B": (605, 296),
    "Qwen-14B": (534, 456),
    "MedGemma-27B": (250, 56),
    "GPT4o": (690, 361),
    "GPT 5": (805, 689),
}

# Third row (Random) in the combined figure
RANDOM_COUNTS = {
    "Qwen-72B": (527, 518),
    "Llama-70B": (605, 538),
    "Qwen-14B": (534, 494),
    "MedGemma-27B": (250, 211),
    "GPT4o": (690, 612),
    "GPT 5": (805, 747),
}


def _extract_letter(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    su = s.upper()
    su = su.replace("<ANSWER>", "").replace("</ANSWER>", "")
    import re

    m = re.search(r"\bOPTION\s*[\(\[]?\s*([A-J])\s*[\)\]]?\b", su)
    if m:
        return m.group(1)
    m = re.search(r"\bANSWER\s*[:\-]?\s*[\(\[]?\s*([A-J])\s*[\)\]]?\b", su)
    if m:
        return m.group(1)
    m = re.fullmatch(r"[\(\[]?\s*([A-J])\s*[\)\]]?", su)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", su)
    if m:
        return m.group(1)
    return None


def _random_r2_counts(repo: Path) -> dict[str, tuple[int, int]]:
    """Compute R1/R2 counts for the Random row on the 933 Origin subset.

    R2 is strict: missing/unparseable predictions are treated as wrong.
    """
    origins = set(
        pd.read_csv(
            repo / "Physician_Labels" / "2026-03_March" / "933_Clinician_Student_Majority_Vote.csv",
            usecols=["Origin"],
        )["Origin"]
        .astype(str)
        .str.strip()
    )

    configs = {
        "Qwen-72B": (
            repo / "After_PT_Removal" / "Qwen2.5-72B-Instruct" / "results" / "predictions" / "Qwen_72B_predictions_on_Random.csv",
            "qwen72b_direct_prediction",
            "qwen72b_extracted_answer",
            "answer_corr",
        ),
        "Llama-70B": (
            repo / "After_PT_Removal" / "Llama-70B" / "results" / "predictions" / "Llama70B_predictions_on_Random.csv",
            "llama70b_direct_prediction",
            "llama70b_extracted_answer",
            "answer_corr",
        ),
        "Qwen-14B": (
            repo / "After_PT_Removal" / "Qwen2.5-14B-Instruct" / "results" / "predictions" / "Qwen_14B_predictions_on_Random.csv",
            "qwen14b_direct_prediction",
            "qwen14b_extracted_answer",
            "answer_corr",
        ),
        "MedGemma-27B": (
            repo / "After_PT_Removal" / "MedGemma-27b-text-it" / "results" / "predictions" / "MedGemma27B_predictions_on_Random.csv",
            "medgemma_direct_prediction",
            "medgemma_extracted_answer",
            "answer_corr",
        ),
        "GPT4o": (
            repo / "After_PT_Removal" / "GPT4o" / "results" / "predictions" / "gpt4o_predictions_on_Random.csv",
            "gpt4o_direct_prediction",
            "gpt_letter",
            "answer_letter",
        ),
        "GPT 5": (
            repo / "After_PT_Removal" / "GPT5" / "results" / "predictions" / "gpt5_predictions_on_Random.csv",
            "gpt5_direct_prediction",
            "gpt_letter",
            "answer_letter",
        ),
    }

    r1_counts = {
        "Qwen-72B": TOP_COUNTS["Qwen-72B"][0],
        "Llama-70B": TOP_COUNTS["Llama-70B"][0],
        "Qwen-14B": TOP_COUNTS["Qwen-14B"][0],
        "MedGemma-27B": TOP_COUNTS["MedGemma-27B"][0],
        "GPT4o": TOP_COUNTS["GPT4o"][0],
        "GPT 5": TOP_COUNTS["GPT 5"][0],
    }

    out: dict[str, tuple[int, int]] = {}
    valid = set("ABCDEFGHIJ")

    def safe_read_csv(path: Path) -> pd.DataFrame:
        # Files can be observed mid-write (e.g., active SLURM run). Retry, then degrade gracefully.
        last_err: Exception | None = None
        for _ in range(4):
            try:
                return pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.2)
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed reading {path}: {last_err or exc}") from exc

    for model, (path, pred_col, parsed_col, answer_col) in configs.items():
        df = safe_read_csv(path)
        df["Origin"] = df["Origin"].astype(str).str.strip()
        sub = df[df["Origin"].isin(origins)].copy()
        pred = sub[pred_col].apply(_extract_letter)
        if parsed_col in sub.columns:
            fallback = sub[parsed_col].apply(_extract_letter)
            pred = pred.where(pred.notna(), fallback)
        truth = sub[answer_col].apply(_extract_letter)

        pred = pred.where(pred.isin(valid))
        truth = truth.where(truth.isin(valid))
        correct = int(((pred == truth) & pred.notna() & truth.notna()).sum())
        out[model] = (r1_counts[model], correct)

    return out


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _cell_bounds(w: int, h: int) -> list[tuple[int, int, int, int]]:
    xs = [round(i * w / 3) for i in range(4)]
    ys = [round(i * h / 2) for i in range(3)]
    order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    bounds: list[tuple[int, int, int, int]] = []
    for r, c in order:
        bounds.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))
    return bounds


def _extract_panel_bbox(cell: np.ndarray) -> tuple[int, int, int, int]:
    # Prefer colored Sankey geometry over black subtitle text.
    mx = cell.max(axis=2).astype(np.int16)
    mn = cell.min(axis=2).astype(np.int16)
    chroma = mx - mn

    # Capture bright Sankey geometry while excluding dark title text.
    mask = (chroma > 35) & (mx > 120)
    yy, xx = np.where(mask)

    # Fallback if color-only detection is too strict.
    if yy.size == 0:
        mask = mn < 245
        yy, xx = np.where(mask)

    if yy.size == 0:
        h, w = cell.shape[:2]
        return 0, 0, w, h

    x0 = int(xx.min())
    x1 = int(xx.max()) + 1
    y0 = int(yy.min())
    y1 = int(yy.max()) + 1

    # Keep generous geometry padding so source/target nodes and top links are
    # never clipped when panels are normalized.
    pad_x = 90
    pad_y = 36
    h, w = cell.shape[:2]
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return x0, y0, x1, y1


def extract_panels(img: Image.Image) -> list[Image.Image]:
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    panels: list[Image.Image] = []
    for x0, y0, x1, y1 in _cell_bounds(w, h):
        cell = arr[y0:y1, x0:x1]
        cx0, cy0, cx1, cy1 = _extract_panel_bbox(cell)
        crop = cell[cy0:cy1, cx0:cx1]
        # Remove residual top/bottom whitespace/header text by tracking rows
        # that actually contain colored Sankey geometry. Use a very low row
        # threshold so thin top links are preserved.
        mx = crop.max(axis=2).astype(np.int16)
        mn = crop.min(axis=2).astype(np.int16)
        mask = (mx > 120) & ((mx - mn) > 35)
        row_counts = mask.sum(axis=1)
        active_rows = np.where(row_counts >= 4)[0]
        if active_rows.size > 0:
            top = max(0, int(active_rows[0]) - 8)
            bottom = min(crop.shape[0], int(active_rows[-1]) + 10)
            crop = crop[top:bottom]
        panels.append(Image.fromarray(crop))
    return panels


def fit_panel(panel: Image.Image, target_w: int, target_h: int, *, inner_pad: int = 16) -> Image.Image:
    avail_w = max(1, target_w - 2 * inner_pad)
    avail_h = max(1, target_h - 2 * inner_pad)
    contained = ImageOps.contain(panel, (avail_w, avail_h), method=Image.Resampling.LANCZOS)
    out = Image.new("RGB", (target_w, target_h), "white")
    ox = inner_pad + (avail_w - contained.width) // 2
    oy = inner_pad + (avail_h - contained.height) // 2
    out.paste(contained, (ox, oy))
    return out


def remove_top_text_ink(panel: Image.Image, band_px: int) -> Image.Image:
    arr = np.array(panel.convert("RGB"))
    h = arr.shape[0]
    band = max(0, min(h, band_px))
    if band == 0:
        return panel
    # Hard-clear a thin top strip where legacy panel titles live.
    clear_band = max(0, min(h, int(band * 0.75)))
    if clear_band > 0:
        arr[:clear_band] = 255
    top = arr[:band]
    dark_ink = top.max(axis=2) < 165
    top[dark_ink] = 255
    arr[:band] = top
    return Image.fromarray(arr)


def crop_top(panel: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return panel
    w, h = panel.size
    top = min(max(0, px), max(0, h - 10))
    return panel.crop((0, top, w, h))


def centered_text(draw: ImageDraw.ImageDraw, x_center: int, y_top: int, text: str, font: ImageFont.ImageFont, fill: str = "#0f2347", line_spacing: int = 8) -> int:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=line_spacing)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text((x_center - tw // 2, y_top), text, font=font, fill=fill, align="center", spacing=line_spacing)
    return th


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    fig_dir = repo / "Figures" / "sankey" / "figures"

    summaries = compute_all_summaries(repo)
    counts = counts_tuple_map(summaries)
    top_counts = {m: counts[m]["Relevant"] for m in MODELS}
    bottom_counts = {m: counts[m]["Irrelevant"] for m in MODELS}
    random_counts = {m: counts[m]["Random"] for m in MODELS}

    top_src = fig_dir / "llm_sankey_all_models_without_physician_relevant.png"
    bottom_src = fig_dir / "llm_sankey_all_models_without_physician.png"
    random_src = fig_dir / "_tmp_random" / "llm_sankey_all_models_without_physician_relevant.png"
    out_path = fig_dir / "llm_sankey_combined.png"

    top_img = Image.open(top_src).convert("RGB")
    bottom_img = Image.open(bottom_src).convert("RGB")
    random_img = Image.open(random_src).convert("RGB")

    top_panels = extract_panels(top_img)
    bottom_panels = extract_panels(bottom_img)
    random_panels = extract_panels(random_img)

    # Normalize all panels to one fixed display size for stable typography.
    panel_w = 1200
    panel_h = 820

    # Keep full extracted flow geometry; avoid extra top trimming that can clip
    # upper links/bars on some panels.
    top_panels = [fit_panel(p, panel_w, panel_h) for p in top_panels]
    bottom_panels = [fit_panel(p, panel_w, panel_h) for p in bottom_panels]
    random_panels = [fit_panel(p, panel_w, panel_h) for p in random_panels]
    # Remove embedded titles from source images
    top_panels = [remove_top_text_ink(p, 120) for p in top_panels]
    bottom_panels = [remove_top_text_ink(p, 120) for p in bottom_panels]
    random_panels = [remove_top_text_ink(p, 120) for p in random_panels]

    margin_x = 28
    margin_y = 18
    col_gap = 20
    section_gap = 24

    font_scale = 2.0
    font_model = load_font(int(34 * font_scale), bold=False)
    font_row_title = load_font(int(34 * font_scale), bold=False)
    font_stats = load_font(int(28 * font_scale), bold=False)

    model_h = int(96 * font_scale)
    row_title_h = int(68 * font_scale)
    row3_title_h = int(120 * font_scale)
    stats_h = int(160 * font_scale)
    gap_10 = int(10 * font_scale)
    gap_8 = int(8 * font_scale)
    stats_line_spacing = int(6 * font_scale)
    row3_line_spacing = int(3 * font_scale)

    width = margin_x * 2 + panel_w * 6 + col_gap * 5
    height = (
        margin_y
        + model_h
        + gap_10
        + row_title_h
        + gap_8
        + stats_h
        + gap_10
        + panel_h
        + section_gap
        + row_title_h
        + gap_8
        + stats_h
        + gap_10
        + panel_h
        + section_gap
        + row3_title_h
        + gap_8
        + stats_h
        + gap_10
        + panel_h
        + margin_y
    )

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x_centers = [margin_x + i * (panel_w + col_gap) + panel_w // 2 for i in range(6)]

    y = margin_y

    # Column headers shown once.
    for i, model in enumerate(MODELS):
        centered_text(draw, x_centers[i], y, model, font_model, fill="#0f2347")
    y += model_h + gap_10

    # Row 1 title (larger)
    centered_text(
        draw,
        width // 2,
        y,
        "Physician-Annotated Relevant Sentences Only",
        font_row_title,
        fill="#0f2347",
        line_spacing=0,
    )
    y += row_title_h + gap_8

    # Row 1 R1/R2 text
    for i, model in enumerate(MODELS):
        r1, r2 = top_counts[model]
        txt = f"R1 Correct: {r1}\nR2 Correct: {r2}"
        centered_text(draw, x_centers[i], y, txt, font_stats, fill="#0f2347", line_spacing=stats_line_spacing)
    y += stats_h + gap_10

    # Row 1 panels
    for i, panel in enumerate(top_panels):
        x = margin_x + i * (panel_w + col_gap)
        canvas.paste(panel, (x, y))
    y += panel_h + section_gap

    # Separator
    draw.line([(margin_x // 2, y - section_gap // 2), (width - margin_x // 2, y - section_gap // 2)], fill="#d8e2f1", width=max(3, int(3 * font_scale // 2)))

    # Row 2 title (larger)
    centered_text(
        draw,
        width // 2,
        y,
        "Physician-Annotated Irrelevant Sentences Only",
        font_row_title,
        fill="#0f2347",
        line_spacing=0,
    )
    y += row_title_h + gap_8

    # Row 2 R1/R2 text only (no repeated model names).
    for i, model in enumerate(MODELS):
        r1, r2 = bottom_counts[model]
        txt = f"R1 Correct: {r1}\nR2 Correct: {r2}"
        centered_text(draw, x_centers[i], y, txt, font_stats, fill="#0f2347", line_spacing=stats_line_spacing)
    y += stats_h + gap_10

    # Row 2 panels
    for i, panel in enumerate(bottom_panels):
        x = margin_x + i * (panel_w + col_gap)
        canvas.paste(panel, (x, y))
    y += panel_h + section_gap

    # Separator
    draw.line([(margin_x // 2, y - section_gap // 2), (width - margin_x // 2, y - section_gap // 2)], fill="#d8e2f1", width=max(3, int(3 * font_scale // 2)))

    # Row 3 title
    centered_text(
        draw,
        width // 2,
        y,
        "With Random Sentences Only\n(Same Number as Physician-Annotated Relevant Sentences)",
        font_row_title,
        fill="#0f2347",
        line_spacing=row3_line_spacing,
    )
    y += row3_title_h + gap_8

    # Row 3 R1/R2 text
    for i, model in enumerate(MODELS):
        r1, r2 = random_counts[model]
        txt = f"R1 Correct: {r1}\nR2 Correct: {r2}"
        centered_text(draw, x_centers[i], y, txt, font_stats, fill="#0f2347", line_spacing=stats_line_spacing)
    y += stats_h + gap_10

    # Row 3 panels
    for i, panel in enumerate(random_panels):
        x = margin_x + i * (panel_w + col_gap)
        canvas.paste(panel, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
