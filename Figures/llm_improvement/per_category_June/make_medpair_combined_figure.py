from __future__ import annotations

import collections
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import norm


_THIS_DIR = Path(__file__).resolve().parent
_LEGACY_PYC_CANDIDATES = [
    _THIS_DIR / "make_medpair_combined_figure.pyc",
    _THIS_DIR / "__pycache__" / "make_medpair_combined_figure.cpython-313.pyc",
]
_MAY27_RAW = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "May27_2026_Data"
    / "Text Relevance Analysis Case View gpt5 phase - 052626.csv"
)
_R1_Q1_BY_SRC = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "Mar2_2026_Data"
    / "Clinician_Student_q1_MJ_accuracy_by_data_source.csv"
)
_R2_MJ_BY_SRC = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "Apr1_2026_Data"
    / "Round2_933_MJ_accuracy_by_data_source.csv"
)
_QWEN72_RAW = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "Jun19_2026_Data"
    / "Qwen72B_Predicted_High.csv"
)
_PT_SOURCE_MAP = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "May27_2026_Data"
    / "May27_2026_Origin_Summary_933.csv"
)
_PT_MODEL = "Physician + Trainee"
_GPT5_REMOVED = "GPT5 Removed"
_QWEN72_REMOVED = "Qwen-72B Removed"

# --- Self-Reported (SR) removal conditions -------------------------------
# The legacy compiled module hard-codes its label -> colour/marker tables and
# its legend order, so SR series are drawn onto the finished figure instead of
# being routed through the dataset CSV.
_SR_ROOT = _THIS_DIR.parent.parent.parent / "After_PT_Removal" / "SR_Predictions"
_SR_COHORT = (
    _THIS_DIR.parent.parent.parent
    / "Physician_Labels"
    / "May27_2026_Data"
    / "May27_2026_Origin_Summary_933.csv"
)
# condition directory -> legend label
_SR_CONDITIONS = {
    "Qwen-72B_Removed": "Qwen-72B SR Removed",
    "Qwen-14B_Removed": "Qwen-14B SR Removed",
    "Llama-70B_Removed": "Llama-70B SR Removed",
}
# legend label -> (colour, marker)
_SR_STYLE = {
    "Qwen-72B SR Removed": ("#8c1d40", "D"),
    "Qwen-14B SR Removed": ("#00796b", "D"),
    "Llama-70B SR Removed": ("#5d4037", "D"),
}
# prettified x-tick label -> SR prediction file stem
_SR_MODEL_FILES = {
    "GPT-4o": "GPT_4o",
    "GPT-5": "GPT_5",
    "Llama-70B": "Llama_70B",
    "Qwen-14B": "Qwen_14B",
    "Qwen-72B": "Qwen_72B",
    # MedGemma-27B's Qwen-14B/Qwen-72B SR runs are truncated (~12-14% answered)
    # and are dropped by the coverage gate; its Llama-70B-removed rerun covers
    # 95.5% and is supplied through _SR_EXTRA_FILES below.
    "MedGemma-27B": "MedGemma_27B",
}
_SR_PRED_COLS = (
    "Extracted_Answer",
    "gpt_letter",
    "gpt4o_direct_prediction",
    "gpt5_direct_prediction",
    "llama70b_direct_prediction",
    "qwen72b_direct_prediction",
    "qwen14b_direct_prediction",
    "medgemma27b_direct_prediction",
)
# Runs that exist only in the per-model results tree and were never copied into
# SR_Predictions/ (its README acknowledges the folder is incomplete). Paths are
# relative to After_PT_Removal/. Keyed by (legend label, pretty model name).
_SR_EXTRA_FILES = {
    ("Llama-70B SR Removed", "Qwen-72B"): (
        "Qwen2.5-72B-Instruct/results/predictions/"
        "[SR]_Qwen72B_predictions_on_llama70b_removed.csv"
    ),
    # Reruns of 2026-07-29 that replaced failed runs: Qwen-14B answered 8 of 933
    # before, GPT-5 and MedGemma-27B returned the literal string "Error" for
    # every item. MedGemma's was sharded and is used here merged.
    ("Llama-70B SR Removed", "Qwen-14B"): (
        "Qwen2.5-14B-Instruct/results/predictions/"
        "[SR]_Qwen14B_predictions_on_llama70b_removed.csv"
    ),
    ("Llama-70B SR Removed", "GPT-5"): (
        "GPT5/results/predictions/[SR]_GPT5_predictions_on_llama70b_removed.csv"
    ),
    ("Llama-70B SR Removed", "MedGemma-27B"): (
        "MedGemma-27b-text-it/results/predictions/"
        "[SR]_MedGemma27B_predictions_on_llama70b_removed_merged.csv"
    ),
}
_AFTER_PT_ROOT = _THIS_DIR.parent.parent.parent / "After_PT_Removal"
# Minimum share of the 933-question cohort a model must have actually answered
# before its SR point is plotted. Coverage per cell ranges from 100% down to
# ~12%; 0.90 keeps the near-complete runs and drops the truncated ones
# (MedGemma-27B at ~12-14%, Llama-70B on Qwen-14B-removed at ~80%).
_SR_MIN_COVERAGE_FRAC = 0.90
_SR_COHORT_N = 933
# Draw order: SR diamonds sit above the legacy markers, and the blue Trainee
# markers sit above everything (applied last, see _raise_trainee_markers_on_top).
_SR_ZORDER = 1200
_TRAINEE_ZORDER = 5000
# Legend wording changes applied to the finished figure (the legacy compiled
# module hard-codes its own labels).
# The legacy module labels the ContextCite conditions plainly as "X Removed".
# Now that the Self-Reported series sit alongside them, the CC ones are marked
# explicitly. ContextCite covers the four open-source models (Qwen-14B,
# Qwen-72B, Llama-70B, MedGemma-27B); GPT-4o and GPT-5 are self-report only, so
# they keep their existing labels.
_LEGEND_RENAMES = {
    "Trainee Removed": "Human Expert Removed",
    "Qwen-72B Removed": "Qwen-72B CC Removed",
    "Qwen-14B Removed": "Qwen-14B CC Removed",
    "Llama-70b Removed": "Llama-70B CC Removed",
    "MedGemma-27B Removed": "MedGemma-27B CC Removed",
}
# Legend keys are normalised to one footprint. The legacy handles are scatter
# markers drawn at s=_DATAPOINT_SIZE (1400 pt^2, ~37pt across), which overlap
# neighbouring rows; _LEGEND_KEY_AREA is the area they are shrunk to and
# _SR_LEGEND_MARKERSIZE is the matching diameter for the SR diamonds.
_LEGEND_KEY_AREA = 760.0
_SR_LEGEND_MARKERSIZE = 29.0
# Axis typography
_AXIS_LABEL_SIZE = 64
_TICK_LABEL_SIZE = 40
_TITLE_SIZE = 72
# Per-panel titles in the 2x2 grid are subtitles, so they sit below _TITLE_SIZE
# (which stays the main title size used by the Total figure).
_PANEL_TITLE_SIZE = 54
_FONT_FAMILY = "Inter Variable"
_FONT_PATH = Path.home() / ".local" / "share" / "fonts" / "InterVariable.ttf"
_DATAPOINT_SIZE = 1400.0
_N_CANDIDATES = np.array(
    [
        33,
        160,
        192,
        249,
        250,
        286,
        295,
        334,
        733,
        747,
        750,
        928,
        930,
        933,
        934,
        1996,
    ],
    dtype=int,
)


def _load_legacy_module():
    for pyc in _LEGACY_PYC_CANDIDATES:
        if pyc.is_file():
            spec = importlib.util.spec_from_file_location("_legacy_medpair_combined", pyc)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise FileNotFoundError(
        "Could not find legacy compiled module. Checked: "
        + ", ".join(str(p) for p in _LEGACY_PYC_CANDIDATES)
    )


def _norm_choice(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    m = re.search(r"[A-J]", s)
    return m.group(0) if m else ""


def _bern_std_pct(acc_pct: float | int | np.floating | None) -> float:
    if acc_pct is None or pd.isna(acc_pct):
        return float("nan")
    p = float(acc_pct) / 100.0
    return 100.0 * float(np.sqrt(max(p * (1.0 - p), 0.0)))


def _compute_pt_condition_rows(
    *,
    raw_csv: Path,
    condition_label: str,
    source_col: str | None = None,
    source_map_csv: Path | None = None,
    origin_strip_suffix: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {raw_csv}")

    raw = pd.read_csv(raw_csv)
    need = {"Origin", "q1", "Correct answer"}
    missing = need - set(raw.columns)
    if missing:
        raise KeyError(f"{raw_csv.name} missing columns: {sorted(missing)}")

    work = raw.copy()
    work["Origin"] = work["Origin"].astype(str).str.strip()
    if origin_strip_suffix:
        work["Origin"] = work["Origin"].str.replace(r"-phase\d+$", "", regex=True)
    work["choice"] = work["q1"].map(_norm_choice)
    work["correct"] = work["Correct answer"].map(_norm_choice)

    rows: list[dict[str, object]] = []
    for origin, g in work.groupby("Origin", sort=False):
        votes = [x for x in g["choice"].tolist() if x]
        c = collections.Counter(votes)
        if not c:
            mv = ""
            tie = True
        else:
            best = max(c.values())
            winners = sorted([k for k, v in c.items() if v == best])
            tie = len(winners) != 1
            mv = winners[0] if not tie else ""

        correct_series = g["correct"].dropna()
        correct = correct_series.iloc[0] if len(correct_series) else ""
        rows.append(
            {
                "Origin": origin,
                "mv_correct": (not tie) and bool(mv) and (mv == correct),
            }
        )

    per_origin = pd.DataFrame(rows)

    if source_col and source_col in work.columns:
        src = (
            work[["Origin", source_col]]
            .dropna(subset=[source_col])
            .drop_duplicates(subset=["Origin"])
            .rename(columns={source_col: "source"})
        )
    elif source_map_csv is not None:
        if not source_map_csv.is_file():
            raise FileNotFoundError(f"Missing source map CSV: {source_map_csv}")
        src_df = pd.read_csv(source_map_csv)
        if "Origin" not in src_df.columns or "data_source_corr_x" not in src_df.columns:
            raise KeyError(
                f"{source_map_csv.name} must include Origin and data_source_corr_x columns."
            )
        src = (
            src_df[["Origin", "data_source_corr_x"]]
            .copy()
            .dropna(subset=["Origin", "data_source_corr_x"])
            .drop_duplicates(subset=["Origin"])
            .rename(columns={"data_source_corr_x": "source"})
        )
    else:
        raise ValueError("Need either source_col or source_map_csv for per-source stats.")

    per_origin = per_origin.merge(src, on="Origin", how="left")
    if per_origin["source"].isna().any():
        n = int(per_origin["source"].isna().sum())
        raise ValueError(f"{condition_label}: missing data-source mapping for {n} origins.")

    per_origin["source"] = per_origin["source"].astype(str).str.strip().str.lower()
    per_origin["mv_correct"] = per_origin["mv_correct"].astype(float)

    overall = 100.0 * float(per_origin["mv_correct"].mean())
    by_src = 100.0 * per_origin.groupby("source")["mv_correct"].mean()

    src_map = {
        "mmlu": float(by_src.get("mmlu", np.nan)),
        "jama": float(by_src.get("jama", np.nan)),
        "medxpert": float(by_src.get("medxpert", np.nan)),
        "medbullets": float(by_src.get("medbullets", np.nan)),
    }

    top = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": condition_label,
        "Total": overall,
        "MMLU": overall,
        "Jama": float("nan"),
        "MedXpert": float("nan"),
        "Medbullets": float("nan"),
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(overall),
        "JAMA_STD": float("nan"),
        "MedXpert_STD": float("nan"),
        "MedBullets_STD": float("nan"),
    }
    bottom = {
        "Base Model": _PT_MODEL,
        "Low+Irr Labelers": condition_label,
        "Total": overall,
        "MMLU": src_map["mmlu"],
        "Jama": src_map["jama"],
        "MedXpert": src_map["medxpert"],
        "Medbullets": src_map["medbullets"],
        "Total_STD": _bern_std_pct(overall),
        "MMLU_STD": _bern_std_pct(src_map["mmlu"]),
        "JAMA_STD": _bern_std_pct(src_map["jama"]),
        "MedXpert_STD": _bern_std_pct(src_map["medxpert"]),
        "MedBullets_STD": _bern_std_pct(src_map["medbullets"]),
    }
    return top, bottom


def _compute_pt_gpt5_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_pt_condition_rows(
        raw_csv=_MAY27_RAW,
        condition_label=_GPT5_REMOVED,
        source_col="data_source_corr_x",
    )


def _compute_pt_qwen72_rows() -> tuple[dict[str, float], dict[str, float]]:
    return _compute_pt_condition_rows(
        raw_csv=_QWEN72_RAW,
        condition_label=_QWEN72_REMOVED,
        source_map_csv=_PT_SOURCE_MAP,
        origin_strip_suffix=True,
    )


def _upsert_row(df: pd.DataFrame, row: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c not in row:
            row[c] = float("nan")
    for c in row:
        if c not in out.columns:
            out[c] = float("nan")
    cond = str(row.get("Low+Irr Labelers", "")).strip()
    mask = (
        out["Base Model"].astype(str).str.strip().eq(_PT_MODEL)
        & out["Low+Irr Labelers"].astype(str).str.strip().eq(cond)
    )
    out = out[~mask]
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out


def _two_prop_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    p1 = k1 / n1
    p2 = k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return float(2.0 * norm.sf(abs(z)))


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.05:
        return "*"
    return ""


def _compute_pt_best_vs_original_sig_map() -> dict[str, str]:
    r1 = pd.read_csv(_R1_Q1_BY_SRC)
    r2 = pd.read_csv(_R2_MJ_BY_SRC)
    g5 = pd.read_csv(_PT_SOURCE_MAP)
    q72 = pd.read_csv(_QWEN72_RAW)
    src_map = pd.read_csv(_PT_SOURCE_MAP)[["Origin", "data_source_corr_x"]].copy()
    src_map["source"] = src_map["data_source_corr_x"].astype(str).str.strip().str.lower()
    src_map = src_map[["Origin", "source"]].drop_duplicates(subset=["Origin"])

    def _agg(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
        out = df[[src_col, "n_origins", "n_correct"]].copy()
        out.columns = ["source", "n", "k"]
        out["source"] = out["source"].astype(str).str.strip().str.lower()
        return out

    orig = _agg(r1, "data_source_corr")
    tr = _agg(r2, "data_source_corr_x")

    g5w = g5.copy()
    g5w["source"] = g5w["data_source_corr_x"].astype(str).str.strip().str.lower()
    g5w["correct"] = g5w["mv_correct"].astype(str).str.lower().map({"true": 1, "false": 0})
    g5agg = g5w.groupby("source", as_index=False).agg(n=("correct", "size"), k=("correct", "sum"))

    q72w = q72.copy()
    q72w["Origin"] = q72w["Origin"].astype(str).str.replace(r"-phase\d+$", "", regex=True)
    q72w["choice"] = q72w["q1"].map(_norm_choice)
    q72w["correct_answer"] = q72w["Correct answer"].map(_norm_choice)
    rows: list[dict[str, object]] = []
    for origin, g in q72w.groupby("Origin", sort=False):
        votes = [x for x in g["choice"].tolist() if x]
        c = collections.Counter(votes)
        if not c:
            tie = True
            mv = ""
        else:
            best = max(c.values())
            winners = sorted([k for k, v in c.items() if v == best])
            tie = len(winners) != 1
            mv = winners[0] if not tie else ""
        ca = g["correct_answer"].dropna()
        ca_val = ca.iloc[0] if len(ca) else ""
        rows.append(
            {
                "Origin": origin,
                "correct": int((not tie) and bool(mv) and (mv == ca_val)),
            }
        )
    q72agg = (
        pd.DataFrame(rows)
        .merge(src_map, on="Origin", how="left")
        .groupby("source", as_index=False)
        .agg(n=("correct", "size"), k=("correct", "sum"))
    )

    scopes = ["overall", "mmlu", "jama", "medxpert", "medbullets"]
    pvals: dict[str, float] = {}

    for scope in scopes:
        if scope == "overall":
            o_n, o_k = int(orig["n"].sum()), int(orig["k"].sum())
            cands = {
                "Trainee Removed": (int(tr["n"].sum()), int(tr["k"].sum())),
                "Qwen-72B Removed": (int(q72agg["n"].sum()), int(q72agg["k"].sum())),
                "GPT5 Removed": (int(g5agg["n"].sum()), int(g5agg["k"].sum())),
            }
        else:
            o = orig[orig["source"] == scope].iloc[0]
            o_n, o_k = int(o["n"]), int(o["k"])
            t = tr[tr["source"] == scope].iloc[0]
            q = q72agg[q72agg["source"] == scope].iloc[0]
            g = g5agg[g5agg["source"] == scope].iloc[0]
            cands = {
                "Trainee Removed": (int(t["n"]), int(t["k"])),
                "Qwen-72B Removed": (int(q["n"]), int(q["k"])),
                "GPT5 Removed": (int(g["n"]), int(g["k"])),
            }

        best_label = max(cands, key=lambda k: 100.0 * cands[k][1] / cands[k][0])
        b_n, b_k = cands[best_label]
        pvals[scope] = _two_prop_pvalue(o_k, o_n, b_k, b_n)

    m = len(scopes)
    return {scope: _sig_stars(min(1.0, p * m)) for scope, p in pvals.items()}


def _scope_from_title(title: str) -> str | None:
    t = title.strip().lower()
    if t.startswith("expert qa") or t == "total":
        return "overall"
    if t == "mmlu":
        return "mmlu"
    if t == "jama":
        return "jama"
    if "medxpert" in t:
        return "medxpert"
    if "medbullets" in t:
        return "medbullets"
    return None


def _find_pt_x(ax) -> float | None:
    labels = [t.get_text().strip().lower() for t in ax.get_xticklabels()]
    x = list(ax.get_xticks())
    for i, lab in enumerate(labels):
        if "physician" in lab:
            return float(x[i])
    return None


def _is_greenish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return g > 0.45 and g >= r + 0.05 and g >= b + 0.05


def _is_blueish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return b > 0.6 and b >= g and b >= r


def _is_grayish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return abs(r - g) < 0.08 and abs(g - b) < 0.08 and 0.35 <= r <= 0.65


def _is_blackish(color) -> bool:
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except Exception:
        return False
    return max(r, g, b) <= 0.18


def _extract_orig_best_by_x(ax) -> dict[float, dict[str, float]]:
    xticks = [float(x) for x in ax.get_xticks()]
    out: dict[float, dict[str, float]] = {x: {} for x in xticks}

    for ln in ax.lines:
        xdata = np.asarray(ln.get_xdata(), dtype=float)
        ydata = np.asarray(ln.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2:
            continue
        if not np.isfinite(xdata).all() or not np.isfinite(ydata).all():
            continue
        # only horizontal reference segments for original / best
        if abs(float(ydata[0] - ydata[1])) > 1e-8:
            continue
        if abs(float(xdata[1] - xdata[0])) < 0.5:
            continue
        mid = float((xdata[0] + xdata[1]) / 2.0)
        nearest = min(xticks, key=lambda t: abs(t - mid))
        if abs(nearest - mid) > 0.8:
            continue

        ls = ln.get_linestyle()
        color = ln.get_color()
        if ls == ":" and _is_grayish(color):
            out[nearest]["best"] = float(ydata[0])
        elif ls == "-" and _is_blackish(color):
            out[nearest]["orig"] = float(ydata[0])

    return out


def _infer_k_n_from_acc(acc_pct: float) -> tuple[int, int]:
    p = float(acc_pct) / 100.0
    best_err = float("inf")
    best_n = int(_N_CANDIDATES[0])
    best_k = int(round(p * best_n))
    for n in _N_CANDIDATES:
        k = int(round(p * int(n)))
        err = abs((k / float(n)) - p)
        if err < best_err - 1e-12:
            best_err = err
            best_n = int(n)
            best_k = k
    return best_k, best_n


def _rewrite_abs_delta_labels_and_get_points(ax) -> list[dict[str, float]]:
    levels = _extract_orig_best_by_x(ax)
    infos: list[dict[str, float]] = []
    y_min, y_max = ax.get_ylim()

    for x in [float(v) for v in ax.get_xticks()]:
        pair = levels.get(x, {})
        if "orig" not in pair or "best" not in pair:
            continue
        orig = float(pair["orig"])
        best = float(pair["best"])
        delta = best - orig

        candidates = []
        for txt in ax.texts:
            s = txt.get_text().strip()
            if "%" not in s:
                continue
            tx, ty = txt.get_position()
            if abs(float(tx) - x) <= 0.9:
                candidates.append((float(ty), txt))

        if candidates:
            _, t = max(candidates, key=lambda z: z[0])
            t.set_text(f"{delta:+.1f}%")
            if delta >= 0:
                t.set_color("#00a000")
                t.set_fontweight("normal")
            else:
                t.set_color("#d62728")
                t.set_fontweight("bold")
            label_y = float(t.get_position()[1])
        else:
            label_y = min(y_max - 1.0, best + max((y_max - y_min) * 0.06, 5.0))
            ax.text(
                x,
                label_y,
                f"{delta:+.1f}%",
                fontsize=28,
                fontweight=("normal" if delta >= 0 else "bold"),
                color=("#00a000" if delta >= 0 else "#d62728"),
                ha="center",
                va="bottom",
                zorder=19,
            )

        infos.append({"x": x, "orig": orig, "best": best, "delta": delta, "label_y": label_y})

    return infos


def _annotate_all_x_sig_on_current_figure(legacy) -> None:
    fig = legacy.plt.gcf()
    for ax in fig.axes:
        scope = _scope_from_title(ax.get_title())
        if scope not in {"mmlu", "jama", "medxpert", "medbullets"}:
            continue
        infos = _rewrite_abs_delta_labels_and_get_points(ax)
        y_min, y_max = ax.get_ylim()
        y_pad = max((y_max - y_min) * 0.085, 6.0)

        # Paired McNemar on the actual per-question outcomes, Bonferroni
        # corrected at 0.01/7 (*) and 0.01/28 (**). Replaces the previous
        # unpaired z-test whose n was reverse-engineered from the rounded
        # accuracy printed on the plot and which applied no correction.
        x_to_pretty = {
            float(x): _pretty_x_label(lbl.get_text())
            for x, lbl in zip(ax.get_xticks(), ax.get_xticklabels())
        }
        for info in infos:
            # stars are shown above green labels (positive improvements only)
            if info["delta"] <= 0:
                continue
            pretty = x_to_pretty.get(float(info["x"]))
            entry = _SIG_MAP.get((scope, pretty)) if pretty else None
            sig = entry["stars"] if entry else ""
            if not sig:
                continue
            y = min(y_max - 1.0, float(info["label_y"]) + y_pad)
            ax.text(
                float(info["x"]),
                y,
                sig,
                fontsize=42,
                fontweight="bold",
                color="#202020",
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.18},
                zorder=21,
            )


def _pretty_x_label(s: str) -> str:
    t = s.strip().lower()
    if t == "gpt4o":
        return "GPT-4o"
    if t in {"qwen-14b", "qwen 14b"}:
        return "Qwen-14B"
    if t in {"qwen 72b", "qwen-72b"}:
        return "Qwen-72B"
    if t == "llama-70b":
        return "Llama-70B"
    if t == "medgemma-27b":
        return "MedGemma-27B"
    if t in {"gpt 5", "gpt5"}:
        return "GPT-5"
    if "physician" in t and "trainee" in t:
        return "Physician +\nTrainee"
    return s


def _restyle_combined_current_figure(legacy) -> None:
    fig = legacy.plt.gcf()
    axes = list(fig.axes)

    def _title_key(ax) -> str:
        return ax.get_title().strip().lower()

    # Only apply when this is the combined multi-panel figure (has the 4 category panels).
    keys = [_title_key(ax) for ax in axes if _title_key(ax)]
    if not all(any(req in k for k in keys) for req in ["mmlu", "jama", "medxpert", "medbullets"]):
        return

    # Apply Inter to every existing text artist. The legacy plot creates many
    # of these artists before this styling pass, so rcParams alone is not enough.
    for text in fig.findobj(match=lambda artist: hasattr(artist, "set_fontfamily")):
        try:
            text.set_fontfamily(_FONT_FAMILY)
        except Exception:
            pass

    # Remove all top-row panels by geometry (robust to title text changes).
    for ax in list(fig.axes):
        if ax.get_legend() is not None:
            continue
        pos = ax.get_position()
        if pos.y0 > 0.66:
            fig.delaxes(ax)

    # Balanced canvas for readability and whitespace.
    fig.set_size_inches(40, 32, forward=True)

    title_to_ax = {}
    for ax in fig.axes:
        k = _title_key(ax)
        if not k:
            continue
        if "mmlu" in k:
            title_to_ax["mmlu"] = ax
        elif "jama" in k:
            title_to_ax["jama"] = ax
        elif "medxpert" in k:
            title_to_ax["medxpert"] = ax
        elif "medbullets" in k:
            title_to_ax["medbullets"] = ax

    # Re-layout remaining main panels into a 2x2 grid.
    # Taller panels with a real gutter between the rows: the previous
    # 0.33-high boxes on a 26in canvas left the markers, delta labels and stars
    # stacked on top of one another.
    # The legend now sits underneath rather than in a right-hand column, so the
    # panels take the full width instead of ~90% of it.
    panel_pos = {
        "mmlu": [0.070, 0.700, 0.420, 0.245],
        "jama": [0.530, 0.700, 0.420, 0.245],
        "medxpert": [0.070, 0.315, 0.420, 0.245],
        "medbullets": [0.530, 0.315, 0.420, 0.245],
    }
    for key, pos in panel_pos.items():
        ax = title_to_ax.get(key)
        if ax is None:
            continue
        ax.set_position(pos)
        ax.set_title(ax.get_title(), fontsize=_PANEL_TITLE_SIZE, fontweight="bold", pad=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=_AXIS_LABEL_SIZE, labelpad=16)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=_AXIS_LABEL_SIZE, labelpad=16)
        ax.tick_params(axis="both", labelsize=_TICK_LABEL_SIZE)
        ax.xaxis.label.set_size(_AXIS_LABEL_SIZE)
        ax.yaxis.label.set_size(_AXIS_LABEL_SIZE)

        pretty = [_pretty_x_label(lbl.get_text()) for lbl in ax.get_xticklabels()]
        ax.set_xticklabels(pretty, rotation=35, ha="right")
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(_TICK_LABEL_SIZE)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(_TICK_LABEL_SIZE)

        # Reduce dominance of baseline guides and keep focus on markers/labels.
        for ln in ax.lines:
            ls = ln.get_linestyle()
            if ls == "-" and _is_blackish(ln.get_color()):
                ln.set_linewidth(3.8)
            elif ls == ":" and _is_grayish(ln.get_color()):
                ln.set_linewidth(2.6)

        for txt in ax.texts:
            s = txt.get_text().strip().lower()
            # The MM/YY row and its caption move to the figure caption: they
            # crowded the axis, and on MMLU the accompanying divider falls off
            # the left edge because every model postdates that benchmark.
            if "dataset release date" in s or re.fullmatch(r"\d{2}/\d{2}", s):
                txt.set_visible(False)
                continue
            if "%" in s and _is_greenish(txt.get_color()):
                txt.set_fontweight("normal")
                txt.set_fontsize(30)
            elif "%" in s and "#d62728" in str(txt.get_color()).lower():
                txt.set_fontsize(30)

        # Keep Trainee (blue) markers above other markers.
        for coll in ax.collections:
            try:
                fcs = coll.get_facecolors()
            except Exception:
                continue
            if len(fcs) == 0:
                continue
            if _is_blueish(fcs[0]):
                # Keep the blue Trainee Removed point unmistakably on top,
                # including above reference lines and all other markers.
                coll.set_zorder(1000)
            else:
                coll.set_zorder(max(5, coll.get_zorder()))
            # Uniform, publication-readable marker sizing.
            try:
                coll.set_sizes(np.array([_DATAPOINT_SIZE]))
            except Exception:
                pass

    # Make legend panel larger and easier to read.
    for ax in list(fig.axes):
        lg = ax.get_legend()
        if lg is None:
            continue
        ax.set_position([0.88, 0.20, 0.12, 0.62])
        lg.set_title(lg.get_title().get_text(), prop={"size": 28})
        for text in lg.get_texts():
            text.set_fontsize(23)


def _load_sig_map() -> dict:
    try:
        sys.path.insert(0, str(_THIS_DIR))
        import mcnemar_significance as MS

        m = MS.compute()
        n1 = sum(1 for v in m.values() if v["stars"] == "*")
        n2 = sum(1 for v in m.values() if v["stars"] == "**")
        print(
            f"Significance: McNemar paired, Bonferroni alpha* ={MS.ALPHA_WITHIN:.2e} "
            f"alpha**={MS.ALPHA_FAMILY:.2e}; {n1} '*', {n2} '**' of {len(m)} cells.",
            flush=True,
        )
        return m
    except Exception as e:  # pragma: no cover
        print(f"Warning: significance map unavailable ({e}); stars suppressed.",
              file=sys.stderr)
        return {}


_SIG_MAP: dict = {}


def _restyle_total_only_current_figure(legacy) -> None:
    """Reduce the combined figure to the single pooled 'Total' panel.

    The legacy layout's top row holds the three physician cohorts; the first of
    them, ``Expert QA (933)``, is the pooled-across-all-four-datasets panel that
    ``MedPAIR_Result_ExpertQA.pdf`` shows as ``Total``. Everything else is
    dropped so the *_total artifact is a summary rather than a copy of the
    per-dataset grid.
    """
    fig = legacy.plt.gcf()

    total_ax = None
    for ax in fig.axes:
        if ax.get_title().strip().lower().startswith("expert qa"):
            total_ax = ax
            break
    if total_ax is None:
        raise RuntimeError("could not find the 'Expert QA (933)' panel to build the Total figure")

    for ax in list(fig.axes):
        if ax is not total_ax:
            fig.delaxes(ax)

    for text in fig.findobj(match=lambda a: hasattr(a, "set_fontfamily")):
        try:
            text.set_fontfamily(_FONT_FAMILY)
        except Exception:
            pass

    # Keep the plotting area compact: 7 categories stretched across a 40in
    # canvas left the x axis very long and the markers marooned in whitespace.
    fig.set_size_inches(24, 11, forward=True)
    total_ax.set_position([0.075, 0.27, 0.615, 0.66])
    # The legacy y-limits leave a wide dead band below 0 for the MM/YY row,
    # which the Total panel hides -- reclaim it so the axis spans the data.
    _y0, _y1 = total_ax.get_ylim()
    total_ax.set_ylim(max(_y0, -6.0), min(_y1, 118.0))
    total_ax.set_title("Total", fontsize=_TITLE_SIZE, fontweight="bold", pad=24)
    total_ax.set_ylabel("Accuracy (%)", fontsize=_AXIS_LABEL_SIZE, labelpad=16)
    total_ax.set_xlabel("")
    total_ax.tick_params(axis="both", labelsize=_TICK_LABEL_SIZE)

    pretty = [_pretty_x_label(lbl.get_text()) for lbl in total_ax.get_xticklabels()]
    total_ax.set_xticklabels(pretty, rotation=40, ha="right")
    for lbl in total_ax.get_xticklabels():
        lbl.set_fontsize(_TICK_LABEL_SIZE)
    for lbl in total_ax.get_yticklabels():
        lbl.set_fontsize(_TICK_LABEL_SIZE)

    for ln in total_ax.lines:
        ls = ln.get_linestyle()
        if ls == "-" and _is_blackish(ln.get_color()):
            ln.set_linewidth(3.8)
        elif ls == ":" and _is_grayish(ln.get_color()):
            ln.set_linewidth(2.6)

    for txt in total_ax.texts:
        s = txt.get_text().strip()
        low = s.lower()
        # The reference Total panel carries neither the release-date caption
        # nor the per-model MM/YY row.
        if "dataset release date" in low or re.fullmatch(r"\d{2}/\d{2}", s):
            txt.set_visible(False)
            continue
        if "%" in low:
            txt.set_fontsize(34)

    for coll in total_ax.collections:
        try:
            fcs = coll.get_facecolors()
        except Exception:
            continue
        if len(fcs) == 0:
            continue
        coll.set_zorder(1000 if _is_blueish(fcs[0]) else max(5, coll.get_zorder()))
        try:
            coll.set_sizes(np.array([_DATAPOINT_SIZE]))
        except Exception:
            pass

    total_ax.margins(x=0.06)
    for spine in ("top", "right"):
        total_ax.spines[spine].set_visible(False)

    if fig.legends:
        lg = fig.legends[0]
        lg.set_bbox_to_anchor((0.70, 0.5), transform=fig.transFigure)


def _orig_best_lines_by_x(ax) -> dict[float, dict]:
    """Like _extract_orig_best_by_x but keeps the Line2D objects so they can move."""
    xticks = [float(x) for x in ax.get_xticks()]
    out: dict[float, dict] = {x: {} for x in xticks}
    for ln in ax.lines:
        xd = np.asarray(ln.get_xdata(), dtype=float)
        yd = np.asarray(ln.get_ydata(), dtype=float)
        if len(xd) != 2 or len(yd) != 2:
            continue
        if not (np.isfinite(xd).all() and np.isfinite(yd).all()):
            continue
        if abs(float(yd[0] - yd[1])) > 1e-8 or abs(float(xd[1] - xd[0])) < 0.5:
            continue
        mid = float((xd[0] + xd[1]) / 2.0)
        nearest = min(xticks, key=lambda t: abs(t - mid))
        if abs(nearest - mid) > 0.8:
            continue
        ls, color = ln.get_linestyle(), ln.get_color()
        if ls == ":" and _is_grayish(color):
            out[nearest]["best"] = float(yd[0])
            out[nearest]["best_line"] = ln
        elif ls == "-" and _is_blackish(color):
            out[nearest]["orig"] = float(yd[0])
    return out


def _refresh_best_and_delta(legacy, sr: dict) -> None:
    """Raise 'Best Performance' to any higher SR diamond and restate the delta.

    The legacy module computes the dotted best-performance line and the green
    improvement label before the SR series exists, so a diamond above that line
    would otherwise be ignored by both.
    """
    fig = legacy.plt.gcf()
    moved = 0
    for ax in fig.axes:
        scope = _scope_from_title(ax.get_title())
        if scope is None:
            continue
        x_by_model = {
            lbl.get_text().strip(): float(x)
            for x, lbl in zip(ax.get_xticks(), ax.get_xticklabels())
        }
        # highest SR value at each x position
        sr_at_x: dict[float, float] = {}
        for per_model in sr.values():
            for pretty, by_source in per_model.items():
                x = x_by_model.get(pretty)
                y = by_source.get(scope)
                if x is None or y is None or not np.isfinite(y):
                    continue
                sr_at_x[x] = max(sr_at_x.get(x, float("-inf")), float(y))

        levels = _orig_best_lines_by_x(ax)

        for x, pair in levels.items():
            if "orig" not in pair or "best" not in pair:
                continue
            orig, best = float(pair["orig"]), float(pair["best"])
            cand = sr_at_x.get(x)
            if cand is not None and cand > best:
                best = cand
                line = pair.get("best_line")
                if line is not None:
                    line.set_ydata([best, best])
                    moved += 1
            # Absolute percentage-point difference in every panel. The legacy
            # Total panel reported a relative change, which put a 21.0 -> 66.8
            # move at "+217.9%" -- a figure that reads as an accuracy and
            # exceeds 100.
            delta = best - orig

            best_txt, best_y = None, float("-inf")
            for txt in ax.texts:
                if "%" not in txt.get_text():
                    continue
                tx, ty = txt.get_position()
                if abs(float(tx) - x) <= 0.9 and float(ty) > best_y:
                    best_txt, best_y = txt, float(ty)
            if best_txt is not None:
                best_txt.set_text(f"{delta:+.1f}%")
                best_txt.set_color("#00a000" if delta >= 0 else "#d62728")
                best_txt.set_fontweight("normal" if delta >= 0 else "bold")
    if moved:
        print(f"Best-performance line raised to an SR value at {moved} position(s).", flush=True)


def _finalize_figure(legacy, key_area: float = _LEGEND_KEY_AREA,
                     sr_markersize: float = _SR_LEGEND_MARKERSIZE) -> None:
    """Apply figure-wide finishing touches after all series are drawn.

    Renames legend entries, enlarges the SR diamond keys, and forces Inter onto
    every text artist -- including ones created after the restyle passes, which
    would otherwise fall back to the default sans-serif.
    """
    fig = legacy.plt.gcf()

    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend()]
    renamed = 0
    for lg in legends:
        for txt in lg.get_texts():
            new = _LEGEND_RENAMES.get(txt.get_text().strip())
            if new:
                txt.set_text(new)
                renamed += 1
        for handle in lg.legend_handles:
            # SR keys are Line2D diamonds; the legacy keys are PathCollections
            # sized in area. Normalise both to one footprint.
            if getattr(handle, "get_marker", None) and handle.get_marker() == "D":
                handle.set_markersize(sr_markersize)
            elif hasattr(handle, "set_sizes"):
                try:
                    handle.set_sizes(np.array([key_area]))
                except Exception:
                    pass

    # Inter last, so nothing created downstream is left on the fallback font.
    for artist in fig.findobj(match=lambda a: hasattr(a, "set_fontfamily")):
        try:
            artist.set_fontfamily(_FONT_FAMILY)
        except Exception:
            pass

    if renamed:
        print(f"Renamed {renamed} legend entr(ies): "
              + ", ".join(f"{k!r} -> {v!r}" for k, v in _LEGEND_RENAMES.items()), flush=True)


def _match_axes_height_to_legend(legacy) -> None:
    """Fit the legend to the plot box so the two span the same height.

    The legend's height is driven by its font size, not by the canvas, so it
    cannot be matched by resizing the figure (doing so just rescales both and
    never converges). Instead rebuild the legend at successively adjusted font
    sizes until its height matches the axes, then align the two exactly.
    """
    fig = legacy.plt.gcf()
    if not fig.legends or not fig.axes:
        return
    ax = fig.axes[0]
    inv = fig.transFigure.inverted()
    target = ax.get_position().height

    fs = float(fig.legends[0].get_texts()[0].get_fontsize())
    for _ in range(6):
        fig.canvas.draw()
        lb = fig.legends[0].get_window_extent().transformed(inv)
        if abs(lb.height - target) / target < 0.03:
            break
        fs = max(10.0, min(48.0, fs * (target / max(lb.height, 1e-6)) ** 0.6))
        old = fig.legends[0]
        handles = list(old.legend_handles)
        labels = [t.get_text() for t in old.get_texts()]
        title = old.get_title().get_text()
        old.remove()
        lg = fig.legend(
            handles, labels, loc="center left", bbox_to_anchor=(0.70, 0.5),
            bbox_transform=fig.transFigure, fontsize=fs, framealpha=1.0,
            frameon=True, fancybox=True, shadow=True, title=title,
            title_fontsize=fs * 1.18, labelspacing=1.1, borderpad=1.0,
            handletextpad=1.2,
        )
        for t in lg.get_texts():
            t.set_fontfamily(_FONT_FAMILY)
        lg.get_title().set_fontfamily(_FONT_FAMILY)

    fig.canvas.draw()
    lb = fig.legends[0].get_window_extent().transformed(inv)
    pos = ax.get_position()
    ax.set_position([pos.x0, lb.y0, pos.width, lb.height])
    print(
        f"Total: legend fitted at {fs:.0f}pt; axes and legend both span "
        f"y {lb.y0:.3f}-{lb.y0 + lb.height:.3f}.",
        flush=True,
    )


def _raise_trainee_markers_on_top(legacy) -> None:
    """Put the blue 'Trainee Removed' markers above every other marker.

    Must run last: both the legacy plot and the SR pass assign their own
    zorders, and the SR diamonds (1200) would otherwise cover the blue dots.
    """
    fig = legacy.plt.gcf()
    raised = 0
    for ax in fig.axes:
        for coll in ax.collections:
            try:
                fcs = coll.get_facecolors()
            except Exception:
                continue
            if len(fcs) == 0:
                continue
            if _is_blueish(fcs[0]):
                coll.set_zorder(_TRAINEE_ZORDER)
                raised += 1
    if raised:
        print(f"Raised {raised} Trainee (blue) marker group(s) to the top.", flush=True)


def _norm_sr_choice(v: object) -> str:
    """Answer letter from an SR prediction cell.

    SR files mix bare letters with ``<answer>Option D</answer>`` wrappers; a
    naive ``[A-J]`` search on the latter matches the ``A`` in ``ANSWER``.
    """
    if pd.isna(v):
        return ""
    s = str(v).strip()
    tagged = re.search(r"<answer>(.*?)</answer>", s, re.IGNORECASE | re.DOTALL)
    inner = tagged.group(1) if tagged else s
    cleaned = re.sub(r"[^A-Za-z]", "", re.sub(r"(?i)\boption\b", "", inner)).upper()
    if len(cleaned) == 1 and "A" <= cleaned <= "J":
        return cleaned
    if tagged:
        m = re.search(r"[A-J]", inner.upper())
        return m.group(0) if m else ""
    # Free-form prose, refusals and the literal "Error" are not answers. A
    # lenient [A-J] scan would score "Error" as E and "I'm sorry..." as I.
    return ""


def _compute_sr_accuracies() -> dict[str, dict[str, dict[str, float]]]:
    """``{legend_label: {pretty_model: {source: accuracy_pct}}}`` over the 933 cohort."""
    if not _SR_COHORT.is_file():
        raise FileNotFoundError(f"Missing SR cohort CSV: {_SR_COHORT}")

    cohort = pd.read_csv(_SR_COHORT)
    cohort["Origin"] = cohort["Origin"].astype(str).str.strip()
    cohort["source"] = cohort["data_source_corr_x"].astype(str).str.strip().str.lower()
    key = cohort.set_index("Origin")[["source", "correct_answer"]]

    out: dict[str, dict[str, dict[str, float]]] = {}
    for cond_dir, label in _SR_CONDITIONS.items():
        per_model: dict[str, dict[str, float]] = {}
        for pretty, stem in _SR_MODEL_FILES.items():
            path = _SR_ROOT / cond_dir / f"{stem}_[SR]_predictions.csv"
            if not path.is_file():
                extra = _SR_EXTRA_FILES.get((label, pretty))
                if extra is None:
                    continue
                path = _AFTER_PT_ROOT / extra
                if not path.is_file():
                    print(f"Warning: missing SR file {path}", file=sys.stderr)
                    continue
            df = pd.read_csv(path)
            # Some files carry several models' predictions side by side (the
            # GPT-5 rerun also holds llama70b_direct_prediction), so never pick
            # a column that names a different model.
            _other = {"GPT-4o": "gpt4o", "GPT-5": "gpt5", "Llama-70B": "llama70b",
                      "Qwen-14B": "qwen14b", "Qwen-72B": "qwen72b",
                      "MedGemma-27B": "medgemma"}
            _bad = [v for k, v in _other.items() if k != pretty]
            pred_col = next(
                (c for c in _SR_PRED_COLS
                 if c in df.columns and not any(b in c.lower() for b in _bad)),
                None,
            )
            if pred_col is None:
                print(f"Warning: no prediction column in {path.name}", file=sys.stderr)
                continue

            df = df.dropna(subset=[pred_col]).copy()
            df["Origin"] = (
                df["Origin"].astype(str).str.strip().str.replace(r"-phase\d+$", "", regex=True)
            )
            df = df.drop_duplicates(subset=["Origin"])
            merged = df[["Origin", pred_col]].merge(
                key, left_on="Origin", right_index=True, how="inner"
            )
            # Refusals and "Error" rows are not answers; drop them so coverage
            # reflects what was actually scored instead of counting them wrong.
            pred = merged[pred_col].map(_norm_sr_choice)
            merged = merged[pred.ne("")].assign(_pred=pred[pred.ne("")])
            coverage = len(merged) / float(_SR_COHORT_N)
            if coverage < _SR_MIN_COVERAGE_FRAC:
                print(
                    f"Warning: skipping {label} / {pretty}: only {len(merged)} of "
                    f"{_SR_COHORT_N} questions answered ({100 * coverage:.1f}% coverage).",
                    file=sys.stderr,
                )
                continue
            if coverage < 1.0:
                print(
                    f"Note: {label} / {pretty} plotted from {len(merged)} of "
                    f"{_SR_COHORT_N} questions ({100 * coverage:.1f}% coverage).",
                    flush=True,
                )

            merged["ok"] = merged["_pred"].eq(merged["correct_answer"].map(_norm_sr_choice))
            by_source = (100.0 * merged.groupby("source")["ok"].mean()).to_dict()
            # "overall" is the pooled accuracy used by the Total panel.
            by_source["overall"] = 100.0 * float(merged["ok"].mean())
            per_model[pretty] = by_source
        if per_model:
            out[label] = per_model
    return out


def _add_sr_series_to_current_figure(legacy, legend_fontsize: float = 52,
                                     legend_title_size: float = 58,
                                     legend_anchor: tuple[float, float] = (0.88, 0.5),
                                     legend_loc: str = "center left",
                                     legend_ncol: int = 1) -> None:
    """Draw the SR conditions onto the laid-out panels and extend the legend."""
    from matplotlib.lines import Line2D

    sr = _compute_sr_accuracies()
    if not sr:
        print("Warning: no SR data available; legend unchanged.", file=sys.stderr)
        return

    fig = legacy.plt.gcf()
    plotted: set[str] = set()

    for ax in fig.axes:
        scope = _scope_from_title(ax.get_title())
        # "overall" is the pooled Total panel of the *_total artifact; the
        # per-dataset pass has already deleted its top row by this point.
        if scope not in {"mmlu", "jama", "medxpert", "medbullets", "overall"}:
            continue

        # Panels have already been restyled, so tick labels are the pretty form.
        x_by_model = {
            lbl.get_text().strip(): float(x)
            for x, lbl in zip(ax.get_xticks(), ax.get_xticklabels())
        }

        for label, per_model in sr.items():
            color, marker = _SR_STYLE[label]
            for pretty, by_source in per_model.items():
                x = x_by_model.get(pretty)
                y = by_source.get(scope)
                if x is None or y is None or not np.isfinite(y):
                    continue
                ax.scatter(
                    [x],
                    [y],
                    c=color,
                    s=_DATAPOINT_SIZE,
                    marker=marker,
                    edgecolors="white",
                    linewidth=0.9,
                    alpha=0.9,
                    zorder=_SR_ZORDER,
                )
                plotted.add(label)

    if not plotted:
        return

    # The legacy module attaches the "Labeler Configuration" legend to the
    # figure (fig.legend), not to an Axes, so it must be rebuilt at figure level.
    if not fig.legends:
        print(
            "Warning: SR points drawn but no figure legend found to extend.",
            file=sys.stderr,
        )
        return

    old = fig.legends[0]
    handles = list(old.legend_handles)
    labels = [t.get_text() for t in old.get_texts()]
    added: list[str] = []
    for label in _SR_CONDITIONS.values():
        if label not in plotted or label in labels:
            continue
        color, marker = _SR_STYLE[label]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                markersize=_SR_LEGEND_MARKERSIZE,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.5,
                color=color,
            )
        )
        labels.append(label)
        added.append(label)

    if not added:
        return

    old.remove()
    new_lg = fig.legend(
        handles,
        labels,
        loc=legend_loc,
        ncol=legend_ncol,
        bbox_to_anchor=legend_anchor,
        bbox_transform=fig.transFigure,
        fontsize=legend_fontsize,
        framealpha=1.0,
        markerscale=1.0,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Labeler Configuration",
        title_fontsize=legend_title_size,
        labelspacing=1.7,
        borderpad=1.2,
        handletextpad=1.2,
    )
    for text in new_lg.get_texts():
        text.set_fontfamily(_FONT_FAMILY)
    new_lg.get_title().set_fontfamily(_FONT_FAMILY)

    print(f"Added SR series to legend: {', '.join(added)}", flush=True)


def main() -> None:
    if not _FONT_PATH.is_file():
        raise FileNotFoundError(f"Inter font file not found: {_FONT_PATH}")
    font_manager.fontManager.addfont(_FONT_PATH)
    legacy = _load_legacy_module()
    legacy.plt.rcParams["font.family"] = _FONT_FAMILY
    global _SIG_MAP
    _SIG_MAP = _load_sig_map()
    original_augment = legacy.augment_physician_trainee
    original_savefig = legacy.plt.savefig

    def _augment_with_gpt5(df_physician: pd.DataFrame, df_dataset: pd.DataFrame):
        out_p, out_d = original_augment(df_physician, df_dataset)
        for label, fn in [
            (_QWEN72_REMOVED, _compute_pt_qwen72_rows),
            (_GPT5_REMOVED, _compute_pt_gpt5_rows),
        ]:
            try:
                top_row, bottom_row = fn()
                out_p = _upsert_row(out_p, top_row)
                out_d = _upsert_row(out_d, bottom_row)
                print(
                    f"Appended {_PT_MODEL} / {label}: "
                    f"top MMLU={top_row['MMLU']:.2f}%, "
                    f"dataset (MMLU/Jama/MedXpert/Medbullets)="
                    f"{bottom_row['MMLU']:.2f}/{bottom_row['Jama']:.2f}/"
                    f"{bottom_row['MedXpert']:.2f}/{bottom_row['Medbullets']:.2f}%",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"Warning: could not append {_PT_MODEL} / {label}: {e}",
                    file=sys.stderr,
                )
        return out_p, out_d

    def _savefig_per_dataset(*args, **kwargs):
        """Pass 1: the 2x2 per-dataset grid -> MedPAIR_Result_combined.pdf."""
        try:
            _annotate_all_x_sig_on_current_figure(legacy)
            # Run styling after annotations so newly created labels also use Inter.
            _restyle_combined_current_figure(legacy)
        except Exception as e:
            print(f"Warning: could not annotate significance labels: {e}", file=sys.stderr)
        try:
            # After restyling: panel positions and pretty x-tick labels are final,
            # and the uniform marker-resize pass will not touch these points.
            _add_sr_series_to_current_figure(legacy, legend_fontsize=40,
                                             legend_title_size=48,
                                             legend_anchor=(0.5, 0.205),
                                             legend_loc="upper center",
                                             legend_ncol=4)
        except Exception as e:
            print(f"Warning: could not add SR series: {e}", file=sys.stderr)
        try:
            _refresh_best_and_delta(legacy, _compute_sr_accuracies())
        except Exception as e:
            print(f"Warning: could not refresh best/delta: {e}", file=sys.stderr)
        try:
            _finalize_figure(legacy, key_area=720.0, sr_markersize=28.0)
        except Exception as e:
            print(f"Warning: could not finalize figure: {e}", file=sys.stderr)
        try:
            # Last: nothing may be drawn over the Trainee markers.
            _raise_trainee_markers_on_top(legacy)
        except Exception as e:
            print(f"Warning: could not raise Trainee markers: {e}", file=sys.stderr)
        return original_savefig(*args, **kwargs)

    combined = _THIS_DIR / "MedPAIR_Result_combined.pdf"
    combined_total = _THIS_DIR / "MedPAIR_Result_combined_total.pdf"

    def _savefig_total(*args, **kwargs):
        """Pass 2: the pooled Total panel -> MedPAIR_Result_combined_total.pdf.

        The legacy module hard-codes its own output path, so redirect it here.
        """
        try:
            _restyle_total_only_current_figure(legacy)
        except Exception as e:
            print(f"Warning: could not build Total figure: {e}", file=sys.stderr)
            return None
        try:
            # Anchor must be passed here: this call rebuilds the legend, so an
            # anchor set earlier in the Total restyle would be discarded.
            _add_sr_series_to_current_figure(legacy, legend_fontsize=36,
                                             legend_title_size=42,
                                             legend_anchor=(0.70, 0.5))
        except Exception as e:
            print(f"Warning: could not add SR series to Total figure: {e}", file=sys.stderr)
        try:
            _refresh_best_and_delta(legacy, _compute_sr_accuracies())
        except Exception as e:
            print(f"Warning: could not refresh best/delta: {e}", file=sys.stderr)
        try:
            _finalize_figure(legacy)
        except Exception as e:
            print(f"Warning: could not finalize figure: {e}", file=sys.stderr)
        try:
            _raise_trainee_markers_on_top(legacy)
        except Exception as e:
            print(f"Warning: could not raise Trainee markers: {e}", file=sys.stderr)
        try:
            _match_axes_height_to_legend(legacy)
        except Exception as e:
            print(f"Warning: could not match axes to legend: {e}", file=sys.stderr)
        args = (str(combined_total),) + tuple(args[1:])
        return original_savefig(*args, **kwargs)

    legacy.augment_physician_trainee = _augment_with_gpt5

    # Point the legacy module at the corrected per-source inputs. Left to its
    # own defaults it falls back to PhysicianEval_Result_Report.csv, whose
    # per-source columns are not per-source accuracies (its MMLU column tracks
    # overall accuracy; its Jama column spans 1.5-36.4 against a true 66-89).
    _corrected_ds = _THIS_DIR / "ExpertQA_933_by_data_source.csv"
    _corrected_phys = _THIS_DIR / "PhysicianEval_933_corrected.csv"
    if _corrected_ds.is_file() and _corrected_phys.is_file():
        legacy._default_dataset_csv = lambda *a, **k: _corrected_ds
        legacy._default_physician_csv = lambda *a, **k: _corrected_phys
        print(f"Using corrected inputs: {_corrected_ds.name}, {_corrected_phys.name}", flush=True)
    else:
        print("Warning: corrected inputs missing; falling back to legacy defaults.",
              file=sys.stderr)

    legacy.plt.savefig = _savefig_per_dataset
    legacy.main()

    # Rebuild from scratch: the per-dataset pass destructively deletes the
    # top-row panels, so the Total panel has to come from a fresh render.
    legacy.plt.close("all")
    legacy.plt.savefig = _savefig_total
    legacy.main()
    print(f"Wrote {combined_total}", flush=True)


if __name__ == "__main__":
    main()
