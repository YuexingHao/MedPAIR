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
    if t.startswith("expert qa"):
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

        for info in infos:
            # stars are requested above green labels (positive improvements only)
            if info["delta"] <= 0:
                continue
            k1, n1 = _infer_k_n_from_acc(info["orig"])
            k2, n2 = _infer_k_n_from_acc(info["best"])
            p = _two_prop_pvalue(k1, n1, k2, n2)
            sig = _sig_stars(p)
            # Display convention requested for this figure: shift the stronger
            # significance levels down by one star; retain existing * results.
            if sig == "***":
                sig = "**"
            elif sig == "**":
                sig = "*"
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
    fig.set_size_inches(40, 26, forward=True)

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
    panel_pos = {
        "mmlu": [0.07, 0.59, 0.38, 0.33],
        "jama": [0.49, 0.59, 0.38, 0.33],
        "medxpert": [0.07, 0.07, 0.38, 0.33],
        "medbullets": [0.49, 0.07, 0.38, 0.33],
    }
    for key, pos in panel_pos.items():
        ax = title_to_ax.get(key)
        if ax is None:
            continue
        ax.set_position(pos)
        ax.set_title(ax.get_title(), fontsize=72, fontweight="bold", pad=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=52, labelpad=14)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=54, labelpad=14)
        ax.tick_params(axis="both", labelsize=38)
        ax.xaxis.label.set_size(52)
        ax.yaxis.label.set_size(54)

        pretty = [_pretty_x_label(lbl.get_text()) for lbl in ax.get_xticklabels()]
        ax.set_xticklabels(pretty, rotation=40, ha="right")
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(36)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(36)

        # Reduce dominance of baseline guides and keep focus on markers/labels.
        for ln in ax.lines:
            ls = ln.get_linestyle()
            if ls == "-" and _is_blackish(ln.get_color()):
                ln.set_linewidth(3.8)
            elif ls == ":" and _is_grayish(ln.get_color()):
                ln.set_linewidth(2.6)

        for txt in ax.texts:
            s = txt.get_text().strip().lower()
            if "dataset release date" in s:
                txt.set_fontsize(34)
                txt.set_fontweight("bold")
                txt.set_transform(ax.transAxes)
                txt.set_position((0.5, 0.90))
                txt.set_ha("center")
                txt.set_va("bottom")
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


def main() -> None:
    if not _FONT_PATH.is_file():
        raise FileNotFoundError(f"Inter font file not found: {_FONT_PATH}")
    font_manager.fontManager.addfont(_FONT_PATH)
    legacy = _load_legacy_module()
    legacy.plt.rcParams["font.family"] = _FONT_FAMILY
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

    def _savefig_with_sig(*args, **kwargs):
        try:
            _annotate_all_x_sig_on_current_figure(legacy)
            # Run styling after annotations so newly created labels also use Inter.
            _restyle_combined_current_figure(legacy)
        except Exception as e:
            print(f"Warning: could not annotate significance labels: {e}", file=sys.stderr)
        return original_savefig(*args, **kwargs)

    legacy.augment_physician_trainee = _augment_with_gpt5
    legacy.plt.savefig = _savefig_with_sig
    legacy.main()

    # Keep the "combined_total" artifact in sync with the latest combined output.
    combined = _THIS_DIR / "MedPAIR_Result_combined.pdf"
    combined_total = _THIS_DIR / "MedPAIR_Result_combined_total.pdf"
    if combined.is_file():
        shutil.copyfile(combined, combined_total)
        print(f"Wrote {combined_total}", flush=True)


if __name__ == "__main__":
    main()
