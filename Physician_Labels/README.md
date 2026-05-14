# Physician_Labels

Ground-truth physician and clinician annotations for sentence relevance. These are used as the gold standard against which model labels and crowd labels are compared, and to define which sentences are “spurious” for the removal experiments.

**Layout (reorganized):**

| Path | Contents |
|------|----------|
| `Mar2_2026_Data/` | March 2026 Centaur batch: raw export, majority-vote tables, and analysis scripts (**stable path** for `933_Clinician_Student_Majority_Vote.csv` used across the repo) |
| `notebooks/` | Accuracy / CDF / 14B comparison notebooks |
| `results/` | Model vs. physician **match-rate** CSVs and `14B_Physician_Comparison_with_Stats.csv` |
| `reference/` | Large reference dumps (e.g. `Centaur_Lab_First_Round_COMPLETE_RAW.csv`) |
| `paths.py` | `Path` constants for Python imports (`MAJORITY_VOTE_933`, `MAR2_DATA`, …) |

---

## Key files

### Latest batch (`Mar2_2026_Data/`)

| File | Description |
|------|-------------|
| `Text Relevance Analysis Case View_022626.csv` | Raw Centaur Lab case view export; `Origin`, `Respondent type`, `Response correct`, `q2`–`q22`, `data_source_corr` |
| `933_Clinician_Student_Majority_Vote.csv` | 933-Origin majority-vote table (expert pipeline) |
| `Clinician_Student_Analysis.py` | Full-cohort analysis; reads `results/14B_Physician_Comparison_with_Stats.csv` for `data_source_corr` |
| `933_Clinician_Student_Analysis.py` | Same pipeline restricted to answerable Origins |

### Reference (`reference/`)

| File | Description |
|------|-------------|
| `Centaur_Lab_First_Round_COMPLETE_RAW.csv` | First-round Centaur Lab dump (copy for local analysis) |

### Model match rates (`results/`)

| File | Description |
|------|-------------|
| `GPT4o_MatchRate.csv` | GPT-4o sentence label match rate vs. physicians |
| `GPT5_MatchRate.csv` | GPT-5 |
| `70B_MatchRate.csv` | Llama 70B |
| `72B_MatchRate.csv` | Qwen 72B |
| `MedGemma_SR_Match_Rate.csv` | MedGemma 27B self-report match rate |
| `14B_Physician_Comparison_with_Stats.csv` | Qwen 14B vs. physician comparison + stats |

---

## Notebooks (`notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `Physician_Fifth_Read_Labels_Accuracy.ipynb` | Fifth-read physician label accuracy |
| `Physician_Labels_Accuracy + CDF_Plot.ipynb` | Accuracy metrics and CDF plots |
| `14B_Labels_Accuracy.ipynb` | Qwen 14B vs. physician ground truth |

See `notebooks/README.md` for the working-directory rule.

---

## Output figures (repo root of this folder)

CDF / confusion figures (`cdf_*.png`, `cdf_*.pdf`, `CNF.png`) remain here for now; you can move them under `figures/` in a later pass if you want that split.

---

## 933 cohort: sentence concordance vs models (join on `Origin` only)

Script: `scripts/compute_933_origin_concordance.py`

- Reads `Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv` and each model’s wide relevancy CSV under `After_PT_Removal/…` (defaults listed in the script).
- Merges **only on `Origin`** (ignores `ID_corr`).
- Binarizes physician `Sentence 1`…`21` scores with `--phys-threshold` (default `0.5`) and compares to model `q1`…`q20` (`label_21` if used for sentence 21).

```bash
python Physician_Labels/scripts/compute_933_origin_concordance.py
python Physician_Labels/scripts/compute_933_origin_concordance.py --per-origin-out Physician_Labels/results/concordance_933_by_origin_long.csv
```

Outputs: `results/concordance_933_by_origin_summary.csv` (and optional long table).

---

## Python: `paths.py`

```python
from Physician_Labels.paths import MAJORITY_VOTE_933, MAR2_DATA, RESULTS
```

(Adjust import path if not running from repo root with `Physician_Labels` on `PYTHONPATH`.)
