# Result_Analysis_Final

Final aggregated accuracy metrics, pairwise comparisons, and figure generation for all models and datasets. **This is the primary destination for reading published results.**

---

## Key Output Files

### Accuracy Metrics
One CSV per dataset (columns: model, round, accuracy, CI, etc.):

| File | Dataset |
|---|---|
| `accuracy_metrics.csv` | All datasets combined |
| `accuracy_metrics_jama.csv` | JAMA Clinical Challenge |
| `accuracy_metrics_medbullets.csv` | MedBullets |
| `accuracy_metrics_medxpert.csv` | MedXpertQA |
| `accuracy_metrics_mmlu.csv` | MMLU (professional medicine) |

### Merged Model Results
| File | Description |
|---|---|
| `all_models_merged.csv` | One row per question; columns for each model's round 1 and round 2 predictions |
| `gpt4o.csv` / `llama70b.csv` / `qwen14b.csv` / `qwen72b.csv` | Per-model prediction exports |

### Pairwise Comparisons
| File | Description |
|---|---|
| `pairwise_comparisons.csv` | Overall McNemar / pairwise test results |
| `pairwise_comparisons_*.csv` | Per-dataset pairwise comparisons |

### Round Comparisons (before vs. after removal)
| File | Description |
|---|---|
| `round_comparisons.csv` | Overall before vs. after accuracy change |
| `round_comparisons_*.csv` | Per-dataset round comparisons |

---

## Scripts and Notebooks

| File | Purpose |
|---|---|
| `Results_Analysis.py` | Main script: loads CSVs, computes accuracy, runs significance tests, exports all metrics |
| `Figure 4 Making.ipynb` | Generates Figure 4 (accuracy comparison plot) |
| `new_dotplot.py` | Dot-plot visualization script |
| `croissant.py` | Croissant metadata export for dataset sharing |

---

## Figures

| File | Description |
|---|---|
| `accuracy_by_source.pdf` / `.png` | Accuracy breakdown by data source |
| `accuracy_by_source_overall.pdf` / `.png` | Overall accuracy by source across models |
| `medical_datasets_comparison.pdf` | Cross-dataset comparison figure |

---

## Subdirectories

| Folder | Contents |
|---|---|
| `Trainee_Accuracy/` | Accuracy metrics for human trainee (med student) baseline |
