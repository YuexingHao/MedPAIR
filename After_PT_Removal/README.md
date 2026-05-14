# After_PT_Removal

This directory contains all per-model experiments that run LLMs before and after removing physician-labeled spurious (low-relevance) context sentences. It also holds shared raw data, aggregated analysis scripts, and the final results.

### OpenAI API key

Do **not** commit API keys. Notebooks and scripts here call the OpenAI SDK with **`OpenAI()`** (no `api_key=` argument), which reads **`OPENAI_API_KEY`** from the environment.

```bash
export OPENAI_API_KEY='sk-...'   # same shell before `jupyter lab` or `python ...`
```

To scan for accidentally committed `sk-proj-...` strings after merges:

```bash
python After_PT_Removal/scripts/_strip_embedded_openai_keys.py
```

---

## Model Subdirectories

| Folder | Model | Key contents |
|---|---|---|
| `GPT4o/` | GPT-4o | Round 1/2 predictions, self-report relevance, concordance results, spurious rate notebooks |
| `GPT5/` | GPT-5 | Sankey diagrams (HTML/PNG), spurious rate analysis, manuscript figure notebooks |
| `Llama-70B/` | Llama 3.1 70B | ContextCite removal CSVs, round 1/2 prediction CSVs, compare scripts |
| `MedGemma-27b-text-it/` | MedGemma 27B | Predictions on all model-removed sets, self-report relevancy notebooks |
| `Qwen2.5-14B-Instruct/` | Qwen 2.5 14B | Round 1/2 results, spurious rate notebook, compare script |
| `Qwen2.5-72B-Instruct/` | Qwen 2.5 72B | Before/after removal results, physician-removed spurious rate, cross-model notebooks |
| `Result_Analysis_Final/` | All models | Final aggregated metrics, figures, pairwise comparisons — **start here for results** |
| `Second_Round_Centaur_Lab_Result_Analysis/` | — | Second-round Centaur Lab raw data and analysis |
| `Qualitative Analysis/` | — | Pre-study and exit survey data + analysis scripts |
| `shared/` | Cross-model | Centaur CSVs, shared GPT-4o prediction runs, second-round removal notebooks, `NLP_Analysis.py` — see below |

---

## Shared bundle (`shared/`)

| Path | Description |
|---|---|
| `shared/paths.py` | `paths.DATA`, `paths.NOTEBOOKS`, `paths.SCRIPTS` — import in notebooks after the bootstrap cell (first cell) |
| `shared/data/` | `Centaur_*.csv`, `Centaur_Lab_*`, `llama70b_answers_second_round_with_origin.csv`, `predictions.csv` |
| `shared/notebooks/` | `After_Removal_Trainee_Labels_GPT4o.ipynb`, `Perplexity.ipynb`, `Spurious_Rate_Visual.ipynb`, `Second_Removal_70b.ipynb`, `Second_Removal_72b.ipynb`, `Second_Removal_Qwen_14b.ipynb` |
| `shared/scripts/` | `NLP_Analysis.py` | Run with cwd `After_PT_Removal` or `shared`; data loads from `shared/data/` |

**Note:** `Second_Removal_72b.ipynb` expects `shared/data/qwen72b_answers_second_round_with_origin.csv` (add it if missing). Jupyter should use working directory `After_PT_Removal`, `shared/`, or `shared/notebooks/` so `paths.py` resolves.

---

## Physician-subset evaluation summaries

These pipelines write ``results/eval_physician_subsets_summary.csv`` (933 / HardQA / Impossible letter accuracy):

| Project folder | Script |
|---|---|
| ``GPT4o/`` | ``scripts/evaluate_predictions_by_physician_subsets_gpt4o.py`` |
| ``GPT5/`` | ``scripts/evaluate_predictions_by_physician_subsets_gpt5.py`` |
| ``Qwen2.5-14B-Instruct/`` | ``results/scripts/evaluate_predictions_by_physician_subsets_14b.py`` |
| ``Qwen2.5-72B-Instruct/`` | ``results/scripts/evaluate_predictions_by_physician_subsets_qwen72b.py`` |
| ``MedGemma-27b-text-it/`` | ``results/scripts/evaluate_predictions_by_physician_subsets_medgemma27b.py`` |
| ``Llama-70B/`` | ``scripts/evaluate_predictions_by_physician_subsets_llama70b.py`` |

**View all projects in one table** (from repo root):

```bash
python After_PT_Removal/scripts/aggregate_physician_subset_evals.py
python After_PT_Removal/scripts/aggregate_physician_subset_evals.py --wide
python After_PT_Removal/scripts/aggregate_physician_subset_evals.py --output /tmp/all_physician_subset_evals.csv
```

Re-run each project’s script first if you need up-to-date numbers.

**MedPAIR PDF figure** (letter accuracy panels) — ``Figures/make_medpair_result_figure.py``; default output ``Figures/MedPAIR_Result.pdf``. Wide physician table: ``After_PT_Removal/scripts/physician_eval_to_result_report.py`` (writes ``Figures/PhysicianEval_Result_Report.csv`` by default). A thin forwarder also exists at ``GPT4o/scripts/make_medpair_result_figure.py``.

---

## Quickstart

To reproduce final results:
1. Open `Result_Analysis_Final/Results_Analysis.py` for accuracy metrics.
2. Open `Result_Analysis_Final/Figure 4 Making.ipynb` for the main comparison figure.
3. Sankey diagrams for GPT-5 are in `GPT5/*.html`.
