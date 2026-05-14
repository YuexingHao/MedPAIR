# Merge

Cross-dataset merge pipeline that combines JAMA, MedBullets, MedXpertQA, and MMLU questions into a unified 2k/4k question set. Also contains sentence attribution scores, LLM annotation runs, and inter-rater reliability analyses.

## Layout

| Path | Contents |
|---|---|
| `data/csv/` | Merged question sets, annotations, alpha tables, and other tabular outputs |
| `figures/` | Exported PDF figures |
| `notebooks/` | Analysis and pipeline notebooks (including `notebooks/meditron/`) |
| `scripts/` | Standalone Python utilities |
| `attribution/14b_contextcite_analysis/` | Qwen 14B ContextCite analysis (stats notebooks, physician comparison CSVs; no per-question `Merge_Q*.csv` tree in-repo) |
| `attribution/qwen72b_contextcite/` | Qwen 72B attribution and physician-match artifacts |
| `paths.py` | `Path` constants for the above (import in notebooks after adding `Merge/` to `sys.path`, or run Jupyter with working directory set to `Merge/`) |

**14B attribution:** Use `data/csv/merged_attribution_scores_14B.csv` (long-form, one row per sentence). If you re-run ContextCite notebooks that emit one CSV per question, outputs go to `data/csv/contextcite_14b_staging/` (gitignored). To rebuild a merged CSV from an extracted archive of `Merge_Q*.csv`, run `python scripts/merge_14b_per_question_csvs.py --input-dir /path/to/folder`.

---

## Key Data Files

CSV paths below are relative to `data/csv/`.

### Merged Question Sets
| File | Description |
|---|---|
| `merged_2k_questions.csv` | 2k merged questions (initial) |
| `merged_2k_questions_standardized.csv` | Standardized version with consistent column names |
| `merged_2k_with_4k_ID.csv` | 2k questions mapped to 4k ID space (use this for cross-referencing) |
| `merged_llm_4k_questions_with_QA_ID.csv` | 4k questions with QA IDs for LLM runs |
| `jama_formatted_questions.csv` | JAMA questions formatted for labeling |
| `medbullets_only.csv` / `medbullets_op4.csv` | MedBullets subset files |
| `MedXpertQA.csv` | MedXpertQA subset |
| `professional_medicine_mmlu.csv` | MMLU professional medicine subset |
| `2k_sentence_seperate.csv` | 2k questions with sentences separated per row |

### Annotation / Relevancy Labels
| File | Description |
|---|---|
| `First_annotated_2k_relevancy.csv` | First-round LLM relevancy annotations on 2k questions |
| `SECOND_annotated_2k_relevancy.csv` | Second-round annotations |
| `THIRD_annotated_2k_relevancy.csv` | Third-round annotations |
| `First_Qwen14_annotated_2k_relevancy.csv` | Qwen 14B first-round annotations |
| `FIRST_GPT4o_Remove_Irrelevant.csv` | GPT-4o first-round irrelevant sentence removal |
| `SECOND_GPT4o_Remove_Irrelevant.csv` | GPT-4o second-round irrelevant sentence removal |
| `REMOVE_IRR_MAJORITY_Vote_GPT4o_Self_Reported_Relevancy_Labels.csv` | GPT-4o majority vote after irrelevance removal |
| `MAJORITY_Vote_GPT4o_Self_Reported_Relevancy_Labels.csv` | GPT-4o majority vote labels |
| `centaur_checkpoint.csv` | Centaur Lab labeling checkpoint |
| `gpt4o_predictions.csv` | GPT-4o predictions on merged set |

### Attribution Scores
| File | Description |
|---|---|
| `merged_attribution_scores_14B.csv` | ContextCite attribution scores (Qwen 14B) for merged set |
| `merged_attribution_scores_14B_answers.csv` | Same, with model answers attached |
| `llama70b_answers_summary.csv` | Llama 70B answers summary |
| `llama70b_attributions_summary.csv` | Llama 70B attribution scores summary |
| `qwen_72b_answers_summary.csv` | Qwen 72B answers summary |

### IRR / Alpha Analysis
| File | Description |
|---|---|
| `general_alpha_results.csv` | Overall Krippendorff's alpha results |
| `qwen14_alpha_results.csv` | Qwen 14B alpha results |
| `data_source_alpha_analysis.csv` | Alpha results broken down by data source |
| `label_difference_results.csv` | Pairwise label difference analysis |

---

## Notebooks

Paths are under `notebooks/` unless noted.

| Notebook | Purpose |
|---|---|
| `DataMerge.ipynb` | Main merge pipeline: combines datasets, standardizes columns, assigns IDs |
| `Match_2k_4k.ipynb` | Matches 2k IDs to 4k question space |
| `GPT4o_Prediction.ipynb` | GPT-4o predictions on merged set |
| `GPT4o_Prediction-RemoveIrrelevant.ipynb` | GPT-4o predictions after removing irrelevant sentences |
| `GPT4o-Self_Report_Sentence_Level_Relevancy.ipynb` | GPT-4o self-report sentence-level relevancy |
| `Merge_ContextCite_14b.ipynb` | ContextCite attribution (Qwen 14B) on merged set |
| `Merge_ContextCite_14bRemove_Irrelevant.ipynb` | ContextCite removal pipeline (14B) |
| `Merge_ContextCite_Qwen_14b.ipynb` | Qwen 14B ContextCite full pipeline |
| `Llama3.1_70B_Stats_Calc.ipynb` | Llama 70B statistical calculations |
| `Qwen14B_Stats_Calc.ipynb` | Qwen 14B statistical calculations |
| `Qwen14B_Self_Report.ipynb` | Qwen 14B self-report relevancy |
| `Result_BarPlots.ipynb` | Bar plot results visualization |
| `Sentence_Attribution_Graph.ipynb` | Sentence attribution score visualization |
| `Landscape-of-Though-Visual.ipynb` | Landscape of thought visualization |

Additional notebooks live under `attribution/14b_contextcite_analysis/` (e.g. `14B_Stats_Calc.ipynb`) and `attribution/qwen72b_contextcite/` (e.g. `Qwen_72B_Stats_Calc.ipynb`).

---

## Scripts

Paths are under `scripts/`.

| Script | Purpose |
|---|---|
| `calculate_krippendorff_alpha.py` | Computes Krippendorff's alpha across annotators |
| `calculate_label_differences.py` | Computes pairwise label differences |
| `analyze_data_source_alpha.py` | Alpha analysis broken down by data source |
| `compare_label_files.py` | Compares two label files for consistency |
| `fix_notebook.py` | Rewrites legacy bare path strings to `paths.CONTEXTCITE_14B_STAGING` |
| `merge_14b_per_question_csvs.py` | Rebuilds `merged_attribution_scores_14B.csv` from a `--input-dir` of `Merge_Q*.csv` (optional `--check-only`, `--parquet`) |

---

## Subdirectories (reference)

| Folder | Contents |
|---|---|
| `notebooks/meditron/` | Meditron-related notebooks |
| `attribution/14b_contextcite_analysis/` | 14B ContextCite analysis (large per-question CSV tree + comparison tables) |
| `attribution/qwen72b_contextcite/` | 72B attribution summaries and physician matching scripts |

LLM self-evaluation runs that previously lived under `LLM_OnItsOwn_LLM_Results/` were relocated into the `After_PT_Removal/` model folders in this repo.
