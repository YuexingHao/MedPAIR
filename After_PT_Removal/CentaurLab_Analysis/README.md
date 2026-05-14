# CentaurLab_Analysis

Crowdsource labeling pipeline using the Centaur Lab platform. Crowd workers labeled sentence relevance across the merged 2k question dataset. This feeds into the physician label comparison and spurious removal experiments.

---

## Folder layout

| Path | Contents |
|------|----------|
| `paths.py` | `DATA`, `TABLES`, `IMAGE_NECESSARY`, `FIGURES` (see file docstring) |
| `data/raw/` | Frozen inputs (May7, standardized 2k questions, ID maps, pre-majority vote, GPT-4o majority labels) |
| `results/tables/` | Pipeline outputs and working tables (Centaur exports, `Sentence_Label_Original_2k.csv`, merges, ContextCite, `1300_IDs.csv`, …) |
| `results/figures/` | PDFs from `CentaurLab_Analysis.ipynb` |
| `results/image_necessary/` | `image_necessary_results.csv` checkpoint from `Image_Necessary/image_necessary_check.py` |
| `notebooks/` | `CentaurLab_Analysis.ipynb`, `Centaur_70B_Alignment.ipynb` |
| `Image_Necessary/` | Image-necessity script + local README |
| `CentaurLab_Analysis.py` | Script export of the main notebook (same path logic as notebooks) |

Notebooks start with a bootstrap cell that finds `paths.py` (must contain `CENTAUR_ROOT`). Run Jupyter with working directory `CentaurLab_Analysis` or `CentaurLab_Analysis/notebooks`.

---

## Key files (by location)

### `data/raw/` (inputs / ID lists)

| File | Description |
|------|-------------|
| `merged_2k_questions_standardized.csv` | Standardized 2k questions (input to labeling) |
| `merged_2k_with_4k_ID.csv` | 2k questions matched to 4k ID space |
| `May7_Data.csv` | Additional labeling batch (May 7) |
| `Pre-Majority_Vote.csv` | Labels before majority vote aggregation |
| `MAJORITY_Vote_GPT4o_Self_Reported_Relevancy_Labels.csv` | GPT-4o self-reported labels (majority vote) |

### `results/tables/` (outputs & working data)

| File | Description |
|------|-------------|
| `1300_IDs.csv` | 1300 question IDs sent to Centaur Lab |
| `Centaur_Lab_Classification.csv` | Full classification output from Centaur Lab (20k+ rows) |
| `Centaur_Lab_Second_Round.csv` | Second-round Centaur Lab labels |
| `Sentence_Label_Original_2k.csv` | Sentence-level labels (updated in place by `image_necessary_check.py` when that script is run) |
| `merged_output.csv` | Full merged labeling output (203k+ rows) |
| `merge_correct_df.csv` | Subset: rows where crowd answer is correct |
| `merge_correct_df_export.csv` | Exported version of the above |
| `less_sentence_df.csv` | Questions with fewer sentences (post-removal) |
| `less_sentence_df_export.csv` | Exported version of the above |
| `less_sentence_df_with_removed_sentences.csv` | Same, with removed sentence text attached |
| `llama70b_ContextCite_Merge.csv` | Llama 70B ContextCite scores merged with labels |
| `70B_ContextCite_Removal.csv` | ContextCite-based removal output for 70B |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/CentaurLab_Analysis.ipynb` | Main analysis: label distribution, majority vote, inter-rater reliability |
| `notebooks/Centaur_70B_Alignment.ipynb` | Alignment between Centaur Lab labels and Llama 70B ContextCite scores |

---

## Output figures (`results/figures/`)

| File | Description |
|------|-------------|
| `centaur_label_proportions_plot.pdf` | Proportion of high/low/irrelevant labels |
| `centaur_majority_vote*.pdf` | Majority vote visualizations (adjusted, aligned, normalized) |
| `correct_vs_wrong_stackplot.pdf` | Correct vs. wrong answers stacked by label |
| `*_relevance_distribution.pdf` | Per-dataset relevance distributions |
| `side_by_side_relevance_distribution.pdf` | Side-by-side comparison across datasets |
| `stacked_relevance_profiles.pdf` | Stacked relevance profile visualization |
| `three_panel_relevance_profiles.pdf` | Three-panel relevance profile figure |
