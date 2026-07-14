# NeuRIPS 2025 Physician Labels Dataset

**Reorganized**: June 21, 2026  
**Status**: Chronologically organized by analysis date

## Folder Structure

```
Physician_Labels/
├── 2026-03_March/              # March 2026 Analysis Data (15 files)
│   ├── 2k_sentence_seperate.csv
│   ├── 933_Clinician_Student_Majority_Vote.csv
│   ├── 933_answerable_TextRelevance_q1_majority_vote.csv
│   ├── 933_answerable_q1_MJ_accuracy_by_data_source_corr_x.csv
│   ├── 933_llm_human_round1_round2_responses.csv
│   ├── Centaur_933_Clinician_Student_Majority_Vote.csv
│   ├── Clinician_Student_Majority_Vote.csv
│   ├── Clinician_Student_TextRelevance_q1_majority_vote.csv
│   ├── Clinician_Student_q1_MJ_accuracy_by_data_source.csv
│   ├── Full_PT_Labels_Morgan_Bouie.csv
│   ├── HardQA_Clinician_Student_Majority_Vote.csv
│   ├── Impossible_Clinician_Student_Majority_Vote.csv
│   ├── Text_Relevance_Analysis_Case_View_022626.csv
│   └── _ppl_high_low_sentences.csv / _ppl_high_low_table.csv
│
├── 2026-04_April/              # April 2026 Analysis Data (5 files)
│   ├── Round2_933_Eval.csv
│   ├── Round2_933_MJ.csv
│   ├── Round2_933_MJ_accuracy_by_data_source.csv
│   ├── gpt5_predictions_High_Relevance_933.csv
│   └── gpt5_predictions_Original_Accuracy_933.csv
│
├── 2026-05_May/                # May 2026 Analysis Data (3 files, 2 subdirs)
│   ├── May15_Data/
│   │   └── Centaur_933_Qwen72B-SR.csv
│   │
│   └── May27_Data/
│       ├── May27_2026_Origin_Summary_933.csv
│       └── Text_Relevance_Analysis_Case_View_gpt5_phase_052626.csv
│
├── 2026-06_June/               # June 2026 Analysis Data (1 file)
│   └── Qwen72B_Predicted_High.csv
│
├── results_archive/            # Compiled Results & Match Rates (10 files)
│   ├── 14B_MatchRate.csv
│   ├── 70B_MatchRate.csv
│   ├── 72B_MatchRate.csv
│   ├── GPT4o_MatchRate.csv
│   ├── GPT4o_SR_Concordance_Result.csv
│   ├── GPT5_MatchRate.csv
│   ├── MedGemma_SR_Match_Rate.csv
│   ├── [SR]Llama70B_annotated_ORIGINAL_Accuracy.csv
│   ├── [SR]Qwen14B_annotated_MedPAIR_relevancy.csv
│   └── [SR]Qwen72B_annotated_MedPAIR_relevancy.csv
│
├── figures_archive/            # Visualization Figures (6 files)
│   ├── CNF.png
│   ├── cdf_comparison_subplots.pdf
│   ├── cdf_physician_only.png
│   ├── cdf_physician_vs_trainee.png
│   ├── cdf_physician_vs_trainee_CI.png
│   └── cdf_qwen_vs_physician.png
│
├── scripts_archive/            # Analysis Scripts & Dependencies
│   └── __pycache__/            # Python compiled bytecode
│
└── README.md                   # This file
```

## Dataset Overview

| Category | Count | Purpose |
|----------|-------|---------|
| Monthly Data | 24 files | Raw/processed data from monthly analysis runs |
| Results | 10 files | Compiled match rates and concordance results |
| Figures | 6 files | CDF plots and comparative visualizations |
| Scripts | 1 dir | Analysis scripts and dependencies |

### By Month

- **March 2026**: 15 files - Initial physician vs. clinician/student labeling comparison
- **April 2026**: 5 files - GPT-5 prediction validation and round 2 evaluation
- **May 2026**: 3 files (split by May 15 & May 27) - Qwen model evaluation and refinements
- **June 2026**: 1 file - Latest model predictions (Qwen72B)

### Key Data Files

#### Core Labeling Data
- `Clinician_Student_Majority_Vote.csv` — Consensus labels from physician + clinician/student majority voting
- `933_Clinician_Student_Majority_Vote.csv` — 933-question variant with majority voting
- `Full_PT_Labels_Morgan_Bouie.csv` — Complete physician labels (Morgan & Bouie annotation)

#### Relevance Analysis
- `*TextRelevance_q1_*.csv` — Text relevance scoring and analysis
- `933_answerable_*.csv` — Answerability assessment for 933-question set

#### Model Performance
- `gpt5_predictions_*.csv` — GPT-5 model predictions (High Relevance and Original Accuracy)
- `Qwen*_*.csv` — Qwen model outputs (14B, 72B variants)
- `[SR]*_*.csv` — Sensitive/Relevant annotated results

#### Metrics & Results
- `*MatchRate.csv` — Agreement/match rates between physician and model predictions
- `*Concordance_Result.csv` — Concordance analysis between labelers
- `*_accuracy_by_data_source.csv` — Accuracy breakdown by data source

### Visualization Figures

- **CDF Plots** — Cumulative distribution function comparisons
  - `cdf_physician_vs_trainee.png` — Physician vs. clinician/student comparison
  - `cdf_qwen_vs_physician.png` — Model vs. physician performance
  - `cdf_physician_only.png` — Physician-only distribution

- **Comparison** — Multi-panel subplot comparison (`cdf_comparison_subplots.pdf`)
- **CNF** — Confusion matrix or classification analysis (`CNF.png`)

## File Naming Conventions

### Data Files
- `{Dataset}_Majority_Vote.csv` — Consensus labels using majority voting
- `{Count}_{Metric}*.csv` — Analysis of specific N items (e.g., "933" = 933-question subset)
- `{Model}_predictions_*.csv` — Model output/predictions
- `[SR]*` — Sensitive/Relevant tagged annotation results

### Date Variants
- Files without date suffix are from earliest run
- Files with date suffixes (`_022626`, `_052626`) indicate analysis date (MMDDYY)

## Usage Guide

### Finding Specific Data
```bash
# Physician labels only
cat 2026-03_March/Full_PT_Labels_Morgan_Bouie.csv

# Latest model predictions
cat 2026-06_June/Qwen72B_Predicted_High.csv

# Consensus labels (physician + clinician/student)
cat 2026-03_March/Clinician_Student_Majority_Vote.csv

# Model performance metrics
ls results_archive/*MatchRate.csv

# All visualizations
ls figures_archive/
```

### Data Analysis Pattern
1. **Base labels**: `Full_PT_Labels_Morgan_Bouie.csv` (March)
2. **Consensus labels**: `Clinician_Student_Majority_Vote.csv` (March)
3. **Model predictions**: `gpt5_predictions_*.csv` (April), `Qwen*_*.csv` (May-June)
4. **Performance metrics**: `results_archive/*MatchRate.csv`
5. **Visualizations**: `figures_archive/cdf_*.png`

## Data Timeline

| Date | Activity | Files | Notes |
|------|----------|-------|-------|
| Mar 2, 2026 | Initial physician labeling + majority voting | 15 files | Full PT labels + 933 subset variants |
| Apr 1, 2026 | Round 2 evaluation + GPT-5 predictions | 5 files | Accuracy by data source analysis |
| May 15, 2026 | Qwen 72B model evaluation | 1 file | SR (Sensitive/Relevant) concordance |
| May 27, 2026 | Continued evaluation + summaries | 1 file | Refined origin summary |
| Jun 19, 2026 | Latest predictions | 1 file | Qwen72B high relevance predictions |

## Key Dimensions

### Labeling Dimensions
- **Answerability** — Can the question be answered?
- **Text Relevance** — How relevant is the text to the question?
- **Accuracy/Correctness** — Is the answer correct/accurate?
- **Classification** — Binary/multi-class categorization

### Comparison Dimensions
- **Physician vs. Clinician/Student** — Expert vs. mid-level labeling agreement
- **Human vs. LLM** — Physician labels vs. model predictions
- **Model vs. Model** — Performance comparison across different LLMs

### Subsets
- **933-question variant** — Specific question subset analysis
- **HardQA** — Difficult question subset
- **Impossible questions** — Unanswerable question subset
- **2000 sentences** — Larger sample analysis

## Related Directories

- **Parent**: `/home/yuexing/NeuRIPS25/` — Main NeuRIPS 2025 project directory
- **Figures**: `figures_archive/` — PNG, PDF visualization outputs
- **Results**: `results_archive/` — Summary metrics and match rates
- **Scripts**: `scripts_archive/` — Analysis code and dependencies

## Notes

- Files are organized chronologically by analysis date (Month_Year)
- Sub-months in May indicate different analysis runs (May15 vs. May27)
- Results are aggregated in `results_archive/` for easy access
- All visualizations consolidated in `figures_archive/`
- Python scripts and dependencies in `scripts_archive/`

---

**Last Updated**: June 21, 2026  
**Organization**: Chronological by analysis date  
**Status**: Complete reorganization from date-based folders to integrated timeline structure
