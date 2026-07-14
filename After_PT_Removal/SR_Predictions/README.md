# SR (Self-Reported) Predictions - Organized by Removed Condition

This directory centralizes all Self-Reported (SR) prediction files from the After_PT_Removal experiments, organized by the model/context that was removed.

## Directory Structure

```
SR_Predictions/
├── Qwen-14B_Removed/          # Models tested with Qwen-14B context removed
│   ├── GPT_4o_[SR]_predictions.csv
│   ├── GPT_5_[SR]_predictions.csv
│   ├── Llama_70B_[SR]_predictions.csv
│   ├── MedGemma_27B_[SR]_predictions.csv
│   ├── Qwen_14B_[SR]_predictions.csv
│   └── Qwen_72B_[SR]_predictions.csv
│
├── Qwen-72B_Removed/          # Models tested with Qwen-72B context removed
│   ├── GPT_4o_[SR]_predictions.csv
│   ├── GPT_5_[SR]_predictions.csv
│   ├── Llama_70B_[SR]_predictions.csv
│   ├── MedGemma_27B_[SR]_predictions.csv
│   ├── Qwen_14B_[SR]_predictions.csv
│   └── Qwen_72B_[SR]_predictions.csv
│
└── Llama-70B_Removed/         # ❌ NOT AVAILABLE
    └── (no files - SR predictions not yet generated for this condition)
```

## Data Summary

| Removed Condition | Models | Total Files | Status |
|-------------------|--------|-------------|--------|
| **Qwen-14B** | GPT-4o, GPT-5, Llama-70B, MedGemma-27B, Qwen-14B, Qwen-72B | 6 | ✅ Complete |
| **Qwen-72B** | GPT-4o, GPT-5, Llama-70B, MedGemma-27B, Qwen-14B, Qwen-72B | 6 | ✅ Complete |
| **Llama-70B** | — | 0 | ❌ Missing |

## File Details

Each CSV file contains Self-Reported (SR) predictions for all 933 QAs, including:
- `Origin`: Question ID
- `Extracted_Answer` / `gpt_letter`: Model's predicted answer
- `answer_corr`: Correct answer (from reference)
- `data_source_corr`: Dataset source (MMLU, JAMA, MedXpert, MedBullets)
- Additional context and response fields

## Timeline

- **May 2025**: Initial SR concordance analysis performed
- **January-March 2026**: SR prediction files generated from notebooks
- **June 9, 2026**: Backup and rerun of analyses
- **June 22, 2026**: Files organized and centralized in this directory

## Usage Notes

### For Table 6 (Appendix.tex)
These files are used to calculate accuracies by dataset:
- Use corresponding folder for each removed condition
- Calculate: # correct predictions / total predictions × 100
- Break down by source: MMLU, JAMA, MedXpert, MedBullets
- Calculate overall accuracy across all 933 QAs

### Why Llama-70B_Removed is Missing
No SR prediction files were generated when Llama-70B context was removed. This would require:
1. Running self-report sentence relevance annotations with Llama-70B context removed
2. Evaluating all models' predictions against human ground truth on that condition

This condition can be added if/when those experiments are run.

## Source Locations

Original file locations (before consolidation):

### Qwen-14B Removed (SR)
- `GPT4o/results/predictions/gpt4o_predictions_on_14b_removed.csv`
- `GPT5/results/predictions/gpt5_predictions_on_Qwen14B_removed.csv`
- `Llama-70B/results/predictions/[SR]_Llama70B_predictions_on_14B.csv`
- `MedGemma-27b-text-it/results/predictions/[SR]_MedGemma27B_predictions_on_14B_progress.csv`
- `Qwen2.5-14B-Instruct/results/predictions/[SR]_Qwen_14B_predictions_14Bprogress.csv`
- `Qwen2.5-72B-Instruct/results/predictions/[SR]_Qwen_72B_predictions_on_14B_progress.csv`

### Qwen-72B Removed (SR)
- `GPT4o/results/predictions/[SR]_gpt4o_predictions_on_qwen72b_removed.csv`
- `GPT5/results/predictions/[SR]_gpt5_predictions_on_72B_removed.csv`
- `Llama-70B/results/predictions/[SR]_Llama70B_predictions_on_72B.csv`
- `MedGemma-27b-text-it/results/predictions/[SR]_MedGemma27B_predictions_on_72B_progress.csv`
- `Qwen2.5-14B-Instruct/results/predictions/[SR]_Qwen_14B_predictions_72Bprogress.csv`
- `Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_72B_SR.csv`

---

**Last updated**: 2026-06-22
