# SR Predictions - Calculated Accuracies for Table 6

## Qwen-14B Removed (SR)

| Model | MMLU | JAMA | MedXpert | MedBullets | General |
|-------|------|------|----------|------------|---------|
| **GPT-4o** | 95.3% | 76.8% | 40.3% | 79.7% | **71.1%** |
| **GPT-5** | 88.1% | 86.6% | 65.7% | 87.0% | **81.8%** |
| **Llama-70B** | 76.7% | 46.6% | 20.3% | 55.8% | **46.0%** |
| **MedGemma-27B** | 41.3% | 11.9% | 3.7% | 16.1% | **14.5%** |
| **Qwen-14B** | 81.9% | 55.5% | 23.9% | 55.6% | **51.7%** |
| **Qwen-72B** | 79.8% | 58.4% | 26.1% | 57.0% | **53.5%** |

### Key Observations
- **GPT-5** shows best overall performance (81.8%)
- **GPT-4o** second best (71.1%)
- **MedGemma-27B** struggles with this condition (14.5%)
- Removing smaller Qwen-14B has moderate impact on open-source models

---

## Qwen-72B Removed (SR)

| Model | MMLU | JAMA | MedXpert | MedBullets | General |
|-------|------|------|----------|------------|---------|
| **GPT-4o** | 95.3% | 72.3% | 38.1% | 76.8% | **68.1%** |
| **GPT-5** | 96.9% | 85.6% | 66.7% | 90.3% | **83.4%** |
| **Llama-70B** | 91.7% | 68.7% | 30.5% | 69.1% | **62.8%** |
| **MedGemma-27B** | 93.3% | 62.4% | 27.3% | 54.8% | **57.9%** |
| **Qwen-14B** | 85.5% | 61.7% | 24.8% | 58.9% | **55.8%** |
| **Qwen-72B** | 82.4% | 64.1% | 25.2% | 58.9% | **56.5%** |

### Key Observations
- **GPT-5** significantly better (83.4% vs 68.1% for GPT-4o)
- **MedGemma-27B** performs much better here (57.9% vs 14.5%)
- **Llama-70B** achieves 62.8% - strong performance
- Removing larger Qwen-72B has greater impact overall
- MMLU appears least affected by context removal (scores remain high)

---

## Llama-70B Removed (SR)

**Status**: ❌ **DATA NOT AVAILABLE**

No SR prediction files exist for when Llama-70B context is removed. To generate this data:

1. Run Self-Report sentence relevance annotations with Llama-70B context removed
2. Generate predictions for all models (GPT-4o, GPT-5, Llama-70B, MedGemma-27B, Qwen-14B, Qwen-72B)
3. Store in `/SR_Predictions/Llama-70B_Removed/` following the naming convention
4. Update this file with calculated accuracies

---

## Comparative Analysis

### Impact of Context Removal

**Qwen-14B Removal** (smaller model):
- Moderate impact on smaller models
- Open-source models more affected than GPT models
- MedGemma-27B particularly vulnerable (14.5% overall)

**Qwen-72B Removal** (larger model):
- Larger impact overall (68.1% → 83.4% for GPT models suggests information loss)
- MedGemma-27B significantly improves (57.9% vs 14.5%)
- Llama-70B shows balanced performance (62.8%)

### Dataset-Specific Patterns

| Dataset | Qwen-14B Removed | Qwen-72B Removed | Observation |
|---------|-----------------|-----------------|-------------|
| **MMLU** | Least impacted | Least impacted | Domain exams most resilient to context loss |
| **JAMA** | Moderate impact | More sensitive | Clinical reasoning needs full context |
| **MedXpert** | Severe impact | Severe impact | Most context-dependent dataset |
| **MedBullets** | Moderate impact | Moderate impact | USMLE-style more robust |

---

## Methodology

Accuracies calculated from SR prediction files using:
```
Accuracy = (# Correct Predictions / Total Questions) × 100

Where:
- Total Questions = 933 QAs across all datasets
- Breakdown by dataset: MMLU (192), JAMA (295), MedXpert (318), MedBullets (207)
- Correct = Model's Extracted_Answer == answer_corr (reference answer)
```

## Files Used

All calculations based on CSV files in:
- `/home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/Qwen-14B_Removed/`
- `/home/yuexing/NeuRIPS25/After_PT_Removal/SR_Predictions/Qwen-72B_Removed/`

For detailed methodology, see individual CSV files or Table 6 in Appendix.pdf

---

**Last Updated**: 2026-06-22
**Data Generated**: January-March 2026
**Organized**: June 22, 2026
