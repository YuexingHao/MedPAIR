# Interrater Reliability (IRR) Findings Summary

**Dataset**: Physician-Clinician-Student Majority Vote Annotations  
**Sample Size**: 933 medical exam questions  
**Date**: March 2026  
**Source File**: `2026-03_March/933_Clinician_Student_Majority_Vote.csv`

---

## Overall IRR Statistics

| Metric | Value |
|--------|-------|
| **Mean IRR** | 0.3751 |
| **Median IRR** | 0.3978 |
| **Std Dev** | 0.2993 |
| **Min** | -0.6000 |
| **Max** | 1.0000 |
| **Range** | 1.6000 |

### Interpretation
- **Average agreement** between clinician/student raters is **0.3751**, indicating **fair to moderate agreement**
- **Median of 0.3978** shows the typical question has moderate disagreement
- **High variance (SD=0.30)** indicates substantial variability in agreement across questions
- **Negative minimum (-0.60)** suggests some questions had worse-than-chance agreement

---

## Agreement Distribution

### Quartile Breakdown
| Quartile | IRR Value |
|----------|-----------|
| **Q1 (25th %ile)** | 0.2025 |
| **Q2 (50th %ile / Median)** | 0.3978 |
| **Q3 (75th %ile)** | 0.5677 |

### IRR Categories
| Category | Count | Percentage |
|----------|-------|-----------|
| **Low IRR** (< 0.33) | 392 | 42.0% |
| **Moderate IRR** (0.33-0.67) | 405 | 43.4% |
| **High IRR** (≥ 0.67) | 136 | 14.6% |

**Key Finding**: Only **14.6% of questions** achieved high agreement (≥ 0.67), while **42% had low agreement** (< 0.33). This suggests substantial rater disagreement is common in this dataset.

---

## Agreement by Data Source

| Source | N | Mean IRR | Median IRR | Notes |
|--------|---|----------|-----------|-------|
| **JAMA** | 295 | 0.3244 | 0.3186 | Lowest mean agreement |
| **MedBullets** | 160 | 0.3961 | 0.4111 | |
| **MedXpert** | 286 | 0.3956 | 0.4247 | |
| **MMLU** | 192 | 0.4050 | 0.4377 | Highest mean agreement |

### Source-Level Insights
- **JAMA** questions have the **lowest IRR** (0.3244), suggesting more nuanced/ambiguous clinical cases
- **MMLU** questions have the **highest IRR** (0.4050), indicating clearer/less ambiguous scenarios
- **Difference**: 0.0806 IRR points between best and worst sources (~25% relative difference)

---

## Key Findings

1. **Moderate but variable agreement**: Mean IRR of 0.38 indicates fair agreement, but this masks substantial variation
2. **Difficulty-dependent**: Only ~15% of questions achieved high agreement; the majority (85%) show moderate-to-low agreement
3. **Source bias**: Clinical case sources (JAMA) are harder to agree on than standardized test questions (MMLU)
4. **Rater disagreement is expected**: In medical education, clinician/student disagreement on sentence relevance is common due to different expertise levels

---

## Implications for Model Evaluation

- **Ground truth labels** in this dataset have inherent disagreement (~0.38 IRR)
- **Model accuracy against majority vote** may be penalized when they disagree with the consensus, even if their selection is clinically valid
- **Source-specific evaluation** may be needed—models might perform differently on high-IRR (MMLU) vs. low-IRR (JAMA) questions
- **Low IRR questions** (< 0.33) may need special handling or exclusion from performance metrics

---

## Files Referenced

- **Data file**: `/home/yuexing/NeuRIPS25/Physician_Labels/2026-03_March/933_Clinician_Student_Majority_Vote.csv`
  - Column: `IRR` (interrater reliability score for each question)
  - Column: `data_source_corr` (JAMA, MedXpert, MMLU, MedBullets)

---

**Generated**: June 22, 2026  
**Analysis Method**: Descriptive statistics on pre-computed IRR scores
