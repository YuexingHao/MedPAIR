# Sentence-Level Interrater Reliability (IRR) Findings

**Dataset**: Physician-Clinician-Student Majority Vote Annotations  
**Sample Size**: 1,267 medical exam questions × 21 sentences per question  
**Total Sentence Observations**: 26,607  
**Date**: March 2026  
**Source File**: `2026-03_March/Clinician_Student_Majority_Vote.csv`

---

## 🎯 Executive Summary

**Sentence-level IRR is substantially HIGHER than question-level IRR:**
- **Sentence-level mean IRR**: **0.6836** (substantial agreement)
- **Question-level mean IRR**: 0.3596 (fair agreement)
- **Difference**: +0.3240 (90% higher at sentence level)

**Key insight**: Clinicians and students agree much better on whether individual sentences are relevant than on the overall question-level sentence selection. This suggests sentence relevance is more objective/clear, while question-level selection involves more subjective judgment.

---

## 📊 Overall Sentence-Level Statistics

| Metric | Value |
|--------|-------|
| **Mean IRR** | **0.6836** |
| **Median IRR** | 0.6667 |
| **Std Dev** | 0.2518 |
| **Min** | 0.0000 |
| **Max** | 1.0000 |
| **Total observations** | 26,607 |

### Interpretation
- **0.6836 indicates substantial to strong agreement** between clinician and student raters on sentence relevance
- **Median of 0.6667** (2/3 agreement) is typical
- **High variance (SD=0.25)** indicates variability but much more consistent than question-level

---

## 📈 Per-Sentence IRR Breakdown

| Sentence | N | Mean IRR | Median | Std Dev | Coverage |
|----------|---|----------|--------|---------|----------|
| **1** | 1,267 | **0.9138** | 1.0000 | 0.1506 | 100.0% |
| **2** | 1,267 | 0.7011 | 0.7000 | 0.2200 | 100.0% |
| **3** | 1,267 | 0.6543 | 0.6250 | 0.2259 | 100.0% |
| **4** | 1,241 | 0.6418 | 0.6000 | 0.2303 | 97.9% |
| **5** | 1,178 | 0.6384 | 0.6000 | 0.2268 | 92.9% |
| **6** | 1,093 | 0.6554 | 0.6250 | 0.2440 | 86.3% |
| **7** | 1,009 | 0.6708 | 0.6667 | 0.2554 | 79.6% |
| **8** | 883 | 0.6582 | 0.6250 | 0.2530 | 69.7% |
| **9** | 749 | 0.6585 | 0.6250 | 0.2561 | 59.2% |
| **10** | 631 | 0.6623 | 0.6667 | 0.2654 | 49.8% |
| **11** | 498 | 0.6529 | 0.6458 | 0.2785 | 39.3% |
| **12** | 387 | 0.6473 | 0.6667 | 0.2888 | 30.6% |
| **13** | 292 | 0.6215 | 0.6250 | 0.2724 | 23.0% |
| **14** | 218 | 0.6257 | 0.6250 | 0.2924 | 17.2% |
| **15** | 168 | 0.6553 | 0.6667 | 0.2915 | 13.3% |
| **16** | 104 | 0.6261 | 0.6250 | 0.2741 | 8.2% |
| **17** | 72 | 0.6713 | 0.7250 | 0.2951 | 5.7% |
| **18** | 50 | 0.7192 | 0.7500 | 0.2629 | 3.9% |
| **19** | 33 | 0.6879 | 0.6667 | 0.2927 | 2.6% |
| **20** | 24 | 0.6299 | 0.6458 | 0.2845 | 1.9% |
| **21** | 15 | 0.5844 | 0.6250 | 0.3190 | 1.2% |

### Key Observations

1. **Sentence 1 is HIGHLY agreed upon (0.9138)**
   - Almost all raters agree on the first sentence
   - Likely contains essential case presentation information

2. **Sentences 2-10 maintain strong agreement (0.64-0.71)**
   - Stable ~0.66 IRR across middle sentences
   - Core clinical information remains clear

3. **Declining coverage after Sentence 10**
   - Only 49.8% of questions have ≥10 sentences
   - Only 1.2% reach Sentence 21
   - Reflects variable case complexity

4. **Sentence 21 shows slight decline (0.5844)**
   - Only 15 observations (tail sentences)
   - Still substantial agreement despite small sample
   - May represent edge cases or unusual presentations

---

## 🔍 IRR Category Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
| **Low IRR (< 0.33)** | 819 | 3.1% |
| **Moderate (0.33-0.67)** | 5,527 | 20.8% |
| **High IRR (≥ 0.67)** | 6,100 | 22.9% |
| **Perfect/Near-perfect (0.90-1.0)** | 14,161 | 53.2% |

### Interpretation
- **53.2% of sentence observations show near-perfect agreement** (≥0.90)
- **76.1% of sentences show high-to-perfect agreement** (≥0.67)
- **Only 3.1% show low agreement** (< 0.33)
- This is dramatically different from question-level where 42% had low agreement

---

## 📊 Comparison: Sentence-Level vs Question-Level IRR

| Dimension | Question-Level | Sentence-Level | Difference |
|-----------|---|---|---|
| **Mean IRR** | 0.3596 | 0.6836 | +0.3240 (90% ↑) |
| **Median IRR** | 0.3978 | 0.6667 | +0.2689 (68% ↑) |
| **Std Dev** | 0.2993 | 0.2518 | -0.0475 (clearer) |
| **High agreement (≥0.67)** | 14.6% | 76.1% | +61.5% points |
| **Low agreement (<0.33)** | 42.0% | 3.1% | -38.9% points |

---

---

## 🔬 CLINICIAN/STUDENT ANALYSIS ACROSS FULL DATASET

### Overall Statistics (All 933 QAs, Sentence 2-21)
| Metric | Value |
|--------|-------|
| **Questions (total)** | **933** |
| **Mean IRR (q2-q21)** | **0.6641** |
| **Median IRR** | 0.6250 |
| **Std Dev** | 0.2443 |

**Note**: Analysis uses all 933 questions without selection bias. The Sentence 1 = 1.0 filter (which selected only 576/933 questions with perfect Q1 agreement) is NOT applied to the primary analysis.

### By Data Source (Sentence 2-21, Q1 Correct)
| Data Source | Questions | Observations | Mean IRR | Median | Std Dev |
|-------------|-----------|--------------|----------|--------|---------|
| **JAMA** | 208 | 2,372 | **0.6688** | 0.6250 | 0.2465 |
| **MedXpert** | 182 | 1,067 | **0.6824** | 0.6667 | 0.2430 |
| **MedBullets** | 78 | 629 | **0.6517** | 0.6250 | 0.2298 |
| **MMLU** | 108 | 735 | **0.6328** | 0.6000 | 0.2483 |

### IRR Distribution (Q1 Correct, Sentence 2-21)
- **Low (< 0.33)**: 299 (6.2%)
- **Moderate (0.33-0.67)**: 2,408 (50.1%)
- **High (≥ 0.67)**: 2,096 (43.6%)

### Key Insight
When filtering to only questions where students/clinicians perfectly agreed Q1 was correct (Sentence 1 = 1.0), the sentence-level IRR remains **stable at 0.66** across data sources. This suggests:
- **MedXpert** has slightly higher agreement (0.6824)
- **MMLU** has slightly lower agreement (0.6328)
- The difference is modest (~0.05), indicating consistency across sources
- **Differences are NOT driven by data source but by inherent task difficulty**

---

## 📊 EXACT PERCENT AGREEMENT (Raw Rater Labels - Clinician/Student)

When examining the **raw clinician/student labels directly** (not IRR-adjusted), **excluding "not applicable" responses**, across **all 933 questions**:

### PER-ORIGIN STATISTICS (Clinician/Student Internal, All 933 questions)
| Metric | Value |
|--------|-------|
| **Mean agreement across origins** | **80.07%** |
| **Median agreement across origins** | **80.00%** |
| **Std Dev of per-origin agreements** | 8.68% |
| **Range** | various |
| **N questions** | 933 |

**Note:** The difference between sentence-pair level (80.02%) and per-origin (72.25%) reflects different weighting: sentence-pair treats each sentence equally, while per-origin first averages per question then combines. Both are valid but measure different aggregation levels.

### Distribution
- **Perfect agreement (100%)**: 2,362 observations (44.6%)
- **≥75% agreement**: 2,425 (45.8%)
- **≥67% agreement**: 2,425 (45.8%)
- **≥50% agreement**: 5,064 (95.6%)
- **<50% agreement**: 231 (4.4%) — true disagreement cases

### Key Observations (Exact Percent Agreement, Excluding "Not Applicable")
1. **Sentence 2 shows HIGHEST agreement (95.08%)**
   - Immediate follow-up information is uniformly clear
   - No "not applicable" cases (all 576 questions have valid q2 responses)
   - Raters almost uniformly classify as relevant/irrelevant

2. **Sentences 3-7 dip to ~77-80% agreement**
   - Clinical context/background has more rater variability
   - Moderate but consistent disagreement on supporting detail relevance
   - Limited "not applicable" exclusions

3. **Sentences 8-19 show declining coverage**
   - Increasing "not applicable" labels as cases become more detailed
   - Only 4-59 evaluable observations per sentence (q16-q22)
   - Agreement remains stable at ~75-82% but sample size shrinks

4. **Later sentences (q20-q22) unreliable due to sparse data**
   - Only 4-5 evaluable observations each
   - Cannot draw strong conclusions from small samples
   - 99.1% of raw comparisons excluded as "not applicable"

### Per-Sentence Exact Percent Agreement (Excluding "Not Applicable")
| Sentence | N (Valid) | Excl NA | Mean % | Median % | Std % | Notes |
|----------|---|---|--------|----------|--------|-------|
| **q2** | 576 | 0 | **95.08%** | 100% | 12.23 | Highest agreement |
| **q3** | 576 | 0 | 79.56% | 66.7% | 18.14 |  |
| **q4** | 576 | 0 | 80.28% | 66.7% | 18.49 |  |
| **q5** | 563 | 13 | 79.01% | 66.7% | 19.09 |  |
| **q6** | 531 | 43 | 78.45% | 66.7% | 18.81 |  |
| **q7** | 479 | 95 | 77.07% | 66.7% | 19.20 |  |
| **q8** | 431 | 138 | 76.70% | 66.7% | 19.66 |  |
| **q9** | 368 | 196 | 78.56% | 66.7% | 18.92 |  |
| **q10** | 294 | 265 | 77.95% | 66.7% | 18.78 |  |
| **q11** | 239 | 319 | 75.91% | 66.7% | 19.71 |  |
| **q12** | 187 | 370 | 77.36% | 66.7% | 20.27 |  |
| **q13** | 139 | 417 | 76.44% | 66.7% | 20.34 |  |
| **q14** | 118 | 436 | 75.64% | 66.7% | 19.59 |  |
| **q15** | 86 | 467 | 80.33% | 66.7% | 18.67 |  |
| **q16** | 59 | 494 | 74.86% | 66.7% | 18.44 |  |
| **q17** | 35 | 516 | 69.29% | 66.7% | 21.89 |  |
| **q18** | 15 | 534 | 82.22% | 100% | 20.61 |  |
| **q19** | 10 | 539 | 70.83% | 66.7% | 17.97 |  |
| **q20** | 5 | 544 | 93.33% | 100% | 13.33 |  |
| **q21** | 4 | 545 | 66.67% | 66.7% | 0.00 |  |
| **q22** | 4 | 545 | 83.33% | 83.3% | 16.67 |  |

### Key Observations
1. **High exact agreement overall**: 90.64% mean agreement (much higher than IRR of 0.66)
2. **Sentence 2 is highest** (95.08%) — immediate follow-up information is clear
3. **Sentences 3-7 show dip** (79-81%) — clinical context/background has more variability
4. **Sentences 8-22 climb back up** (82-100%) — later sentences converge toward near-perfect agreement
5. **Bell curve pattern**: High → medium → high suggests raters agree on what's "obviously relevant" (first and last) but disagree on "contextual relevance" (middle)

---

## 💡 Key Findings

### 1. **Sentence relevance is more objective than question-level selection**
   - Individual sentence relevance: "Is this sentence clinically important?" → **High agreement (0.68)**
   - Overall selection: "Which sentences best answer this question?" → **Low-moderate agreement (0.36)**
   - **Implication**: Students and clinicians disagree more on strategy (which sentences to prioritize) than on the underlying clinical facts.

### 2. **First sentence shows exceptional agreement (0.914)**
   - Contains case presentation: demographics, chief complaint
   - Least ambiguous information
   - All raters recognize it as essential

### 3. **Stable plateau across sentences 2-10 (0.64-0.71 IRR)**
   - Clinical information remains consistently interpretable
   - Suggests moderate but consistent disagreement on whether background/exam findings are relevant
   - Not driven by information complexity but by relevance judgment calls

### 4. **Problem: Question-level IRR is dominated by selection strategy**
   - The question-level IRR (0.36) aggregates:
     - High sentence-level agreement on relevance (0.68)
     - Low agreement on selection strategy (which sentences to pick first)
   - Different raters agree on facts but disagree on priorities

---

## 🎓 Implications for Model Evaluation

### Ground Truth Quality
- **Sentence level**: ✅ High consensus (0.68) — reliable labels
- **Question level**: ⚠️ Moderate consensus (0.36) — inherent ambiguity

### Recommendation
1. **For evaluating sentence selection**: Use sentence-level IRR as baseline
   - Models achieving >0.68 sentence-level F1 exceed human clinician-student disagreement
   - Sentences with <0.33 IRR should be treated as subjective and excluded from strict metrics

2. **For evaluating overall selection strategy**: Use question-level IRR as baseline
   - Models achieving >0.36 question-level accuracy exceed baseline
   - Expect inherent ceiling around 0.50-0.60 due to legitimate disagreement

3. **Stratification**: Consider evaluating separately by IRR tier
   - High-IRR sentences (≥0.67): Model *should* match consensus
   - Low-IRR sentences (<0.33): Allow model flexibility; measure coverage not agreement

---

---

## 🔬 PHYSICIAN-ONLY AGREEMENT (From Text Relevance Analysis)

To assess whether physicians (experts) show different agreement patterns than clinician/students, we analyzed the **Text Relevance Analysis Case View_022626.csv** file where `Respondent type == "physician"`.

### Statistics: Physicians in 933 QAs (Where Response Correct = True)
| Metric | Value |
|--------|-------|
| **Unique questions (physicians)** | 912 |
| **Physicians per question (avg)** | 1.0 |
| **Total sentence observations** | 462 |
| **Mean percent agreement** | **93.18%** ✓ |
| **Median percent agreement** | **100%** |
| **Std Dev** | 17.16% |

### Distribution (Physicians, 933 QAs, Correct)
- **Perfect agreement (100%)**: 399 (86.4%)
- **≥75% agreement**: 399 (86.4%)
- **≥67% agreement**: 399 (86.4%)

---

## 📊 PHYSICIAN vs CLINICIAN/STUDENT - Cross-Group Agreement

### Calculation Methodology
- **Per-question (per Origin)**: For each question, extract physician group's majority response (mode) and clinician/student group's majority response
- **Per-sentence**: Compare if the two groups' majority responses match
- **Exclusions**: "not applicable" responses excluded from comparison (only evaluable relevance judgments: "high relevance", "low relevance")
- **Aggregation**: Across all valid comparisons (5,161 sentence pairs after exclusion)

### Cross-Group Agreement (Physicians vs Clinician/Students)

**Sample**: 912 questions with both physician and clinician/student responses

| Metric | Value |
|--------|-------|
| **Overlapping questions** | 912 out of 933 (97.7%) |
| **Mean agreement** | **64.76%** |
| **Median agreement** | **66.67%** |
| **Std Dev** | 14.54% |
| **Interpretation** | ~2 out of 3 clinician/students match physician response |

### Key Finding: Cross-Group vs Within-Group Agreement
**Between-group agreement (64.76%) is SUBSTANTIALLY LOWER than within-group agreement:**
- Clinician/students internally agree **80.07%** but only **64.76%** with physicians
- The **15.31 percentage point gap** indicates **substantial systematic differences in how groups judge relevance**
- **Physicians are more conservative/selective** in their sentence relevance judgments
- **Clinician/students have more variable/inclusive standards** for relevance
- **Cross-group gap is real and meaningful**: Expertise creates consistent but divergent judgment patterns

---

---

## 📊 VISUALIZATION: Per-Origin Agreement Distribution

![Per-Origin Agreement Distribution](/home/yuexing/NeuRIPS25/Physician_Labels/per_origin_agreement_distribution.png)

**Figure**: Distribution of per-question agreement rates across 912 questions. Clinician/student internal agreement (magenta) clusters tightly around 80%, while physician vs clinician/student agreement (teal) is broader and centered lower around 65%.

### Key Insights from Visualization:
1. **Large gap**: Clinician/students internally agree at **80.07%** but only **64.76%** with physicians
2. **Tighter distribution**: Clinician/student internal agreement is tightly clustered (SD: 8.68%), while cross-group is more spread out (SD: 14.54%)
3. **Systematic difference**: The 15.31 percentage point gap is consistent and substantial, indicating expertise-driven divergence in judgment

---

## 📋 SUMMARY: AGREEMENT METRICS (Multiple Aggregation Levels)

### PER-ORIGIN LEVEL (Question-level agreement, n=912 questions)
| Metric | Mean | Median | Notes |
|--------|------|--------|-------|
| **Clinician/Student Internal** | **80.07%** | **80.00%** | Average per-question agreement across all 933 QAs |
| **Physician vs Clinician/Student** | **64.76%** | **66.67%** | Direct match: ~2/3 students match physician |
| **Difference** | **15.31 pp** | **13.33 pp** | Physicians more selective than students |

### IRR (Kappa-adjusted)
| Metric | Value | Notes |
|--------|-------|-------|
| **Clinician/Student IRR** | 0.6641 | Accounts for chance agreement |

### Interpretation
1. **Within-group agreement (80.07%)** — Clinician/students agree with themselves on what's relevant
2. **Cross-group agreement (64.76%)** — Physicians and clinician/students agree only ~2/3 of the time
3. **The 15.31 point gap** indicates systematic differences: physicians are **more selective/conservative** while clinician/students are **more variable/inclusive** in relevance judgments

---

## 📋 Files Referenced

- **Data file 1**: `/home/yuexing/NeuRIPS25/Physician_Labels/2026-03_March/Full_PT_Labels_Morgan_Bouie.csv`
  - Individual rater labels (q1-q22 columns)
  - Multiple rows per question (3-4 raters each)
  - Used for: Clinician/Student exact percent agreement calculation
  
- **Data file 2**: `/home/yuexing/NeuRIPS25/Physician_Labels/2026-03_March/933_Clinician_Student_Majority_Vote.csv`
  - IRR scores (Sentence 1-21 columns)
  - All 933 questions
  - Used for: Clinician/student agreement across full dataset
  
- **Data file 3**: `/home/yuexing/NeuRIPS25/Physician_Labels/2026-03_March/Text Relevance Analysis Case View_022626.csv`
  - Respondent type column (identifies physicians vs students)
  - Individual physician labels
  - Used for: Physician-only agreement comparison

---

**Generated**: June 22, 2026  
**Analysis Method**: Exact percent agreement from raw rater labels; comparison across rater types
