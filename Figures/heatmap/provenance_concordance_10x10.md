# Provenance: `concordance_10x10_SR_CC_jaccard_high.csv`

This matrix is rebuilt by:

- `/home/yuexing/NeuRIPS25/Figures/concordance_rate/update_cc_concordance_and_heatmap.py`

The transposed heatmap figure is rebuilt by:

- `/home/yuexing/NeuRIPS25/Figures/heatmap/make_concordance_heatmap_transposed.py`

## Upstream data inputs

Human high-sentence sets (MV > 0.66):

- `/home/yuexing/NeuRIPS25/Physician_Labels/Mar2_2026_Data/933_Clinician_Student_Majority_Vote.csv`

CC top-K sentence sets:

- `/home/yuexing/NeuRIPS25/Merge/attribution/14b_contextcite_analysis/qwen14b_contextcite_topk_summary.csv` (`qwen14b_topK_sentence_ids`)
- `/home/yuexing/NeuRIPS25/Merge/attribution/qwen72b_contextcite/qwen72b_contextcite_topk_summary.csv` (`qwen72b_topK_sentence_ids`)
- `/home/yuexing/NeuRIPS25/Merge/attribution/llama70b_contextcite/llama70b_contextcite_topk_summary.csv` (`llama70b_topK_sentence_ids`)

SR high-sentence sets:

- `/home/yuexing/NeuRIPS25/Physician_Labels/results/GPT4o_MatchRate.csv` (`label_1..21`, id `ID`)
- `/home/yuexing/NeuRIPS25/Physician_Labels/results/GPT5_MatchRate.csv` (`q1..q21`, id `ID_corr`)
- `/home/yuexing/NeuRIPS25/Physician_Labels/results/[SR]Qwen14B_annotated_MedPAIR_relevancy.csv` (`q1..q21`, id `ID`)
- `/home/yuexing/NeuRIPS25/Physician_Labels/results/[SR]Qwen72B_annotated_MedPAIR_relevancy.csv` (`q1..q21`, id `ID`)
- `/home/yuexing/NeuRIPS25/Physician_Labels/results/[SR]Llama70B_annotated_ORIGINAL_Accuracy.csv` (`q1..q21`, id `ID`)
- `/home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/data/raw/MedGemma_SR_Match_Rate.csv` (`q1..q21`, id `ID`)

## Formula used for each matrix cell

For each pair of labelers A and B and each shared Origin:

- `Jaccard(A, B) = |High_A ∩ High_B| / |High_A ∪ High_B|`

The reported matrix entry is:

- `100 * mean_origin(Jaccard(A, B))`

Diagonal entries are fixed to `100.0`.

## Outputs

- `/home/yuexing/NeuRIPS25/Figures/heatmap/concordance_10x10_SR_CC_jaccard_high.csv`
- `/home/yuexing/NeuRIPS25/Figures/heatmap/concordance_heatmap_transposed.pdf`
- `/home/yuexing/NeuRIPS25/Figures/heatmap/concordance_heatmap_transposed.png`
