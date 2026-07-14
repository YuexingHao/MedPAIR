# Three-Row Input Mapping (Relevant / Irrelevant / Random)

This document defines exactly how each row is fed into model reruns.

## Row 1: Relevant

- Meaning: keep physician-trainee relevant content (the rerun column is `New_Sentences`).
- Canonical input CSV:
  - `/home/yuexing/NeuRIPS25/After_PT_Removal/shared/data/Centaur_Lab_Second_Round.csv`
- Context column used for GPT models:
  - `New_Sentences`
- Output predictions:
  - Qwen-72B: `Qwen_72B_predictions_trainee_irr_removed.csv`
  - Qwen-14B: `Qwen_14B_predictions_trainee_irr_removed.csv`
  - Llama-70B: `Llama70B_predictions_on_trainee_irr_removed.csv`
  - MedGemma-27B: `MedGemma27B_predictions_on_trainee_irr_removed.csv`
  - GPT-4o: `gpt4o_predictions_on_trainee_irr_removed.csv`
  - GPT-5: `gpt5_predictions_on_trainee_irr_removed.csv`

## Row 2: Irrelevant

- Meaning: physician MJ low/irrelevant-only context.
- Canonical input CSV:
  - `/home/yuexing/NeuRIPS25/After_PT_Removal/shared/data/Centaur_Lab_First_Round_933_MJ_LowIRR_as_FilteredSentences_for_rerun.csv`
- Context column used for GPT models:
  - `Filtered_Sentences`
- Output predictions:
  - Qwen-72B: `Qwen_72B_predictions_on_MJ_LowIRR.csv`
  - Qwen-14B: `Qwen_14B_predictions_on_MJ_LowIRR.csv`
  - Llama-70B: `Llama70B_predictions_on_MJ_LowIRR.csv`
  - MedGemma-27B: `MedGemma27B_predictions_on_MJ_LowIRR.csv`
  - GPT-4o: `gpt4o_predictions_on_MJ_LowIRR_expert933_subset_from_existing1300.csv`
  - GPT-5: `gpt5_predictions_on_MJ_LowIRR.csv`

## Row 3: Random

- Meaning: random-sentence control context.
- Canonical input CSV:
  - `/home/yuexing/NeuRIPS25/After_PT_Removal/shared/data/Centaur_Lab_First_Round_1300_Random_as_NewSentences_for_rerun.csv`
- Context column used for GPT models:
  - `Random_Sentences`
- Output predictions:
  - Qwen-72B: `Qwen_72B_predictions_on_Random.csv`
  - Qwen-14B: `Qwen_14B_predictions_on_Random.csv`
  - Llama-70B: `Llama70B_predictions_on_Random.csv`
  - MedGemma-27B: `MedGemma27B_predictions_on_Random.csv`
  - GPT-4o: `gpt4o_predictions_on_Random.csv`
  - GPT-5: `gpt5_predictions_on_Random.csv`

## Notes

- Open-source model reruns use:
  - `/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts/__pycache__/rerun_physician_irrelevant_predictions.cpython-313.pyc`
- GPT-4o reruns use:
  - `/home/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts/rerun_mj_lowirr_gpt4o.py`
- GPT-5 reruns use:
  - `/home/yuexing/NeuRIPS25/After_PT_Removal/GPT5/scripts/predict_gpt5_on_context.py`
