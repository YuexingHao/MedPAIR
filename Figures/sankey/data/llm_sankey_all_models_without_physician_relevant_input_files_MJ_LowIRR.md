# llm_sankey_all_models_without_physician_relevant (MJ_LowIRR Inputs)

The Sankey figure generation run uses the following **round-2 rerun** files (`*_on_MJ_LowIRR.csv`):

- `NeuRIPS25/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_MJ_LowIRR.csv`
- `NeuRIPS25/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_MJ_LowIRR.csv`
- `NeuRIPS25/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_MJ_LowIRR.csv`
- `NeuRIPS25/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_on_MJ_LowIRR.csv`
- `NeuRIPS25/After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_MJ_LowIRR.csv`
- `NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_MJ_LowIRR.csv`

Notes:
- The wrapper script used is:
  - `NeuRIPS25/Figures/sankey/notebooks/make_llm_sankey_all_models_without_physician_relevant_mj_lowirr.py`
- Round-1 behavior (original baseline mapping) remains unchanged from the compiled base generator.
