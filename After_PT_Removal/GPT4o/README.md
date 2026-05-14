# GPT4o (After_PT_Removal)

Post–pre-training removal experiments for GPT-4o: round comparisons, “removed content” predictions, SR concordance, and figures.

## Layout

| Path | Contents |
|---|---|
| `paths.py` | `Path` constants — import in notebooks after the bootstrap cell (first cell) or run Jupyter with cwd under this folder |
| `data/raw/` | Merge / round CSVs (`merged_2k_with_4k_ID.csv`, `GPT4o_Round*.csv`, `GPT4o_Both_Rounds.csv`, `GPT4o_Before_After_Results.csv`) |
| `data/sr_concordance/` | Inputs and result for self-report vs physician concordance |
| `results/predictions/remove_low_irr/` | Remove–low/irrelevant experiment inputs (e.g. `Qwen72B_annotated_MedPAIR_relevancy.csv`, `GPT4o_predictions.csv`, `GPT4o_analysis_results.csv`) |
| `results/predictions/` | GPT-4o prediction CSVs (`gpt4o_predictions_on_*_removed.csv`), including Llama 70B (`gpt4o_predictions_on_llama70b_removed.csv`), Qwen 72B SR (`[SR]_gpt4o_predictions_on_qwen72b_removed.csv`), MedGemma (`gpt4o_predictions_on_MedGemma_removed.csv`), trainee removed (`gpt4o_predictions_on_trainee_removed.csv`, `gpt4o_predictions_on_trainee_irr_removed.csv`). GPT-5 outputs belong under `../GPT5/results/predictions/`. |
| `results/tables/` | Summary tables (`Result_Report.csv`, `model_performance_transposed.csv`, `MedGemma_SR_Match_Rate.csv`) |
| `results/figures/` | Exported PNG/PDF figures |
| `notebooks/` | Analysis notebooks (root + `sr_concordance/`) |
| `results/notebooks/remove_low_irr/` | Remove–low/irrelevant notebooks (e.g. Llama/Qwen GPT-4o runs) |
| `scripts/` | `gpt4o_compare.py`, `gpt4o_sr_concordance.py`, plotting helpers under `scripts/remove_low_irr/` |

## Scripts

- `scripts/gpt4o_compare.py` — Round1 vs Round2 comparison (writes `data/raw/GPT4o_Before_After_Results.csv`).
- `scripts/gpt4o_sr_concordance.py` — SR concordance pipeline (reads/writes under `data/sr_concordance/`).
- `scripts/evaluate_predictions_by_physician_subsets_gpt4o.py` — letter accuracy on `results/predictions/*.csv` vs physician subsets (933 / HardQA / Impossible); writes `results/eval_physician_subsets_summary.csv`.

Run from repo: `python scripts/gpt4o_compare.py` with cwd `After_PT_Removal/GPT4o`.

## OpenAI API key

Prediction notebooks under `results/notebooks/` use `openai.OpenAI()`, which requires **`export OPENAI_API_KEY='sk-...'`** in the terminal before you start Jupyter or run a script. See also `scripts/fill_hardqa_gpt4o_predictions.py` (`--api-key-file` supported).
