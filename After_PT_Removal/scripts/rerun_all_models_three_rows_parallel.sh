#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yuexing/NeuRIPS25"
SCRIPT_DIR="$ROOT/After_PT_Removal/scripts"
LOG_DIR="$ROOT/After_PT_Removal/results/slurm_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=0
SKIP_GPT=0
ROW_FILTER="all"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-gpt) SKIP_GPT=1 ;;
    --row=*)
      ROW_FILTER="${arg#*=}"
      ROW_FILTER="${ROW_FILTER,,}"
      case "$ROW_FILTER" in
        relevant|irrelevant|random|all) ;;
        *)
          echo "Invalid --row value: $ROW_FILTER" >&2
          echo "Allowed: relevant, irrelevant, random, all" >&2
          exit 2
          ;;
      esac
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--dry-run] [--skip-gpt] [--row=relevant|irrelevant|random|all]" >&2
      exit 2
      ;;
  esac
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_MAP="$LOG_DIR/rerun_all_models_three_rows_${TS}.tsv"
echo -e "job_name\tjob_id\trow\tmodel\toutput" > "$JOB_MAP"

echo "============================================================"
echo "Three-row rerun mapping (explicit):"
echo "  Relevant  -> input: shared/data/Centaur_Lab_Second_Round.csv ; context=New_Sentences"
echo "  Irrelevant-> input: shared/data/Centaur_Lab_First_Round_933_MJ_LowIRR_as_FilteredSentences_for_rerun.csv ; context=Filtered_Sentences"
echo "  Random    -> input: shared/data/Centaur_Lab_First_Round_1300_Random_as_NewSentences_for_rerun.csv ; context=Random_Sentences"
echo "  Row filter: ${ROW_FILTER}"
echo "Full mapping doc: $SCRIPT_DIR/three_row_input_mapping.md"
echo "============================================================"

row_enabled() {
  local row_lc="${1,,}"
  [[ "$ROW_FILTER" == "all" || "$ROW_FILTER" == "$row_lc" ]]
}

submit_file_job() {
  local row="$1"
  local model="$2"
  local out="$3"
  local sbatch_file="$4"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] sbatch $sbatch_file"
    echo -e "${model}_${row}\tDRYRUN\t${row}\t${model}\t${out}" >> "$JOB_MAP"
    return 0
  fi
  local jid
  jid="$(sbatch --parsable "$sbatch_file")"
  echo "[SUBMIT] $jid  $sbatch_file"
  echo -e "${model}_${row}\t${jid}\t${row}\t${model}\t${out}" >> "$JOB_MAP"
}

submit_wrap_job() {
  local row="$1"
  local model="$2"
  local out="$3"
  local job_name="$4"
  local partition="$5"
  local cpus="$6"
  local gres="$7"
  local time_lim="$8"
  local cmd="$9"
  local log_file="${LOG_DIR}/slurm-${job_name}-%j.out"

  local wrap_cmd
  wrap_cmd="bash -lc 'source /home/yuexing/miniconda/etc/profile.d/conda.sh; conda activate base; ${cmd}'"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] sbatch --job-name=${job_name} --partition=${partition} --cpus-per-task=${cpus} ${gres:+--gres=${gres}} --time=${time_lim} --output=${log_file} --wrap=${wrap_cmd}"
    echo -e "${job_name}\tDRYRUN\t${row}\t${model}\t${out}" >> "$JOB_MAP"
    return 0
  fi

  local jid
  if [[ -n "$gres" ]]; then
    jid="$(sbatch --parsable \
      --job-name="$job_name" \
      --partition="$partition" \
      --nodes=1 \
      --gres="$gres" \
      --cpus-per-task="$cpus" \
      --mem=0 \
      --time="$time_lim" \
      --output="$log_file" \
      --wrap="$wrap_cmd")"
  else
    jid="$(sbatch --parsable \
      --job-name="$job_name" \
      --partition="$partition" \
      --nodes=1 \
      --cpus-per-task="$cpus" \
      --mem=16G \
      --time="$time_lim" \
      --output="$log_file" \
      --wrap="$wrap_cmd")"
  fi
  echo "[SUBMIT] $jid  $job_name"
  echo -e "${job_name}\t${jid}\t${row}\t${model}\t${out}" >> "$JOB_MAP"
}

RERUN_PYC="$ROOT/After_PT_Removal/shared/scripts/__pycache__/rerun_physician_irrelevant_predictions.cpython-313.pyc"
IN_RELEVANT="$ROOT/After_PT_Removal/shared/data/Centaur_Lab_Second_Round.csv"
IN_IRRELEVANT="$ROOT/After_PT_Removal/shared/data/Centaur_Lab_First_Round_933_MJ_LowIRR_as_FilteredSentences_for_rerun.csv"
IN_RANDOM="$ROOT/After_PT_Removal/shared/data/Centaur_Lab_First_Round_1300_Random_as_NewSentences_for_rerun.csv"

# ---------------------------
# Open-source: Relevant row
# ---------------------------
if row_enabled "relevant"; then
  submit_wrap_job "Relevant" "Qwen-72B" \
    "$ROOT/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_trainee_irr_removed.csv" \
    "qwen72b-relevant" "pi_mghassem" "16" "gpu:4" "08:00:00" \
    "python $RERUN_PYC --input-csv $IN_RELEVANT --model qwen72b --qwen72b-out $ROOT/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_trainee_irr_removed.csv --expert-933-only"

  submit_wrap_job "Relevant" "Qwen-14B" \
    "$ROOT/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_trainee_irr_removed.csv" \
    "qwen14b-relevant" "pi_mghassem" "8" "gpu:2" "08:00:00" \
    "python $RERUN_PYC --input-csv $IN_RELEVANT --model qwen14b --qwen14b-out $ROOT/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_trainee_irr_removed.csv --expert-933-only"

  submit_wrap_job "Relevant" "Llama-70B" \
    "$ROOT/After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_trainee_irr_removed.csv" \
    "llama70b-relevant" "pi_mghassem" "16" "gpu:4" "08:00:00" \
    "python $RERUN_PYC --input-csv $IN_RELEVANT --model llama70b --llama70b-out $ROOT/After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_trainee_irr_removed.csv --expert-933-only"

  submit_wrap_job "Relevant" "MedGemma-27B" \
    "$ROOT/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_trainee_irr_removed.csv" \
    "medgemma-relevant" "pi_mghassem" "12" "gpu:4" "12:00:00" \
    "python $RERUN_PYC --input-csv $IN_RELEVANT --model medgemma --medgemma-out $ROOT/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_trainee_irr_removed.csv --expert-933-only"
fi

# ---------------------------
# Open-source: Irrelevant + Random rows (existing sbatch scripts)
# ---------------------------
if row_enabled "irrelevant"; then
  submit_file_job "Irrelevant" "Qwen-72B" \
    "$ROOT/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_MJ_LowIRR.csv" \
    "$SCRIPT_DIR/rerun_mj_qwen72b.sbatch"
  submit_file_job "Irrelevant" "Qwen-14B" \
    "$ROOT/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_on_MJ_LowIRR.csv" \
    "$SCRIPT_DIR/rerun_mj_qwen14b.sbatch"
  submit_file_job "Irrelevant" "Llama-70B" \
    "$ROOT/After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_MJ_LowIRR.csv" \
    "$SCRIPT_DIR/rerun_mj_llama70b.sbatch"
  submit_file_job "Irrelevant" "MedGemma-27B" \
    "$ROOT/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_MJ_LowIRR.csv" \
    "$SCRIPT_DIR/rerun_mj_medgemma.sbatch"
fi

if row_enabled "random"; then
  submit_file_job "Random" "Qwen-72B" \
    "$ROOT/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_Random.csv" \
    "$SCRIPT_DIR/rerun_random_qwen72b.sbatch"
  submit_file_job "Random" "Qwen-14B" \
    "$ROOT/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_on_Random.csv" \
    "$SCRIPT_DIR/rerun_random_qwen14b.sbatch"
  submit_file_job "Random" "Llama-70B" \
    "$ROOT/After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_Random.csv" \
    "$SCRIPT_DIR/rerun_random_llama70b.sbatch"
  submit_file_job "Random" "MedGemma-27B" \
    "$ROOT/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_Random.csv" \
    "$SCRIPT_DIR/rerun_random_medgemma.sbatch"
fi

# ---------------------------
# Closed-source: GPT4o / GPT5
# ---------------------------
if [[ "$SKIP_GPT" -eq 0 ]]; then
  if row_enabled "relevant"; then
    submit_wrap_job "Relevant" "GPT4o" \
      "$ROOT/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_trainee_irr_removed.csv" \
      "gpt4o-relevant" "pi_mghassem" "4" "" "24:00:00" \
      "python $ROOT/After_PT_Removal/shared/scripts/rerun_mj_lowirr_gpt4o.py --input-csv $IN_RELEVANT --output-csv $ROOT/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_trainee_irr_removed.csv --context-column New_Sentences --force-rerun"

    submit_wrap_job "Relevant" "GPT5" \
      "$ROOT/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_trainee_irr_removed.csv" \
      "gpt5-relevant" "pi_mghassem" "4" "" "24:00:00" \
      "python $ROOT/After_PT_Removal/GPT5/scripts/predict_gpt5_on_context.py --input-csv $IN_RELEVANT --output-csv $ROOT/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_trainee_irr_removed.csv --context-column New_Sentences --prediction-column gpt5_direct_prediction --answer-column answer_corr --force-rerun"
  fi

  if row_enabled "irrelevant"; then
    submit_wrap_job "Irrelevant" "GPT4o" \
      "$ROOT/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_MJ_LowIRR_expert933_subset_from_existing1300.csv" \
      "gpt4o-irrelevant" "pi_mghassem" "4" "" "24:00:00" \
      "python $ROOT/After_PT_Removal/shared/scripts/rerun_mj_lowirr_gpt4o.py --input-csv $IN_IRRELEVANT --output-csv $ROOT/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_MJ_LowIRR_expert933_subset_from_existing1300.csv --context-column Filtered_Sentences --force-rerun"

    submit_wrap_job "Irrelevant" "GPT5" \
      "$ROOT/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_MJ_LowIRR.csv" \
      "gpt5-irrelevant" "pi_mghassem" "4" "" "24:00:00" \
      "python $ROOT/After_PT_Removal/GPT5/scripts/predict_gpt5_on_context.py --input-csv $IN_IRRELEVANT --output-csv $ROOT/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_MJ_LowIRR.csv --context-column Filtered_Sentences --prediction-column gpt5_direct_prediction --answer-column answer_corr --force-rerun"
  fi

  if row_enabled "random"; then
    submit_wrap_job "Random" "GPT4o" \
      "$ROOT/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_Random.csv" \
      "gpt4o-random" "pi_mghassem" "4" "" "24:00:00" \
      "python $ROOT/After_PT_Removal/shared/scripts/rerun_mj_lowirr_gpt4o.py --input-csv $IN_RANDOM --output-csv $ROOT/After_PT_Removal/GPT4o/results/predictions/gpt4o_predictions_on_Random.csv --context-column Random_Sentences --force-rerun"

    submit_wrap_job "Random" "GPT5" \
      "$ROOT/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_Random.csv" \
      "gpt5-random" "pi_mghassem" "4" "" "24:00:00" \
      "python $ROOT/After_PT_Removal/GPT5/scripts/predict_gpt5_on_context.py --input-csv $IN_RANDOM --output-csv $ROOT/After_PT_Removal/GPT5/results/predictions/gpt5_predictions_on_Random.csv --context-column Random_Sentences --prediction-column gpt5_direct_prediction --answer-column answer_corr --force-rerun"
  fi
else
  echo "[INFO] --skip-gpt enabled: GPT4o/GPT5 jobs were not submitted."
fi

echo
echo "Job map written to: $JOB_MAP"
echo "Monitor with: squeue -u yuexing"
echo "After completion, regenerate figures with:"
echo "  cd $ROOT"
echo "  python Figures/sankey/notebooks/make_llm_sankey_all_models_without_physician_relevant_llama_trainee.py"
echo "  python Figures/sankey/notebooks/make_llm_sankey_all_models_without_physician_llama_mj_only.py"
echo "  python Figures/sankey/notebooks/make_llm_sankey_all_models_without_physician_relevant_random.py"
echo "  python Figures/sankey/notebooks/make_llm_barplot_sankey_combined.py"
