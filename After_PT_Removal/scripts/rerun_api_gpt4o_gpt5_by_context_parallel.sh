#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yuexing/NeuRIPS25"
SCRIPT_DIR="$ROOT/After_PT_Removal/scripts"
LOG_DIR="$ROOT/After_PT_Removal/results/slurm_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=0
CONTEXT_FILTER="all"  # filtered | clinician | random | all
WORKERS=4

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --context=*)
      CONTEXT_FILTER="${arg#*=}"
      CONTEXT_FILTER="${CONTEXT_FILTER,,}"
      case "$CONTEXT_FILTER" in
        filtered|clinician|random|all) ;;
        *)
          echo "Invalid --context value: $CONTEXT_FILTER" >&2
          echo "Allowed: filtered, clinician, random, all" >&2
          exit 2
          ;;
      esac
      ;;
    --workers=*)
      WORKERS="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--dry-run] [--context=filtered|clinician|random|all] [--workers=N]" >&2
      exit 2
      ;;
  esac
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_MAP="$LOG_DIR/rerun_api_gpt4o_gpt5_by_context_${TS}.tsv"
echo -e "job_name\tjob_id\tmodel\tcontext_key\tcontext_column\tinput_csv\toutput_csv" > "$JOB_MAP"

IN_LOWIRR="$ROOT/After_PT_Removal/shared/data/Centaur_Lab_First_Round_933_MJ_LowIRR_as_FilteredSentences_for_rerun.csv"
IN_RANDOM="$ROOT/After_PT_Removal/shared/data/Centaur_Lab_First_Round_1300_Random_as_NewSentences_for_rerun.csv"

GPT4_SCRIPT="$ROOT/After_PT_Removal/shared/scripts/rerun_mj_lowirr_gpt4o.py"
GPT5_SCRIPT="$ROOT/After_PT_Removal/GPT5/scripts/predict_gpt5_on_context.py"
OUT4_DIR="$ROOT/After_PT_Removal/GPT4o/results/predictions"
OUT5_DIR="$ROOT/After_PT_Removal/GPT5/results/predictions"

echo "============================================================"
echo "Raw API rerun mapping (GPT-4o + GPT-5 only; no plotting):"
echo "  filtered : input=$IN_LOWIRR ; context=Filtered_Sentences"
echo "  clinician: input=$IN_LOWIRR ; context=Clinician_Student_MJ_Low/IRR_Sentences"
echo "  random   : input=$IN_RANDOM ; context=Random_Sentences"
echo "  filter   : $CONTEXT_FILTER"
echo "  workers  : $WORKERS"
echo "============================================================"

context_enabled() {
  local key="${1,,}"
  [[ "$CONTEXT_FILTER" == "all" || "$CONTEXT_FILTER" == "$key" ]]
}

submit_wrap_job() {
  local model="$1"
  local context_key="$2"
  local context_col="$3"
  local input_csv="$4"
  local output_csv="$5"
  local job_name="$6"
  local cmd="$7"

  local log_file="${LOG_DIR}/slurm-${job_name}-%j.out"
  local wrap_cmd="bash -lc 'source /home/yuexing/miniconda/etc/profile.d/conda.sh; conda activate base; ${cmd}'"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] sbatch --job-name=${job_name} --partition=pi_mghassem --cpus-per-task=4 --mem=16G --time=24:00:00 --output=${log_file} --wrap=${wrap_cmd}"
    echo -e "${job_name}\tDRYRUN\t${model}\t${context_key}\t${context_col}\t${input_csv}\t${output_csv}" >> "$JOB_MAP"
    return 0
  fi

  local jid
  jid="$(sbatch --parsable \
    --job-name="$job_name" \
    --partition="pi_mghassem" \
    --nodes=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time="24:00:00" \
    --output="$log_file" \
    --wrap="$wrap_cmd")"
  echo "[SUBMIT] $jid  $job_name"
  echo -e "${job_name}\t${jid}\t${model}\t${context_key}\t${context_col}\t${input_csv}\t${output_csv}" >> "$JOB_MAP"
}

submit_context_pair() {
  local context_key="$1"
  local context_col="$2"
  local input_csv="$3"
  local out_tag="$4"

  local out4="${OUT4_DIR}/gpt4o_predictions_ctx_${out_tag}.csv"
  local out5="${OUT5_DIR}/gpt5_predictions_ctx_${out_tag}.csv"

  local cmd4="python $GPT4_SCRIPT --input-csv $input_csv --output-csv $out4 --context-column \"$context_col\" --workers $WORKERS --force-rerun"
  local cmd5="python $GPT5_SCRIPT --input-csv $input_csv --output-csv $out5 --context-column \"$context_col\" --prediction-column gpt5_direct_prediction --answer-column answer_corr --workers $WORKERS --force-rerun"

  submit_wrap_job "GPT4o" "$context_key" "$context_col" "$input_csv" "$out4" "gpt4o-api-${context_key}" "$cmd4"
  submit_wrap_job "GPT5" "$context_key" "$context_col" "$input_csv" "$out5" "gpt5-api-${context_key}" "$cmd5"
}

if context_enabled "filtered"; then
  submit_context_pair "filtered" "Filtered_Sentences" "$IN_LOWIRR" "filtered_sentences"
fi
if context_enabled "clinician"; then
  submit_context_pair "clinician" "Clinician_Student_MJ_Low/IRR_Sentences" "$IN_LOWIRR" "clinician_student_mj_lowirr_sentences"
fi
if context_enabled "random"; then
  submit_context_pair "random" "Random_Sentences" "$IN_RANDOM" "random_sentences"
fi

echo
echo "Job map written to: $JOB_MAP"
echo "Monitor with: squeue -u yuexing"
echo "Script path: $SCRIPT_DIR/rerun_api_gpt4o_gpt5_by_context_parallel.sh"
