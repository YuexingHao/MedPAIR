#!/bin/bash
#SBATCH --job-name=llama70b_cc_mb
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/llama70b_cc_mb_%j.out
#SBATCH --error=logs/llama70b_cc_mb_%j.err
#SBATCH --chdir=/home/yuexing/NeuRIPS25/Merge/attribution/llama70b_contextcite

# Run Llama-3.1-70B-Instruct ContextCite over the 298 MedBullets QAs
# (Merge Q1035 - Merge Q1332 in merged_llm_4k_questions_with_QA_ID.csv).
# Outputs:
#   llama70b_attribution_scores_medbullets/attributions_summary_1035_1332.csv
#   llama70b_attribution_scores_medbullets/Merge_Q{N}.csv  (per-QA)

set -euo pipefail

mkdir -p logs llama70b_attribution_scores_medbullets

# Use the base conda env where context_cite + transformers + bitsandbytes live.
source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

CSV=/orcd/home/002/yuexing/NeuRIPS25/Merge/data/csv/merged_llm_4k_questions_with_QA_ID.csv

echo "[$(date)] Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi || true

PYTHONUNBUFFERED=1 python generate_attributions.py \
    --csv_path "$CSV" \
    --model_id meta-llama/Llama-3.1-70B-Instruct \
    --dtype float16 \
    --load_in_8bit \
    --device_map cuda:0 \
    --output_dir llama70b_attribution_scores_medbullets \
    --flush_every 5 \
    --start_id 1035 \
    --end_id 1332

echo "[$(date)] DONE."
