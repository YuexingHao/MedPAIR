#!/bin/bash
#SBATCH --job-name=medgemma27b_sr_shard
#SBATCH --partition=pi_mghassem
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuexing@cs.stanford.edu

# 1300 rows at ~23 s/row is ~8.4 h in one job, past the wall limit. Four shards
# of ~325 rows land at ~2.1 h each and run concurrently.
#
# Each shard writes its own partial file - parallel jobs must never write the
# shared prediction CSV. Merge afterwards with:
#   python ../shared/scripts/merge_sr_partials.py \
#     --target "results/predictions/[SR]_MedGemma27B_predictions_on_llama70b_removed.csv" \
#     --pred-col medgemma27b_direct_prediction \
#     --partials "results/predictions/partials/medgemma_shard_*.csv"
#
# Submit from /home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it

set -euo pipefail

mkdir -p logs results/predictions/partials

echo "=========================================="
echo "Array Job: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/orcd/compute/mghassem/001/gobi1/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd /home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it

echo ""
echo "Starting MedGemma-27B SR shard $SLURM_ARRAY_TASK_ID of 4..."
echo "Start time: $(date)"
echo ""

python -u notebooks/run_medgemma_sr_inference_batched.py \
  --num-shards 4 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --batch-size 16 \
  --partial-out "results/predictions/partials/medgemma_shard_${SLURM_ARRAY_TASK_ID}.csv" \
  "$@"

echo ""
echo "Shard $SLURM_ARRAY_TASK_ID complete at: $(date)"
echo "=========================================="
