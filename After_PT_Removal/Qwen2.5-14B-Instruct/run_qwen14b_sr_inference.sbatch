#!/bin/bash
#SBATCH --job-name=qwen14b_sr_inference
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuexing@cs.stanford.edu

# Submit from /home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-14B-Instruct
# so the relative logs/ paths above resolve.

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

# The model is read from the lab's shared HF cache (local snapshot, no download).
export HF_HOME=/orcd/compute/mghassem/001/gobi1/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Do NOT set CUDA_VISIBLE_DEVICES here - Slurm already scopes it to the
# allocated GPUs.

cd /home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-14B-Instruct

echo ""
echo "Starting Qwen2.5-14B-Instruct SR inference..."
echo "Start time: $(date)"
echo ""

python -u notebooks/run_qwen14b_sr_inference_updated.py "$@"

echo ""
echo "Inference complete at: $(date)"
echo "=========================================="
