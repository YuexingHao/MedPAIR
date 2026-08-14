#!/bin/bash
#SBATCH --job-name=medgemma27b_sr_inference
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuexing@cs.stanford.edu

# Submit from /home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it
# so the relative logs/ paths above resolve.

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

# /home/yuexing/venv no longer exists; conda base carries torch/transformers.
source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

# The model is read from the lab's shared HF cache (local snapshot, no download).
export HF_HOME=/orcd/compute/mghassem/001/gobi1/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Do NOT set CUDA_VISIBLE_DEVICES here - Slurm already scopes it to the
# allocated GPUs, and overriding it pointed the job at devices it did not own.

cd /home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it

echo ""
echo "Starting MedGemma-27B-Text-It SR inference..."
echo "Start time: $(date)"
echo ""

python -u notebooks/run_medgemma_sr_inference_updated.py "$@"

echo ""
echo "Inference complete at: $(date)"
echo "=========================================="
