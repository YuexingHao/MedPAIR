#!/bin/bash
#SBATCH --job-name=medgemma27b_sr_batched
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
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

source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/orcd/compute/mghassem/001/gobi1/huggingface
# vLLM spawns worker processes for tensor parallelism; keep them off the shared
# tmpfs and quiet about telemetry.

export DO_NOT_TRACK=1

# Do NOT set CUDA_VISIBLE_DEVICES here - Slurm already scopes it to the
# allocated GPUs, and vLLM reads the allocation from it.

cd /home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it

echo ""
echo "Starting MedGemma-27B SR inference (batched)..."
echo "Start time: $(date)"
echo ""

python -u notebooks/run_medgemma_sr_inference_batched.py "$@"

echo ""
echo "Inference complete at: $(date)"
echo "=========================================="
