#!/bin/bash
#SBATCH --job-name=qwen72b_sr_inference
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuexing@cs.stanford.edu

# Setup logging directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Allocated GPUs: $SLURM_GPUS"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Number of Nodes: $SLURM_NNODES"
echo "=========================================="

# Module setup
module load cuda/12.1
module load cudnn/8.9.1

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Activate virtual environment (adjust path as needed)
source /home/yuexing/venv/bin/activate

# Navigate to project directory
cd /home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-72B-Instruct

# Run inference script
echo ""
echo "Starting Qwen2.5-72B-Instruct SR inference..."
echo "Start time: $(date)"
echo ""

python notebooks/run_qwen72b_sr_inference_updated.py

echo ""
echo "Inference complete at: $(date)"
echo "=========================================="
