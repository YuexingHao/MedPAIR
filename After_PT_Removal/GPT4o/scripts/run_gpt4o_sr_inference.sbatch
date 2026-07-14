#!/bin/bash
#SBATCH --job-name=gpt4o_sr_inference
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuexing@cs.stanford.edu

mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "=========================================="

source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

# OPENAI_API_KEY must be supplied by the environment.
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Please set: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

cd /home/yuexing/NeuRIPS25/After_PT_Removal/GPT4o/scripts

echo ""
echo "Starting GPT-4o SR inference via OpenAI API..."
echo "Start time: $(date)"
echo ""

python run_gpt4o_sr_inference.py

echo ""
echo "Inference complete at: $(date)"
echo "=========================================="
