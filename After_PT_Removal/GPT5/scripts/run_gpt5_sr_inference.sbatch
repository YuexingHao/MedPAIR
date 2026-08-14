#!/bin/bash
#SBATCH --job-name=gpt5_sr_inference
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuexing@cs.stanford.edu

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "=========================================="

source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

# Key resolution order: inherited OPENAI_API_KEY, else a private key file.
# Prefer the file - it keeps the secret out of shell history and out of
# `scontrol show job` / `ps` output. Create it once with:
#   install -m 600 /dev/null ~/.config/openai/key
#   printf '%s' 'sk-...' > ~/.config/openai/key
KEY_FILE="${OPENAI_KEY_FILE:-$HOME/.config/openai/key}"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -r "$KEY_FILE" ]; then
    OPENAI_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
    export OPENAI_API_KEY
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: no OpenAI key found."
    echo "Either export OPENAI_API_KEY, or write it to $KEY_FILE (mode 600)."
    exit 1
fi

cd /home/yuexing/NeuRIPS25/After_PT_Removal/GPT5/scripts

echo ""
echo "Starting GPT-5 SR inference via OpenAI API..."
echo "Start time: $(date)"
echo ""

python -u run_gpt5_sr_inference.py "$@"

echo ""
echo "Inference complete at: $(date)"
echo "=========================================="
