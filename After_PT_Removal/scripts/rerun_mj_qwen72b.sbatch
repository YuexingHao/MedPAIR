#!/bin/bash
#SBATCH --job-name=qwen72b-mj-lowirr
#SBATCH --partition=pi_mghassem,mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/results/slurm_logs/slurm-qwen72b-mj-lowirr-%j.out

set -euo pipefail
source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

python /orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts/__pycache__/rerun_physician_irrelevant_predictions.cpython-313.pyc \
  --input-csv /orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/shared/data/Centaur_Lab_First_Round_933_MJ_LowIRR_as_NewSentences_for_rerun.csv \
  --model qwen72b \
  --qwen72b-out /orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_MJ_LowIRR.csv \
  --expert-933-only
