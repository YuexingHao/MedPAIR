#!/bin/bash
#SBATCH --job-name=medgemma-mj-lowirr
#SBATCH --partition=pi_mghassem
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/results/slurm_logs/slurm-medgemma-mj-lowirr-%j.out

set -euo pipefail
source /home/yuexing/miniconda/etc/profile.d/conda.sh
conda activate base

python /orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/shared/scripts/__pycache__/rerun_physician_irrelevant_predictions.cpython-313.pyc \
  --input-csv /orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/shared/data/Centaur_Lab_First_Round_933_MJ_LowIRR_as_NewSentences_for_rerun.csv \
  --model medgemma \
  --medgemma-out /orcd/home/002/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_MJ_LowIRR.csv \
  --expert-933-only
