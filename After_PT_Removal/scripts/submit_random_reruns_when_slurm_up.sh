#!/usr/bin/env bash
set -euo pipefail
for i in {1..180}; do
  if timeout 8s squeue -u yuexing >/dev/null 2>&1; then
    echo "SLURM reachable at $(date -u '+%F %T UTC')"
    sbatch /home/yuexing/NeuRIPS25/After_PT_Removal/scripts/rerun_random_qwen72b.sbatch
    sbatch /home/yuexing/NeuRIPS25/After_PT_Removal/scripts/rerun_random_llama70b.sbatch
    sbatch /home/yuexing/NeuRIPS25/After_PT_Removal/scripts/rerun_random_qwen14b.sbatch
    sbatch /home/yuexing/NeuRIPS25/After_PT_Removal/scripts/rerun_random_medgemma.sbatch
    exit 0
  fi
  echo "[$i/180] SLURM unreachable or timed out at $(date -u '+%H:%M:%S UTC'), retrying in 20s"
  sleep 20
done
echo "Timed out waiting for SLURM controller after ~60 minutes" >&2
exit 2
