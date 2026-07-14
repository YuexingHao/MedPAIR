#!/usr/bin/env bash
set -euo pipefail
LOG="/home/yuexing/NeuRIPS25/After_PT_Removal/results/slurm_logs/monitor_random_open_models.log"
FINAL="/home/yuexing/NeuRIPS25/After_PT_Removal/results/slurm_logs/monitor_random_open_models_final.tsv"
JOB_ID="15590098"

echo "[$(date -u '+%F %T UTC')] monitor started for job ${JOB_ID}" >> "$LOG"
while squeue -h -j "$JOB_ID" | grep -q .; do
  python - <<'PY' >> "$LOG"
import pandas as pd, os, datetime
p='/home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_Random.csv'
ts=datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
if os.path.exists(p):
    df=pd.read_csv(p)
    print(f"[{ts}] medgemma rows={len(df)} pred_nonnull={df['medgemma_direct_prediction'].notna().sum()} ext_nonnull={df['medgemma_extracted_answer'].notna().sum()}")
else:
    print(f"[{ts}] medgemma file missing")
PY
  sleep 300
done

echo "[$(date -u '+%F %T UTC')] medgemma job done; writing final summary" >> "$LOG"
python - <<'PY' > "$FINAL"
import pandas as pd
files={
'Llama70B':('/home/yuexing/NeuRIPS25/After_PT_Removal/Llama-70B/results/predictions/Llama70B_predictions_on_Random.csv','llama70b_direct_prediction','llama70b_extracted_answer'),
'MedGemma27B':('/home/yuexing/NeuRIPS25/After_PT_Removal/MedGemma-27b-text-it/results/predictions/MedGemma27B_predictions_on_Random.csv','medgemma_direct_prediction','medgemma_extracted_answer'),
'Qwen72B':('/home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-72B-Instruct/results/predictions/Qwen_72B_predictions_on_Random.csv','qwen72b_direct_prediction','qwen72b_extracted_answer'),
'Qwen14B':('/home/yuexing/NeuRIPS25/After_PT_Removal/Qwen2.5-14B-Instruct/results/predictions/Qwen_14B_predictions_on_Random.csv','qwen14b_direct_prediction','qwen14b_extracted_answer'),
}
print('model\trows\tpred_nonnull\textracted_nonnull')
for m,(p,pcol,ecol) in files.items():
    df=pd.read_csv(p)
    print(f"{m}\t{len(df)}\t{df[pcol].notna().sum()}\t{df[ecol].notna().sum()}")
PY

echo "[$(date -u '+%F %T UTC')] final summary written to $FINAL" >> "$LOG"
