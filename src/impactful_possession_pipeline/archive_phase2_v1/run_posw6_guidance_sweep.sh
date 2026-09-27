#!/usr/bin/env bash
set -euo pipefail

cd /home/peter/fb-proj

RUN_DIR="impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048"
CHECKPOINT="${RUN_DIR}/best_forward_simulator.pt"
SWEEP_DIR="${RUN_DIR}/guidance_sweep"
LOG_FILE="${SWEEP_DIR}/guidance_sweep.log"

mkdir -p "${SWEEP_DIR}"
: > "${LOG_FILE}"

for guidance in 0 1 2 4 8 12; do
  echo "START guidance=${guidance} date=$(date -Is)" | tee -a "${LOG_FILE}"
  /home/peter/miniconda3/envs/fb-env/bin/python -u -m impactful_possession_pipeline.evaluate_rollout_diagnostics \
    --checkpoint "${CHECKPOINT}" \
    --out-json "${SWEEP_DIR}/rollout_5matches_guidance_${guidance}.json" \
    --max-files 5 \
    --cut-minute 45 \
    --end-minute 60 \
    --min-events-per-minute 33 \
    --max-events-per-minute 49 \
    --possession-guidance-strength "${guidance}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo "END guidance=${guidance} date=$(date -Is)" | tee -a "${LOG_FILE}"
done
