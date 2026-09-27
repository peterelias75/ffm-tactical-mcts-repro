#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GUIDANCE MAX_FILES DEVICE [extra evaluate_rollout_diagnostics args...]" >&2
  exit 2
fi

GUIDANCE="$1"
MAX_FILES="$2"
DEVICE="$3"
shift 3
EXTRA_ARGS=("$@")

cd /home/peter/fb-proj

RUN_DIR="impactful_possession_pipeline/runs/forward_simulator_full_3400matches_posw6_finetune_4epochs_20260629_maxlen2048"
CHECKPOINT="${RUN_DIR}/best_forward_simulator.pt"
OUT_DIR="${RUN_DIR}/${GUIDANCE_COMPARE_DIR:-guidance_compare_30matches}"
OUT_JSON="${OUT_DIR}/rollout_${MAX_FILES}matches_guidance_${GUIDANCE}.json"
LOG_FILE="${OUT_DIR}/rollout_${MAX_FILES}matches_guidance_${GUIDANCE}.log"

mkdir -p "${OUT_DIR}"

echo "START guidance=${GUIDANCE} max_files=${MAX_FILES} device=${DEVICE} date=$(date -Is)" > "${LOG_FILE}"
/home/peter/miniconda3/envs/fb-env/bin/python -u -m impactful_possession_pipeline.evaluate_rollout_diagnostics \
  --checkpoint "${CHECKPOINT}" \
  --out-json "${OUT_JSON}" \
  --max-files "${MAX_FILES}" \
  --cut-minute 45 \
  --end-minute 60 \
  --device "${DEVICE}" \
  --min-events-per-minute 33 \
  --max-events-per-minute 49 \
  --possession-guidance-strength "${GUIDANCE}" \
  "${EXTRA_ARGS[@]}" \
  >> "${LOG_FILE}" 2>&1
echo "END guidance=${GUIDANCE} max_files=${MAX_FILES} device=${DEVICE} date=$(date -Is)" >> "${LOG_FILE}"
