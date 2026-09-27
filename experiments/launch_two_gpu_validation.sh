#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/peter/miniconda3/envs/fb-env/bin/python
CHECKPOINT="$ROOT/impactful_possession_pipeline/runs/phase1_v3_value_head_finetune_w64/best_forward_simulator.pt"
HOLDOUT="$ROOT/impactful_possession_pipeline/runs/pinned_diagnostic_holdout_25.json"
N_FORWARD="${N_FORWARD:-25}"
N_PLANNER="${N_PLANNER:-10}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$ROOT/paper_reproduction/runs/$STAMP"

mkdir -p "$RUN_ROOT/E01_forward_model" "$RUN_ROOT/E05_guarded_planner"
nvidia-smi -q > "$RUN_ROOT/nvidia-smi.txt"
"$PY" -m pip freeze > "$RUN_ROOT/python-freeze.txt"

FORWARD_CMD=("$PY" -u -m impactful_possession_pipeline.evaluate_pipeline
  --checkpoint "$CHECKPOINT" --max-matches "$N_FORWARD" --seed 42
  --out-json "$RUN_ROOT/E01_forward_model/results.json")
PLANNER_CMD=("$PY" -u -m impactful_possession_pipeline.evaluate_confirmed_planner_pipeline
  --checkpoint "$CHECKPOINT" --holdout-json "$HOLDOUT"
  --max-matches "$N_PLANNER" --cut-minute 45 --step-minutes 5
  --confirm-top-k 1 --confirm-rollout-seeds 3 --seed 20260729 --resume
  --out-json "$RUN_ROOT/E05_guarded_planner/results.json")

printf '%q ' env CUDA_VISIBLE_DEVICES=0 "${FORWARD_CMD[@]}" > "$RUN_ROOT/E01_forward_model/command.sh"
printf '\n' >> "$RUN_ROOT/E01_forward_model/command.sh"
printf '%q ' env CUDA_VISIBLE_DEVICES=1 "${PLANNER_CMD[@]}" > "$RUN_ROOT/E05_guarded_planner/command.sh"
printf '\n' >> "$RUN_ROOT/E05_guarded_planner/command.sh"

(
  cd "$ROOT"
  env CUDA_VISIBLE_DEVICES=0 "${FORWARD_CMD[@]}" > "$RUN_ROOT/E01_forward_model/stdout.log" 2>&1
  printf 'complete\n' > "$RUN_ROOT/E01_forward_model/status"
) &
FORWARD_PID=$!
printf '%s\n' "$FORWARD_PID" > "$RUN_ROOT/E01_forward_model/pid"

(
  cd "$ROOT"
  env CUDA_VISIBLE_DEVICES=1 "${PLANNER_CMD[@]}" > "$RUN_ROOT/E05_guarded_planner/stdout.log" 2>&1
  printf 'complete\n' > "$RUN_ROOT/E05_guarded_planner/status"
) &
PLANNER_PID=$!
printf '%s\n' "$PLANNER_PID" > "$RUN_ROOT/E05_guarded_planner/pid"

printf '%s\n' "$RUN_ROOT" > "$ROOT/paper_reproduction/LATEST_RUN"
printf 'running\n' > "$RUN_ROOT/E01_forward_model/status"
printf 'running\n' > "$RUN_ROOT/E05_guarded_planner/status"
printf 'run_root=%s forward_pid=%s planner_pid=%s\n' "$RUN_ROOT" "$FORWARD_PID" "$PLANNER_PID"

