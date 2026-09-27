#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$(sed -n '1p' "$ROOT/paper_reproduction/LATEST_RUN")"
printf 'run_root=%s\n' "$RUN_ROOT"
for EXP in E01_forward_model E05_guarded_planner; do
  printf '\n[%s] status=%s pid=%s\n' "$EXP" "$(sed -n '1p' "$RUN_ROOT/$EXP/status")" "$(sed -n '1p' "$RUN_ROOT/$EXP/pid")"
  tail -n 8 "$RUN_ROOT/$EXP/stdout.log" 2>/dev/null || true
done

