#!/usr/bin/env bash
set -euo pipefail
GPU_INDEX="${1:?usage: monitor_gpu_memory.sh GPU_INDEX LOG_FILE [LIMIT_MIB]}"
LOG_FILE="${2:?usage: monitor_gpu_memory.sh GPU_INDEX LOG_FILE [LIMIT_MIB]}"
LIMIT_MIB="${3:-9500}"
printf 'timestamp_utc,gpu_index,memory_used_mib,memory_total_mib,utilization_pct,temperature_c,status\n' > "$LOG_FILE"
while true; do
  LINE="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -i "$GPU_INDEX")"
  USED="$(printf '%s\n' "$LINE" | cut -d, -f2 | tr -d ' ')"
  STATUS=ok
  if (( USED >= LIMIT_MIB )); then STATUS=over_limit; fi
  printf '%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LINE" "$STATUS" >> "$LOG_FILE"
  sleep 30
done
