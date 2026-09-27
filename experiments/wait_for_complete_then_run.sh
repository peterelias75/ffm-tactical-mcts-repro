#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RESULT_JSON COMMAND_SH POLL_SECONDS" >&2
  exit 2
fi

result_json=$1
command_sh=$2
poll_seconds=$3

echo "waiting for complete=true in ${result_json}" 
while true; do
  if [[ -s "$result_json" ]] && rg -q '"complete"[[:space:]]*:[[:space:]]*true' "$result_json"; then
    echo "dependency complete at $(date -u +%Y-%m-%dT%H:%M:%SZ); running ${command_sh}"
    exec bash "$command_sh"
  fi
  sleep "$poll_seconds"
done
