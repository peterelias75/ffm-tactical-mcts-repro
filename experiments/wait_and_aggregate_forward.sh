#!/usr/bin/env bash
set -euo pipefail
OUT_JSON="${1:?output JSON required}"
shift
INPUTS=("$@")
while true; do
  READY=1
  for FILE in "${INPUTS[@]}"; do
    if [[ ! -s "$FILE" ]]; then READY=0; break; fi
    if ! /home/peter/miniconda3/envs/fb-env/bin/python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["summary"].get("complete") else 1)' "$FILE"; then
      READY=0
      break
    fi
  done
  if (( READY == 1 )); then break; fi
  sleep 60
done
CMD=(/home/peter/miniconda3/envs/fb-env/bin/python -u -m paper_reproduction.aggregate_forward_results --out-json "$OUT_JSON")
for FILE in "${INPUTS[@]}"; do CMD+=(--input "$FILE"); done
"${CMD[@]}"
