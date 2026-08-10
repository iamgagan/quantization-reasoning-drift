#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
for cfg in "r1d-1.5b-3bit r1d-3bit" "dsr-1.5b-3bit dsr-3bit" "r1d-1.5b-bf16 r1d-bf16" "dsr-1.5b-bf16 dsr-bf16"; do
  set -- $cfg
  echo "=== starting $2 at $(date) ==="
  $PY run_eval.py --model-path models/$1 --tag $2 --n 50 --max-tokens 8192
done
echo "=== ALL DONE at $(date) ==="
