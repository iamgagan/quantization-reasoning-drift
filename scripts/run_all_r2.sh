#!/bin/bash
# Round 2: full MATH-500, 16384 cap, online loop-aware early stopping.
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p results_r2
for cfg in "r1d-1.5b-3bit r1d-3bit" "dsr-1.5b-3bit dsr-3bit" "r1d-1.5b-bf16 r1d-bf16" "dsr-1.5b-bf16 dsr-bf16"; do
  set -- $cfg
  echo "=== starting $2 at $(date) ==="
  $PY run_eval2.py --model-path models/$1 --tag $2 --n 500 --max-tokens 16384 --out-dir results_r2
done
echo "=== ROUND 2 ALL DONE at $(date) ==="
