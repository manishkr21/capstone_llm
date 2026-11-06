#!/usr/bin/env bash
# Train the given configs (default: all Phase-7 experiments) with P jobs in parallel on one GPU.
#   P=5 bash run_queue.sh e5_a4_partial e1_a4_lorma ...
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
P=${P:-5}
CFGS=${*:-"e1_a3_lorma e2_a1_full e3_a4_partial e4_a2_qlora e5_a3_ia3 e6_a1_prefix e7_a4_prompt"}
mkdir -p logs
for c in $CFGS; do
  n=$(grep -m1 run_name configs/$c.yaml | awk '{print $2}')
  [ -f runs/$n/summary.json ] || echo $c
done | xargs -P $P -I{} bash -c "$PY train.py --config configs/{}.yaml > logs/{}.log 2>&1 && echo 'ok   {}' || echo 'FAIL {}'"
echo "QUEUE DONE"
