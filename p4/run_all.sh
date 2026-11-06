#!/usr/bin/env bash
# Phase 4: train the listed experiment configs, P at a time on one GPU, then write one zip per run.
#   bash run_all.sh configs/e1_a1_full.yaml configs/e2_a4_lorma.yaml ...      (default: configs/e[1-7]_*.yaml)
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
P=${P:-4}
mkdir -p logs
CFGS=${@:-$(ls configs/e[1-7]_*.yaml)}
printf "%s\n" $CFGS | xargs -P $P -I{} bash -c '$0 train.py --config {} > logs/$(basename {} .yaml).log 2>&1 && echo "ok   {}" || echo "FAIL {}"' $PY
for c in $CFGS; do
  n=$(grep -m1 run_name $c | awk '{print $2}')
  [ -f runs/$n/test_probs.npy ] && $PY make_submission.py --run runs/$n
done
