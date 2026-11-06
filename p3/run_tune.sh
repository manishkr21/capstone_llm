#!/usr/bin/env bash
# Tuning round: 5-fold CV for configs/tune/*.yaml (folds in parallel), then submissions + comparison.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
P=${P:-8}
mkdir -p logs
DIR=${DIR:-configs/tune}
for f in 0 1 2 3 4; do for c in $DIR/*.yaml; do
  n=$(grep -m1 run_name $c | awk '{print $2}')
  [ -f runs/$n/fold$f/summary.json ] || echo "$c $f"
done; done | xargs -P $P -L1 bash -c '$0 train.py --config $1 --folds $2 > logs/$(basename $1 .yaml)_f$2.log 2>&1 && echo "ok   $1 fold $2" || echo "FAIL $1 fold $2"' $PY
for c in $DIR/*.yaml; do
  n=$(grep -m1 run_name $c | awk '{print $2}')
  [ -f runs/$n/cv_summary.json ] && $PY make_submission.py --run runs/$n
done
$PY compare.py > /dev/null && echo "comparison updated"
