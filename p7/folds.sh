#!/usr/bin/env bash
# 2-fold out-of-fold runs (train.fold) of the given configs, P jobs in parallel: run names F_<run_name>_f{0,1}.
#   P=4 bash folds.sh e1_a3_lorma e5_a3_ia3
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
P=${P:-4}
for c in $*; do
  n=$(grep -m1 run_name configs/$c.yaml | awk '{print $2}')
  for k in 0 1; do [ -f runs/F_${n}_f$k/summary.json ] || echo "$c $n $k"; done
done | xargs -P $P -L1 bash -c '$0 train.py --config configs/$1.yaml --set run_name=F_$2_f$3 "train.fold=[$3,2]" > logs/f_$1_f$3.log 2>&1 && echo "ok   $2 f$3" || echo "FAIL $2 f$3"' $PY
echo "FOLDS DONE"
