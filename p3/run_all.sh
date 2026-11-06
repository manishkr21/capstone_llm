#!/usr/bin/env bash
# Phase 3: 5-fold CV for every experiment (folds run in parallel on one GPU), then the KD
# experiment (its fold-k teacher is E1 fold k), then one fold-ensemble submission per experiment.
#   bash run_all.sh            # everything
#   P=4 bash run_all.sh        # fewer parallel folds
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
P=${P:-8}
mkdir -p logs

job() {  # "<config> <fold> [overrides...]"
  set -- $1
  local cfg=$1 fold=$2; shift 2
  local name=$(basename $cfg .yaml)$( [ $# -gt 0 ] && echo "_${1##*=}" | tr '/' '_' )
  $PY train.py --config configs/$cfg.yaml --folds $fold ${@:+--set "$@"} > logs/${name}_f$fold.log 2>&1 \
    && echo "ok   $cfg fold $fold $*" || echo "FAIL $cfg fold $fold $* (logs/${name}_f$fold.log)"
}
export -f job
export PY

jobs() {
  for f in 0 1 2 3 4; do
    for c in e1_a4_lorma e0_a4_lorma_sentonly e2_a1_lora e3_a3_rslora e4_a2_ia3 e5_a4_qlora e7_a3_adalora e8_a4_prefix e9_a1_prompt; do
      [ -f runs/$(grep -m1 run_name configs/$c.yaml | awk '{print $2}')/fold$f/summary.json ] || echo "$c $f"
    done
    # same recipe as E1 on the ORIGINAL Phase-1 A4 (backbone comparison)
    [ -f runs/E1_a4_lorma_origP1/fold$f/summary.json ] || \
      echo "e1_a4_lorma $f run_name=E1_a4_lorma_origP1 model.phase1_ckpt=runs/A4_efficient/best.pt"
  done
}
jobs | xargs -P $P -I{} bash -c 'job "{}"'
seq 0 4 | xargs -P 5 -I{} bash -c 'job "e6_kd_a4_small {}"'

for r in runs/*/; do
  [ -f $r/cv_summary.json ] && $PY make_submission.py --run ${r%/}
done
$PY compare.py
