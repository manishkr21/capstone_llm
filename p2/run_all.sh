#!/usr/bin/env bash
# Phase 2 (L-NER): train the six experiments, then build one Codabench zip per experiment.
#   bash run_all.sh            # all six
#   bash run_all.sh e2 e3      # only these (E6 needs E1 to exist: it is the teacher)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
mkdir -p logs

ALL=(e1_a3_full e2_a4_lorma e3_a1_lora e4_a2_adapter e5_a4_partial e6_kd_a3_small)
if [ $# -gt 0 ]; then
  SEL=()
  for want in "$@"; do for c in "${ALL[@]}"; do [[ $c == ${want}* ]] && SEL+=("$c"); done; done
else
  SEL=("${ALL[@]}")
fi

$PY scripts/check_alignment.py
for c in "${SEL[@]}"; do
  run=$(grep '^run_name:' configs/$c.yaml | awk '{print $2}')
  echo "=== $run"
  $PY train.py --config configs/$c.yaml 2>&1 | tee logs/$run.log
  $PY predict.py --run runs/$run 2>&1 | tee -a logs/$run.log
done
$PY scripts/compare.py
ls -l submissions/
