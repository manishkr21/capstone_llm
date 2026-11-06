#!/usr/bin/env bash
# Retrain A1-A4 on the cleaned corpus (configs/clean.yaml, slides/data_audit_retrain.tex).
#   R1: from scratch, 3000 steps = 196.6M tokens (same as the first Phase-1 runs)
#   R2: continue R1 for 12,260 steps, 1B tokens seen in total (configs/clean_r2.yaml)
# Two queues run in parallel on one shared GPU (four at once ran out of memory).  Effective batch is unchanged
# (65,536 tokens/step); only the micro-batch / accumulation split and torch.compile
# differ, for throughput (A2 cannot be compiled: its encoder length changes per step).
#   bash run_clean.sh                               # queues "a4_efficient a1_gpt" and "a3_prefixlm a2_encdec"
#   QUEUES="a4_efficient a1_gpt|a3_prefixlm" bash run_clean.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
QUEUES=${QUEUES:-"a4_efficient a1_gpt|a3_prefixlm a2_encdec"}

[ -f data_clean/corpus/train.jsonl ] || $PY scripts/01_prepare_data.py --config configs/base.yaml configs/clean.yaml
[ -f data_clean/tokens/train.bin ]   || $PY scripts/03_tokenize.py --config configs/base.yaml configs/clean.yaml

speed() {
  case $1 in
    a2_encdec) echo "train.micro_batch_size=32 train.grad_accum_steps=2 train.compile=false" ;;
    *)         echo "train.micro_batch_size=16 train.grad_accum_steps=4 train.compile=true" ;;
  esac
}

one_arch() {
  local ARCH=$1 BASE R1 R2
  BASE=$($PY -c "import sys;sys.path.insert(0,'.');from legal_lm.config import load_config;print(load_config(['configs/$ARCH.yaml'])['run_name'])")
  R1=${BASE}_clean; R2=${BASE}_clean_1B
  $PY scripts/04_train.py --config configs/$ARCH.yaml configs/clean.yaml --set run_name=$R1 $(speed $ARCH) || return 1
  $PY scripts/04_train.py --config configs/$ARCH.yaml configs/clean.yaml configs/clean_r2.yaml \
      --set run_name=$R2 train.init_from=runs/$R1/last.pt $(speed $ARCH) || return 1
  for RUN in $R1 $R2; do
    $PY scripts/05_generate.py --run runs/$RUN --split dev || return 1
    $PY scripts/06_evaluate.py --pred runs/$RUN/predictions/dev_t0.8_k50_p0.95_rp1.1/prediction.json \
        --prompts data_clean/dev/dev_prompts.json --refs data_clean/dev/dev_references.json || return 1
    $PY scripts/05_generate.py --run runs/$RUN --split test || return 1
  done
}

run_queue() {
  for ARCH in $1; do
    one_arch $ARCH > logs_clean/$ARCH.log 2>&1 && echo "$ARCH done" || echo "$ARCH FAILED (logs_clean/$ARCH.log)"
  done
}

mkdir -p logs_clean
IFS='|' read -ra QS <<< "$QUEUES"
for Q in "${QS[@]}"; do
  run_queue "$Q" &
  sleep 120   # let the first compile finish before the next queue starts
done
wait
echo "all done"
