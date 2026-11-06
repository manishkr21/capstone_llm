#!/bin/bash
# usage: ./run_queue.sh cfg1 cfg2 ...   (runs the configs one after another, each on top of base.yaml)
cd "$(dirname "$0")"
source ../cenv/bin/activate
for c in "$@"; do
  n=$(basename "$c" .yaml)
  python train.py --config configs/base.yaml "configs/$n.yaml" > "logs/$n.log" 2>&1
done
