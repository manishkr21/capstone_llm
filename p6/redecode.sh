#!/bin/bash
# usage: ./redecode.sh TAG "gen.key=value ..." run_cfg1 run_cfg2 ...   (decode finished runs with other settings)
cd "$(dirname "$0")"; source ../cenv/bin/activate
tag=$1; sets=$2; shift 2
for c in "$@"; do
  python train.py --skip-train --config configs/base.yaml "configs/$c.yaml" --set gen.tag=$tag $sets > "logs/${c}_dec_$tag.log" 2>&1
done
