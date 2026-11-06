#!/usr/bin/env bash
# Cached full-coverage document models: (1) layer ablation on V1 (A4) features; (2) 3-encoder fusion.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
until [ -f cache/feats/V1_e6_16ch_test.npz ]; do sleep 30; done
for L in mean attn gru lstm transformer; do
  echo "start D_v1_$L $(date +%H:%M)"
  $PY doc_train.py --feats V1_e6_16ch --layer $L --name D_v1_$L 2>&1 | grep -v -i warn | tail -1
done
until grep -q "embed done" logs/embed.log; do sleep 30; done
for L in lstm gru transformer attn; do
  echo "start D_fuse3_$L $(date +%H:%M)"
  $PY doc_train.py --feats V1_e6_16ch V2_e7_16ch V3_a1_partial --layer $L --name D_fuse3_$L 2>&1 | grep -v -i warn | tail -1
done
echo "doc models done $(date +%H:%M)"
