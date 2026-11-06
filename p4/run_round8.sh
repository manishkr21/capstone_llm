#!/usr/bin/env bash
# One at a time: W1, W2 -> embed new encoders -> fusion models -> stacker.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
log(){ echo "$* $(date +%H:%M)"; }
for c in configs/w1_a4_full_16ch.yaml configs/w2_a4_top6_16ch.yaml; do
  n=$(grep -m1 run_name $c | awk '{print $2}')
  log "start $n"
  $PY train.py --config $c > logs/$(basename $c .yaml).log 2>&1 && echo "ok   $n" || echo "FAIL $n"
  [ -f runs/$n/test_probs.npy ] && $PY make_submission.py --run runs/$n 2>&1 | grep "run(s)"
done
for r in W1_a4_full_16ch W2_a4_top6_16ch E1_a1_full V6_kd_ens3 A_e6_first; do
  [ -f runs/$r/best.pt ] || continue
  [ -f cache/feats/${r}_test.npz ] && continue
  log "embed $r"
  $PY embed_chunks.py --run runs/$r > logs/embed_$r.log 2>&1 && echo "ok   embed $r" || echo "FAIL embed $r"
done
have(){ for r in "$@"; do [ -f cache/feats/${r}_test.npz ] && echo -n "$r "; done; }
F5=$(have V1_e6_16ch V2_e7_16ch V3_a1_partial W1_a4_full_16ch W2_a4_top6_16ch)
FALL=$(have V1_e6_16ch V2_e7_16ch V3_a1_partial W1_a4_full_16ch W2_a4_top6_16ch E1_a1_full V6_kd_ens3 A_e6_first)
log "start D_fuse5_lstm ($F5)";  $PY doc_train.py --feats $F5 --layer lstm --name D_fuse5_lstm 2>&1 | grep seed_f1
log "start D_fuseall_lstm ($FALL)"; $PY doc_train.py --feats $FALL --layer lstm --name D_fuseall_lstm 2>&1 | grep seed_f1
log "start D_fuseall_gru"; $PY doc_train.py --feats $FALL --layer gru --name D_fuseall_gru 2>&1 | grep seed_f1
$PY stack_submission.py --l2 0.01 --name STACK_round8 2>&1 | grep -E "nested|in-sample"
log "round8 done"
