#!/usr/bin/env bash
# Improvement round, one run at a time: V1-V5, then V6 = KD from an ensemble of the 3 best single models.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
for c in configs/v1_e6_16ch.yaml configs/v2_e7_16ch.yaml configs/v3_a1_partial.yaml configs/v4_a3_partial.yaml configs/v5_e6_attnpool.yaml; do
  n=$(grep -m1 run_name $c | awk '{print $2}')
  if [ ! -f runs/$n/summary.json ]; then
    echo "start $n $(date +%H:%M)"
    $PY train.py --config $c > logs/$(basename $c .yaml).log 2>&1 && echo "ok   $n" || { echo "FAIL $n"; continue; }
  fi
  $PY make_submission.py --run runs/$n 2>&1 | grep "run(s)"
done
# V6: ensemble teacher = mean logits of the 3 best single models (by clean-validation macro-F1)
TOP=$($PY - <<'PY'
import json, glob
rs = []
for f in glob.glob("runs/*/summary.json"):
    s = json.load(open(f))
    if s["run"].startswith(("E", "V")) and not s["run"].startswith(("E8", "V6")):
        rs.append((s["val"]["macro_f1"], s["run"]))
print(" ".join(r for _, r in sorted(rs, reverse=True)[:3]))
PY
)
echo "V6 teacher members: $TOP"
for r in $TOP; do [ -f runs/$r/train_logits.npy ] || $PY dump_logits.py --run runs/$r > logs/dump_$r.log 2>&1; done
$PY -c "
import numpy as np, sys
m = [np.load(f'runs/{r}/train_logits.npy') for r in sys.argv[1:]]
lp = np.mean([x - np.logaddexp(x[:, :1], x[:, 1:]) for x in m], 0)   # mean log-probabilities
np.save('runs/teacher_ens3_logits.npy', lp); print('teacher logits', lp.shape, 'members', sys.argv[1:])
" $TOP
cat > configs/v6_kd_ens3.yaml <<EOF2
# V6: KD student (A4, 6 of 10 blocks, full FT) learning from the ensemble of the 3 best single models: $TOP
inherit: e8_kd_a4_small.yaml
run_name: V6_kd_ens3
distill: {teacher_logits: runs/teacher_ens3_logits.npy, alpha: 0.5, temperature: 2.0}
EOF2
echo "start V6_kd_ens3 $(date +%H:%M)"
$PY train.py --config configs/v6_kd_ens3.yaml > logs/v6_kd_ens3.log 2>&1 && echo "ok   V6_kd_ens3" || echo "FAIL V6_kd_ens3"
$PY make_submission.py --run runs/V6_kd_ens3 2>&1 | grep "run(s)"
echo "queue done $(date +%H:%M)"
