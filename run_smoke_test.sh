#!/usr/bin/env bash
# End-to-end check of the whole Phase-1 pipeline on a tiny synthetic dataset.
# Takes a few minutes on a laptop CPU/MPS. Nothing here needs the real dataset.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
SMOKE=configs/smoke.yaml

$PY tests/test_models.py
$PY scripts/make_toy_data.py --out data_toy/hf
$PY scripts/01_prepare_data.py   --config configs/base.yaml $SMOKE
$PY scripts/02_train_tokenizer.py --config configs/base.yaml $SMOKE
$PY scripts/03_tokenize.py        --config configs/base.yaml $SMOKE

for ARCH in a1_gpt a2_encdec a3_prefixlm a4_efficient; do
  $PY scripts/04_train.py --config configs/$ARCH.yaml $SMOKE --set train.resume=false
done
for RUN in A1_gpt A2_encdec A3_prefixlm A4_efficient; do
  $PY scripts/05_generate.py --run runs_toy/$RUN --split dev
  $PY scripts/06_evaluate.py --pred runs_toy/$RUN/predictions/dev_t0.8_k50_p0.95_rp1.1/prediction.json \
      --prompts data_toy/dev/dev_prompts.json --refs data_toy/dev/dev_references.json --no-bertscore
done
$PY scripts/05_generate.py --run runs_toy/A1_gpt --split test
$PY scripts/07_compare.py --runs-dir runs_toy
$PY scripts/08_plot_resources.py --runs-dir runs_toy
echo "SMOKE TEST PASSED"
