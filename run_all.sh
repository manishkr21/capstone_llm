#!/usr/bin/env bash
# Full Phase-1 pipeline on the real data: prepare once, then train/generate/evaluate
# all four architectures.  Each step can also be run on its own (see README).
#   bash run_all.sh                 # everything
#   ARCHS="a1_gpt a4_efficient" bash run_all.sh   # a subset
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
ARCHS=${ARCHS:-"a1_gpt a2_encdec a3_prefixlm a4_efficient"}

[ -f data/corpus/train.jsonl ]       || $PY scripts/01_prepare_data.py --config configs/base.yaml
[ -f data/tokenizer/legal_bpe.json ] || $PY scripts/02_train_tokenizer.py --config configs/base.yaml --compare-gpt2
[ -f data/tokens/train.bin ]         || $PY scripts/03_tokenize.py --config configs/base.yaml

for ARCH in $ARCHS; do
  RUN=$($PY -c "import sys,yaml;sys.path.insert(0,'.');from legal_lm.config import load_config;print(load_config(['configs/$ARCH.yaml'])['run_name'])")
  $PY scripts/04_train.py --config configs/$ARCH.yaml
  $PY scripts/05_generate.py --run runs/$RUN --split dev
  $PY scripts/06_evaluate.py --pred runs/$RUN/predictions/dev_t0.8_k50_p0.95_rp1.1/prediction.json
  $PY scripts/05_generate.py --run runs/$RUN --split test
done
$PY scripts/07_compare.py --runs-dir runs
$PY scripts/08_plot_resources.py --runs-dir runs
