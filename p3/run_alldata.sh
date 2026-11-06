#!/usr/bin/env bash
# Final models: retrain each listed config on all 90 documents for the median best epoch of its
# cross-validation (train.py --all-data) -> runs/<run>_all/fold_all/test_out.pt.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-../cenv/bin/python}
P=${P:-9}
mkdir -p logs
printf "%s\n" "$@" | xargs -P $P -I{} bash -c '$0 train.py --config {} --all-data > logs/all_$(basename {} .yaml).log 2>&1 && echo "ok   {}" || echo "FAIL {}"' $PY
