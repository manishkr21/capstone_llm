"""Step 1: stream the legal corpus, mix the four sources, split train/val by document,
build the local dev set (512-word prompts + true 128-word continuations) and fetch
the official test prompts.

    python scripts/01_prepare_data.py --config configs/base.yaml
"""
import argparse

import _bootstrap  # noqa: F401
from legal_lm.config import add_config_args, config_from_args
from legal_lm.data.corpus import prepare_corpus
from legal_lm.monitor import monitor_from_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    cfg = config_from_args(ap.parse_args())
    with monitor_from_config(cfg, f"{cfg['paths']['data_dir']}/logs/resources_01_prepare.csv", phase="prepare"):
        prepare_corpus(cfg)
