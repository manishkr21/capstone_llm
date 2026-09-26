"""Step 3: encode train/val documents into flat uint16 token files for fast training.

    python scripts/03_tokenize.py --config configs/base.yaml
"""
import argparse

import _bootstrap  # noqa: F401
from legal_lm.config import add_config_args, config_from_args
from legal_lm.data.dataset import tokenize_splits
from legal_lm.data.tokenizer import LegalTokenizer
from legal_lm.monitor import monitor_from_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    cfg = config_from_args(ap.parse_args())
    with monitor_from_config(cfg, f"{cfg['paths']['data_dir']}/logs/resources_03_tokenize.csv", phase="tokenize"):
        tokenize_splits(cfg, LegalTokenizer.from_config(cfg["tokenizer"]))
