"""Step 4: train one architecture from scratch.

    python scripts/04_train.py --config configs/a1_gpt.yaml
    python scripts/04_train.py --config configs/a4_efficient.yaml --set train.max_steps=5000
"""
import argparse

import _bootstrap  # noqa: F401
from legal_lm.config import add_config_args, config_from_args
from legal_lm.trainer import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default="configs/a1_gpt.yaml")
    train(config_from_args(ap.parse_args()))
