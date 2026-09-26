"""Print parameter counts of the four architectures (no data needed).

    python scripts/00_model_summary.py
"""
import argparse

import _bootstrap  # noqa: F401
from legal_lm.config import load_config
from legal_lm.models import ModelConfig, build_model
from legal_lm.utils import human

CONFIGS = ["configs/a1_gpt.yaml", "configs/a2_encdec.yaml", "configs/a3_prefixlm.yaml", "configs/a4_efficient.yaml"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    print(f"{'run':<22}{'arch':<11}{'total':>10}{'non-emb':>10}{'in 50M-300M':>13}  description")
    for path in args.configs:
        cfg = load_config([path], args.set)
        mcfg = ModelConfig.from_dict(cfg["model"], vocab_size=cfg["tokenizer"]["vocab_size"],
                                     max_seq_len=cfg["train"]["block_size"])
        model = build_model(mcfg)
        c = model.param_counts()
        ok = "yes" if 50e6 <= c["total"] <= 300e6 else "NO"
        print(f"{cfg['run_name']:<22}{mcfg.arch:<11}{human(c['total']):>10}{human(c['non_embedding']):>10}"
              f"{ok:>13}  {model.description}")
