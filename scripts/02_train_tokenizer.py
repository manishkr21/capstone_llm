"""Step 2: learn a byte-level BPE vocabulary on the training split and report its
efficiency (tokens per word) on held-out text.  Compare against GPT-2 with --compare-gpt2.

    python scripts/02_train_tokenizer.py --config configs/base.yaml
    python scripts/02_train_tokenizer.py --set tokenizer.vocab_size=16000 tokenizer.path=data/tokenizer/bpe16k.json
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from legal_lm.config import add_config_args, config_from_args
from legal_lm.data.tokenizer import LegalTokenizer, iter_corpus_texts, tokenizer_efficiency, train_bpe
from legal_lm.monitor import monitor_from_config
from legal_lm.utils import save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--compare-gpt2", action="store_true", help="also measure the GPT-2 tokenizer")
    args = ap.parse_args()
    cfg = config_from_args(args)
    tc, root = cfg["tokenizer"], Path(cfg["paths"]["data_dir"])

    log = root / "logs" / "resources_02_tokenizer.csv"
    with monitor_from_config(cfg, log, phase="tokenizer"):
        if tc["type"] == "bpe":
            texts = iter_corpus_texts(root / "corpus" / "train.jsonl", tc["train_docs"])
            tok = train_bpe(texts, tc["vocab_size"], tc["min_frequency"], tc["path"])
            print(f"saved {tc['path']}")
        else:
            tok = LegalTokenizer.from_config(tc)

        val = [json.loads(line)["text"] for line, _ in zip(open(root / "corpus" / "val.jsonl"), range(500))]
        report = {"legal_bpe" if tc["type"] == "bpe" else tc["type"]: tokenizer_efficiency(tok, val)}
        if args.compare_gpt2:
            report["gpt2"] = tokenizer_efficiency(LegalTokenizer.from_config({"type": "gpt2"}), val)
        print(json.dumps(report, indent=2))
        save_json(report, Path(tc["path"]).with_suffix(".efficiency.json"))
