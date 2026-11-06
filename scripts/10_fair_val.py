"""Build a validation set that is leak-free for BOTH corpora: documents of the first
val split (data/corpus/val.jsonl) whose word 13-grams overlap neither the first nor
the cleaned train split by more than 50%.  Old (R0) and new (R1/R2) checkpoints can
then be compared on text neither of them was trained on.

    python scripts/10_fair_val.py --out data_clean/fair_val
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401
from legal_lm.data.clean import _bounded_imap, _ngrams_work
from legal_lm.data.dataset import tokenize_jsonl
from legal_lm.data.tokenizer import LegalTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", default="data/corpus/val.jsonl")
    ap.add_argument("--train", nargs="+", default=["data/corpus/train.jsonl", "data_clean/corpus/train.jsonl"])
    ap.add_argument("--out", default="data_clean/fair_val")
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--mod", type=int, default=8)
    ap.add_argument("--max-overlap", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        grams = []
        for path in args.train:
            def texts(p=path):
                with open(p) as f:
                    for ln in f:
                        yield (json.loads(ln)["text"], args.k, args.mod)
            for g in tqdm(_bounded_imap(pool, _ngrams_work, texts(), inflight=4096, chunksize=64),
                          desc=Path(path).parent.parent.name, unit="doc"):
                grams.append(g)
        ref = np.unique(np.concatenate(grams))
        del grams
        docs = [json.loads(ln) for ln in open(args.val)]
        res = pool.map(_ngrams_work, [(d["text"], args.k, args.mod) for d in docs], chunksize=16)
    kept = 0
    with open(out / "val.jsonl", "w") as f:
        for d, g in zip(docs, res):
            ov = float((ref[np.minimum(np.searchsorted(ref, g), len(ref) - 1)] == g).mean()) if len(g) else 0.0
            if ov <= args.max_overlap:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1
    tok = LegalTokenizer.from_config({"type": "bpe", "path": "data/tokenizer/legal_bpe.json"})
    info = tokenize_jsonl(out / "val.jsonl", out / "val.bin", tok)
    print(f"kept {kept}/{len(docs)} old val docs; {info['n_tokens']:,} tokens {info['per_source']}")
    json.dump({"kept": kept, "total": len(docs), **info}, open(out / "meta.json", "w"), indent=2)
