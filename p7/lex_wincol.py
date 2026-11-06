"""Add window-level pool-normalised per-citation BM25 to cache/lex_{split}.npz: wc{w}_g{gamma}."""
import numpy as np

from pcr import P7_ROOT
from pcr.data import load
from pcr.lexical import per_citation_wincol

for split in ["dev", "train", "test"]:
    q, pool = load(split)
    p = P7_ROOT / "cache" / f"lex_{split}.npz"
    lx = dict(np.load(p))
    qids, cids = list(lx["qids"]), list(lx["cids"])
    for w in (60, 100):
        for g, M in per_citation_wincol(qids, [q[i]["text"] for i in qids], [pool[c] for c in cids], cids, w,
                                        ngram=(1, 3)).items():
            lx[f"wc{w}_g{int(g)}"] = M
    np.savez_compressed(p, **lx)
    print(split, "done", flush=True)
