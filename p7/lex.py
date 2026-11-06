"""Cache lexical score matrices for every split: cache/lex_{split}.npz with qids, cids, ctx (BM25 over the
concatenated citation contexts), percit (per-citation BM25, max), and the judgment years."""
import numpy as np

from pcr import P7_ROOT
from pcr.data import load, year
from pcr.lexical import bm25, per_citation, windows

for split in ["dev", "train", "test"]:
    q, pool = load(split)
    qids, cids = list(q), list(pool)
    docs = [pool[c] for c in cids]
    qt = [q[i]["text"] for i in qids]
    ctx = bm25([" ".join(windows(t, 80)) for t in qt], docs, ngram=(1, 3))
    out = {"qids": np.array(qids), "cids": np.array(cids), "ctx": ctx.astype(np.float32),
           "qyear": np.array([year(t) or 9999 for t in qt]), "cyear": np.array([year(d) or 0 for d in docs])}
    for w in (40, 60, 100):
        for agg in ("max", "lse"):
            out[f"pc{w}_{agg}"] = per_citation(qt, docs, w, agg, ngram=(1, 3))
    np.savez_compressed(P7_ROOT / "cache" / f"lex_{split}.npz", **out)
    print(split, "done", flush=True)
