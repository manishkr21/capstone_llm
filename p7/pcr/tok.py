"""Token cache shared by all four Phase-1 models (same 32k BPE): cache/tok.pkl

  docs[id]          -> int32 tokens of the whole case (every id of every pool; queries are in their pools)
  wins[split][qid]  -> list of int32 token arrays, one per citation window (+-w words around <CITATION>)
"""
import pickle

import numpy as np

from . import P7_ROOT
from .data import load
from .lexical import windows

PATH = P7_ROOT / "cache" / "tok.pkl"


def build(tok, w=60, win_len=256):
    docs, wins = {}, {}
    for split in ("train", "dev", "test"):
        q, pool = load(split)
        todo = [k for k in pool if k not in docs]
        for k, e in zip(todo, tok.tk.encode_batch([pool[k] for k in todo])):
            docs[k] = np.array(e.ids, dtype=np.int32)
        wins[split] = {}
        for k, v in q.items():
            ws = windows(v["text"], w)
            wins[split][k] = [np.array(e.ids[:win_len], dtype=np.int32) for e in tok.tk.encode_batch(ws)]
        print(split, len(docs), flush=True)
    pickle.dump({"docs": docs, "wins": wins, "w": w}, open(PATH, "wb"))


def get():
    return pickle.load(open(PATH, "rb"))


def chunks(ids, L, cap, mode="spread", rng=None):
    """Split a document into L-token chunks; keep <= cap of them (first, evenly spread, or random)."""
    cs = [ids[i:i + L] for i in range(0, max(1, len(ids)), L)]
    if len(cs) <= cap:
        return cs
    if mode == "first":
        return cs[:cap]
    if mode == "random":
        keep = sorted({0} | set(rng.choice(np.arange(1, len(cs)), cap - 1, replace=False).tolist()))
        return [cs[i] for i in keep]
    return [cs[int(round(i))] for i in np.linspace(0, len(cs) - 1, cap)]
