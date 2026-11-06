"""Candidate lists and evidence pairs for the cross-encoder: cache/cross_{split}.pkl

For every (query, case) in a query's list, the ``m`` (citation window, case chunk) pairs with the highest cosine
under a trained bi-encoder run (default E2_a1_full) -- the passages the cross-encoder reads.
  train: top-40 by the lexical first stage (per-citation BM25, row + column z-score) + every gold case
  dev / test: top-30 of a stack run's scores (runs/stack/<stack>_{split}.npz)

    python cross_prep.py --run E2_a1_full --stack STK4_rc_blend4
"""
import argparse
import importlib.util
import pickle

import numpy as np
import torch

from pcr import P7_ROOT
from pcr import tok as T
from pcr.data import load
from pcr.ltr import row_col

_s = importlib.util.spec_from_file_location("p7_train", P7_ROOT / "train.py")  # (p2/ also has a train.py on sys.path)
_t = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_t)


def lists(split, stack, k_train, k_eval):
    lx = _t.lex(split)
    qids, cids = list(lx["qids"]), list(lx["cids"])
    col = {c: j for j, c in enumerate(cids)}
    own = [(i, col[q]) for i, q in enumerate(qids) if q in col]
    out = {}
    if split == "train":
        q, _ = load("train")
        F = sum(row_col(lx["pc60_max"], own))
        for i, qi in enumerate(qids):
            top = [cids[j] for j in np.argsort(-F[i])[:k_train + 1] if cids[j] != qi][:k_train]
            out[qi] = list(dict.fromkeys(top + sorted(q[qi]["rel"])))
        return out
    d = np.load(P7_ROOT / "runs" / "stack" / f"{stack}_{split}.npz")
    by = {}
    for p, (i, j) in zip(d["pred"], d["pairs"]):
        by.setdefault(qids[i], []).append((p, cids[j]))
    return {qi: [c for _, c in sorted(v, key=lambda x: -x[0])[:k_eval]] for qi, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="E2_a1_full")
    ap.add_argument("--stack", default="STK4_rc_blend4")
    ap.add_argument("--k-train", type=int, default=40)
    ap.add_argument("--k-eval", type=int, default=30)
    ap.add_argument("--m", type=int, default=2)
    a = ap.parse_args()
    device = torch.device("cuda")
    model, cfg = _t.load_run(a.run, device)
    model.eval()
    toks, tc = T.get(), cfg["train"]
    for split in ("dev", "test", "train"):
        L = lists(split, a.stack, a.k_train, a.k_eval)
        cases = sorted({c for v in L.values() for c in v})
        chs, c_own = [], []
        for k, c in enumerate(cases):
            cs = T.chunks(toks["docs"][c], tc["chunk_len"], tc["eval_chunks"], "spread")
            chs += cs
            c_own += [k] * len(cs)
        wins, w_own = [], []
        qids = list(L)
        for k, q in enumerate(qids):
            wins += toks["wins"][split][q]
            w_own += [k] * len(toks["wins"][split][q])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            C = model.embed(chs, device, tc["max_tokens"])
            W = model.embed(wins, device, tc["max_tokens"])
        c_own, w_own = np.array(c_own), np.array(w_own)
        c_start = {c: int(np.flatnonzero(c_own == k)[0]) for k, c in enumerate(cases)}
        c_n = {c: int((c_own == k).sum()) for k, c in enumerate(cases)}
        recs = {}
        for k, q in enumerate(qids):
            wsel = np.flatnonzero(w_own == k)
            Wq = W[torch.as_tensor(wsel, device=device)]
            recs[q] = []
            for c in L[q]:
                s0, n = c_start[c], c_n[c]
                sim = (Wq @ C[s0:s0 + n].T).float().cpu().numpy()        # (windows, chunks)
                flat = np.argsort(-sim, axis=None)[:a.m]
                recs[q].append((c, [(int(f // n), int(f % n), float(sim.flat[f])) for f in flat]))
        pickle.dump(recs, open(P7_ROOT / "cache" / f"cross_{split}.pkl", "wb"))
        if split != "test":
            q, _ = load(split)
            hit = sum(len(set(L[x]) & q[x]["rel"]) for x in L)
            print(split, "lists", len(L), "cases", len(cases), "gold covered", round(hit / sum(len(q[x]["rel"]) for x in L), 3), flush=True)
        else:
            print(split, "lists", len(L), flush=True)


if __name__ == "__main__":
    main()
