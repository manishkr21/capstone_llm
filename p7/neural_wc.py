"""Window-level pool-normalised neural scores of trained bi-encoder runs: runs/<run>/{dev,test}_wc.npz

Per citation window: the max over each case's chunks of the cosine (as in the model), z-scored over cases
(row) and over all windows of the split (column); per query the max over its windows of row + gamma * column
(the same normalisation as pcr.lexical.per_citation_wincol).

    python neural_wc.py E2_a1_full E1_a3_lorma ...
"""
import sys

import numpy as np
import torch

from pcr import P7_ROOT
from pcr import tok as T
from pcr.lexical import zscore
import importlib.util
_s = importlib.util.spec_from_file_location("p7_train", P7_ROOT / "train.py")  # (p2/ also has a train.py on sys.path)
_t = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_t)

GAMMA = 2.0


@torch.no_grad()
def window_scores(model, split, toks, tc, device):
    lx = _t.lex(split)
    qids, cids = list(lx["qids"]), list(lx["cids"])
    chs, c_own = [], []
    for k, c in enumerate(cids):
        cs = T.chunks(toks["docs"][c], tc["chunk_len"], tc["eval_chunks"], "spread")
        chs += cs
        c_own += [k] * len(cs)
    wins, w_own = [], []
    for k, q in enumerate(qids):
        wins += toks["wins"][split][q]
        w_own += [k] * len(toks["wins"][split][q])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        C = model.embed(chs, device, tc["max_tokens"])
        W = model.embed(wins, device, tc["max_tokens"])
    c_own = torch.tensor(c_own, device=device)
    SW = torch.empty(len(wins), len(cids))
    for s in range(0, len(wins), 512):
        sim = W[s:s + 512] @ C.T
        SW[s:s + 512] = sim.new_full((sim.size(0), len(cids)), -1e4).scatter_reduce(
            1, c_own.unsqueeze(0).expand_as(sim), sim, "amax").float().cpu()
    SW, w_own = SW.numpy().astype(np.float64), np.array(w_own)
    col = {c: j for j, c in enumerate(cids)}
    lo = SW.min()
    for k, q in enumerate(qids):
        if q in col:
            SW[w_own == k, col[q]] = lo
    R = zscore(SW)
    M = R + GAMMA * zscore(R.T).T
    return np.stack([M[w_own == k].max(0) for k in range(len(qids))]).astype(np.float32), qids, cids


def main():
    device = torch.device("cuda")
    toks = T.get()
    for run in sys.argv[1:]:
        model, cfg = _t.load_run(run, device)
        model.eval()
        for split in ("dev", "test"):
            S, qids, cids = window_scores(model, split, toks, cfg["train"], device)
            np.savez_compressed(P7_ROOT / "runs" / run / f"{split}_wc.npz", S=S, qids=qids, cids=cids)
        print(run, "done", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
