"""Learning-to-rank stack: lexical features + neural scores of fold-run pairs -> LightGBM lambdarank.

    python stack.py --name STK1 --neural E2_a1_full E1_a3_lorma
    python stack.py --name STK0 --neural            # lexical features only

Each name in --neural refers to the fold runs runs/F_<name>_f0, _f1 (train.fold=[k,2]): on train the
held-out fold's scores are used (no query is scored by a model trained on it), on dev / test the two fold
models' z-scored matrices are averaged. The ranker re-orders the top-``depth`` per-citation BM25 cases; it is
trained on train, scored on dev; the test model is refit on train + dev (``--refit``) with the same settings.
``--blend RUN ... --w W``: the ranker's z-scored output + W x the mean neural score of full runs (row z-score + column z-score, as for the lexical features)
(a neural model as a feature inside the ranker did not help: its out-of-fold train scores behave differently
from the dev / test ones).
"""
import argparse
import json

import lightgbm as lgb
import numpy as np

from pcr import P7_ROOT
from pcr import tok as T
from pcr.data import load, micro_f1
from pcr.lexical import zscore
from pcr.ltr import features, labels, rank_from, row_col
import importlib.util
_s = importlib.util.spec_from_file_location("p7_train", P7_ROOT / "train.py")  # (p2/ also has a train.py on sys.path)
_t = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_t)


def neural(name, split, lx):
    runs = [P7_ROOT / "runs" / f"F_{name}_f{k}" for k in (0, 1)]
    if split == "train":
        S = np.zeros(lx["ctx"].shape, dtype=np.float32)
        for r in runs:
            d = np.load(r / "train_scores.npz")
            assert list(d["qids"]) == list(lx["qids"]) and list(d["cids"]) == list(lx["cids"])
            h = d["heldout"]
            S[h] = zscore(d["S"])[h]
        return S
    mats = []
    for r in runs:
        d = np.load(r / f"{split}_scores.npz")
        assert list(d["qids"]) == list(lx["qids"]) and list(d["cids"]) == list(lx["cids"])
        mats.append(zscore(d["S"]))
    return np.mean(mats, 0)


def split_data(split, names, toks, depth, first="pc60_max"):
    q, _ = load(split)
    lx = _t.lex(split)
    ntok = np.array([len(toks["docs"][c]) for c in lx["cids"]])
    nwin = np.array([len(toks["wins"][split][x]) for x in lx["qids"]])
    X, g, pairs = features(lx, [neural(n, split, lx) for n in names], ntok, nwin, depth, first)
    y = labels(lx, q, pairs) if split != "test" else None
    return X, g, pairs, y, lx, q


def blend(pred, runs, split, pairs, w):
    if not runs:
        return pred
    lx = _t.lex(split)
    col = {c: j for j, c in enumerate(lx["cids"])}
    own = [(i, col[q]) for i, q in enumerate(lx["qids"]) if q in col]
    N = np.mean([sum(row_col(np.load(P7_ROOT / "runs" / r / f"{split}_scores.npz")["S"], own)) for r in runs], 0)
    return (pred - pred.mean()) / pred.std() + w * np.array([N[i, j] for i, j in pairs])


def ranker(a):
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=a.trees, learning_rate=a.lr, num_leaves=a.leaves,
                          min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          lambdarank_truncation_level=20, n_jobs=8, random_state=a.seed, verbose=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--neural", nargs="*", default=[])
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--first", default="pc60_max", help="lexical score whose row + column z-score picks the candidates")
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--blend", nargs="*", default=[])
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--refit", action="store_true", help="test model trained on train + dev")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    toks = T.get()
    tr = split_data("train", a.neural, toks, a.depth, a.first)
    dv = split_data("dev", a.neural, toks, a.depth, a.first)
    pred = np.zeros(len(dv[0]))
    for s in range(a.seeds):  # a few seeds of the (subsampled) ranker, averaged
        a.seed = s
        pred += ranker(a).fit(tr[0], tr[3], group=tr[1]).predict(dv[0])
    pred = blend(pred, a.blend, "dev", dv[2], a.w)
    m = micro_f1(rank_from(pred, dv[4], dv[2]), dv[5])
    rep = {"name": a.name, "neural": a.neural, "blend": a.blend, "w": a.w, "depth": a.depth, "first": a.first, "trees": a.trees, "lr": a.lr, "leaves": a.leaves,
           "refit": a.refit, "dev": {k: round(v, 4) for k, v in m.items()}}
    print(json.dumps(rep), flush=True)
    if a.no_zip:
        return
    te = split_data("test", a.neural, toks, a.depth, a.first)
    X = np.concatenate([tr[0], dv[0]]) if a.refit else tr[0]
    y = np.concatenate([tr[3], dv[3]]) if a.refit else tr[3]
    g = np.concatenate([tr[1], dv[1]]) if a.refit else tr[1]
    pt = np.zeros(len(te[0]))
    for s in range(a.seeds):
        a.seed = s
        pt += ranker(a).fit(X, y, group=g).predict(te[0])
    pt = blend(pt, a.blend, "test", te[2], a.w)
    ranked = rank_from(pt, te[4], te[2])
    assert len(ranked) == len(te[4]["qids"])
    rep["zip"] = str(_t.write_zip(a.name, ranked, set(te[4]["cids"])))
    out = P7_ROOT / "runs" / "stack"
    out.mkdir(exist_ok=True)
    json.dump(rep, open(out / f"{a.name}.json", "w"), indent=1)
    for split, p_, d_ in (("dev", pred, dv), ("test", pt, te)):  # per-pair scores, for the cross-encoder / blends
        np.savez_compressed(out / f"{a.name}_{split}.npz", pred=p_, pairs=np.array(d_[2]))
    print(rep["zip"], flush=True)


if __name__ == "__main__":
    main()
