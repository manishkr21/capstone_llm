"""Lexical component 2: a multi-label MLP over TF-IDF features (all 100 statutes learned jointly).

    python lexical_mlp.py --name M1_lexmlp --hidden 1024 --seed 0

TF-IDF (word 1-2 grams, fitted on the training split only, cached in cache/tfidf_*.npz) -> sparse linear layer
-> ReLU -> dropout -> 100 logits, BCE with sqrt positive weights, AdamW. Trained on the training split only;
the epoch is chosen on the same fixed 2,000-document validation subset the neural runs use for selection.
Writes runs/<name>/val_probs.npy and test_probs.npy for stack.py.
"""
import argparse
import json
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from lsi import P5_ROOT
from lsi.data import CACHE, load_splits, multi_hot

import importlib.util
_s = importlib.util.spec_from_file_location("p5_sub", P5_ROOT / "make_submission.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def tfidf_matrices(max_features):
    paths = {k: CACHE / f"tfidf{max_features}_{k}.npz" for k in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()):
        return {k: sp.load_npz(p).tocsr() for k, p in paths.items()}
    from sklearn.feature_extraction.text import TfidfVectorizer
    s = load_splits()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features, sublinear_tf=True, dtype=np.float32)
    out = {"train": vec.fit_transform([d["text"] for d in s["train"]])}
    out["val"] = vec.transform([d["text"] for d in s["val"]])
    out["test"] = vec.transform([d["text"] for d in s["test"]])
    for k, X in out.items():
        X.sort_indices()
        sp.save_npz(paths[k], X)
    return out


def to_torch(X, device):
    X = X.tocoo()
    idx = torch.from_numpy(np.vstack([X.row, X.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(X.data), X.shape, device=device)


@torch.no_grad()
def predict(W1, b1, W2, b2, X, device, bs=2048):
    out = []
    for i in range(0, X.shape[0], bs):
        h = F.relu(torch.sparse.mm(to_torch(X[i:i + bs], device), W1) + b1)
        out.append(torch.sigmoid(h @ W2 + b2).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-features", type=int, default=400000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    M = tfidf_matrices(args.max_features)
    s = load_splits()
    Ytr = torch.tensor(multi_hot([d["labels"] for d in s["train"]]), dtype=torch.float32)
    Yva = multi_hot([d["labels"] for d in s["val"]])
    sel = sorted(random.Random(1).sample(range(len(Yva)), 2000))   # same selection subset as train.py
    d_in = M["train"].shape[1]
    W1 = (torch.randn(d_in, args.hidden, device=device) * 0.01).requires_grad_()
    b1 = torch.zeros(args.hidden, device=device, requires_grad=True)
    W2 = (torch.randn(args.hidden, 100, device=device) * 0.02).requires_grad_()
    b2 = torch.zeros(100, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W1, b1, W2, b2], lr=args.lr, weight_decay=args.wd)
    freq = Ytr.mean(0).clamp(min=1e-4)
    pw = ((1 - freq) / freq).sqrt().clamp(max=30).to(device)
    n = M["train"].shape[0]
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            x = to_torch(M["train"][b], device)
            h = F.dropout(F.relu(torch.sparse.mm(x, W1) + b1), args.dropout, training=True)
            loss = F.binary_cross_entropy_with_logits(h @ W2 + b2, Ytr[b].to(device), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = predict(W1, b1, W2, b2, M["val"][sel], device)
        thr, f1 = _m.best_global_threshold(Yva[sel], pv)
        print(f"  epoch {ep + 1}  loss {loss.item():.4f}  sel macro-F1 {f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            best_state = [t.detach().clone() for t in (W1, b1, W2, b2)]
    pv = predict(*best_state, M["val"], device)
    pt = predict(*best_state, M["test"], device)
    out = P5_ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_probs.npy", pv.astype(np.float32)); np.save(out / "test_probs.npy", pt.astype(np.float32))
    sc = _m.cross_fit(Yva, pv, "per_label", 5)
    summary = {"run": args.name, "kind": "lexical MLP over TF-IDF (auxiliary stacking input)", "args": vars(args),
               "val_crossfit": {k: round(v, 4) for k, v in sc.items() if k != "per_label_f1"}, "sel_best": round(best, 4),
               "train_min": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
