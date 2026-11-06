"""Document models on cached chunk features (all chunks of a judgment, several encoders fused).

    python doc_train.py --feats V1_e6_16ch --layer gru --name D_v1_gru
    python doc_train.py --feats V1_e6_16ch V2_e7_16ch V3_a1_partial --layer lstm --name D_fuse3_lstm

Features: cache/feats/<run>_<split>.npz from embed_chunks.py (the last <=48 chunks of every document,
mean-pooled by the fine-tuned encoder of <run>).  With several runs the chunk vectors are concatenated
(the chunking is identical, so chunk j of a document is the same text in every encoder).

Document layer (--layer): mean | attn (attention pooling) | gru | lstm (bidirectional, then attention
pooling) | transformer (2 layers, 4 heads, then attention pooling).  Chunk position is embedded counted
from the end of the document.  Each seed is trained on the training split and selected on the clean
validation split; the class-1 probabilities of the seeds are averaged and saved like any other run
(runs/<name>/val_probs.npy, test_probs.npy, summary.json), so make_submission.py and
stack_submission.py can use it.
"""
import argparse
import json
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cj import P4_ROOT
from cj.data import TokenDocs
import importlib.util

_spec = importlib.util.spec_from_file_location("p4_train", P4_ROOT / "train.py")  # not p2/train.py
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)


def load_feats(runs, split):
    zs = [np.load(P4_ROOT / "cache" / "feats" / f"{r}_{split}.npz") for r in runs]
    offs = zs[0]["offs"]
    assert all((z["offs"] == offs).all() for z in zs), "chunking differs between encoders"
    return torch.from_numpy(np.concatenate([z["vecs"] for z in zs], 1)), offs


def batch(V, offs, idx, device):
    lens = [int(offs[i + 1] - offs[i]) for i in idx]
    K, d = max(lens), V.shape[1]
    x = torch.zeros(len(idx), K, d)
    valid = torch.zeros(len(idx), K, dtype=torch.bool)
    for b, (i, n) in enumerate(zip(idx, lens)):
        x[b, K - n:] = V[offs[i]:offs[i + 1]].float()   # right-aligned: the last chunk is always at K-1
        valid[b, K - n:] = True
    return x.to(device), valid.to(device)


class DocModel(nn.Module):
    def __init__(self, d_in, layer, h=256, dropout=0.2, max_chunks=64):
        super().__init__()
        self.layer = layer
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, h), nn.GELU())
        self.pos = nn.Embedding(max_chunks, h)
        if layer == "gru":
            self.rnn = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        elif layer == "lstm":
            self.rnn = nn.LSTM(h, h // 2, batch_first=True, bidirectional=True)
        elif layer == "transformer":
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, dropout, batch_first=True,
                                                                       norm_first=True), 2)
        self.score = nn.Linear(h, 1)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 2))

    def forward(self, x, valid):
        B, K, _ = x.shape
        from_end = torch.arange(K - 1, -1, -1, device=x.device).clamp(max=self.pos.num_embeddings - 1)
        z = self.inp(x) + self.pos(from_end)[None]
        if self.layer in ("gru", "lstm"):
            z = self.rnn(z)[0]
        elif self.layer == "transformer":
            z = self.tf(z, src_key_padding_mask=~valid)
        if self.layer == "mean":
            doc = (z * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        else:
            a = self.score(z).squeeze(-1).masked_fill(~valid, float("-inf")).softmax(-1)
            doc = (a.unsqueeze(-1) * z).sum(1)
        return self.out(doc)


@torch.no_grad()
def probs(model, V, offs, device, bs=256):
    model.eval()
    out = []
    for s in range(0, len(offs) - 1, bs):
        x, v = batch(V, offs, range(s, min(len(offs) - 1, s + bs)), device)
        out.append(model(x, v).float().softmax(-1)[:, 1].cpu())
    model.train()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--layer", default="gru", choices=["mean", "attn", "gru", "lstm", "transformer"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    device = torch.device("cuda")
    Vtr, otr = load_feats(args.feats, "train")
    Vva, ova = load_feats(args.feats, "val")
    Vte, ote = load_feats(args.feats, "test")
    ytr = torch.from_numpy(TokenDocs("train").labels).long()
    yva = TokenDocs("val").labels
    w = (len(ytr) / (2 * torch.bincount(ytr, minlength=2).float())).to(device)
    run_dir = P4_ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed_val, seed_test, seed_f1 = [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DocModel(Vtr.shape[1], args.layer, args.hidden, args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = args.epochs * ((len(ytr) + args.bs - 1) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
        best, bv, bt = -1, None, None
        rng = np.random.default_rng(seed)
        for ep in range(args.epochs):
            perm = rng.permutation(len(ytr))
            for s in range(0, len(perm), args.bs):
                idx = perm[s:s + args.bs]
                x, v = batch(Vtr, otr, idx, device)
                loss = F.cross_entropy(model(x, v), ytr[idx].to(device), weight=w)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            pv = probs(model, Vva, ova, device)
            thr, f1 = _train.best_threshold(yva, pv)
            if f1 > best:
                best, bv, bt = f1, pv, probs(model, Vte, ote, device)
        print(f"  {args.name} seed {seed}: best val macro-F1 {best:.4f}", flush=True)
        seed_val.append(bv)
        seed_test.append(bt)
        seed_f1.append(best)
    pv, pt = np.mean(seed_val, 0), np.mean(seed_test, 0)
    thr, f1 = _train.best_threshold(yva, pv)
    sc = _train.macro_f1(yva, (pv >= thr).astype(int))
    np.save(run_dir / "val_probs.npy", pv)
    np.save(run_dir / "test_probs.npy", pt)
    n_par = sum(p.numel() for p in model.parameters())
    summary = {"run": args.name, "feats": args.feats, "layer": args.layer, "val": sc,
               "val_f1_at_0.5": _train.macro_f1(yva, (pv >= 0.5).astype(int))["macro_f1"], "threshold": thr,
               "seed_f1_mean": statistics.mean(seed_f1), "seed_f1_std": statistics.pstdev(seed_f1),
               "total_params": n_par, "trainable_params": n_par, "trainable_pct": 100.0,
               "train_min": round((time.time() - t0) / 60, 1), "data": {"strategy": "cached-all", "n_chunks": 48}}
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=1)
    json.dump(vars(args), open(run_dir / "config.json", "w"), indent=1)
    print(json.dumps({"run": args.name, "seed_f1": f"{summary['seed_f1_mean']:.4f} +- {summary['seed_f1_std']:.4f}",
                      "seed_avg_f1": round(sc["macro_f1"], 4)}), flush=True)


if __name__ == "__main__":
    main()
