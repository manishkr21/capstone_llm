"""Second-stage stacking: a small MLP over the statute logits of several base runs learns to combine them
and to exploit statute co-occurrence (e.g. statute j is likely when i and k are predicted).

    python stack.py --run runs/V2_e5_ep5 runs/E1_a4_lorma ... --name STACK1
    python stack.py --all --name STACK_all

The base runs never trained on the validation split, so their validation logits are out-of-sample.
The meta-model is trained on the validation split with K-fold cross-validation: every validation document
gets an out-of-fold (OOF) probability from a meta-model that did not see it. Per-statute thresholds are then
fitted on the OOF probabilities, and the honest score is the 2-fold cross-fitted macro-F1 of those OOF
probabilities (make_submission.cross_fit). For the test set the K fold meta-models are averaged.
"""
import argparse
import importlib.util
import json
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lsi import P5_ROOT
from lsi.data import N_LABELS, TokenDocs, load_splits

_s = importlib.util.spec_from_file_location("p5_sub", P5_ROOT / "make_submission.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


class Meta(nn.Module):
    def __init__(self, n_in, hidden, drop, base_cols):
        super().__init__()
        self.base_cols = base_cols  # (R, 100) column indices: run r's logit for statute j
        self.net = nn.Sequential(nn.Dropout(drop), nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, N_LABELS))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.mix = nn.Parameter(torch.full((len(base_cols),), 1.0 / len(base_cols)))

    def forward(self, x):
        base = (x[:, self.base_cols] * self.mix[None, :, None]).sum(1)   # weighted average of base logits
        return base + self.net(x)                                        # + learned correction


def train_meta(Xtr, Ytr, Xva, base_cols, args, device, seed):
    torch.manual_seed(seed)
    m = Meta(Xtr.shape[1], args.hidden, args.dropout, base_cols).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    Xt, Yt = torch.tensor(Xtr, device=device), torch.tensor(Ytr, dtype=torch.float32, device=device)
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            loss = F.binary_cross_entropy_with_logits(m(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(Xva, device=device))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every finished run in runs/")
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--len-feats", action="store_true", help="add log document length and chunk count as inputs")
    args = ap.parse_args()
    runs = args.run
    if args.all:
        runs = sorted("runs/" + r.name for r in (P5_ROOT / "runs").iterdir()
                      if (r / "val_probs.npy").exists() and (r / "test_probs.npy").exists())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val, test = TokenDocs("val"), TokenDocs("test")
    Y = val.labels
    Xv = np.concatenate([logit(np.load(P5_ROOT / r / "val_probs.npy")) for r in runs], 1).astype(np.float32)
    Xt = np.concatenate([logit(np.load(P5_ROOT / r / "test_probs.npy")) for r in runs], 1).astype(np.float32)
    if args.len_feats:
        def lf(d):
            L = np.diff(d.offs).astype(np.float32)
            return np.stack([np.log1p(L) / 10, np.minimum(np.ceil(L / 512), 8) / 8], 1)
        Xv = np.concatenate([Xv, lf(val)], 1)
        Xt = np.concatenate([Xt, lf(test)], 1)
    base_cols = torch.arange(len(runs) * N_LABELS, device=device).view(len(runs), N_LABELS)

    rng = np.random.default_rng(0)
    fold = rng.permutation(len(Y)) % args.folds
    oof = np.zeros(Y.shape, dtype=np.float32)
    test_p = np.zeros((len(Xt), N_LABELS), dtype=np.float32)
    for k in range(args.folds):
        tr, va = fold != k, fold == k
        for s in range(args.seeds):
            Xboth = np.concatenate([Xv[va], Xt])
            p = train_meta(Xv[tr], Y[tr], Xboth, base_cols, args, device, seed=100 * k + s)
            oof[va] += p[: va.sum()] / args.seeds
            test_p += p[va.sum():] / (args.seeds * args.folds)

    mean_p = np.mean([np.load(P5_ROOT / r / "val_probs.npy") for r in runs], 0)
    keep = ("macro_f1", "macro_p", "macro_r", "micro_f1", "subset_acc")
    plain = _m.cross_fit(Y, mean_p, "per_label", 5)
    stacked = _m.cross_fit(Y, oof, "per_label", 5)
    thr = _m.fit(Y, oof, "per_label", 5)
    D = _m.apply(test_p, thr)
    report = {"name": args.name, "runs": runs, "meta": {k: getattr(args, k) for k in ("folds", "hidden", "dropout", "lr", "wd", "epochs", "seeds")},
              "mean_ensemble_crossfit": {k: round(plain[k], 4) for k in keep},
              "stacked_crossfit": {k: round(stacked[k], 4) for k in keep},
              "test_labels_per_doc": round(float(D.sum(1).mean()), 2)}
    print(json.dumps(report))
    out = P5_ROOT / "runs" / "submissions_meta"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**report, "thresholds": thr.round(4).tolist()}, open(out / f"{args.name}.json", "w"), indent=1)
    np.save(out / f"{args.name}_val_oof.npy", oof)
    np.save(out / f"{args.name}_test.npy", test_p)
    if args.no_zip:
        return
    preds = [{"id": str(i), "pred": [int(j) for j in np.flatnonzero(row)]} for i, row in zip(test.doc_ids, D)]
    assert [p["id"] for p in preds] == [str(r["id"]) for r in load_splits()["test"]]
    assert all(p["pred"] and len(set(p["pred"])) == len(p["pred"]) for p in preds)
    pred_path = out / f"{args.name}_prediction.json"
    json.dump(preds, open(pred_path, "w"))
    zp = P5_ROOT / "submissions" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "prediction.json")
    print(f"-> {zp}")


if __name__ == "__main__":
    main()
