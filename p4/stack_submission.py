"""Learned stacking: a logistic regression over the logits of every trained run -> Codabench zip.

    python stack_submission.py                       # all runs with val/test probabilities
    python stack_submission.py --l2 0.01 --name STACK_all

The stacker is fitted on the clean validation split only (dev.json minus the test documents), so it
learns how much to trust each model; the decision threshold is tuned on the same split.  Honest
estimate (nested: stacker + threshold fitted on one half of validation, scored on the other, 5 seeds x
2 halves): printed before the zip is written.
"""
import argparse
import glob
import json
import time
import zipfile

import numpy as np
import torch

from cj import P4_ROOT
from cj.data import TokenDocs, load_splits
import importlib.util

_spec = importlib.util.spec_from_file_location("p4_train", P4_ROOT / "train.py")  # not p2/train.py
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)


def logits(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_lr(X, y, l2):
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b)


def predict(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="*", default=None, help="default: every run with val_probs + test_probs")
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--name", default="STACK_all")
    args = ap.parse_args()
    runs = args.run or sorted(p.split("/")[-2] for p in glob.glob(str(P4_ROOT / "runs" / "*" / "test_probs.npy")))
    val, test = TokenDocs("val"), TokenDocs("test")
    y = val.labels
    Xv = np.stack([logits(np.load(P4_ROOT / "runs" / r / "val_probs.npy")) for r in runs], 1)
    Xt = np.stack([logits(np.load(P4_ROOT / "runs" / r / "test_probs.npy")) for r in runs], 1)

    scores = []
    for seed in range(5):
        idx = np.random.default_rng(seed).permutation(len(y))
        A, B = idx[:len(y) // 2], idx[len(y) // 2:]
        for a, b in ((A, B), (B, A)):
            w, c = fit_lr(Xv[a], y[a], args.l2)
            t, _ = _train.best_threshold(y[a], predict(Xv[a], w, c))
            scores.append(_train.macro_f1(y[b], (predict(Xv[b], w, c) >= t).astype(int))["macro_f1"])
    print(f"{len(runs)} runs; nested split-half macro-F1 {np.mean(scores):.4f} +- {np.std(scores):.4f}")

    w, c = fit_lr(Xv, y, args.l2)
    thr, f1 = _train.best_threshold(y, predict(Xv, w, c))
    pt = predict(Xt, w, c)
    preds = [{"id": str(i), "pred": int(p >= thr)} for i, p in zip(test.doc_ids, pt)]
    assert [p["id"] for p in preds] == [str(r["id"]) for r in load_splits()["test"]]
    out = P4_ROOT / "runs" / f"{args.name}_prediction.json"
    json.dump(preds, open(out, "w"))
    json.dump({"runs": runs, "weights": dict(zip(runs, map(float, w))), "bias": c, "threshold": thr,
               "l2": args.l2, "nested_f1": [float(np.mean(scores)), float(np.std(scores))]},
              open(P4_ROOT / "runs" / f"{args.name}_stacker.json", "w"), indent=1)
    zp = P4_ROOT / "submissions" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, "prediction.json")
    top = sorted(zip(map(float, w), runs), reverse=True)[:5]
    print(f"in-sample val F1 {f1:.4f} @thr {thr:.2f}; test predicted 1: {np.mean(pt >= thr):.1%}; "
          f"largest weights {[(r, round(v, 2)) for v, r in top]} -> {zp}")


if __name__ == "__main__":
    main()
