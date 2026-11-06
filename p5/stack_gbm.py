"""Per-statute gradient-boosted stacker (LightGBM), an alternative to the shared MLP in stack.py.

    python stack_gbm.py --name GBM1

For statute j the features are: every base run's logit for j, the across-run mean logit of all 100
statutes (label co-occurrence), and the across-run std of the logit for j (model disagreement).
K-fold on the validation split gives out-of-fold probabilities (honest), as in stack.py; the K fold models
are averaged for the test set. Optionally blended with the MLP stacker's OOF / test probabilities.
"""
import argparse
import importlib.util
import json
import time
import zipfile

import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed

from lsi import P5_ROOT
from lsi.data import N_LABELS, TokenDocs, load_splits

_s = importlib.util.spec_from_file_location("p5_sub", P5_ROOT / "make_submission.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def features(L, j):
    """L: (R, n, 100) logits -> (n, R + 100 + 1) features for statute j."""
    return np.concatenate([L[:, :, j].T, L.mean(0), L[:, :, j].std(0)[:, None]], 1)


def fit_label(j, Lv, Lt, Y, fold, K, params):
    Xv, Xt = features(Lv, j), features(Lt, j)
    oof, tp = np.zeros(len(Y), np.float32), np.zeros(len(Xt), np.float32)
    for k in range(K):
        tr, va = fold != k, fold == k
        m = lgb.LGBMClassifier(**params)
        m.fit(Xv[tr], Y[tr, j])
        oof[va] = m.predict_proba(Xv[va])[:, 1]
        tp += m.predict_proba(Xt)[:, 1] / K
    return j, oof, tp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--exclude", nargs="*", default=["T4_tfidf_cnb"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--blend", default=None, help="MLP stacker name in runs/submissions_meta to average with")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    runs = sorted("runs/" + r.name for r in (P5_ROOT / "runs").iterdir()
                  if (r / "val_probs.npy").exists() and (r / "test_probs.npy").exists() and r.name not in args.exclude)
    val, test = TokenDocs("val"), TokenDocs("test")
    Y = val.labels
    Lv = np.stack([logit(np.load(P5_ROOT / r / "val_probs.npy")) for r in runs]).astype(np.float32)
    Lt = np.stack([logit(np.load(P5_ROOT / r / "test_probs.npy")) for r in runs]).astype(np.float32)
    fold = np.random.default_rng(0).permutation(len(Y)) % args.folds
    params = dict(n_estimators=args.n_estimators, learning_rate=args.lr, num_leaves=args.leaves, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=1, verbose=-1)
    res = Parallel(n_jobs=args.jobs)(delayed(fit_label)(j, Lv, Lt, Y, fold, args.folds, params) for j in range(N_LABELS))
    oof, tp = np.zeros(Y.shape, np.float32), np.zeros((Lt.shape[1], N_LABELS), np.float32)
    for j, o, t in res:
        oof[:, j], tp[:, j] = o, t
    keep = ("macro_f1", "macro_p", "macro_r", "micro_f1", "subset_acc")
    report = {"name": args.name, "n_runs": len(runs), "gbm_crossfit": {k: round(v, 4) for k, v in _m.cross_fit(Y, oof, "per_label", 5).items() if k in keep}}
    if args.blend:
        meta = P5_ROOT / "runs" / "submissions_meta"
        mo, mt = np.load(meta / f"{args.blend}_val_oof.npy"), np.load(meta / f"{args.blend}_test.npy")
        oof, tp = (oof + mo) / 2, (tp + mt) / 2
        report["blend_with"] = args.blend
        report["blend_crossfit"] = {k: round(v, 4) for k, v in _m.cross_fit(Y, oof, "per_label", 5).items() if k in keep}
    report["minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(report), flush=True)
    meta = P5_ROOT / "runs" / "submissions_meta"
    np.save(meta / f"{args.name}_val_oof.npy", oof); np.save(meta / f"{args.name}_test.npy", tp)
    if args.no_zip:
        return
    thr = _m.fit(Y, oof, "per_label", 5)
    D = _m.apply(tp, thr)
    preds = [{"id": str(i), "pred": [int(x) for x in np.flatnonzero(row)]} for i, row in zip(test.doc_ids, D)]
    assert [p["id"] for p in preds] == [str(r["id"]) for r in load_splits()["test"]]
    assert all(p["pred"] and len(set(p["pred"])) == len(p["pred"]) for p in preds)
    json.dump({**report, "runs": runs, "thresholds": thr.round(4).tolist()}, open(meta / f"{args.name}.json", "w"), indent=1)
    pp = meta / f"{args.name}_prediction.json"
    json.dump(preds, open(pp, "w"))
    zp = P5_ROOT / "submissions" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pp, "prediction.json")
    print(f"-> {zp}")


if __name__ == "__main__":
    main()
