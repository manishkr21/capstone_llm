"""Lexical component trained on train + validation, with out-of-fold validation predictions.

    python tfidf_cv.py --name T6_tfidf_cv

Validation document i is predicted by a TF-IDF + logistic-regression model trained on the training split plus
the other K-1 validation folds (never on i), so its validation probabilities stay out-of-sample for stack.py;
the test probabilities come from one model trained on train + all of validation (~20% more labelled data).
"""
import argparse
import json
import time

import numpy as np
from joblib import parallel_backend
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lsi import P5_ROOT
from lsi.data import load_splits, multi_hot

import importlib.util
_s = importlib.util.spec_from_file_location("p5_sub", P5_ROOT / "make_submission.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def fit_predict(X, Y, Xp, C, jobs):
    clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=300, class_weight="balanced", solver="liblinear"))
    with parallel_backend("threading", n_jobs=jobs):
        clf.fit(X, Y)
    return [clf.predict_proba(x).astype(np.float32) for x in Xp]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Ytr, Yva = multi_hot([d["labels"] for d in tr]), multi_hot([d["labels"] for d in va])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True, dtype=np.float32)
    vec.fit([d["text"] for d in tr])                         # vocabulary from the training split, as before
    Xtr, Xva, Xte = (vec.transform([d["text"] for d in s]).tocsr() for s in (tr, va, te))
    for X in (Xtr, Xva, Xte):
        X.sort_indices()
    fold = np.random.default_rng(42).permutation(len(va)) % args.folds
    pv = np.zeros(Yva.shape, np.float32)
    for k in range(args.folds):
        keep = fold != k
        X = vstack([Xtr, Xva[keep]]).tocsr(); X.sort_indices()
        pv[fold == k] = fit_predict(X, np.concatenate([Ytr, Yva[keep]]), [Xva[fold == k]], args.C, args.jobs)[0]
        print(f"  fold {k} done  {(time.time() - t0) / 60:.1f} min", flush=True)
    X = vstack([Xtr, Xva]).tocsr(); X.sort_indices()
    pt = fit_predict(X, np.concatenate([Ytr, Yva]), [Xte], args.C, args.jobs)[0]
    out = P5_ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_probs.npy", pv); np.save(out / "test_probs.npy", pt)
    sc = _m.cross_fit(Yva, pv, "per_label", 5)
    summary = {"run": args.name, "kind": "tfidf_lr on train + val folds (auxiliary stacking input)", "args": vars(args),
               "val_crossfit": {k: round(v, 4) for k, v in sc.items() if k != "per_label_f1"},
               "train_min": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
