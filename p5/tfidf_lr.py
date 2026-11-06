"""Lexical component for the hybrid stack: TF-IDF features + one-vs-rest logistic regression per statute.

    python tfidf_lr.py --name T1_tfidf_word                 # word 1-2 grams
    python tfidf_lr.py --name T2_tfidf_char --analyzer char_wb --ngram 3 5

Trained on the training split only (competition data, no pretrained model); writes runs/<name>/val_probs.npy
and test_probs.npy in the same format as the neural runs, so stack.py / make_submission.py can use it.
It is an auxiliary input to the stacker next to the Phase-1 models, not a standalone submission.
"""
import argparse
import json
import time

import numpy as np
from joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lsi import P5_ROOT
from lsi.data import load_splits, multi_hot

import importlib.util
_s = importlib.util.spec_from_file_location("p5_sub", P5_ROOT / "make_submission.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--analyzer", default="word")
    ap.add_argument("--ngram", type=int, nargs=2, default=[1, 2])
    ap.add_argument("--max-features", type=int, default=400000)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--model", default="lr", choices=["lr", "svm", "cnb"],
                    help="lr: logistic regression; svm: linear SVM (decision values squashed by a sigmoid); "
                         "cnb: complement naive Bayes (per-statute binary)")
    args = ap.parse_args()
    t0 = time.time()
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Ytr, Yva = multi_hot([d["labels"] for d in tr]), multi_hot([d["labels"] for d in va])
    vec = TfidfVectorizer(analyzer=args.analyzer, ngram_range=tuple(args.ngram), min_df=3,
                          max_features=args.max_features, sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform([d["text"] for d in tr])
    Xva, Xte = vec.transform([d["text"] for d in va]), vec.transform([d["text"] for d in te])
    for X in (Xtr, Xva, Xte):
        X.sort_indices()
    if args.model == "lr":
        base = LogisticRegression(C=args.C, max_iter=300, class_weight="balanced", solver="liblinear")
    elif args.model == "svm":
        from sklearn.svm import LinearSVC
        base = LinearSVC(C=args.C, class_weight="balanced")
    else:
        from sklearn.naive_bayes import ComplementNB
        base = ComplementNB(alpha=args.C)
    clf = OneVsRestClassifier(base)
    with parallel_backend("threading", n_jobs=args.jobs):
        clf.fit(Xtr, Ytr)
    if args.model == "svm":
        sig = lambda z: 1 / (1 + np.exp(-2 * z))
        pv, pt = sig(clf.decision_function(Xva)).astype(np.float32), sig(clf.decision_function(Xte)).astype(np.float32)
    else:
        pv, pt = clf.predict_proba(Xva).astype(np.float32), clf.predict_proba(Xte).astype(np.float32)
    out = P5_ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_probs.npy", pv)
    np.save(out / "test_probs.npy", pt)
    sc = _m.cross_fit(Yva, pv, "per_label", 5)
    summary = {"run": args.name, "kind": "tfidf_lr (auxiliary stacking input)", "args": vars(args),
               "vocab": len(vec.vocabulary_), "val_crossfit": {k: round(v, 4) for k, v in sc.items() if k != "per_label_f1"},
               "train_min": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
