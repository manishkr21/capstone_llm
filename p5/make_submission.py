"""Test predictions -> prediction.json (a JSON list of {"id", "pred": [statute ids]}) + Codabench zip.

    python make_submission.py --run runs/E5_a4_partial                       # one run
    python make_submission.py --run runs/E5_a4_partial runs/E1_a4_lorma --name ENS_e5_e1
    python make_submission.py --run ... --thr global                         # one threshold for all statutes

Thresholds (fitted on the validation split, never on test):
  global    : one threshold for all 100 statutes
  per_label : one threshold per statute; statutes with fewer than --min-pos validation positives are
              shrunk towards the global threshold
The reported "honest" score fits thresholds on one half of the validation split and scores the other half
(2-fold cross-fitting), so it does not reuse the documents the thresholds were tuned on.
Every document gets at least one statute (the highest-scoring one if none passes its threshold).
With several runs the statute probabilities are averaged (optionally weighted, --weights).
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from lsi import P5_ROOT
from lsi.data import N_LABELS, TokenDocs, load_splits
import importlib.util

_spec = importlib.util.spec_from_file_location("p5_train", P5_ROOT / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
f1_scores, best_global_threshold = _train.f1_scores, _train.best_global_threshold

GRID = np.linspace(0.02, 0.95, 94)


def per_label_thresholds(Y, P, min_pos=5):
    g_thr, _ = best_global_threshold(Y, P)
    thr = np.full(N_LABELS, g_thr)
    for j in range(N_LABELS):
        y, p = Y[:, j].astype(bool), P[:, j]
        if y.sum() == 0:
            continue
        pred = p[None, :] >= GRID[:, None]                       # (grid, n)
        tp = (pred & y).sum(1)
        f1 = 2 * tp / (pred.sum(1) + y.sum())
        best = float(GRID[int(np.argmax(f1))])
        w = min(1.0, y.sum() / (2.0 * min_pos))                  # few positives -> trust the global threshold more
        thr[j] = w * best + (1 - w) * g_thr
    return thr


def apply(P, thr):
    D = (P >= thr).astype(np.uint8)
    empty = D.sum(1) == 0
    D[empty, P[empty].argmax(1)] = 1
    return D


def fit(Y, P, mode, min_pos):
    return per_label_thresholds(Y, P, min_pos) if mode == "per_label" else np.full(N_LABELS, best_global_threshold(Y, P)[0])


def cross_fit(Y, P, mode, min_pos, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    a, b = idx[: len(Y) // 2], idx[len(Y) // 2:]
    D = np.zeros_like(Y)
    D[b] = apply(P[b], fit(Y[a], P[a], mode, min_pos))
    D[a] = apply(P[a], fit(Y[b], P[b], mode, min_pos))
    return f1_scores(Y, D)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--thr", choices=["per_label", "global"], default="per_label")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--test-from", nargs="*", default=None,
                    help="test probabilities from these runs instead (e.g. *_all twins trained on train+val); "
                         "thresholds still come from the --run validation probabilities")
    args = ap.parse_args()
    runs = [P5_ROOT / r for r in args.run]
    w = np.asarray(args.weights or [1.0] * len(runs), dtype=np.float64)
    w = w / w.sum()
    val, test = TokenDocs("val"), TokenDocs("test")
    pv = sum(wi * np.load(r / "val_probs.npy") for wi, r in zip(w, runs))
    test_runs = [P5_ROOT / r for r in args.test_from] if args.test_from else runs
    assert len(test_runs) == len(runs)
    pt = sum(wi * np.load(r / "test_probs.npy") for wi, r in zip(w, test_runs))
    Y = val.labels
    thr = fit(Y, pv, args.thr, args.min_pos)
    in_sample = f1_scores(Y, apply(pv, thr))
    honest = cross_fit(Y, pv, args.thr, args.min_pos)
    D = apply(pt, thr)
    preds = [{"id": str(i), "pred": [int(j) for j in np.flatnonzero(row)]} for i, row in zip(test.doc_ids, D)]

    expected = [str(r["id"]) for r in load_splits()["test"]]
    assert [p["id"] for p in preds] == expected, "test ids out of order"
    assert all(p["pred"] and len(set(p["pred"])) == len(p["pred"]) and all(0 <= j < N_LABELS for j in p["pred"]) for p in preds)

    name = args.name or runs[0].name
    keep = ("macro_f1", "macro_p", "macro_r", "micro_f1", "subset_acc")
    report = {"name": name, "runs": [str(r.relative_to(P5_ROOT)) for r in runs], "weights": w.tolist(), "thr_mode": args.thr,
              "val_in_sample": {k: round(in_sample[k], 4) for k in keep},
              "val_crossfit": {k: round(honest[k], 4) for k in keep},
              "test_labels_per_doc": round(float(D.sum(1).mean()), 2), "val_labels_per_doc": round(float(Y.sum(1).mean()), 2)}
    print(json.dumps(report))
    out_dir = P5_ROOT / "runs" / "submissions_meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({**report, "thresholds": thr.round(4).tolist()}, open(out_dir / f"{name}.json", "w"), indent=1)
    if args.no_zip:
        return
    pred_path = out_dir / f"{name}_prediction.json"
    json.dump(preds, open(pred_path, "w"))
    (P5_ROOT / "submissions").mkdir(exist_ok=True)
    zip_path = P5_ROOT / "submissions" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "prediction.json")
    print(f"-> {zip_path}")


if __name__ == "__main__":
    main()
