"""Test predictions -> prediction.json (a JSON list of {"id", "pred"}) + Codabench zip.

    python make_submission.py --run runs/E2_a4_lorma
    python make_submission.py --run runs/E1_a1_full runs/E2_a4_lorma --name ENS_e1_e2

With several runs the class-1 probabilities are averaged.  The decision threshold is the one that
maximises macro-F1 on the averaged clean-validation probabilities (never the test set).  The
writer checks one integer 0/1 prediction for every test id.
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from cj import P4_ROOT
from cj.data import TokenDocs, load_splits
import importlib.util

_spec = importlib.util.spec_from_file_location("p4_train", P4_ROOT / "train.py")  # not p2/train.py
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
best_threshold, macro_f1 = _train.best_threshold, _train.macro_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    runs = [P4_ROOT / r for r in args.run]
    val, test = TokenDocs("val"), TokenDocs("test")
    pv = np.mean([np.load(r / "val_probs.npy") for r in runs], 0)
    pt = np.mean([np.load(r / "test_probs.npy") for r in runs], 0)
    thr, f1 = best_threshold(val.labels, pv)
    preds = [{"id": str(i), "pred": int(p >= thr)} for i, p in zip(test.doc_ids, pt)]

    expected = [r["id"] for r in load_splits()["test"]]
    assert [p["id"] for p in preds] == [str(i) for i in expected], "test ids out of order"
    assert all(p["pred"] in (0, 1) for p in preds) and len(preds) == len(expected)

    name = args.name or runs[0].name
    out = P4_ROOT / "submissions"
    out.mkdir(exist_ok=True)
    pred_path = P4_ROOT / "runs" / f"{name}_prediction.json"
    json.dump(preds, open(pred_path, "w"))
    zip_path = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "prediction.json")
    share = np.mean([p["pred"] for p in preds])
    print(f"{len(runs)} run(s): val macro-F1 {f1:.4f} @thr {thr:.2f} (P/R {macro_f1(val.labels, (pv >= thr).astype(int))['macro_p']:.3f}/"
          f"{macro_f1(val.labels, (pv >= thr).astype(int))['macro_r']:.3f}); test predicted 1: {share:.1%} -> {zip_path}")


if __name__ == "__main__":
    main()
