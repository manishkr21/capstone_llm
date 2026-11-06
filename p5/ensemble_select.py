"""Greedy forward selection of an ensemble (Caruana et al., 2004) on the cross-fitted validation macro-F1.

    python ensemble_select.py --max 8            # considers every finished run in runs/

Starts from the best single run and repeatedly adds the run (repetition allowed = implicit weights) that
most improves the 2-fold cross-fitted macro-F1 of the averaged probabilities with per-statute thresholds.
"""
import argparse
import importlib.util
import json
from collections import Counter

import numpy as np

from lsi import P5_ROOT
from lsi.data import TokenDocs

_s = importlib.util.spec_from_file_location("p5_sub", P5_ROOT / "make_submission.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    Y = TokenDocs("val").labels
    runs = sorted(r.name for r in (P5_ROOT / "runs").iterdir()
                  if (r / "val_probs.npy").exists() and (r / "test_probs.npy").exists() and r.name not in args.exclude)
    P = {r: np.load(P5_ROOT / "runs" / r / "val_probs.npy") for r in runs}
    score = lambda members: _m.cross_fit(Y, np.mean([P[r] for r in members], 0), "per_label", 5)["macro_f1"]
    single = {r: score([r]) for r in runs}
    for r, s in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"  {r:22s} {s:.4f}")
    chosen = [max(single, key=single.get)]
    best = single[chosen[0]]
    while len(chosen) < args.max:
        cand = {r: score(chosen + [r]) for r in runs}
        r, s = max(cand.items(), key=lambda kv: kv[1])
        if s <= best + 1e-4:
            break
        chosen.append(r)
        best = s
        print(f"+ {r:22s} -> {s:.4f}")
    w = Counter(chosen)
    print(json.dumps({"members": list(w), "weights": [w[r] for r in w], "crossfit_macro_f1": round(best, 4)}))


if __name__ == "__main__":
    main()
