"""Minimum-Bayes-risk selection across the summaries of several runs (no extra model involved).

    python mbr.py --name MBR1 --cand E2_a1_lora:rp F1_a4_full:rp E6_a1_adalora:rp ...

Each candidate is ``RUN:TAG`` (TAG "" = the run's own decode, val_pred.json / test_pred.json; otherwise
val_pred_TAG.json / test_pred_TAG.json). For every document the candidate with the highest mean
agreement (ROUGE-1 + ROUGE-2 F1, word-level) with the other candidates is chosen: the "consensus"
summary, which tends to keep the content most systems agree on and to drop idiosyncratic errors.
Scored on the validation documents; the same rule picks the test summaries and writes a zip.
"""
import argparse
import json
import time
import zipfile
from collections import Counter

import numpy as np

from summ import P6_ROOT
from summ.data import load_splits
from summ.metrics import score


def ngrams(text, n):
    w = text.lower().split()
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def f1(a, b):
    o = sum((a & b).values())
    if not o:
        return 0.0
    p, r = o / max(1, sum(a.values())), o / max(1, sum(b.values()))
    return 2 * p * r / (p + r)


def pick(cands, weights):
    """cands: list of texts for one document -> index of the consensus candidate."""
    g = [(ngrams(c, 1), ngrams(c, 2)) for c in cands]
    k = len(cands)
    util = np.zeros(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                util[i] += weights[j] * (f1(g[i][0], g[j][0]) + f1(g[i][1], g[j][1]))
    return int(np.argmax(util))


def load(run, tag, split):
    suffix = f"_{tag}" if tag else ""
    return {r["id"]: r["summary"] for r in json.load(open(P6_ROOT / "runs" / run / f"{split}_pred{suffix}.json"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    cands = [c.split(":") if ":" in c else (c, "") for c in args.cand]
    w = args.weights or [1.0] * len(cands)
    val = load_splits()["val"]
    V = [load(r, t, "val") for r, t in cands]
    ids = [d["id"] for d in val if all(d["id"] in v for v in V)]
    refs = {d["id"]: " ".join(d["summary"].split()) for d in val}
    chosen = [pick([v[i] for v in V], w) for i in ids]
    hyps = [V[c][i] for c, i in zip(chosen, ids)]
    sc = score(hyps, [refs[i] for i in ids])
    report = {"name": args.name, "cands": args.cand, "n_val": len(ids), "val_metrics": {k: round(v, 4) for k, v in sc.items()},
              "picked_share": {args.cand[k]: round(chosen.count(k) / len(ids), 3) for k in range(len(cands))}}
    print(json.dumps(report), flush=True)
    out = P6_ROOT / "runs" / "mbr"
    out.mkdir(exist_ok=True)
    if args.no_zip:
        return
    T = [load(r, t, "test") for r, t in cands]
    test_ids = [d["id"] for d in load_splits()["test"]]
    preds = [{"id": i, "summary": T[pick([t[i] for t in T], w)][i]} for i in test_ids]
    assert all(p["summary"].strip() for p in preds)
    pp = out / f"{args.name}_test_pred.json"
    json.dump(preds, open(pp, "w"), indent=1)
    json.dump(report, open(out / f"{args.name}.json", "w"), indent=1)
    zp = P6_ROOT / "submissions" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pp, "prediction.json")
    print(f"-> {zp}")


if __name__ == "__main__":
    main()
