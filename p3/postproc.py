"""Decoding-time post-processing for macro-F1, evaluated on out-of-fold emissions.

    python postproc.py --name C3_ens_t7_t8_t10_t9                  # uses runs/ensembles/<name>.json + _oof.pt
    python postproc.py --name C3_ens_t7_t8_t10_t9 --submit block=PO,Dissent scale=2

Two knobs (no retraining):
  block  : labels never emitted (emission = -inf).  Macro-F1 averages over every label present in
           gold OR predictions, so predicting a label the test set does not contain adds a class
           with F1 = 0.  PrecedentOverruled (3 training sentences, 1 doc) and Dissent (2 of 90 docs)
           are the candidates.
  scale  : CRF transition scores multiplied by ``scale`` (> 1 gives longer segments; predicted
           segments are ~30% shorter than gold).
  bias   : optional per-class bias from calibrate.py (--use-bias).

Evaluation: (a) out-of-fold macro-F1 on all 90 documents; (b) "test-like" macro-F1, averaged
over --draws random sets of 5 CL + 5 IT out-of-fold documents (the shape of the real test
set), each scored on its own so a label absent from a draw only counts if it is predicted.
"""
import argparse
import importlib.util
import itertools
import json
import time
import zipfile

import numpy as np
import torch
import yaml

from rr import P3_ROOT
from rr.data import load_labels, load_split

_spec = importlib.util.spec_from_file_location("p3_cal", P3_ROOT / "calibrate.py")
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)
SHORT = {"PO": "PrecedentOverruled", "Dissent": "Dissent"}
NEG = -1e9


def counts(gold_idx, pred_idx, n):
    """(3, n): true positives, gold count, predicted count per label."""
    tp = np.bincount(gold_idx[gold_idx == pred_idx], minlength=n)
    return np.stack([tp, np.bincount(gold_idx, minlength=n), np.bincount(pred_idx, minlength=n)])


def macro_f1(c):
    tp, g, p = c
    present = (g + p) > 0
    prec = np.divide(tp, p, out=np.zeros(len(tp)), where=p > 0)
    rec = np.divide(tp, g, out=np.zeros(len(tp)), where=g > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    return f1[present].mean()


def decode(emit, crf, block_idx, scale, bias):
    e = emit + bias
    if block_idx:
        e = e.copy()
        e[:, block_idx] = NEG
    trans, start, end = crf
    return np.array(_cal.viterbi_np(e, trans * scale, start, end))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="ensemble name in runs/ensembles (from calibrate.py)")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--use-bias", action="store_true", help="also try the calibrate.py bias")
    ap.add_argument("--submit", nargs="*", default=None,
                    help="write a test zip for these settings, e.g. block=PO,Dissent scale=2 bias=0")
    args = ap.parse_args()
    ens = json.load(open(P3_ROOT / "runs" / "ensembles" / f"{args.name}.json"))
    oof = torch.load(P3_ROOT / "runs" / "ensembles" / f"{args.name}_oof.pt", weights_only=False)
    cfg0 = yaml.safe_load(open(P3_ROOT / ens["members"][0] / "config.yaml"))
    data_dir = P3_ROOT / cfg0["data"]["dir"]
    labels = load_labels(data_dir)
    n = len(labels)
    l2i = {l: i for i, l in enumerate(labels)}
    docs = load_split(data_dir, "train") + load_split(data_dir, "dev")
    crfs = [tuple(c[k].numpy().astype(np.float64) for k in ("trans", "start", "end")) for c in oof["crfs"]]
    emits = {i: e.numpy().astype(np.float64) for i, e in oof["emits"].items()}
    cal_bias = np.array([ens.get("final_bias", {}).get(l, 0.0) for l in labels])

    if args.submit is not None:
        kv = dict(x.split("=") for x in args.submit)
        block = [SHORT.get(b, b) for b in kv.get("block", "").split(",") if b]
        scale, use_b = float(kv.get("scale", 1)), kv.get("bias", "0") == "1"
        outs = [torch.load(f, map_location="cpu") for r in ens["members"]
                for f in sorted((P3_ROOT / r).glob("fold*/test_out.pt"))]
        crf = _cal.mean_crf([o["crf"] for o in outs], n)
        crf_np = tuple(getattr(crf, k).detach().numpy().astype(np.float64) for k in ("trans", "start", "end"))
        preds = {}
        for doc_id in outs[0]["emissions"]:
            emit = torch.stack([torch.log_softmax(o["emissions"][doc_id], -1) for o in outs]).mean(0).numpy()
            idx = decode(emit.astype(np.float64), crf_np, [l2i[b] for b in block], scale, cal_bias if use_b else 0.0)
            preds[doc_id] = [labels[j] for j in idx]
        test_docs = load_split(data_dir, "test")
        problems = _cal._sub.check(preds, test_docs, labels)
        if problems:
            raise SystemExit(f"format problems: {problems}")
        tag = f"{args.name}_pp_" + ("blk" + "+".join(kv.get("block", "").split(",")) + "_" if block else "") + \
              f"s{scale:g}" + ("_bias" if use_b else "")
        pred_path = P3_ROOT / "runs" / "ensembles" / f"{tag}_prediction.json"
        json.dump(preds, open(pred_path, "w"), ensure_ascii=False)
        zip_path = P3_ROOT / "submissions" / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pred_path, "prediction.json")
        from collections import Counter
        print(f"{len(outs)} fold models -> {zip_path}\n  predicted label counts: "
              f"{dict(Counter(l for v in preds.values() for l in v).most_common())}")
        return

    gold = {d["id"]: np.array([l2i[l] for l in d["labels"]]) for d in docs}
    by_dom = {dom: [d["id"] for d in docs if d["domain"] == dom] for dom in ("CL", "IT")}
    rng = np.random.default_rng(0)
    draws = [np.r_[rng.choice(len(by_dom["CL"]), 5, replace=False),
                   len(by_dom["CL"]) + rng.choice(len(by_dom["IT"]), 5, replace=False)] for _ in range(args.draws)]
    order = by_dom["CL"] + by_dom["IT"]
    blocks = {"none": [], "PO": ["PO"], "PO+Dissent": ["PO", "Dissent"]}
    rows = []
    for (bname, blist), scale, use_b in itertools.product(blocks.items(), args.scales,
                                                           [False, True] if args.use_bias else [False]):
        bidx = [l2i[SHORT[b]] for b in blist]
        C = np.stack([counts(gold[i], decode(emits[i], crfs[oof["fold_of"][i]], bidx, scale,
                                             cal_bias if use_b else 0.0), n) for i in order])  # (docs, 3, n)
        sims = np.array([macro_f1(C[dr].sum(0)) for dr in draws])
        seg = np.mean([len(np.flatnonzero(np.diff(decode(emits[i], crfs[oof["fold_of"][i]], bidx, scale,
                                                           cal_bias if use_b else 0.0)))) + 1 for i in order])
        rows.append({"block": bname, "scale": scale, "bias": use_b, "oof_all": round(macro_f1(C.sum(0)), 4),
                     "test_like_mean": round(sims.mean(), 4), "test_like_p10": round(np.percentile(sims, 10), 4),
                     "test_like_p90": round(np.percentile(sims, 90), 4), "segments_per_doc": round(seg, 1)})
        print(rows[-1], flush=True)
    gold_seg = np.mean([len(np.flatnonzero(np.diff(gold[i]))) + 1 for i in order])
    rows.sort(key=lambda r: -r["test_like_mean"])
    out = {"name": args.name, "gold_segments_per_doc": round(gold_seg, 1), "results": rows}
    json.dump(out, open(P3_ROOT / "runs" / "ensembles" / f"{args.name}_postproc.json", "w"), indent=2)
    print(f"\ngold segments per doc {gold_seg:.1f}; best by test-like mean:")
    for r in rows[:6]:
        print(" ", r)


if __name__ == "__main__":
    main()
