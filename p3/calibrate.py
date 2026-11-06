"""Per-class bias calibration for macro-F1, fitted on out-of-fold emissions.

    python calibrate.py --run runs/T8_e2_causal_ep20_tau05 runs/T9_e3_ep20_tau05 --name C1_ens_t8_t9

1. Out-of-fold emissions: fold-k models score the fold-k held-out documents (several runs are
   ensembled by averaging log-softmax emissions and CRF parameters, as in make_submission.py).
2. A bias b[label] is added to the emissions before Viterbi.  b is fitted by coordinate ascent
   over a grid to maximise macro-F1; labels with fewer than --min-gold out-of-fold sentences keep b = 0.
3. Honest estimate: nested CV -- for each fold, b is fitted on the other folds' documents and
   applied to that fold, so the reported "calibrated" score never sees its own labels.
4. The final b (fitted on all out-of-fold documents) is applied to the fold-ensemble test
   emissions -> submissions/<name>_<date>_<time>.zip.
"""
import argparse
import importlib.util
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

from rr import P3_ROOT
from rr.data import cv_folds, encode_doc, load_labels, load_split
from rr.metrics import evaluate
from rr.model import CRF

_spec = importlib.util.spec_from_file_location("p3_train", P3_ROOT / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
_spec2 = importlib.util.spec_from_file_location("p3_sub", P3_ROOT / "make_submission.py")
_sub = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_sub)


def mean_crf(states, n):
    crf = CRF(n)
    crf.load_state_dict({k: torch.stack([s[k] for s in states]).mean(0) for k in states[0]})
    return crf


def oof_emissions(runs, docs, folds, labels, device, cache):
    if cache.exists():
        return torch.load(cache)
    l2i = {l: i for i, l in enumerate(labels)}
    emits, fold_of, crfs = {}, {}, []
    for k, dev_ids in enumerate(folds):
        dev_docs = [d for d in docs if d["id"] in set(dev_ids)]
        per, states = {d["id"]: [] for d in dev_docs}, []
        for run in runs:
            model, tok, cfg = _train.load_fold_model(P3_ROOT / run / f"fold{k}", len(labels), device)
            for d in dev_docs:
                enc = encode_doc(tok, d, cfg["data"]["max_sent_tokens"], l2i)
                per[d["id"]].append(torch.log_softmax(_train.doc_emissions(model, enc, cfg, device).cpu(), -1))
            states.append({n: v.detach().cpu() for n, v in model.crf.state_dict().items()})
            del model
            torch.cuda.empty_cache()
        for i, e in per.items():
            emits[i], fold_of[i] = torch.stack(e).mean(0), k
        crfs.append({k2: torch.stack([s[k2] for s in states]).mean(0) for k2 in states[0]})
        print(f"  fold {k}: out-of-fold emissions for {len(dev_docs)} docs", flush=True)
    out = {"emits": emits, "fold_of": fold_of, "crfs": crfs}
    torch.save(out, cache)
    return out


def viterbi_np(emit, trans, start, end):
    """Same decoding as rr.model.CRF.viterbi, in numpy (fast enough for the bias search)."""
    T = emit.shape[0]
    back = np.zeros((T, emit.shape[1]), dtype=np.int64)
    score = start + emit[0]
    for t in range(1, T):
        s = score[:, None] + trans
        back[t] = s.argmax(0)
        score = s.max(0) + emit[t]
    best = [int((score + end).argmax())]
    for t in range(T - 1, 0, -1):
        best.append(int(back[t, best[-1]]))
    return best[::-1]


def decode_all(doc_ids, oof, bias, labels, crf_cache):
    b = bias.numpy()
    preds = {}
    for i in doc_ids:
        trans, start, end = crf_cache[oof["fold_of"][i]]
        preds[i] = [labels[j] for j in viterbi_np(oof["np"][i] + b, trans, start, end)]
    return preds


def fit_bias(doc_list, oof, labels, crf_cache, grid, passes, min_gold):
    ids = [d["id"] for d in doc_list]
    gold = {}
    for d in doc_list:
        for l in d["labels"]:
            gold[l] = gold.get(l, 0) + 1
    free = [j for j, l in enumerate(labels) if gold.get(l, 0) >= min_gold]
    bias = torch.zeros(len(labels))
    score = lambda b: evaluate(doc_list, decode_all(ids, oof, b, labels, crf_cache))["all"]["macro_f1"]
    best = score(bias)
    for _ in range(passes):
        improved = False
        for j in free:
            for v in grid:
                if v == bias[j]:
                    continue
                b = bias.clone()
                b[j] = v
                s = score(b)
                if s > best + 1e-6:
                    best, bias, improved = s, b, True
        if not improved:
            break
    return bias, best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--min-gold", type=int, default=20)
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    cfg0 = yaml.safe_load(open(P3_ROOT / args.run[0] / "config.yaml"))
    data_dir = P3_ROOT / cfg0["data"]["dir"]
    labels = load_labels(data_dir)
    docs = load_split(data_dir, "train") + load_split(data_dir, "dev")
    folds = cv_folds(docs, cfg0["cv"]["n_folds"], cfg0["cv"]["seed"])
    out_dir = P3_ROOT / "runs" / "ensembles"
    out_dir.mkdir(exist_ok=True)
    oof = oof_emissions(args.run, docs, folds, labels, device, out_dir / f"{args.name}_oof.pt")
    crf_cache = [tuple(c[k].numpy().astype(np.float64) for k in ("trans", "start", "end")) for c in oof["crfs"]]
    oof["np"] = {i: e.numpy().astype(np.float64) for i, e in oof["emits"].items()}
    all_ids = [d["id"] for d in docs]
    base = evaluate(docs, decode_all(all_ids, oof, torch.zeros(len(labels)), labels, crf_cache))

    nested = {}
    for k, dev_ids in enumerate(folds):  # honest estimate: b never sees the fold it is applied to
        inner = [d for d in docs if d["id"] not in set(dev_ids)]
        b, _ = fit_bias(inner, oof, labels, crf_cache, args.grid, args.passes, args.min_gold)
        nested.update(decode_all(dev_ids, oof, b, labels, crf_cache))
        print(f"  nested fold {k}: bias {[round(float(x), 2) for x in b]}", flush=True)
    cal = evaluate(docs, nested)
    bias, fit_score = fit_bias(docs, oof, labels, crf_cache, args.grid, args.passes, args.min_gold)

    res = {"name": args.name, "members": args.run,
           "oof_uncalibrated": {k: round(v["macro_f1"], 4) for k, v in base.items()},
           "oof_calibrated_nested": {k: round(v["macro_f1"], 4) for k, v in cal.items()},
           "calibrated_P_R": [round(cal["all"]["macro_p"], 4), round(cal["all"]["macro_r"], 4)],
           "uncalibrated_P_R": [round(base["all"]["macro_p"], 4), round(base["all"]["macro_r"], 4)],
           "final_bias": {l: round(float(b), 2) for l, b in zip(labels, bias)}, "in_sample_fit": round(fit_score, 4)}
    json.dump(res, open(out_dir / f"{args.name}.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
    if args.no_submit:
        return

    # test: fold-ensemble emissions (+ bias), averaged CRF
    outs = [torch.load(f, map_location="cpu") for r in args.run for f in sorted((P3_ROOT / r).glob("fold*/test_out.pt"))]
    crf = mean_crf([o["crf"] for o in outs], len(labels))
    preds = {}
    for doc_id in outs[0]["emissions"]:
        emit = torch.stack([torch.log_softmax(o["emissions"][doc_id], -1) for o in outs]).mean(0)
        preds[doc_id] = [labels[i] for i in crf.viterbi(emit + bias)]
    test_docs = load_split(data_dir, "test")
    problems = _sub.check(preds, test_docs, labels)
    if problems:
        raise SystemExit(f"format problems: {problems}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pred_path = out_dir / f"{args.name}_prediction.json"
    json.dump(preds, open(pred_path, "w"), ensure_ascii=False)
    zip_path = P3_ROOT / "submissions" / f"{args.name}_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "prediction.json")
    print(f"{len(outs)} fold models -> {zip_path}")


if __name__ == "__main__":
    main()
