"""Out-of-fold score of an ensemble of several runs (same CV folds).

    python ensemble_cv.py --run runs/E1_a4_lorma runs/E3_a3_rslora --name T5_ens_e1_e3

For fold k, each run's fold-k model scores the fold-k held-out documents; log-softmax emissions
and CRF parameters are averaged across runs and decoded with Viterbi (the same rule as
make_submission.py uses on the test set).  Writes runs/<name>/cv_summary.json-style output to
runs/ensembles/<name>.json.
"""
import argparse
import importlib.util
import json
import statistics

import torch
import yaml

from rr import P3_ROOT
from rr.data import cv_folds, encode_doc, load_labels, load_split
from rr.metrics import evaluate
from rr.model import CRF

_spec = importlib.util.spec_from_file_location("p3_train", P3_ROOT / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    cfg0 = yaml.safe_load(open(P3_ROOT / args.run[0] / "config.yaml"))
    data_dir = P3_ROOT / cfg0["data"]["dir"]
    labels = load_labels(data_dir)
    l2i = {l: i for i, l in enumerate(labels)}
    docs = load_split(data_dir, "train") + load_split(data_dir, "dev")
    folds = cv_folds(docs, cfg0["cv"]["n_folds"], cfg0["cv"]["seed"])
    preds, fold_f1 = {}, []
    for k, dev_ids in enumerate(folds):
        dev_docs = [d for d in docs if d["id"] in set(dev_ids)]
        emits, crfs = {d["id"]: [] for d in dev_docs}, []
        for run in args.run:
            model, tok, cfg = _train.load_fold_model(P3_ROOT / run / f"fold{k}", len(labels), device)
            for d in dev_docs:
                enc = encode_doc(tok, d, cfg["data"]["max_sent_tokens"], l2i)
                emits[d["id"]].append(torch.log_softmax(_train.doc_emissions(model, enc, cfg, device).cpu(), -1))
            crfs.append({n: v.detach().cpu() for n, v in model.crf.state_dict().items()})
            del model
            torch.cuda.empty_cache()
        crf = CRF(len(labels))
        crf.load_state_dict({n: torch.stack([c[n] for c in crfs]).mean(0) for n in crfs[0]})
        fold_preds = {i: [labels[j] for j in crf.viterbi(torch.stack(e).mean(0))] for i, e in emits.items()}
        preds.update(fold_preds)
        fold_f1.append(evaluate(dev_docs, fold_preds)["all"]["macro_f1"])
        print(f"fold {k}: {fold_f1[-1]:.4f}", flush=True)
    sc = evaluate(docs, preds)
    out = {"run": args.name, "members": args.run,
           "oof": {k: {m: round(v[m], 4) for m in ("macro_f1", "macro_p", "macro_r", "accuracy")} for k, v in sc.items()},
           "oof_per_label": {l: v["f1"] for l, v in sc["all"]["per_label"].items()},
           "fold_macro_f1_mean": round(statistics.mean(fold_f1), 4),
           "fold_macro_f1_std": round(statistics.pstdev(fold_f1), 4)}
    (P3_ROOT / "runs" / "ensembles").mkdir(exist_ok=True)
    json.dump(out, open(P3_ROOT / "runs" / "ensembles" / f"{args.name}.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("run", "oof", "fold_macro_f1_mean", "fold_macro_f1_std")}, indent=1))


if __name__ == "__main__":
    main()
