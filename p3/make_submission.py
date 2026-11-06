"""Fold-ensemble test predictions -> prediction.json + Codabench zip.

    python make_submission.py --run runs/E1_a4_lorma

Averages the test emissions and CRF parameters of the fold models (runs/<run>/fold*/test_out.pt),
decodes with Viterbi, checks the format (every test doc, one official label per sentence, same
order) and writes submissions/<run>_<date>_<time>.zip with prediction.json at the root.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import torch

from rr import P3_ROOT
from rr.data import load_labels, load_split
from rr.model import CRF


def ensemble(run_dirs):
    outs = [torch.load(f, map_location="cpu") for r in run_dirs for f in sorted(Path(r).glob("fold*/test_out.pt"))]
    if not outs:
        raise FileNotFoundError(f"no fold*/test_out.pt in {run_dirs}")
    labels = outs[0]["labels"]
    crf = None
    if all(o["crf"] is not None for o in outs):
        crf = CRF(len(labels))
        crf.load_state_dict({k: torch.stack([o["crf"][k] for o in outs]).mean(0) for k in outs[0]["crf"]})
    preds = {}
    for doc_id in outs[0]["emissions"]:
        emit = torch.stack([torch.log_softmax(o["emissions"][doc_id], -1) for o in outs]).mean(0)
        idx = crf.viterbi(emit) if crf is not None else emit.argmax(-1).tolist()
        preds[doc_id] = [labels[i] for i in idx]
    return preds, len(outs)


def check(preds, test_docs, labels):
    problems = []
    for d in test_docs:
        p = preds.get(d["id"])
        if p is None:
            problems.append(f"missing {d['id']}")
        elif len(p) != len(d["text"]):
            problems.append(f"{d['id']}: {len(p)} labels for {len(d['text'])} sentences")
        elif any(l not in labels for l in p):
            problems.append(f"{d['id']}: unknown label")
    extra = set(preds) - {d["id"] for d in test_docs}
    if extra:
        problems.append(f"unknown ids {sorted(extra)[:3]}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True, help="one run dir, or several to ensemble across runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data-dir", default="../capstone_llm/data/hf/Phase_3_RR")
    args = ap.parse_args()
    data_dir = P3_ROOT / args.data_dir
    labels = load_labels(data_dir)
    test_docs = load_split(data_dir, "test")
    preds, n = ensemble([P3_ROOT / r for r in args.run])
    problems = check(preds, test_docs, labels)
    if problems:
        raise SystemExit(f"format problems: {problems}")
    name = args.name or Path(args.run[0]).name
    out_dir = P3_ROOT / "submissions"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pred_path = P3_ROOT / Path(args.run[0]) / "prediction.json"
    with open(pred_path, "w") as f:
        json.dump(preds, f, ensure_ascii=False)
    zip_path = out_dir / f"{name}_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "prediction.json")
    print(f"{n} fold models, {len(preds)} test docs, {sum(map(len, preds.values()))} sentences -> {zip_path}")


if __name__ == "__main__":
    main()
