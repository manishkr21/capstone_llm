"""Predict entity spans with a trained run and package a Codabench submission.

    python predict.py --run runs/E1_a3_full                # test -> prediction.json + zip
    python predict.py --run runs/E1_a3_full --split dev    # re-score the held-out docs
    python predict.py --run runs/E1_a3_full --set infer.decode=argmax infer.stride=256

The zip (prediction.json at the archive root) is written to runs/<run>/test/ and copied
to submissions/<run>_<date>_<time>.zip.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime

import torch

from lner import P2_ROOT
from lner.data import bio_tags, load_labels, load_test, load_train, split_dev
from lner.engine import load_run, predict_docs, prepare_docs
from lner.metrics import strict_scores
from legal_lm.config import apply_overrides
from legal_lm.utils import save_json


def check_submission(preds, texts, labels):
    assert set(preds) == set(texts), "prediction IDs must match the test IDs exactly"
    n = 0
    for doc, ents in preds.items():
        for s, e, l, *rest in ents:
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(texts[doc]), (doc, s, e)
            assert l in labels, l
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override infer.* settings")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = (P2_ROOT / args.run).resolve()

    probe = __import__("yaml").safe_load(open(run_dir / "config.yaml"))
    data_dir = P2_ROOT / probe["data"]["dir"]
    labels = load_labels(data_dir)
    tags = bio_tags(labels)
    model, tok, cfg = load_run(run_dir, tags, device, args.ckpt)
    apply_overrides(cfg, args.set)
    presplit, case_norm = cfg["data"]["presplit"], cfg["data"].get("case_norm", False)

    if args.split == "dev":
        texts, gold = load_train(data_dir)
        _, dev_ids = split_dev(list(texts), cfg["data"]["n_dev"], cfg["data"]["split_seed"])
        assert dev_ids, "this run was trained without a dev split"
        docs = prepare_docs(tok, {k: texts[k] for k in dev_ids}, None, presplit, None, case_norm)
        scores = strict_scores({k: gold[k] for k in dev_ids}, predict_docs(model, docs, tags, cfg["infer"], device), labels)
        print({k: v for k, v in scores.items() if k != "per_label"})
        for l, v in scores["per_label"].items():
            print(f"  {l:10s} P {v['p']:.3f} R {v['r']:.3f} F1 {v['f1']:.3f}  (gold {v['gold']}, pred {v['pred']})")
        return

    texts = load_test(data_dir)
    docs = prepare_docs(tok, texts, None, presplit, None, case_norm)
    preds = predict_docs(model, docs, tags, cfg["infer"], device)
    preds = {k: preds[k] for k in texts}  # official ID order
    n = check_submission(preds, texts, labels)
    out_dir = run_dir / "test"
    pred_path = out_dir / "prediction.json"
    save_json(preds, pred_path, indent=None)
    stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    zip_path = out_dir / f"submission_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    sub_dir = P2_ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    final = sub_dir / f"{cfg['run_name']}_{stamp}.zip"
    shutil.copy(zip_path, final)
    print(f"{n} entities in {len(preds)} documents -> {final}")


if __name__ == "__main__":
    main()
