"""Train one Phase-3 experiment with document-level cross-validation.

    python train.py --config configs/e1_a4_lorma.yaml                 # all 5 folds
    python train.py --config configs/e1_a4_lorma.yaml --folds 0 2     # some folds
    python train.py --config configs/e1_a4_lorma.yaml --set train.lr=5e-4 run_name=E1_lr5e-4

The official train + dev documents (90: 45 CL, 45 IT) are split into ``cv.n_folds`` folds,
stratified by domain.  Fold k trains on the other folds and selects its best epoch on fold k.
Writes runs/<run_name>/fold<k>/: best.pt (trainable parameters only), summary.json,
dev_prediction.json and test_out.pt (test emissions + CRF, for the fold ensemble), then
runs/<run_name>/cv_summary.json once every fold is done (out-of-fold scores).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from rr import P3_ROOT
from rr.data import cv_folds, encode_doc, load_labels, load_split
from rr.metrics import confusion, evaluate
from rr.model import build_model, make_controller, trainable_state
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import save_json, set_seed


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def load_fold_model(fold_dir, n_labels, device):
    fold_dir = Path(fold_dir)
    cfg = yaml.safe_load(open(fold_dir.parent / "config.yaml"))
    model, tok, _ = build_model(cfg, n_labels)
    state = torch.load(fold_dir / "best.pt", map_location="cpu")
    missing = set(state) - set(model.state_dict())
    if missing:
        raise RuntimeError(f"checkpoint keys not in model: {sorted(missing)[:5]}")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), tok, cfg


@torch.no_grad()
def doc_emissions(model, doc, cfg, device):
    ic = cfg["infer"]
    with autocast(device):
        return model.forward_doc(doc, 0, len(doc["sents"]), cfg["data"]["context"], ic["window"],
                                 ic.get("max_windows", 32)).float()


def decode(model, emit):
    return model.crf.viterbi(emit) if model.crf is not None else emit.argmax(-1).tolist()


@torch.no_grad()
def predict(model, enc_docs, cfg, labels, device):
    model.eval()
    out = {d["id"]: [labels[i] for i in decode(model, doc_emissions(model, d, cfg, device))] for d in enc_docs}
    model.train()
    return out


def run_fold(cfg, k, folds, docs, test_docs, labels, device, run_dir):
    tc, dc = cfg["train"], cfg["data"]
    all_data = k == "all"  # final model: every document, fixed epochs, no evaluation
    kseed = 99 if all_data else k
    set_seed(cfg["seed"] + kseed)
    fold_dir = run_dir / ("fold_all" if all_data else f"fold{k}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    dev_ids = set() if all_data else set(folds[k])
    train_docs = [d for d in docs if d["id"] not in dev_ids]
    dev_docs = [d for d in docs if d["id"] in dev_ids]

    model, tok, info = build_model(cfg, len(labels))
    model.to(device)
    l2i = {l: i for i, l in enumerate(labels)}
    tr, dv, te = ([encode_doc(tok, d, dc["max_sent_tokens"], l2i) for d in ds]
                  for ds in (train_docs, dev_docs, test_docs))

    teacher = None
    if cfg.get("distill"):
        teacher, _, _ = load_fold_model(P3_ROOT / cfg["distill"]["teacher_run"] / f"fold{k}", len(labels), device)
        for p in teacher.parameters():
            p.requires_grad = False

    # logit adjustment (Menon et al., 2021): add tau * log prior to the scores during training
    counts = torch.ones(len(labels))
    for d in tr:
        counts += torch.bincount(d["y"], minlength=len(labels)).float()
    log_prior = (counts / counts.sum()).log().to(device)
    tau = float(tc.get("logit_adjust_tau", 0.0))

    chunk = tc["chunk_sents"]
    steps_per_epoch = math.ceil(sum(math.ceil(len(d["sents"]) / chunk) for d in tr) / tc["chunks_per_step"])
    total = steps_per_epoch * tc["epochs"]
    ctrl = make_controller(model, cfg["peft"], total)
    head_ids = {id(p) for m in model.head_modules() for p in m.parameters()}
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    hd_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": float(tc["lr"])},
                             {"params": hd_params, "lr": float(tc["head_lr"])}],
                            weight_decay=float(tc["weight_decay"]))
    base_lrs = [g["lr"] for g in opt.param_groups]
    warmup = int(tc["warmup_frac"] * total)
    rng = random.Random(cfg["seed"] + kseed)
    log = open(fold_dir / "metrics.jsonl", "w")
    best, bad, step, t0 = -1.0, 0, 0, time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(tc["epochs"]):
        chunks = []
        for d in tr:  # every doc is cut into ceil(n / chunk) random windows of consecutive sentences
            n = len(d["sents"])
            for _ in range(math.ceil(n / chunk)):
                s = rng.randint(0, max(0, n - chunk))
                chunks.append((d, s, min(n, s + chunk)))
        rng.shuffle(chunks)
        model.train()
        tot_loss = 0.0
        for i in range(0, len(chunks), tc["chunks_per_step"]):
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step, total, warmup)
            batch = chunks[i:i + tc["chunks_per_step"]]
            for d, s, e in batch:
                with autocast(device):
                    emit = model.forward_doc(d, s, e, dc["context"], tc["window"], tc.get("max_windows", 16)).float()
                y = d["y"][s:e].to(device)
                adj = emit + tau * log_prior
                loss = model.crf.nll(adj, y) if model.crf is not None else F.cross_entropy(adj, y)
                if teacher is not None:  # Hinton et al. (2015): KL to the teacher's softened scores
                    T, a = cfg["distill"]["temperature"], cfg["distill"]["alpha"]
                    with torch.no_grad(), autocast(device):
                        t_emit = teacher.forward_doc(d, s, e, dc["context"], tc["window"], tc.get("max_windows", 16))
                    kd = F.kl_div(F.log_softmax(emit / T, -1), F.log_softmax(t_emit.float() / T, -1),
                                  log_target=True, reduction="batchmean") * T * T
                    loss = a * loss + (1 - a) * kd
                if ctrl is not None:
                    loss = loss + ctrl.penalty()
                (loss / len(batch)).backward()
                tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["grad_clip"])
            if ctrl is not None:
                ctrl.step(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if all_data:
            log.write(json.dumps({"epoch": epoch + 1, "loss": tot_loss / len(chunks)}) + "\n")
            log.flush()
            print(f"  all-data epoch {epoch + 1:>2}  loss {tot_loss / len(chunks):.4f}", flush=True)
            continue
        preds = predict(model, dv, cfg, labels, device)
        sc = evaluate(dev_docs, preds)
        rec = {"epoch": epoch + 1, "loss": tot_loss / len(chunks), "dev_macro_f1": sc["all"]["macro_f1"],
               "dev_CL": sc["CL"]["macro_f1"], "dev_IT": sc["IT"]["macro_f1"], "min": (time.time() - t0) / 60}
        if ctrl is not None:
            rec["adalora_rank_mean"] = sum(ctrl.ranks()) / len(ctrl.ranks())
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  fold {k} epoch {epoch + 1:>2}  loss {rec['loss']:.4f}  dev macro-F1 {rec['dev_macro_f1']:.4f}"
              f" (CL {rec['dev_CL']:.4f}, IT {rec['dev_IT']:.4f})", flush=True)
        if rec["dev_macro_f1"] > best:
            best, bad = rec["dev_macro_f1"], 0
            torch.save(trainable_state(model), fold_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    train_min = (time.time() - t0) / 60
    if all_data:
        torch.save(trainable_state(model), fold_dir / "best.pt")
        test_emit = {d["id"]: doc_emissions(model, d, cfg, device).cpu() for d in te}
        crf = None if model.crf is None else {n: v.detach().cpu() for n, v in model.crf.state_dict().items()}
        torch.save({"emissions": test_emit, "crf": crf, "labels": labels}, fold_dir / "test_out.pt")
        summary = {"run": cfg["run_name"], "fold": "all", **info, "epochs": tc["epochs"], "train_docs": len(tr),
                   "train_min": round(train_min, 2), "steps": step}
        save_json(summary, fold_dir / "summary.json")
        del model, teacher
        torch.cuda.empty_cache()
        return summary
    model.load_state_dict(torch.load(fold_dir / "best.pt"), strict=False)
    t1 = time.time()
    preds = predict(model, dv, cfg, labels, device)
    infer_s = time.time() - t1
    sc = evaluate(dev_docs, preds)
    save_json(preds, fold_dir / "dev_prediction.json")
    test_emit = {d["id"]: doc_emissions(model, d, cfg, device).cpu() for d in te}
    crf = None if model.crf is None else {n: v.detach().cpu() for n, v in model.crf.state_dict().items()}
    torch.save({"emissions": test_emit, "crf": crf, "labels": labels}, fold_dir / "test_out.pt")
    summary = {"run": cfg["run_name"], "fold": k, **info, "best_dev_macro_f1": best, "dev": sc,
               "confusion": confusion(dev_docs, preds), "train_min": round(train_min, 2), "steps": step,
               "dev_infer_s": round(infer_s, 2),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
               "ckpt_mb": round((fold_dir / "best.pt").stat().st_size / 1e6, 1)}
    save_json(summary, fold_dir / "summary.json")
    del model, teacher
    torch.cuda.empty_cache()
    return summary


def aggregate(run_dir, docs, n_folds):
    """Out-of-fold scores: every document is predicted by the fold model that did not train on it."""
    preds, per_fold = {}, []
    for k in range(n_folds):
        f = run_dir / f"fold{k}"
        if not (f / "summary.json").exists():
            return None
        preds.update(json.load(open(f / "dev_prediction.json")))
        per_fold.append(json.load(open(f / "summary.json")))
    sc = evaluate(docs, preds)
    fold_f1 = [s["dev"]["all"]["macro_f1"] for s in per_fold]
    out = {"run": run_dir.name,
           "oof": {k: {m: round(v[m], 4) for m in ("macro_f1", "macro_p", "macro_r", "accuracy")} for k, v in sc.items()},
           "oof_per_label": {l: v["f1"] for l, v in sc["all"]["per_label"].items()},
           "fold_macro_f1_mean": round(statistics.mean(fold_f1), 4),
           "fold_macro_f1_std": round(statistics.pstdev(fold_f1), 4),
           "trainable_params": per_fold[0]["trainable_params"], "total_params": per_fold[0]["total_params"],
           "trainable_pct": per_fold[0]["trainable_pct"],
           "train_min_per_fold": round(sum(s["train_min"] for s in per_fold) / n_folds, 2),
           "peak_gpu_gb": max(s["peak_gpu_gb"] or 0 for s in per_fold), "ckpt_mb": per_fold[0]["ckpt_mb"],
           "dev_infer_s": round(sum(s["dev_infer_s"] for s in per_fold) / n_folds, 2)}
    save_json(out, run_dir / "cv_summary.json")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=str(P3_ROOT / "configs" / "e1_a4_lorma.yaml"))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--all-data", action="store_true",
                    help="train one model on all 90 docs for the median best epoch of this run's CV folds")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.all_data:
        cv_dir = P3_ROOT / cfg.get("runs_dir", "runs") / cfg["run_name"]
        best_epochs = []
        for m in sorted(cv_dir.glob("fold[0-9]*/metrics.jsonl")):
            recs = [json.loads(x) for x in open(m)]
            best_epochs.append(max(recs, key=lambda r: r["dev_macro_f1"])["epoch"])
        if not best_epochs:
            raise SystemExit(f"run the cross-validation of {cfg['run_name']} first (no fold metrics)")
        cfg["train"]["epochs"] = int(statistics.median(best_epochs))
        print(f"[{cfg['run_name']}] all-data: CV best epochs {best_epochs} -> {cfg['train']['epochs']} epochs")
        cfg["run_name"] += "_all"
        if cfg.get("distill"):
            raise SystemExit("all-data KD needs an all-data teacher; not supported")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = P3_ROOT / cfg["data"]["dir"]
    labels = load_labels(data_dir)
    docs = load_split(data_dir, "train") + load_split(data_dir, "dev")
    test_docs = load_split(data_dir, "test")
    n_folds = cfg["cv"]["n_folds"]
    folds = cv_folds(docs, n_folds, cfg["cv"]["seed"])
    run_dir = P3_ROOT / cfg.get("runs_dir", "runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if args.all_data:
        s = run_fold(cfg, "all", folds, docs, test_docs, labels, device, run_dir)
        print(f"[{cfg['run_name']}] trained on {s['train_docs']} docs for {s['epochs']} epochs", flush=True)
        return
    for k in (args.folds if args.folds is not None else range(n_folds)):
        s = run_fold(cfg, k, folds, docs, test_docs, labels, device, run_dir)
        print(f"[{cfg['run_name']}] fold {k}: dev macro-F1 {s['dev']['all']['macro_f1']:.4f}", flush=True)
    agg = aggregate(run_dir, docs, n_folds)
    if agg:
        print(json.dumps({k: agg[k] for k in ("run", "oof", "fold_macro_f1_mean", "fold_macro_f1_std")}, indent=1))


if __name__ == "__main__":
    main()
