"""Train one Phase-4 experiment (court judgment prediction).

    python train.py --config configs/e2_a4_lorma.yaml
    python train.py --config configs/e2_a4_lorma.yaml --set data.strategy=first run_name=E2_first

Model selection uses the clean validation split only (dev.json minus the documents that are also in
the test set, see cj/data.py).  Every ``train.eval_every`` optimizer steps the model scores the
validation set; the best state (by macro-F1 at the validation-tuned decision threshold) is kept.
Writes runs/<run_name>/: config.yaml, metrics.jsonl, best.pt (trainable parameters only),
val_probs.npy, test_probs.npy, summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from cj import P4_ROOT
from cj.data import TokenDocs, batch_chunks, select_chunks
from cj.model import build_model, make_controller, trainable_state
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import save_json, set_seed


def macro_f1(y, pred):
    out = []
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum())
        p, g = int((pred == c).sum()), int((y == c).sum())
        prec, rec = (tp / p if p else 0.0), (tp / g if g else 0.0)
        out.append((prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0))
    return {"macro_f1": (out[0][2] + out[1][2]) / 2, "macro_p": (out[0][0] + out[1][0]) / 2,
            "macro_r": (out[0][1] + out[1][1]) / 2}


def best_threshold(y, p1):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [macro_f1(y, (p1 >= t).astype(int))["macro_f1"] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def forward_docs(model, docs_tokens, dc, pad_id, device, max_chunks):
    """Logits for a list of documents (token arrays); chunks are encoded in slices of max_chunks."""
    chunks = [select_chunks(t, dc["strategy"], dc["n_chunks"], dc["chunk_len"]) for t in docs_tokens]
    ids, mask, doc_idx = batch_chunks(chunks, pad_id)
    vecs = []
    for i in range(0, ids.size(0), max_chunks):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vecs.append(model.encoder(ids[i:i + max_chunks].to(device), mask[i:i + max_chunks].to(device)))
    return model.doc_logits(torch.cat(vecs).float(), doc_idx.to(device), len(docs_tokens)).float()


@torch.no_grad()
def predict(model, data: TokenDocs, cfg, pad_id, device, bs=16):
    model.eval()
    probs = []
    for i in range(0, len(data), bs):
        toks = [data.tokens(j) for j in range(i, min(len(data), i + bs))]
        probs.append(forward_docs(model, toks, cfg["data"], pad_id, device, cfg["train"]["max_chunks"]).softmax(-1)[:, 1].cpu())
    model.train()
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=str(P4_ROOT / "configs" / "e2_a4_lorma.yaml"))
    ap.add_argument("--max-train-docs", type=int, default=None, help="smoke tests")
    args = ap.parse_args()
    cfg = config_from_args(args)
    tc, dc = cfg["train"], cfg["data"]
    set_seed(cfg["seed"])
    device = torch.device("cuda")
    run_dir = P4_ROOT / cfg.get("runs_dir", "runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)

    train, val, test = TokenDocs("train"), TokenDocs("val"), TokenDocs("test")
    order = list(range(len(train)))[: args.max_train_docs]
    model, tok, ctrl, info = build_model(cfg)
    model.to(device)
    pad = tok.eos_id

    teacher_logits = None
    if cfg.get("distill"):
        teacher_logits = torch.from_numpy(np.load(P4_ROOT / cfg["distill"]["teacher_logits"])).float()

    y_train = torch.from_numpy(train.labels).long()
    counts = torch.bincount(y_train[order], minlength=2).float()
    class_w = (counts.sum() / (2 * counts)).to(device) if tc.get("class_weight", True) else torch.ones(2, device=device)

    steps_per_epoch = math.ceil(len(order) / tc["docs_per_step"])
    total = steps_per_epoch * tc["epochs"]
    ctrl = make_controller(model, ctrl, cfg["peft"], total)
    head_ids = {id(p) for m in model.head_modules() for p in m.parameters()}
    bb_params = [p for p in model.parameters() if id(p) not in head_ids]  # requires_grad may change (gradual)
    opt = torch.optim.AdamW([{"params": bb_params, "lr": float(tc["lr"])},
                             {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": float(tc["head_lr"])}],
                            weight_decay=float(tc["weight_decay"]))
    base = [g["lr"] for g in opt.param_groups]
    warm = int(tc["warmup_frac"] * total)
    rng = random.Random(cfg["seed"])
    log = open(run_dir / "metrics.jsonl", "w")
    best, best_thr, step, t0 = -1.0, 0.5, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(tc["epochs"]):
        rng.shuffle(order)
        for b in range(0, len(order), tc["docs_per_step"]):
            batch = order[b:b + tc["docs_per_step"]]
            for g, lr0 in zip(opt.param_groups, base):
                g["lr"] = lr0 * lr_factor(step, total, warm)
            tot_loss = 0.0
            for m in range(0, len(batch), tc["docs_per_micro"]):
                mb = batch[m:m + tc["docs_per_micro"]]
                logits = forward_docs(model, [train.tokens(i) for i in mb], dc, pad, device, tc["max_chunks"])
                y = y_train[mb].to(device)
                loss = F.cross_entropy(logits, y, weight=class_w)
                if teacher_logits is not None:
                    T, a = cfg["distill"]["temperature"], cfg["distill"]["alpha"]
                    t = teacher_logits[mb].to(device)
                    kd = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(t / T, -1), log_target=True,
                                  reduction="batchmean") * T * T
                    loss = a * loss + (1 - a) * kd
                if ctrl is not None and hasattr(ctrl, "penalty") and not isinstance(ctrl.penalty(), float):
                    loss = loss + ctrl.penalty()
                (loss * len(mb) / len(batch)).backward()
                tot_loss += loss.item() * len(mb) / len(batch)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["grad_clip"])
            if ctrl is not None:
                ctrl.step(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % tc["log_every"] == 0:
                print(f"  step {step}/{total}  loss {tot_loss:.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
            if step % tc["eval_every"] == 0 or step == total:
                pv = predict(model, val, cfg, pad, device)
                thr, f1t = best_threshold(val.labels, pv)
                f05 = macro_f1(val.labels, (pv >= 0.5).astype(int))["macro_f1"]
                rec = {"step": step, "epoch": epoch + 1, "loss": tot_loss, "val_f1_at_0.5": f05, "val_f1_tuned": f1t,
                       "thr": thr, "min": (time.time() - t0) / 60}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"  [val] step {step}  macro-F1 {f05:.4f} @0.5, {f1t:.4f} @thr {thr:.2f}", flush=True)
                if f1t > best:
                    best, best_thr = f1t, thr
                    torch.save(trainable_state(model), run_dir / "best.pt")
                    np.save(run_dir / "val_probs.npy", pv)
    train_min = (time.time() - t0) / 60
    model.load_state_dict(torch.load(run_dir / "best.pt"), strict=False)
    pv = np.load(run_dir / "val_probs.npy")
    t1 = time.time()
    pt = predict(model, test, cfg, pad, device)
    test_s = time.time() - t1
    np.save(run_dir / "test_probs.npy", pt)
    sc = macro_f1(val.labels, (pv >= best_thr).astype(int))
    summary = {"run": cfg["run_name"], **info, "val": sc, "val_f1_at_0.5": macro_f1(val.labels, (pv >= 0.5).astype(int))["macro_f1"],
               "threshold": best_thr, "steps": step, "train_min": round(train_min, 1), "test_infer_s": round(test_s, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "ckpt_mb": round((run_dir / "best.pt").stat().st_size / 1e6, 1), "data": dc}
    save_json(summary, run_dir / "summary.json")
    print(json.dumps({k: summary[k] for k in ("run", "val", "threshold", "train_min")}))


if __name__ == "__main__":
    main()
