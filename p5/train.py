"""Train one Phase-5 experiment (multi-label statute identification, 100 statute ids).

    python train.py --config configs/e1_a4_lorma.yaml
    python train.py --config configs/e1_a4_lorma.yaml --set data.strategy=first run_name=E1_first

Loss: binary cross-entropy per statute, optionally with positive weights (``train.pos_weight``:
none | sqrt | inv, from training label frequencies) to help rare statutes, which macro-F1 rewards.
Model selection: every ``train.eval_every`` steps the model scores a fixed subset of the validation
split (``train.sel_docs``); the state with the best macro-F1 at the best single global threshold is kept.
At the end the best state scores the whole validation split and the test split.
Writes runs/<run_name>/: config.yaml, metrics.jsonl, best.pt, val_probs.npy, test_probs.npy, summary.json.
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

from lsi import P5_ROOT
from lsi.data import N_LABELS, TokenDocs, batch_chunks, select_chunks
from lsi.model import build_model, make_controller, trainable_state
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import save_json, set_seed


def f1_scores(Y, P):
    """Y, P: (n, L) 0/1. Macro/micro F1 over labels (macro over labels present in gold or prediction),
    plus subset accuracy."""
    tp = (Y & P).sum(0).astype(float)
    g, p = Y.sum(0).astype(float), P.sum(0).astype(float)
    prec = np.divide(tp, p, out=np.zeros_like(tp), where=p > 0)
    rec = np.divide(tp, g, out=np.zeros_like(tp), where=g > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = (g + p) > 0
    micro = 2 * tp.sum() / max(1.0, g.sum() + p.sum())
    return {"macro_f1": float(f1[present].mean()), "macro_p": float(prec[present].mean()),
            "macro_r": float(rec[present].mean()), "micro_f1": float(micro),
            "subset_acc": float((Y == P).all(1).mean()), "per_label_f1": f1}


def decide(probs, thr, min_one=True):
    P = (probs >= thr).astype(np.uint8)
    if min_one:  # every document cites at least one statute
        empty = P.sum(1) == 0
        P[empty, probs[empty].argmax(1)] = 1
    return P


def best_global_threshold(Y, probs):
    grid = np.linspace(0.05, 0.9, 35)
    scores = [f1_scores(Y, decide(probs, t))["macro_f1"] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def forward_docs(model, docs_tokens, dc, pad_id, device, max_chunks):
    chunks = [select_chunks(t, dc["strategy"], dc["n_chunks"], dc["chunk_len"]) for t in docs_tokens]
    ids, mask, doc_idx = batch_chunks(chunks, pad_id)
    n_prompt = getattr(model, "n_prompt", 0)
    if n_prompt:  # prompt tuning: placeholder positions that PromptEmbedding replaces with learned vectors
        ids = torch.cat([torch.full((ids.size(0), n_prompt), pad_id, dtype=ids.dtype), ids], 1)
        mask = torch.cat([torch.zeros((mask.size(0), n_prompt), dtype=torch.bool), mask], 1)
    vecs = []
    for i in range(0, ids.size(0), max_chunks):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vecs.append(model.encoder(ids[i:i + max_chunks].to(device), mask[i:i + max_chunks].to(device)))
    return model.doc_logits(torch.cat(vecs).float(), doc_idx.to(device), len(docs_tokens)).float()


@torch.no_grad()
def predict(model, data: TokenDocs, cfg, pad_id, device, idx=None, bs=16):
    model.eval()
    idx = list(range(len(data))) if idx is None else list(idx)
    # sort by length so a batch holds similar-sized documents (much less padding), then restore order
    order = sorted(range(len(idx)), key=lambda k: len(data.tokens(idx[k])))
    out = np.zeros((len(idx), N_LABELS), dtype=np.float32)
    for i in range(0, len(order), bs):
        sl = order[i:i + bs]
        toks = [data.tokens(idx[k]) for k in sl]
        out[sl] = torch.sigmoid(forward_docs(model, toks, cfg["data"], pad_id, device, cfg["train"]["max_chunks"])).cpu().numpy()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=str(P5_ROOT / "configs" / "e1_a4_lorma.yaml"))
    ap.add_argument("--max-train-docs", type=int, default=None, help="smoke tests")
    ap.add_argument("--all-data", action="store_true",
                    help="train on train + validation for the fixed schedule (no selection), keep the last state, "
                         "score only the test split; thresholds come from the train-only twin run")
    args = ap.parse_args()
    cfg = config_from_args(args)
    tc, dc = cfg["train"], cfg["data"]
    set_seed(cfg["seed"])
    device = torch.device("cuda")
    run_dir = P5_ROOT / cfg.get("runs_dir", "runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)

    train, val, test = TokenDocs("train"), TokenDocs("val"), TokenDocs("test")
    if args.all_data:
        train = TokenDocs.concat(train, val)
        cfg["run_name"] += "_all"
        run_dir = P5_ROOT / cfg.get("runs_dir", "runs") / cfg["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    order = list(range(len(train)))
    random.Random(0).shuffle(order)
    order = order[: args.max_train_docs]
    sel = sorted(random.Random(1).sample(range(len(val)), min(tc["sel_docs"], len(val))))
    model, tok, ctrl, info = build_model(cfg)
    model.to(device)
    pad = tok.eos_id

    Y_train = torch.from_numpy(train.labels.astype(np.float32))
    freq = Y_train[order].mean(0).clamp(min=1e-4)
    pw = {"none": None, "sqrt": ((1 - freq) / freq).sqrt(), "inv": (1 - freq) / freq}[tc.get("pos_weight", "none")]
    if pw is not None:
        pw = pw.clamp(max=tc.get("pos_weight_max", 30.0)).to(device)
    teacher = None
    if cfg.get("distill"):
        teacher = torch.from_numpy(np.load(P5_ROOT / cfg["distill"]["teacher_logits"])).float()

    steps_per_epoch = math.ceil(len(order) / tc["docs_per_step"])
    total = steps_per_epoch * tc["epochs"]
    ctrl = make_controller(model, ctrl, cfg["peft"], total)
    head_ids = {id(p) for m in model.head_modules() for p in m.parameters()}
    bb_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": float(tc["lr"])},
                             {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": float(tc["head_lr"])}],
                            weight_decay=float(tc["weight_decay"]))
    base = [g["lr"] for g in opt.param_groups]
    warm = int(tc["warmup_frac"] * total)
    rng = random.Random(cfg["seed"])
    log = open(run_dir / "metrics.jsonl", "w")
    best, best_thr, step, t0 = -1.0, 0.5, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    Yv_sel = val.labels[sel]
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
                y = Y_train[mb].to(device)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
                if teacher is not None:
                    T, a = cfg["distill"]["temperature"], cfg["distill"]["alpha"]
                    soft = torch.sigmoid(teacher[mb].to(device) / T)
                    kd = F.binary_cross_entropy_with_logits(logits / T, soft) * T * T
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
            if args.all_data:
                if step == total:
                    torch.save(trainable_state(model), run_dir / "best.pt")
                continue
            if step % tc["eval_every"] == 0 or step == total:
                pv = predict(model, val, cfg, pad, device, idx=sel)
                thr, f1t = best_global_threshold(Yv_sel, pv)
                s05 = f1_scores(Yv_sel, decide(pv, 0.5))
                rec = {"step": step, "epoch": epoch + 1, "loss": tot_loss, "sel_macro_f1_at_0.5": s05["macro_f1"],
                       "sel_micro_f1_at_0.5": s05["micro_f1"], "sel_macro_f1_tuned": f1t, "thr": thr,
                       "min": (time.time() - t0) / 60}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"  [val-sel] step {step}  macro-F1 {s05['macro_f1']:.4f} @0.5, {f1t:.4f} @thr {thr:.2f}"
                      f"  micro {s05['micro_f1']:.4f}", flush=True)
                if f1t > best:
                    best, best_thr = f1t, thr
                    torch.save(trainable_state(model), run_dir / "best.pt")
    train_min = (time.time() - t0) / 60
    model.load_state_dict(torch.load(run_dir / "best.pt"), strict=False)
    if args.all_data:
        np.save(run_dir / "test_probs.npy", predict(model, test, cfg, pad, device))
        save_json({"run": cfg["run_name"], **info, "train_docs": len(train), "steps": step,
                   "train_min": round(train_min, 1), "data": dc}, run_dir / "summary.json")
        print(json.dumps({"run": cfg["run_name"], "train_docs": len(train), "train_min": round(train_min, 1)}))
        return
    lim = 64 if args.max_train_docs else None   # smoke tests score only a few documents
    pv = predict(model, val, cfg, pad, device, idx=range(lim) if lim else None)
    np.save(run_dir / "val_probs.npy", pv)
    t1 = time.time()
    pt = predict(model, test, cfg, pad, device, idx=range(lim) if lim else None)
    test_s = time.time() - t1
    np.save(run_dir / "test_probs.npy", pt)
    Yv = val.labels[:len(pv)]
    thr, _ = best_global_threshold(Yv, pv)
    sc = f1_scores(Yv, decide(pv, thr))
    summary = {"run": cfg["run_name"], **info,
               "val": {k: v for k, v in sc.items() if k != "per_label_f1"}, "global_threshold": thr,
               "val_at_0.5": {k: v for k, v in f1_scores(Yv, decide(pv, 0.5)).items() if k != "per_label_f1"},
               "sel_best_macro_f1": best, "steps": step, "train_min": round(train_min, 1), "test_infer_s": round(test_s, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "ckpt_mb": round((run_dir / "best.pt").stat().st_size / 1e6, 1), "data": dc}
    save_json(summary, run_dir / "summary.json")
    print(json.dumps({k: summary[k] for k in ("run", "val", "global_threshold", "train_min")}))


if __name__ == "__main__":
    main()
