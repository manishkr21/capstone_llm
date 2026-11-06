"""Train one Phase-7 experiment (prior case retrieval, citation-window bi-encoder on a Phase-1 model).

    python train.py --config configs/e1_a4_lorma.yaml
    python train.py --config configs/e1_a4_lorma.yaml --set train.epochs=0 run_name=Z0_a4   # zero-shot

Each step: train.queries query cases; for each, n_pos gold cases and n_hard lexical hard negatives (top
of the per-citation BM25 ranking that are not gold); every query is scored against every case in the batch
(multi-positive InfoNCE, its own case removed). Optional distillation (``distill``): KL between the
student's and a trained teacher run's score distributions over the same batch.
Selection: micro-F1@7 on dev after every epoch (neural scores alone). At the end the best state scores dev
and test; scores are also fused with the lexical ranker (z-score sum + a penalty for cases dated after the
query; weights picked on dev). Writes runs/<run_name>/: config.yaml, metrics.jsonl, best.pt,
{dev,test}_scores.npz (+ train_scores.npz for a fold run), summary.json and submissions/<run_name>_{neural,fused}_<stamp>.zip.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pcr import P7_ROOT
from pcr.data import K, load, micro_f1, topk
from pcr.lexical import zscore
from pcr.model import build_model, info_nce, trainable_state
from pcr import tok as T
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import set_seed

LEX = "pc60_max"


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def lex(split):
    return dict(np.load(P7_ROOT / "cache" / f"lex_{split}.npz"))


def batch_scores(model, qids, cands, toks, split, tc, device, rng, grad=True):
    wins, w_owner = [], []
    for k, q in enumerate(qids):
        ws = toks["wins"][split][q]
        if len(ws) > tc["max_wins"]:
            ws = [ws[i] for i in rng.choice(len(ws), tc["max_wins"], replace=False)]
        wins += ws
        w_owner += [k] * len(ws)
    chs, c_owner = [], []
    for k, c in enumerate(cands):
        cs = T.chunks(toks["docs"][c], tc["chunk_len"], tc["train_chunks"], "random", rng)
        chs += cs
        c_owner += [k] * len(cs)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        W = model.embed(wins, device, tc["max_tokens"])
        C = model.embed(chs, device, tc["max_tokens"])
        return model.score(W, torch.tensor(w_owner, device=device), len(qids), C,
                           torch.tensor(c_owner, device=device), len(cands))


@torch.no_grad()
def score_split(model, split, toks, cfg, device):
    """Neural (n_queries, n_cases) score matrix in the order of the lexical cache."""
    lx, tc = lex(split), cfg["train"]
    qids, cids = list(lx["qids"]), list(lx["cids"])
    model.eval()
    chs, c_owner = [], []
    for k, c in enumerate(cids):
        cs = T.chunks(toks["docs"][c], tc["chunk_len"], tc["eval_chunks"], "spread")
        chs += cs
        c_owner += [k] * len(cs)
    wins, w_owner = [], []
    for k, q in enumerate(qids):
        wins += toks["wins"][split][q]
        w_owner += [k] * len(toks["wins"][split][q])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        C = model.embed(chs, device, tc["max_tokens"])
        W = model.embed(wins, device, tc["max_tokens"])
    w_owner, c_owner = torch.tensor(w_owner, device=device), torch.tensor(c_owner, device=device)
    S = torch.empty(len(qids), len(cids))
    for s in range(0, len(qids), 32):  # blocks of queries keep the (windows x chunks) matrix small
        sel = (w_owner >= s) & (w_owner < s + 32)
        S[s:s + 32] = model.score(W[sel], w_owner[sel] - s, min(32, len(qids) - s), C, c_owner, len(cids)).float().cpu()
    model.train()
    return S.numpy(), qids, cids


def fuse(S, lx, lam, pen):
    viol = (lx["cyear"][None, :] > lx["qyear"][:, None]).astype(np.float32)
    return zscore(S) + lam * zscore(lx[LEX]) - pen * viol


def fit_fusion(S, lx, queries):
    best = (-1, 0.0, 0.0)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for pen in (0.0, 0.5, 1.0, 2.0):
            f = micro_f1(topk(fuse(S, lx, lam, pen), list(lx["qids"]), list(lx["cids"])), queries)["f1"]
            if f > best[0]:
                best = (f, lam, pen)
    return best


def write_zip(name, ranked, pool_ids):
    for q, r in ranked.items():
        assert len(r) == K and len(set(r)) == K and q not in r and all(c in pool_ids for c in r)
    zp = P7_ROOT / "submissions" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(ranked, indent=1))
    return zp


def load_run(name, device):
    cfg = yaml.safe_load(open(P7_ROOT / "runs" / name / "config.yaml"))
    model, _, _ = build_model(cfg)
    model.load_state_dict(torch.load(P7_ROOT / "runs" / name / "best.pt", map_location="cpu"), strict=False)
    return model.to(device), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=str(P7_ROOT / "configs" / "base.yaml"))
    cfg = config_from_args(ap.parse_args())
    set_seed(cfg["seed"])
    device = torch.device("cuda")
    run_dir = P7_ROOT / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    tc = cfg["train"]
    model, tok, params = build_model(cfg)
    model.to(device)
    print(json.dumps(params), flush=True)
    toks = T.get()
    teacher = None
    if cfg.get("distill"):
        teacher, _ = load_run(cfg["distill"]["teacher"], device)
        teacher.eval()

    tq, _ = load("train")
    lxt = lex("train")
    t_qids, t_cids = list(lxt["qids"]), list(lxt["cids"])
    rel = {q: sorted(tq[q]["rel"]) for q in t_qids if tq[q]["rel"]}
    hard = {}
    for i, q in enumerate(t_qids):
        if q in rel:
            order = np.argsort(-lxt[LEX][i])[: tc["hard_pool"] + len(rel[q]) + 1]
            hard[q] = [t_cids[j] for j in order if t_cids[j] != q and t_cids[j] not in tq[q]["rel"]][: tc["hard_pool"]]
    train_q = sorted(rel)
    fold = tc.get("fold")  # [k, n]: train on the queries outside fold k of n; score the held-out ones (stacking)
    if fold:
        train_q = [q for i, q in enumerate(train_q) if i % fold[1] != fold[0]]
    dq, dpool = load("dev")

    params_t = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_t, lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train_q) // tc["queries"]
    total = steps_per_epoch * tc["epochs"]
    warm = max(1, int(tc["warmup_frac"] * total))
    rng = np.random.default_rng(cfg["seed"])
    log = open(run_dir / "metrics.jsonl", "a")
    best, step, t0 = -1.0, 0, time.time()

    def evaluate(ep):
        nonlocal best
        S, qids, cids = score_split(model, "dev", toks, cfg, device)
        m = micro_f1(topk(S, qids, cids), dq)
        rec = {"epoch": ep, "step": step, "dev_f1": round(m["f1"], 4), "dev_p": round(m["p"], 4),
               "dev_r": round(m["r"], 4), "min": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if m["f1"] > best:
            best = m["f1"]
            torch.save(trainable_state(model), run_dir / "best.pt")

    if tc["epochs"] == 0:
        evaluate(0)
    for ep in range(1, tc["epochs"] + 1):
        random.Random(cfg["seed"] + ep).shuffle(train_q)
        for b in range(steps_per_epoch):
            qs = train_q[b * tc["queries"]:(b + 1) * tc["queries"]]
            cands = []
            for q in qs:
                pos = [rel[q][i] for i in rng.choice(len(rel[q]), min(tc["n_pos"], len(rel[q])), replace=False)]
                neg = [hard[q][i] for i in rng.choice(min(len(hard[q]), tc["hard_pool"]), tc["n_hard"], replace=False)]
                cands += pos + neg
            cands = list(dict.fromkeys(cands))
            for g in opt.param_groups:
                g["lr"] = tc["lr"] * lr_factor(step, total, warm)
            srng = np.random.default_rng(cfg["seed"] * 100003 + step)  # windows/chunks; a teacher sees the same ones
            S = batch_scores(model, qs, cands, toks, "train", tc, device, srng)
            pos = torch.tensor([[c in tq[q]["rel"] for c in cands] for q in qs], device=device)
            own = torch.tensor([[c == q for c in cands] for q in qs], device=device)
            loss = info_nce(S.float(), pos, own, tc["tau"])
            if teacher is not None:
                dc = cfg["distill"]
                St = batch_scores(teacher, qs, cands, toks, "train", tc, device,
                                  np.random.default_rng(cfg["seed"] * 100003 + step), grad=False)
                Td = dc["temperature"] * tc["tau"]
                kl = F.kl_div(F.log_softmax((S.float() / Td).masked_fill(own, -1e4), 1),
                              F.softmax((St.float() / Td).masked_fill(own, -1e4), 1), reduction="batchmean")
                loss = dc["alpha"] * loss + (1 - dc["alpha"]) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_t, tc["grad_clip"])
            opt.step()
            step += 1
            if step % tc["log_every"] == 0:
                rec = {"epoch": ep, "step": step, "loss": round(loss.item(), 4), "lr": opt.param_groups[0]["lr"],
                       "min": round((time.time() - t0) / 60, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
        evaluate(ep)

    model.load_state_dict(torch.load(run_dir / "best.pt"), strict=False)
    out = {"run_name": cfg["run_name"], "params": params, "train_min": round((time.time() - t0) / 60, 1)}
    t1 = time.time()
    Sd, qids, cids = score_split(model, "dev", toks, cfg, device)
    out["dev_encode_sec"] = round(time.time() - t1, 1)
    lxd = lex("dev")
    out["dev_neural"] = {k: round(v, 4) for k, v in micro_f1(topk(Sd, qids, cids), dq).items()}
    f, lam, pen = fit_fusion(Sd, lxd, dq)
    out["dev_fused"] = {"f1": round(f, 4), "lam": lam, "pen": pen}
    out["dev_lexical"] = round(micro_f1(topk(fuse(Sd * 0, lxd, 1.0, pen), qids, cids), dq)["f1"], 4)
    St, tq_ids, tc_ids = score_split(model, "test", toks, cfg, device)
    np.savez_compressed(run_dir / "dev_scores.npz", S=Sd, qids=qids, cids=cids)
    np.savez_compressed(run_dir / "test_scores.npz", S=St, qids=tq_ids, cids=tc_ids)
    out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    out["ckpt_mb"] = round((run_dir / "best.pt").stat().st_size / 1e6, 1)
    if fold:
        Str, trq, trc = score_split(model, "train", toks, cfg, device)
        np.savez_compressed(run_dir / "train_scores.npz", S=Str, qids=trq, cids=trc,
                            heldout=np.array([q not in set(train_q) for q in trq]))
        json.dump(out, open(run_dir / "summary.json", "w"), indent=1)
        print(json.dumps(out), flush=True)
        return
    pool = set(tc_ids)
    out["zip_neural"] = str(write_zip(f"{cfg['run_name']}_neural", topk(St, tq_ids, tc_ids), pool))
    out["zip_fused"] = str(write_zip(f"{cfg['run_name']}_fused", topk(fuse(St, lex("test"), lam, pen), tq_ids, tc_ids), pool))
    json.dump(out, open(run_dir / "summary.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
