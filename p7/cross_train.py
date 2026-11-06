"""Cross-encoder reranker on a Phase-1 backbone (reads a citation window and a case passage together).

    python cross_train.py --config configs/c1_a1_cross.yaml

Input: [citation window] " Cited judgment:" [case chunk]; pooled state -> linear -> relevance score. A (query,
case) pair is scored by the max over its ``m`` evidence pairs (cross_prep.py: the window/chunk pairs a bi-encoder
matched best). Listwise loss per query: one gold case against ``n_neg`` non-gold cases of its candidate list.
Selection: dev micro-F1@7 of the stack's top-30 re-ordered by z(stack) + w * z(cross), w in a small grid.
Writes runs/<run_name>/: config.yaml, metrics.jsonl, best.pt, {dev,test}_cross.npz, summary.json and
submissions/<run_name>_<stamp>.zip (the blend at the selected w).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from pcr import P7_ROOT
from pcr import tok as T
from pcr.data import K, load, micro_f1
from pcr.model import build_model, trainable_state
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import set_seed

_s = importlib.util.spec_from_file_location("p7_train", P7_ROOT / "train.py")  # (p2/ also has a train.py on sys.path)
_t = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_t)

WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 1e3)  # 1e3 ~ cross-encoder order alone


def seqs_for(split, q, recs_q, toks, sep, tc):
    """Token sequences of a query's candidate list: list of (case, [seq, ...])."""
    out = []
    for c, pairs in recs_q:
        cs = T.chunks(toks["docs"][c], tc["chunk_len"], tc["eval_chunks"], "spread")
        out.append((c, [np.concatenate([toks["wins"][split][q][w][: tc["win_len"]], sep, cs[j]]) for w, j, _ in pairs]))
    return out


def score_lists(model, items, device, tc, grad=False):
    """items: list of (case, [seq, ...]) -> (n_items,) max over each item's evidence pairs."""
    flat, own = [], []
    for k, (_, ss) in enumerate(items):
        flat += ss
        own += [k] * len(ss)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        s = model.embed(flat, device, tc["max_tokens"], normalize=False).squeeze(-1)
    own = torch.tensor(own, device=device)
    return s.new_full((len(items),), -1e4).scatter_reduce(0, own, s, "amax")


@torch.no_grad()
def score_split(model, split, recs, toks, sep, tc, device):
    model.eval()
    out = {}
    qs = list(recs)
    for b in range(0, len(qs), 16):
        items, idx = [], []
        for q in qs[b:b + 16]:
            it = seqs_for(split, q, recs[q], toks, sep, tc)
            items += it
            idx += [(q, c) for c, _ in it]
        s = score_lists(model, items, device, tc).float().cpu().numpy()
        for (q, c), v in zip(idx, s):
            out.setdefault(q, {})[c] = float(v)
    model.train()
    return out


def stack_scores(stack, split):
    lx = _t.lex(split)
    qids, cids = list(lx["qids"]), list(lx["cids"])
    d = np.load(P7_ROOT / "runs" / "stack" / f"{stack}_{split}.npz")
    out = {}
    for p, (i, j) in zip(d["pred"], d["pairs"]):
        out.setdefault(qids[i], {})[cids[j]] = float(p)
    return out


def blend_rank(base, cross, w):
    """Per query: the cross-scored cases re-ordered by z(base) + w z(cross) (z over the query's list)."""
    ranked = {}
    for q, cs in cross.items():
        c = list(cs)
        b = np.array([base[q][x] for x in c])
        x = np.array([cs[y] for y in c])
        s = (b - b.mean()) / (b.std() + 1e-9) + w * (x - x.mean()) / (x.std() + 1e-9)
        ranked[q] = [c[i] for i in np.argsort(-s)[:K]]
    return ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=str(P7_ROOT / "configs" / "c1_a1_cross.yaml"))
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
    sep = np.array(tok.tk.encode(" Cited judgment:").ids, dtype=np.int32)
    recs = {s: pickle.load(open(P7_ROOT / "cache" / f"cross_{s}.pkl", "rb")) for s in ("train", "dev", "test")}
    tq, _ = load("train")
    dq, _ = load("dev")
    base_dev = stack_scores(cfg["stack"], "dev")
    train_q = [q for q in recs["train"] if tq[q]["rel"]]

    params_t = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_t, lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train_q) // tc["queries"]
    total = steps_per_epoch * tc["epochs"]
    warm = max(1, int(tc["warmup_frac"] * total))
    rng = np.random.default_rng(cfg["seed"])
    log = open(run_dir / "metrics.jsonl", "a")
    best, step, t0 = (-1.0, None), 0, time.time()

    def evaluate(ep):
        nonlocal best
        cross = score_split(model, "dev", recs["dev"], toks, sep, tc, device)
        res = {w: micro_f1(blend_rank(base_dev, cross, w), dq)["f1"] for w in WEIGHTS}
        w = max(res, key=res.get)
        rec = {"epoch": ep, "step": step, "dev_cross_only": round(res[1e3], 4),
               "dev_blend": {str(k): round(v, 4) for k, v in res.items()}, "min": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if res[w] > best[0]:
            best = (res[w], w)
            torch.save(trainable_state(model), run_dir / "best.pt")

    for ep in range(1, tc["epochs"] + 1):
        random.Random(cfg["seed"] + ep).shuffle(train_q)
        for b in range(steps_per_epoch):
            items, groups = [], []
            for q in train_q[b * tc["queries"]:(b + 1) * tc["queries"]]:
                lst = recs["train"][q]
                pos = [r for r in lst if r[0] in tq[q]["rel"]]
                neg = [r for r in lst if r[0] not in tq[q]["rel"] and r[0] != q]
                pick = [pos[rng.integers(len(pos))]] + [neg[i] for i in rng.choice(len(neg), min(tc["n_neg"], len(neg)), replace=False)]
                items += seqs_for("train", q, pick, toks, sep, tc)
                groups.append(len(pick))
            for g in opt.param_groups:
                g["lr"] = tc["lr"] * _t.lr_factor(step, total, warm)
            s = score_lists(model, items, device, tc, grad=True).float()
            loss, o = 0.0, 0
            for n in groups:  # gold case first in every group
                loss = loss + F.cross_entropy(s[o:o + n].unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
                o += n
            loss = loss / len(groups)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_t, tc["grad_clip"])
            opt.step()
            step += 1
            if step % tc["log_every"] == 0:
                rec = {"epoch": ep, "step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if tc.get("eval_every") and step % tc["eval_every"] == 0:
                evaluate(ep)
        evaluate(ep)

    model.load_state_dict(torch.load(run_dir / "best.pt"), strict=False)
    out = {"run_name": cfg["run_name"], "params": params, "train_min": round((time.time() - t0) / 60, 1),
           "dev_best": round(best[0], 4), "w": best[1], "dev_stack": round(micro_f1(blend_rank(base_dev, {q: dict.fromkeys(base_dev[q], 0.0) for q in recs["dev"]}, 0.0), dq)["f1"], 4)}
    for split in ("dev", "test"):
        cross = score_split(model, split, recs[split], toks, sep, tc, device)
        pickle.dump(cross, open(run_dir / f"{split}_cross.pkl", "wb"))
    base_test = stack_scores(cfg["stack"], "test")
    cross_test = pickle.load(open(run_dir / "test_cross.pkl", "rb"))
    _, pool = load("test")
    out["zip"] = str(_t.write_zip(cfg["run_name"], blend_rank(base_test, cross_test, best[1]), set(pool)))
    json.dump(out, open(run_dir / "summary.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
