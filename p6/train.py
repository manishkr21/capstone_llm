"""Train one Phase-6 summarization experiment, then decode the validation and test documents.

    python train.py --config configs/base.yaml configs/a4_lorma.yaml
    python train.py --config configs/base.yaml configs/a4_lorma.yaml --set train.epochs=1 run_name=dbg

Loss: token cross-entropy on the summary (optional label smoothing); with ``kd.teacher`` set, the
student also matches the teacher's temperature-softened next-token distribution on the same summary
(word-level knowledge distillation, Hinton et al. 2015 / Kim & Rush 2016):
    loss = alpha * CE + (1 - alpha) * T^2 * KL(teacher_T || student_T)
Selection: teacher-forced validation loss after every epoch; the best state is kept.
Writes runs/<run_name>/: config.yaml, metrics.jsonl, best.pt, val_pred.json, test_pred.json,
summary.json, and submissions/<run_name>_<stamp>.zip (list format [{"id", "summary"}]).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from summ import P6_ROOT
from summ.data import CACHE, Docs, build_cache, condense, load_splits
from summ.metrics import score
from summ.model import build_summarizer, make_controller, trainable_state
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import set_seed


# ---------------------------------------------------------------------- sources
def _condense_one(a):
    sents, budget, strategy, head_frac = a
    return condense(sents, budget, strategy, head_frac).astype(np.int32)


def sources(split: str, cfg: dict):
    """Condensed source token arrays for every document of ``split`` (A2: lists of chunks)."""
    m, d = cfg["model"], cfg["data"]
    budget = m["src_len"] * m.get("n_chunks", 1)
    path = CACHE / f"src_{split}_{d['strategy']}_{budget}_{d.get('head_frac', 0.15)}.pkl"
    if path.exists():
        out = pickle.load(open(path, "rb"))
    else:
        docs = Docs(split)
        with Pool(24) as pool:
            out = pool.map(_condense_one, [(docs.sents(i), budget, d["strategy"], d.get("head_frac", 0.15))
                                           for i in range(len(docs))], chunksize=16)
        pickle.dump(out, open(path, "wb"))
    if "n_chunks" in m:  # A2: fusion-in-decoder chunks
        L = m["src_len"]
        out = [[x[i:i + L] for i in range(0, max(1, len(x)), L)] for x in out]
    return out


def postprocess(text: str) -> str:
    text = " ".join(text.split())
    if text and text[-1] not in ".?!\"')":
        cut = max(text.rfind(". "), text.rfind("? "))
        if cut > len(text) * 0.6:
            text = text[: cut + 1]
    return text


@torch.no_grad()
def decode_split(model, tok, srcs, g, device, bs=None, log_every=5):
    model.eval()
    bs = bs or g.get("batch_size", 32)
    order = sorted(range(len(srcs)), key=lambda i: -(sum(len(c) for c in srcs[i]) if isinstance(srcs[i], list) else len(srcs[i])))
    out, t0 = [None] * len(srcs), time.time()
    for n, s in enumerate(range(0, len(order), bs)):
        idx = order[s:s + bs]
        b = model.make_batch([srcs[i] for i in idx], None, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = model.generate(b, g)
        for i, r in zip(idx, res):
            out[i] = postprocess(tok.decode(r))
        if n % log_every == 0:
            print(f"    decoded {min(s + bs, len(order))}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    return out


@torch.no_grad()
def val_loss(model, srcs, tgts, device, bs):
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(srcs), bs):
        b = model.make_batch(srcs[s:s + bs], tgts[s:s + bs], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.loss(b)
        k = int((b["labels"] != -100).sum())
        tot += float(loss) * k
        n += k
    return tot / n


def load_run(run_name, device="cuda"):
    """Rebuild a trained run (Phase-1 weights + adaptation) and load its best trainable state."""
    rd = P6_ROOT / "runs" / run_name
    cfg = yaml.safe_load(open(rd / "config.yaml"))
    model, tok, _ = build_summarizer(cfg)
    missing = model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return model.to(device), tok, cfg


def write_submission(name, ids, summaries, tag=""):
    preds = [{"id": i, "summary": s if s.strip() else "The appeal was disposed of."} for i, s in zip(ids, summaries)]
    rd = P6_ROOT / "runs" / name
    pp = rd / f"test_pred{tag}.json"
    json.dump(preds, open(pp, "w"), indent=1)
    zp = P6_ROOT / "submissions" / f"{name}{tag}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pp, "prediction.json")
    return zp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=str(P6_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--skip-train", action="store_true", help="only decode an existing run")
    args = ap.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda")
    name = cfg["run_name"]
    rd = P6_ROOT / "runs" / name
    rd.mkdir(parents=True, exist_ok=True)
    tc, g = cfg["train"], cfg["gen"]

    model, tok, info = build_summarizer(cfg)
    if cfg["model"].get("init_run"):  # continue from another run's fine-tuned weights (e.g. a longer source)
        sd = torch.load(P6_ROOT / "runs" / cfg["model"]["init_run"] / "best.pt", map_location="cpu")
        pe = sd.get("lm.pos_emb.weight")
        if pe is not None and pe.shape != model.lm.pos_emb.weight.shape:  # A1: stretch the learned table again
            sd["lm.pos_emb.weight"] = F.interpolate(pe.T[None], size=model.lm.pos_emb.weight.size(0),
                                                    mode="linear", align_corners=True)[0].T
        model.load_state_dict(sd, strict=False)
    print(json.dumps(info), flush=True)
    build_cache(tok)
    tr, va, te = Docs("train"), Docs("val"), Docs("test")
    if tc.get("max_train_docs"):
        keep = list(range(min(len(tr), tc["max_train_docs"])))
    else:
        keep = list(range(len(tr)))
    s_tr, s_va, s_te = sources("train", cfg), sources("val", cfg), sources("test", cfg)
    t_tr, t_va = [tr.summary(i) for i in range(len(tr))], [va.summary(i) for i in range(len(va))]
    model.to(device)

    teacher = None
    if cfg.get("kd", {}).get("teacher"):
        teacher, _, tcfg = load_run(cfg["kd"]["teacher"], device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        ts_tr = sources("train", tcfg)
        print(f"teacher {cfg['kd']['teacher']} loaded", flush=True)

    if not args.skip_train:
        yaml.safe_dump(cfg, open(rd / "config.yaml", "w"), sort_keys=False)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.01), betas=(0.9, 0.98))
        mb, acc = tc["micro_batch"], tc["grad_accum"]
        steps_per_epoch = math.ceil(len(keep) / (mb * acc))
        total = steps_per_epoch * tc["epochs"]
        warm = tc.get("warmup", 100)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
        ctrl = make_controller(model, cfg["peft"], total)
        log = open(rd / "metrics.jsonl", "w")
        best, step, t0 = float("inf"), 0, time.time()
        rng = random.Random(cfg.get("seed", 0))
        kd = cfg.get("kd", {})
        for ep in range(tc["epochs"]):
            model.train()
            order = keep[:]
            rng.shuffle(order)
            for s0 in range(0, len(order), mb * acc):
                group = order[s0:s0 + mb * acc]
                for a in range(0, len(group), mb):
                    idx = group[a:a + mb]
                    b = model.make_batch([s_tr[i] for i in idx], [t_tr[i] for i in idx], device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, logits = model.loss(b, tc.get("label_smoothing", 0.0))
                        if teacher is not None:
                            tb = teacher.make_batch([ts_tr[i] for i in idx], [t_tr[i] for i in idx], device)
                            with torch.no_grad():
                                tl = teacher.target_logits(tb)
                            V, T = model.cfg.vocab_size, kd.get("T", 2.0)
                            m = b["labels"] != -100
                            ls = F.log_softmax(logits[..., :V][m].float() / T, -1)
                            lt = F.log_softmax(tl[..., :V][m].float() / T, -1)
                            kl = (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
                            loss = kd.get("alpha", 0.5) * loss + (1 - kd.get("alpha", 0.5)) * kl
                        if ctrl is not None:
                            loss = loss + ctrl.penalty()
                    (loss * len(idx) / len(group)).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if ctrl is not None:
                    ctrl.step(step)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    rec = {"step": step, "epoch": ep, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0],
                           "min": round((time.time() - t0) / 60, 1)}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(rec, flush=True)
            vl = val_loss(model, s_va, t_va, device, tc.get("eval_batch", 8))
            rec = {"epoch": ep + 1, "step": step, "val_loss": round(vl, 4), "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            if vl < best:
                best = vl
                torch.save(trainable_state(model), rd / "best.pt")
        train_min = round((time.time() - t0) / 60, 1)
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    else:
        train_min, peak, best = None, None, None
    del teacher
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"), strict=False)
    model.to(device)

    nv = cfg["gen"].get("val_docs", 300)
    t1 = time.time()
    hyp_va = decode_split(model, tok, s_va[:nv], g, device)
    dec_s_per_doc = (time.time() - t1) / nv
    hyp_te = decode_split(model, tok, s_te, g, device)
    del model
    torch.cuda.empty_cache()
    refs = [" ".join(d["summary"].split()) for d in load_splits()["val"]][:nv]
    tag = f"_{g['tag']}" if args.skip_train else ""
    json.dump([{"id": i, "summary": h} for i, h in zip(va.ids, hyp_va)], open(rd / f"val_pred{tag}.json", "w"), indent=1)
    sc = score(hyp_va, refs)
    zp = write_submission(name, te.ids, hyp_te, tag)
    summary = {"run": name, **info, "val_loss": best, "val_metrics": {k: round(v, 4) for k, v in sc.items()},
               "train_min": train_min, "peak_gb": peak, "decode_s_per_doc": round(dec_s_per_doc, 3),
               "gen": g, "zip": zp.name}
    json.dump(summary, open(rd / ("summary.json" if not args.skip_train else f"summary_{g.get('tag', 'redecode')}.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
