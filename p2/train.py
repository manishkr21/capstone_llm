"""Fine-tune one Phase-1 model for L-NER (one experiment = one config).

    python train.py --config configs/e1_a3_full.yaml
    python train.py --config configs/e2_a4_lorma.yaml --set train.lr=5e-4 run_name=E2_lr5e-4
    python train.py --config configs/e1_a3_full.yaml --all-data   # no dev split, keep last epoch

Writes runs/<run_name>/: config.yaml, metrics.jsonl, best.pt (trainable params only),
summary.json (dev scores, per-label F1, params, time, memory) and dev_prediction.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
import yaml

from lner import P2_ROOT
from lner.data import bio_tags, load_labels, load_train, split_dev
from lner.engine import (batch_tensors, build_run_model, load_run, predict_docs, prepare_docs,
                         train_windows, trainable_state)
from lner.metrics import strict_scores
from legal_lm.config import add_config_args, config_from_args
from legal_lm.utils import save_json, set_seed


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def token_loss(logits, y, class_w, tc):
    """Weighted cross-entropy, or focal loss (Lin et al., 2017): (1 - p_y)^gamma * CE."""
    logp = F.log_softmax(logits.view(-1, logits.size(-1)), -1).gather(1, y.view(-1, 1)).squeeze(1)
    w = class_w[y.view(-1)]
    if tc.get("loss", "ce") == "focal":
        w = w * (1 - logp.exp()).pow(tc.get("focal_gamma", 2.0))
    return -(w * logp).sum() / class_w[y.view(-1)].sum()


def layer_depth(name: str, bb) -> int:
    """0 = embeddings, 1..L = transformer blocks (A2: encoder then decoder), L + 1 = final norms."""
    parts = name.split(".")
    for key, offset in (("blocks", 0), ("encoder", 0), ("decoder", len(bb.lm.encoder) if bb.arch == "encdec" else 0)):
        if key in parts:
            i = parts.index(key)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return offset + int(parts[i + 1]) + 1
    if "tok_emb" in name or "pos_emb" in name:
        return 0
    return len(bb.all_blocks) + 1


def main():
    ap = argparse.ArgumentParser()
    add_config_args(ap, default=str(P2_ROOT / "configs" / "e1_a3_full.yaml"))
    ap.add_argument("--all-data", action="store_true", help="train on all 70 docs (no dev), save last epoch")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.all_data:
        cfg["run_name"] += "_all"
        cfg["data"]["n_dev"] = 0
    tc, ic = cfg["train"], cfg["infer"]
    for k in ("lr", "head_lr", "weight_decay"):  # YAML reads "1e-3" (no dot) as a string
        tc[k] = float(tc[k])
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = P2_ROOT / cfg.get("runs_dir", "runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data_dir = P2_ROOT / cfg["data"]["dir"]
    labels = load_labels(data_dir)
    tags = bio_tags(labels)
    tag2id = {t: i for i, t in enumerate(tags)}
    texts, gold = load_train(data_dir)
    train_ids, dev_ids = split_dev(list(texts), cfg["data"]["n_dev"], cfg["data"]["split_seed"])
    if cfg["data"].get("max_train_docs"):  # learning curve: same dev docs, fewer training docs
        train_ids = train_ids[:cfg["data"]["max_train_docs"]]

    model, tok, info = build_run_model(cfg, tags)
    model.to(device)
    print(f"[{cfg['run_name']}] {model.backbone.arch} / {cfg['peft']['method']}: "
          f"{info['trainable_params']:,} of {info['total_params']:,} params trainable ({info['trainable_pct']}%)")
    assert tc["window"] <= model.backbone.max_len and ic["window"] <= model.backbone.max_len

    presplit, case_norm = cfg["data"]["presplit"], cfg["data"].get("case_norm", False)
    train_docs = prepare_docs(tok, {k: texts[k] for k in train_ids}, gold, presplit, tag2id, case_norm)
    dev_docs = prepare_docs(tok, {k: texts[k] for k in dev_ids}, gold, presplit, tag2id, case_norm)
    dev_gold = {k: gold[k] for k in dev_ids}

    teacher = None
    if cfg.get("distill"):
        dc = cfg["distill"]
        teacher, _, tcfg = load_run(P2_ROOT / dc["teacher_run"], tags, device)
        assert tcfg["data"]["presplit"] == presplit and tcfg["data"].get("case_norm", False) == case_norm, \
            "teacher and student must tokenize identically"
        assert model.crf is None, "distillation uses the softmax head"
        for p in teacher.parameters():
            p.requires_grad = False
        t_info = sum(p.numel() for p in teacher.parameters())
        print(f"  distilling from {dc['teacher_run']} ({t_info:,} params), alpha={dc['alpha']} T={dc['temperature']}")

    head_params = list(model.head.parameters()) + (list(model.crf.parameters()) if model.crf else [])
    head_ids = {id(p) for p in head_params}
    body = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": tc["head_lr"], "weight_decay": 0.0}]
    # layer-wise LR decay: block l of L gets lr * llrd^(L + 1 - depth); embeddings are deepest-decayed
    n_layers, decay = len(model.backbone.all_blocks), tc.get("llrd") or 1.0
    by_lr = {}
    for n, p in body:
        lr = tc["lr"] * decay ** (n_layers + 1 - layer_depth(n, model.backbone))
        by_lr.setdefault((lr, p.dim() >= 2), []).append(p)
    for (lr, is_matrix), ps in sorted(by_lr.items()):
        groups.append({"params": ps, "lr": lr, "weight_decay": tc["weight_decay"] if is_matrix else 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.98), eps=1e-6)

    class_w = torch.ones(len(tags), device=device)
    if tc.get("label_weight") == "inv_sqrt":  # rare tags weigh more: sqrt(n_O / n_tag), capped
        counts = torch.bincount(torch.tensor([t for d in train_docs for t in d["tags"]]), minlength=len(tags)).float()
        class_w = (counts[0] / counts.clamp(min=1)).sqrt().clamp(max=tc.get("label_weight_cap", 10.0)).to(device)
    class_w[0] = tc.get("o_weight", 1.0)
    rng = random.Random(cfg["seed"])
    steps_per_epoch = math.ceil(len(train_windows(train_docs, tc["window"], tc["stride"], random.Random(0)))
                                / tc["batch_size"])
    total = steps_per_epoch * tc["epochs"]
    warmup = int(tc.get("warmup_frac", 0.1) * total)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    best, best_epoch, bad, step, t0, n_tok = -1.0, -1, 0, 0, time.time(), 0
    log = open(run_dir / "metrics.jsonl", "w")
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        wins = train_windows(train_docs, tc["window"], tc["stride"], rng)
        run_loss = 0.0
        for b in range(0, len(wins), tc["batch_size"]):
            ids, y = batch_tensors(wins[b:b + tc["batch_size"]], device)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, total, warmup)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(ids).float()
                loss = model.crf.nll(logits, y) if model.crf else token_loss(logits, y, class_w, tc)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(ids).float()
                    T = dc["temperature"]
                    kd = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(t_logits / T, -1),
                                  log_target=True, reduction="none").sum(-1).mean() * T * T
                    loss = dc["alpha"] * loss + (1 - dc["alpha"]) * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["grad_clip"])
            opt.step()
            step += 1
            n_tok += ids.numel()
            run_loss += loss.item()
        rec = {"epoch": epoch, "step": step, "train_loss": round(run_loss / steps_per_epoch, 4),
               "elapsed_s": round(time.time() - t0, 1)}
        if dev_docs:
            scores = strict_scores(dev_gold, predict_docs(model, dev_docs, tags, ic, device), labels)
            rec.update({k: v for k, v in scores.items() if k != "per_label"})
            if scores["macro_f1"] > best:
                best, best_epoch, bad = scores["macro_f1"], epoch, 0
                torch.save(trainable_state(model), run_dir / "best.pt")
            else:
                bad += 1
        print("  " + json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if tc.get("patience") and bad >= tc["patience"]:
            print(f"  early stop: no dev improvement for {bad} epochs")
            break
    train_s = time.time() - t0
    if not dev_docs:
        torch.save(trainable_state(model), run_dir / "best.pt")

    summary = {"run": cfg["run_name"], "arch": model.backbone.arch, "method": cfg["peft"]["method"],
               "distilled_from": (cfg.get("distill") or {}).get("teacher_run"), **info,
               "best_epoch": best_epoch, "train_minutes": round(train_s / 60, 2),
               "train_tokens_per_s": round(n_tok / train_s),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
               "checkpoint_mb": round((run_dir / "best.pt").stat().st_size / 1e6, 1)}
    if dev_docs:
        state = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(state, strict=False)
        t1 = time.time()
        preds = predict_docs(model, dev_docs, tags, ic, device)
        summary["dev_infer_s"] = round(time.time() - t1, 2)
        summary["dev"] = strict_scores(dev_gold, preds, labels)
        save_json(preds, run_dir / "dev_prediction.json")
        print(f"  best dev macro-F1 {best:.4f} at epoch {best_epoch}")
    save_json(summary, run_dir / "summary.json")


if __name__ == "__main__":
    main()
