"""One training loop for all four architectures.

Features: AdamW with decoupled weight decay, linear warm-up + cosine decay,
gradient accumulation, mixed precision (bf16 or fp16 + GradScaler), gradient clipping,
periodic validation on a fixed set of blocks, best/last checkpoints, automatic
resume after a Colab/Kaggle disconnect, optional knowledge distillation from a
teacher checkpoint, a JSONL log of every metric, and a per-minute log of RAM/VRAM/CPU
usage (resources.csv, see legal_lm/monitor.py).
"""
from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from legal_lm.data.dataset import TokenBlocks
from legal_lm.data.tokenizer import LegalTokenizer
from legal_lm.models import LegalLM, ModelConfig, build_model
from legal_lm.monitor import monitor_from_config, summarize_resources
from legal_lm.utils import atomic_torch_save, human, load_json, pick_device, save_json, set_seed

PARAM_MIN, PARAM_MAX = 50_000_000, 300_000_000  # competition rule for the Phase-1 model


# --------------------------------------------------------------------------- helpers
def build_from_config(cfg: dict, vocab_size: int) -> LegalLM:
    mcfg = ModelConfig.from_dict(cfg["model"], vocab_size=vocab_size, max_seq_len=cfg["train"]["block_size"])
    return build_model(mcfg)


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a checkpoint. Returns (model, cfg, tokenizer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tok = LegalTokenizer.from_config(cfg["tokenizer"])
    model = build_from_config(cfg, tok.vocab_size)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, tok


def lr_at(step: int, t: dict) -> float:
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    progress = min(1.0, (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"]))
    return t["min_lr"] + 0.5 * (1 + math.cos(math.pi * progress)) * (t["lr"] - t["min_lr"])


def make_optimizer(model, t: dict, device):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, **extra)


def amp_setup(device, preference: str):
    """Pick the autocast context and whether a GradScaler is needed."""
    if device.type != "cuda" or preference == "float32":
        return nullcontext(), torch.float32
    bf16_ok = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if (preference == "bfloat16" or (preference == "auto" and bf16_ok)) else torch.float16
    return torch.autocast("cuda", dtype=dtype), dtype


def sequence_loss(logits, targets, vocab):
    return F.cross_entropy(logits[..., :vocab].float().reshape(-1, vocab), targets.reshape(-1))


def distill_loss(student_logits, teacher_logits, vocab, temperature):
    """Hinton et al. (2015): KL between softened teacher and student distributions."""
    s = F.log_softmax(student_logits[..., :vocab].float() / temperature, -1).reshape(-1, vocab)
    t = F.log_softmax(teacher_logits[..., :vocab].float() / temperature, -1).reshape(-1, vocab)
    return F.kl_div(s, t, log_target=True, reduction="batchmean") * temperature ** 2


@torch.no_grad()
def evaluate(model, data: TokenBlocks, t: dict, vocab: int, ctx, device, prefix_len: int) -> float:
    """Continuation loss on a fixed set of validation blocks (same blocks every time)."""
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(t["eval_batches"]):
        x = data.sample(t["micro_batch_size"], t["block_size"] + 1, rng).to(device)
        with ctx:
            logits = model(x, prefix_len)
        losses.append(sequence_loss(logits, x[:, prefix_len:], vocab).item())
    model.train()
    return float(np.mean(losses))


# --------------------------------------------------------------------------- main loop
def train(cfg: dict) -> dict:
    t = cfg["train"]
    out_dir = Path(cfg["paths"]["runs_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = pick_device(t.get("device", "auto"))
    ctx, amp_dtype = amp_setup(device, t.get("dtype", "auto"))

    tok = LegalTokenizer.from_config(cfg["tokenizer"])
    vocab = tok.vocab_size
    model = build_from_config(cfg, vocab).to(device)
    counts = model.param_counts()
    print(f"[{cfg['run_name']}] {model.description}")
    print(f"  params: total {human(counts['total'])}  non-embedding {human(counts['non_embedding'])}"
          f"  device {device}  amp {amp_dtype}")
    if t.get("enforce_param_budget", True) and not PARAM_MIN <= counts["total"] <= PARAM_MAX:
        raise ValueError(f"{counts['total']:,} params is outside the allowed 50M-300M range")

    tok_dir = Path(cfg["paths"]["data_dir"]) / "tokens"
    meta = load_json(tok_dir / "meta.json")
    if meta["vocab_size"] != vocab:
        raise ValueError("token bins were made with a different tokenizer; rerun 03_tokenize.py")
    train_data = TokenBlocks(tok_dir / "train.bin", meta["dtype"])
    val_data = TokenBlocks(tok_dir / "val.bin", meta["dtype"])
    L = t["block_size"] + 1
    eval_prefix = max(1, int(t["block_size"] * cfg["eval"]["prefix_frac"]))

    teacher, dcfg = None, cfg.get("distill") or {}
    if dcfg.get("teacher_ckpt"):
        teacher, tcfg, _ = load_checkpoint(dcfg["teacher_ckpt"], device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  distilling from {dcfg['teacher_ckpt']} ({human(teacher.param_counts()['total'])} params)")

    optimizer = make_optimizer(model, t, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    step, best_val, bad_evals = 0, float("inf"), 0

    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if t.get("resume", True) and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"  resumed from {last_path} at step {step}")

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log_file = open(out_dir / "metrics.jsonl", "a")

    def log(record):
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    def save(path, with_optimizer):
        state = {"model": model.state_dict(), "config": cfg, "step": step, "best_val": best_val,
                 "param_counts": counts}
        if with_optimizer:
            state.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        atomic_torch_save(state, path)

    tokens_per_step = t["micro_batch_size"] * t["grad_accum_steps"] * t["block_size"]
    model.train()
    fwd = torch.compile(model) if t.get("compile") else model
    start, t0, tokens_timed = time.time(), time.time(), 0
    monitor = monitor_from_config(cfg, out_dir / "resources.csv", device, phase="train").start()
    try:
        while step < t["max_steps"]:
            rng_np = np.random.default_rng(cfg["seed"] * 100_003 + step)  # reproducible after resume
            rng_py = random.Random(cfg["seed"] * 7 + step)
            lr = lr_at(step, t)
            for group in optimizer.param_groups:
                group["lr"] = lr

            for _ in range(t["grad_accum_steps"]):
                x = train_data.sample(t["micro_batch_size"], L, rng_np).to(device, non_blocking=True)
                p = model.sample_prefix_len(L, rng_py)
                with ctx:
                    logits = fwd(x, p)
                    ce = sequence_loss(logits, x[:, p:], vocab)
                    loss = ce
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(x, p)
                        kd = distill_loss(logits, t_logits, vocab, dcfg.get("temperature", 2.0))
                        loss = dcfg.get("alpha", 0.5) * ce + (1 - dcfg.get("alpha", 0.5)) * kd
                scaler.scale(loss / t["grad_accum_steps"]).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens_timed += tokens_per_step
            monitor.update(step=step)

            if step % t["log_interval"] == 0 or step == 1:
                dt = time.time() - t0
                tps = tokens_timed / max(dt, 1e-9)
                print(f"  step {step:>6}/{t['max_steps']}  loss {ce.item():.4f}  lr {lr:.2e}"
                      f"  grad {grad_norm:.2f}  {tps:,.0f} tok/s")
                log({"type": "train", "step": step, "loss": ce.item(), "total_loss": loss.item(),
                     "lr": lr, "grad_norm": grad_norm, "tokens_per_sec": tps,
                     "tokens_seen": step * tokens_per_step})
                t0, tokens_timed = time.time(), 0

            if step % t["eval_interval"] == 0 or step == t["max_steps"]:
                monitor.update(phase="eval")
                val = evaluate(model, val_data, t, vocab, ctx, device, eval_prefix)
                monitor.update(phase="checkpoint")
                improved = val < best_val
                if improved:
                    best_val, bad_evals = val, 0
                    save(best_path, with_optimizer=False)
                else:
                    bad_evals += 1
                print(f"  [eval] step {step}  val continuation loss {val:.4f}  ppl {math.exp(val):.2f}"
                      f"{'  *best*' if improved else ''}")
                log({"type": "eval", "step": step, "val_loss": val, "val_ppl": math.exp(val),
                     "tokens_seen": step * tokens_per_step})
                save(last_path, with_optimizer=True)
                monitor.update(phase="train")
                patience = t.get("early_stop_patience")
                if patience and bad_evals >= patience:
                    print(f"  early stopping: no improvement in {patience} evaluations")
                    break
    finally:  # also runs after a crash or Ctrl-C, so the last minute is written
        monitor.stop()
        log_file.close()
    elapsed = time.time() - start
    summary = {
        "run_name": cfg["run_name"], "arch": cfg["model"]["arch"], "description": model.description,
        "params_total": counts["total"], "params_non_embedding": counts["non_embedding"],
        "best_val_loss": best_val, "best_val_ppl": math.exp(best_val) if best_val < 50 else None,
        "steps": step, "tokens_seen": step * tokens_per_step, "train_seconds_this_session": elapsed,
        "block_size": t["block_size"], "eval_prefix_len": eval_prefix,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "distilled_from": dcfg.get("teacher_ckpt"),
        "compute": summarize_resources(out_dir / "resources.csv"),  # all sessions of this run
    }
    save_json(summary, out_dir / "summary.json")
    print(f"  done: best val loss {best_val:.4f}; checkpoints in {out_dir}")
    return summary
