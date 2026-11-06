"""Score checkpoints on a validation token file (same blocks, same prefix as training eval).

Used for R0: the first Phase-1 checkpoints re-scored on the leak-free val set of the
cleaned corpus, so old and new models are compared on identical data.

    python scripts/09_eval_val.py --val data_clean/tokens/val.bin \
        --ckpt runs/A1_gpt/best1.pt runs/A4_efficient/best.pt --out runs/r0_clean_val.json
"""
import argparse
import json
import math

import _bootstrap  # noqa: F401
from legal_lm.data.dataset import TokenBlocks
from legal_lm.trainer import amp_setup, evaluate, load_checkpoint
from legal_lm.utils import pick_device, save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--eval-batches", type=int, default=None, help="default: the run's train.eval_batches")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    ctx, _ = amp_setup(device, "auto")
    results = {}
    for path in args.ckpt:
        model, cfg, tok = load_checkpoint(path, device)
        t = dict(cfg["train"])
        if args.eval_batches:
            t["eval_batches"] = args.eval_batches
        val = TokenBlocks(args.val, "uint16")
        prefix = max(1, int(t["block_size"] * cfg["eval"]["prefix_frac"]))
        loss = evaluate(model, val, t, tok.vocab_size, ctx, device, prefix)
        results[path] = {"run": cfg["run_name"], "val_loss": loss, "val_ppl": math.exp(loss)}
        print(json.dumps(results[path]))
    if args.out:
        save_json(results, args.out)
