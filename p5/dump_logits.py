"""Teacher statute logits on the training split, for the knowledge-distillation experiment.

    python dump_logits.py --run runs/E5_a4_partial                      # -> runs/E5_a4_partial/train_logits.npy
    python dump_logits.py --run runs/E5_a4_partial runs/E2_a1_full      # ensemble teacher -> runs/TEACHER_ens/
"""
import argparse
import importlib.util

import numpy as np
import torch
import yaml

from lsi import P5_ROOT
from lsi.data import TokenDocs
from lsi.model import build_model

_spec = importlib.util.spec_from_file_location("p5_train", P5_ROOT / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)


@torch.no_grad()
def dump_one(run, train, bs):
    cfg = yaml.safe_load(open(run / "config.yaml"))
    model, tok, _, _ = build_model(cfg)
    state = torch.load(run / "best.pt", map_location="cpu")
    missing = set(state) - set(model.state_dict())
    assert not missing, f"checkpoint keys not in model: {sorted(missing)[:5]}"
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()
    out = []
    for i in range(0, len(train), bs):
        toks = [train.tokens(j) for j in range(i, min(len(train), i + bs))]
        out.append(_train.forward_docs(model, toks, cfg["data"], tok.eos_id, device, cfg["train"]["max_chunks"]).cpu().numpy())
        if i % (bs * 400) == 0:
            print(f"  {run.name}: {i:,}/{len(train):,}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="TEACHER_ens", help="folder under runs/ for an ensemble teacher")
    args = ap.parse_args()
    train = TokenDocs("train")
    logits = np.mean([dump_one(P5_ROOT / r, train, args.bs) for r in args.run], 0)
    out = P5_ROOT / args.run[0] if len(args.run) == 1 else P5_ROOT / "runs" / args.out
    
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_logits.npy", logits.astype(np.float32))
    micro = _train.f1_scores(train.labels, (logits > 0).astype(np.uint8))["micro_f1"]
    print(f"saved {logits.shape} teacher logits to {out}; teacher train micro-F1 @0 {micro:.3f}")


if __name__ == "__main__":
    main()
