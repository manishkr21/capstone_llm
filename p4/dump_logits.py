"""Teacher logits on the training split, for the knowledge-distillation experiment.

    python dump_logits.py --run runs/E1_a1_full      # -> runs/E1_a1_full/train_logits.npy  (31,677 x 2)
"""
import argparse

import numpy as np
import torch
import yaml

from cj import P4_ROOT
from cj.data import TokenDocs
from cj.model import build_model
import importlib.util

_spec = importlib.util.spec_from_file_location("p4_train", P4_ROOT / "train.py")  # not p2/train.py
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
forward_docs = _train.forward_docs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    run = P4_ROOT / args.run
    cfg = yaml.safe_load(open(run / "config.yaml"))
    model, tok, _, _ = build_model(cfg)
    state = torch.load(run / "best.pt", map_location="cpu")
    missing = set(state) - set(model.state_dict())
    assert not missing, f"checkpoint keys not in model: {sorted(missing)[:5]}"
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()
    train = TokenDocs("train")
    out = []
    for i in range(0, len(train), args.bs):
        toks = [train.tokens(j) for j in range(i, min(len(train), i + args.bs))]
        out.append(forward_docs(model, toks, cfg["data"], tok.eos_id, device, cfg["train"]["max_chunks"]).cpu().numpy())
        if i % (args.bs * 200) == 0:
            print(f"  {i:,}/{len(train):,}", flush=True)
    logits = np.concatenate(out)
    np.save(run / "train_logits.npy", logits)
    acc = ((logits.argmax(1)) == train.labels).mean()
    print(f"saved {logits.shape} teacher logits; teacher train accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
