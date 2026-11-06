"""Embed every chunk of every document with a fine-tuned run's encoder (for the cached document models).

    python embed_chunks.py --run runs/V1_e6_16ch            # -> cache/feats/V1_e6_16ch_{train,val,test}.npz

The encoder (Phase-1 backbone + the run's adaptation, as saved in best.pt) is applied to the last
``--max-chunks`` 512-token chunks of each document (all of them for 95%+ of judgments) and each chunk is
mean-pooled, exactly as in training.  Output per split: vecs (total_chunks, d) float16 and offs (n_docs+1).
"""
import argparse

import numpy as np
import torch
import yaml

from cj import P4_ROOT
from cj.data import TokenDocs, batch_chunks, select_chunks
from cj.model import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-chunks", type=int, default=48)
    ap.add_argument("--bs", type=int, default=64, help="chunks per forward")
    args = ap.parse_args()
    run = P4_ROOT / args.run
    cfg = yaml.safe_load(open(run / "config.yaml"))
    model, tok, _, _ = build_model(cfg)
    state = torch.load(run / "best.pt", map_location="cpu")
    assert not (set(state) - set(model.state_dict()))
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda")
    enc = model.encoder.to(device).eval()
    out_dir = P4_ROOT / "cache" / "feats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        data = TokenDocs(split)
        vecs, offs = [], [0]
        pending = []

        def flush():
            ids, mask, _ = batch_chunks([pending], tok.eos_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vecs.append(enc(ids.to(device), mask.to(device)).float().cpu().numpy().astype(np.float16))
            pending.clear()

        for i in range(len(data)):
            chunks = select_chunks(data.tokens(i), "last", args.max_chunks, cfg["data"]["chunk_len"])
            offs.append(offs[-1] + len(chunks))
            for c in chunks:
                pending.append(c)
                if len(pending) >= args.bs:
                    flush()
            if i % 5000 == 0:
                print(f"  {split} {i:,}/{len(data):,}", flush=True)
        if pending:
            flush()
        V = np.concatenate(vecs)
        assert V.shape[0] == offs[-1]
        np.savez(out_dir / f"{run.name}_{split}.npz", vecs=V, offs=np.asarray(offs, dtype=np.int64))
        print(f"  saved {split}: {len(data):,} docs, {V.shape[0]:,} chunks x {V.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
