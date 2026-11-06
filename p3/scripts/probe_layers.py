"""How much rhetorical-role information does each Phase-1 backbone (and each of its layers) hold?

    python scripts/probe_layers.py --ckpt runs/A4_efficient_clean_1B/best.pt --attention causal

Frozen backbone, sentences packed into 512-token windows as in the RR model, mean-pooled hidden
state of every layer.  A linear softmax classifier (class-balanced loss) is trained per layer on the
same 5 document-level folds as train.py, and the out-of-fold macro-F1 over the 90 documents is
reported.  No context model, no CRF: it measures the sentence representation alone.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rr import P3_ROOT  # noqa: E402
from rr.data import cv_folds, encode_doc, load_labels, load_split, pack, window_batch  # noqa: E402
from rr.metrics import macro_scores  # noqa: E402
from lner.backbones import Backbone, load_phase1  # noqa: E402


@torch.no_grad()
def layer_vectors(bb, tok, docs, l2i, device, window=512):
    """{doc_id: (n_layers+1, n_sents, d)}; layer 0 = embeddings, last = final (normed) output."""
    blocks = bb.all_blocks
    caught = []
    if bb.arch == "efficient":  # the Phase-2 wrapper calls attn/mlp directly: read the residual stream entering
        # each block (input of norm1) -> layer 0 = embeddings, layer l = output of block l-1
        hooks = [b.norm1.register_forward_pre_hook(lambda m, i: caught.append(i[0])) for b in blocks]
    else:  # block outputs; layer 0 is then block 1's output
        hooks = [b.register_forward_hook(lambda m, i, o: caught.append(o[0] if isinstance(o, tuple) else o))
                 for b in blocks]
    out = {}
    for d in docs:
        e = encode_doc(tok, d, 128, l2i)
        groups = pack(e["sents"], "window", window)
        per_layer = None
        for g0 in range(0, len(groups), 32):
            ids, own = window_batch(e["sents"], groups[g0:g0 + 32], tok.eos_id)
            ids, own = ids.to(device), own.to(device)
            caught.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                final = bb(ids)
            hs = [h.float() for h in caught[-len(blocks):]] + [final.float()]
            n = int(own.max()) + 1
            o = own.reshape(-1)
            keep = o >= 0
            cnt = torch.zeros(n, device=device).index_add_(0, o[keep], torch.ones_like(o[keep], dtype=torch.float))
            vecs = torch.stack([torch.zeros(n, h.size(-1), device=device).index_add_(0, o[keep], h.reshape(-1, h.size(-1))[keep])
                                / cnt[:, None] for h in hs])
            per_layer = vecs if per_layer is None else torch.cat([per_layer, vecs], 1)
        out[d["id"]] = per_layer.cpu()
    for h in hooks:
        h.remove()
    return out


def probe(X, y, Xd, n_cls, device, epochs=300):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xd = ((X - mu) / sd).to(device), ((Xd - mu) / sd).to(device)
    y = y.to(device)
    w = (torch.bincount(y, minlength=n_cls).float() + 1).reciprocal().sqrt()
    lin = torch.nn.Linear(X.size(1), n_cls).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(lin(X), y, weight=w / w.mean()).backward()
        opt.step()
    return lin(Xd).argmax(-1).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Phase-1 checkpoint, relative to capstone_llm/")
    ap.add_argument("--attention", default="causal")
    ap.add_argument("--encdec-output", default="decoder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--random", action="store_true", help="same architecture, random weights (control)")
    args = ap.parse_args()
    device = torch.device("cuda")
    data_dir = P3_ROOT / "../capstone_llm/data/hf/Phase_3_RR"
    labels = load_labels(data_dir)
    l2i = {l: i for i, l in enumerate(labels)}
    docs = load_split(data_dir, "train") + load_split(data_dir, "dev")
    folds = cv_folds(docs, 5, 0)
    lm, tok = load_phase1(args.ckpt)
    if args.random:
        torch.manual_seed(0)
        lm.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    bb = Backbone(lm, args.attention, None, args.encdec_output).to(device).eval()
    V = layer_vectors(bb, tok, docs, l2i, device)
    n_layers = next(iter(V.values())).size(0)
    gold = {d["id"]: [l2i[l] for l in d["labels"]] for d in docs}
    res = {}
    for L in range(n_layers):
        preds = {}
        for dev_ids in folds:
            tr = [d for d in docs if d["id"] not in set(dev_ids)]
            X = torch.cat([V[d["id"]][L] for d in tr])
            y = torch.tensor([t for d in tr for t in gold[d["id"]]])
            for d in docs:
                if d["id"] in set(dev_ids):
                    preds[d["id"]] = probe(X, y, V[d["id"]][L], len(labels), device).tolist()
        g = [labels[t] for d in docs for t in gold[d["id"]]]
        p = [labels[t] for d in docs for t in preds[d["id"]]]
        res[L] = round(macro_scores(g, p)["macro_f1"], 4)
        print(f"  layer {L:>2}: OOF macro-F1 {res[L]:.4f}", flush=True)
    best = max(res, key=res.get)
    print(json.dumps({"ckpt": args.ckpt, "attention": args.attention, "best_layer": best, "best": res[best],
                      "last": res[n_layers - 1]}))
    if args.out:
        json.dump({"ckpt": args.ckpt, "attention": args.attention, "per_layer": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
