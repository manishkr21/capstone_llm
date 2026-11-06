"""Token-level ensemble: several trained runs decode together, averaging next-token log-probabilities.

    python ensemble.py --name ENS1 --runs E2_a1_lora F1_a4_full E3_a3_adapter --set gen.beams=1

All Phase-1 models share one tokenizer, so their distributions over the next summary token can be mixed
at every step, while each model reads the document in its own layout (A1/A3/A4 condensed source, A2
chunks) with its own KV cache.  Writes runs/ens/<name>_{val,test}_pred.json, <name>.json and a zip.
"""
import argparse
import json
import time
import zipfile

import torch
import yaml

from summ import P6_ROOT
from summ.data import Docs, load_splits
from summ.metrics import score
from summ.model import decode
import importlib.util
_s = importlib.util.spec_from_file_location("p6_train", P6_ROOT / "train.py")  # (p2/ also has a train.py on sys.path)
_t = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_t)
load_run, postprocess, sources = _t.load_run, _t.postprocess, _t.sources


def run_split(models, srcs_per_model, g, weights, tok, device, bs, n=None):
    n = n or len(srcs_per_model[0])
    lens = [len(x) if not isinstance(x, list) else sum(map(len, x)) for x in srcs_per_model[0][:n]]
    order = sorted(range(n), key=lambda i: -lens[i])
    out, t0 = [None] * n, time.time()
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        batches = [m.make_batch([sr[i] for i in idx], None, device) for m, sr in zip(models, srcs_per_model)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = decode(models, batches, g, weights)
        for i, r in zip(idx, res):
            out[i] = postprocess(tok.decode(r))
        print(f"    {min(s + bs, n)}/{n}  {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="gen.key=value overrides of configs/base.yaml gen")
    ap.add_argument("--val-docs", type=int, default=300)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    g = yaml.safe_load(open(P6_ROOT / "configs" / "base.yaml"))["gen"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        g[k.split(".", 1)[1]] = yaml.safe_load(v)
    models, cfgs, tok = [], [], None
    for r in args.runs:
        m, tok, cfg = load_run(r, device)
        m.eval()
        models.append(m)
        cfgs.append(cfg)
    w = args.weights
    if w:
        w = [x / sum(w) for x in w]
    bs = g.get("batch_size", 32) // max(1, g.get("beams", 1))
    hv = run_split(models, [sources("val", c) for c in cfgs], g, w, tok, device, bs, args.val_docs)
    val = load_splits()["val"][: args.val_docs]
    sc = score(hv, [" ".join(d["summary"].split()) for d in val])
    out = P6_ROOT / "runs" / "ens"
    out.mkdir(exist_ok=True)
    rep = {"name": args.name, "runs": args.runs, "weights": w, "gen": g, "n_val": len(hv),
           "val_metrics": {k: round(v, 4) for k, v in sc.items()}}
    print(json.dumps(rep), flush=True)
    json.dump([{"id": d["id"], "summary": h} for d, h in zip(val, hv)], open(out / f"{args.name}_val_pred.json", "w"), indent=1)
    json.dump(rep, open(out / f"{args.name}.json", "w"), indent=1)
    if args.no_test:
        return
    ht = run_split(models, [sources("test", c) for c in cfgs], g, w, tok, device, bs)
    ids = Docs("test").ids
    preds = [{"id": i, "summary": h if h.strip() else "The appeal was dismissed."} for i, h in zip(ids, ht)]
    pp = out / f"{args.name}_test_pred.json"
    json.dump(preds, open(pp, "w"), indent=1)
    zp = P6_ROOT / "submissions" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pp, "prediction.json")
    print(f"-> {zp}", flush=True)


if __name__ == "__main__":
    main()
