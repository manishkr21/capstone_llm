"""Where do the strict errors come from? Gold/prediction matching by overlap.

    python scripts/error_analysis.py runs/E2_a4_lorma --set infer.propagate=10

Every gold entity is one of:
  exact          start, end and label match (true positive)
  wrong_label    same span, different label
  boundary       overlaps a prediction with the same label, but start or end differ
  bound+label    overlaps a prediction with another label and other boundaries
  missed         no prediction overlaps it
Every prediction that overlaps no gold entity is 'spurious'.
"micro-F1 if fixed" = micro-F1 when that error type is turned into true positives.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from lner import P2_ROOT  # noqa: E402
from lner.data import bio_tags, load_labels, load_train, split_dev  # noqa: E402
from lner.engine import load_run, predict_docs, prepare_docs  # noqa: E402
from legal_lm.config import apply_overrides  # noqa: E402


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", choices=["dev", "train"], default="dev")
    args = ap.parse_args()
    run_dir = (P2_ROOT / args.run).resolve()
    data_dir = P2_ROOT / yaml.safe_load(open(run_dir / "config.yaml"))["data"]["dir"]
    labels = load_labels(data_dir)
    tags = bio_tags(labels)
    device = torch.device("cuda")
    model, tok, cfg = load_run(run_dir, tags, device)
    apply_overrides(cfg, args.set)
    texts, gold = load_train(data_dir)
    train_ids, dev_ids = split_dev(list(texts), cfg["data"]["n_dev"], cfg["data"]["split_seed"])
    ids = dev_ids if args.split == "dev" else train_ids
    docs = prepare_docs(tok, {k: texts[k] for k in ids}, None, cfg["data"]["presplit"], None,
                        cfg["data"].get("case_norm", False))
    preds = predict_docs(model, docs, tags, cfg["infer"], device)

    cat, per_label, conf, region = Counter(), defaultdict(Counter), Counter(), Counter()
    spurious, spur_label = 0, Counter()
    n_pred = 0
    for doc in ids:
        G = [tuple(g[:3]) for g in gold[doc]]
        P = [tuple(p[:3]) for p in preds[doc]]
        n_pred += len(P)
        pset = set(P)
        for g in G:
            ov = [p for p in P if overlap(g, p)]
            if g in pset:
                c = "exact"
            elif any(p[:2] == g[:2] for p in ov):
                c = "wrong_label"
                conf[(g[2], next(p[2] for p in ov if p[:2] == g[:2]))] += 1
            elif any(p[2] == g[2] for p in ov):
                c = "boundary"
            elif ov:
                c = "bound+label"
                conf[(g[2], ov[0][2])] += 1
            else:
                c = "missed"
            cat[c] += 1
            per_label[g[2]][c] += 1
            region[("header" if g[0] < 1500 else "body", c == "exact")] += 1
        for p in P:
            if not any(overlap(p, g) for g in G):
                spurious += 1
                spur_label[p[2]] += 1

    n_gold = sum(cat.values())
    tp = cat["exact"]

    def f1(tp_, n_pred_):
        p, r = tp_ / n_pred_, tp_ / n_gold
        return 2 * p * r / (p + r)

    print(f"{args.run} on {args.split} ({len(ids)} docs): gold {n_gold}, predicted {n_pred}, "
          f"micro-F1 {f1(tp, n_pred):.4f}")
    print(f"\n{'gold entities':14s} {'count':>6s} {'share':>6s}   micro-F1 if fixed")
    for c in ("exact", "wrong_label", "boundary", "bound+label", "missed"):
        fixed = "" if c == "exact" else f"{f1(tp + cat[c], n_pred):.4f}"
        print(f"{c:14s} {cat[c]:6d} {cat[c] / n_gold:6.1%}   {fixed}")
    print(f"{'spurious pred':14s} {spurious:6d}          {f1(tp, n_pred - spurious):.4f}  (if removed)")
    print(f"\nall boundary errors fixed (boundary + bound+label as their own label) -> "
          f"{f1(tp + cat['boundary'], n_pred):.4f}; plus all spurious removed -> "
          f"{f1(tp + cat['boundary'], n_pred - spurious):.4f}")

    print(f"\n{'label':10s} {'gold':>5s} " + " ".join(f"{c:>11s}" for c in ("exact", "wrong_label", "boundary", "bound+label", "missed")) + "  spurious")
    for l in labels:
        row = per_label[l]
        n = sum(row.values())
        print(f"{l:10s} {n:5d} " + " ".join(f"{row[c] / max(n, 1):11.0%}" for c in ("exact", "wrong_label", "boundary", "bound+label", "missed")) + f"  {spur_label[l]:8d}")
    print("\nmost frequent label confusions (gold -> predicted):",
          ", ".join(f"{a}->{b} {n}" for (a, b), n in conf.most_common(8)))
    for r in ("header", "body"):
        n = region[(r, True)] + region[(r, False)]
        print(f"{r:6s} (first 1500 chars = header): {n} gold, {region[(r, True)] / max(n, 1):.0%} exact")


if __name__ == "__main__":
    main()
