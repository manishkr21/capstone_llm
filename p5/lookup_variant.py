"""NOT USED (duplicate texts carry the same label set only 35% of the time in train vs dev). Optional post-processing: test documents whose text occurs verbatim in train.json or dev.json get the
statute labels of that labelled copy (union over copies); all other predictions are unchanged.

    python lookup_variant.py --pred runs/submissions_meta/ENS2_greedy4_prediction.json --name ENS2_greedy4_lookup

On the validation split this rule can be checked against train copies (dev documents that also occur in train).
"""
import argparse
import json
import time
import zipfile
from collections import defaultdict

from lsi import P5_ROOT
from lsi.data import DATA, text_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    known = defaultdict(set)
    train = json.load(open(DATA / "Train" / "train.json"))
    dev = json.load(open(DATA / "Train" / "dev.json"))
    for r in train + dev:
        known[text_hash(r["text"])] |= set(r["labels"])
    # sanity check: how well does "copy the labels of a train duplicate" match the dev labels?
    tr_only = defaultdict(set)
    for r in train:
        tr_only[text_hash(r["text"])] |= set(r["labels"])
    hits = [(set(r["labels"]), tr_only[text_hash(r["text"])]) for r in dev if text_hash(r["text"]) in tr_only]
    exact = sum(a == b for a, b in hits)
    print(f"dev docs with a train copy: {len(hits)}; identical label sets: {exact} ({exact / max(1, len(hits)):.1%})")
    test = {r["id"]: text_hash(r["text"]) for r in json.load(open(DATA / "Test" / "test.json"))}
    preds = json.load(open(args.pred))
    changed = 0
    for p in preds:
        h = test[p["id"]]
        if h in known:
            p["pred"] = sorted(int(x) for x in known[h])
            changed += 1
    assert all(p["pred"] and len(set(p["pred"])) == len(p["pred"]) for p in preds)
    print(f"{changed} of {len(preds)} test predictions replaced by known labels")
    out = P5_ROOT / "runs" / "submissions_meta" / f"{args.name}_prediction.json"
    json.dump(preds, open(out, "w"))
    zp = P5_ROOT / "submissions" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, "prediction.json")
    print(f"-> {zp}")


if __name__ == "__main__":
    main()
