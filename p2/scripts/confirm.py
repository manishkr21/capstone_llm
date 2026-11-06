"""Mean dev macro-F1 over 3 dev splits for each candidate, with each post-processing option.

    python scripts/confirm.py          # reads sweeps/f_<candidate>_s<seed>/

Single 10-document dev splits move by +-0.02, so candidates are ranked by the mean of
three disjoint-ish splits (split_seed 0, 1, 2).
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from lner import P2_ROOT  # noqa: E402
from lner.data import bio_tags, load_labels, load_train, split_dev, tags_to_spans  # noqa: E402
from lner.decoding import decode  # noqa: E402
from lner.engine import doc_logits, load_run, merge_spans, prepare_docs, propagate  # noqa: E402
from lner.metrics import strict_scores  # noqa: E402

POST = {"none": (None, None), "merge": (" ", None), "prop5": (None, 5), "prop10": (None, 10),
        "merge+prop5": (" ", 5), "merge+prop10": (" ", 10)}


def main():
    device = torch.device("cuda")
    cache = P2_ROOT / "sweeps" / "confirm_scores.json"
    scores = json.load(open(cache)) if cache.exists() else {}
    texts = gold = None
    for run_dir in sorted((P2_ROOT / "sweeps").glob("f_*_s[0-9]")):
        if run_dir.name in scores or not (run_dir / "summary.json").exists():
            continue
        import yaml
        cfg0 = yaml.safe_load(open(run_dir / "config.yaml"))
        data_dir = P2_ROOT / cfg0["data"]["dir"]
        labels = load_labels(data_dir)
        tags = bio_tags(labels)
        if texts is None:
            texts, gold = load_train(data_dir)
        model, tok, cfg = load_run(run_dir, tags, device)
        _, dev_ids = split_dev(list(texts), cfg["data"]["n_dev"], cfg["data"]["split_seed"])
        docs = prepare_docs(tok, {k: texts[k] for k in dev_ids}, None, cfg["data"]["presplit"], None,
                            cfg["data"].get("case_norm", False))
        res = {}
        lg = {d["id"]: doc_logits(model, d["ids"], cfg["infer"]["window"], cfg["infer"]["stride"], device) for d in docs}
        base = {d["id"]: tags_to_spans(d["text"], d["spans"], decode(lg[d["id"]], tags, "viterbi", crf=model.crf), tags)
                for d in docs}
        for name, (merge, prop) in POST.items():
            preds = {}
            for d in docs:
                ents = base[d["id"]]
                if merge:
                    ents = merge_spans(d["text"], ents, merge)
                if prop:
                    ents = propagate(d["text"], ents, prop)
                preds[d["id"]] = ents
            res[name] = strict_scores({k: gold[k] for k in dev_ids}, preds, labels)["macro_f1"]
        scores[run_dir.name] = res
        json.dump(scores, open(cache, "w"), indent=1)
        print(run_dir.name, res, flush=True)
        del model
        torch.cuda.empty_cache()

    table = defaultdict(lambda: defaultdict(list))
    for name, res in scores.items():
        cand = re.match(r"f_(.+)_s\d$", name).group(1)
        for post, f in res.items():
            table[cand][post].append(f)
    print(f"\n{'candidate':16s}" + "".join(f"{p:>16s}" for p in POST) + "   (mean +- std over dev splits)")
    for cand in sorted(table, key=lambda c: -statistics.mean(table[c]["none"])):
        row = table[cand]
        print(f"{cand:16s}" + "".join(
            f"{statistics.mean(row[p]):9.4f}+-{statistics.pstdev(row[p]):.3f}" for p in POST) + f"   n={len(row['none'])}")


if __name__ == "__main__":
    main()
