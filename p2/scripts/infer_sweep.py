"""Inference-only tuning on a trained run: decoding, stride, post-processing. No retraining.

    python scripts/infer_sweep.py runs/E2_a4_lorma runs/E1_a3_full

Logits are computed once per (run, stride); decode / merge / propagate variants reuse them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from lner import P2_ROOT  # noqa: E402
from lner.data import bio_tags, load_labels, load_train, split_dev, tags_to_spans  # noqa: E402
from lner.decoding import decode  # noqa: E402
from lner.engine import doc_logits, load_run, merge_spans, prepare_docs, propagate  # noqa: E402
from lner.metrics import strict_scores  # noqa: E402

POST = [  # (name, decode, merge_gap, propagate)
    ("viterbi", "viterbi", None, None),
    ("argmax", "argmax", None, None),
    ("viterbi+merge_space", "viterbi", " ", None),
    ("viterbi+merge_space_dash", "viterbi", " -", None),
    ("viterbi+propagate5", "viterbi", None, 5),
    ("viterbi+propagate10", "viterbi", None, 10),
    ("viterbi+merge+propagate10", "viterbi", " ", 10),
]


def main():
    device = torch.device("cuda")
    for run in sys.argv[1:]:
        run_dir = (P2_ROOT / run).resolve()
        import yaml
        cfg0 = yaml.safe_load(open(run_dir / "config.yaml"))
        data_dir = P2_ROOT / cfg0["data"]["dir"]
        labels = load_labels(data_dir)
        tags = bio_tags(labels)
        model, tok, cfg = load_run(run_dir, tags, device)
        texts, gold = load_train(data_dir)
        _, dev_ids = split_dev(list(texts), cfg["data"]["n_dev"], cfg["data"]["split_seed"])
        docs = prepare_docs(tok, {k: texts[k] for k in dev_ids}, None, cfg["data"]["presplit"], None,
                            cfg["data"].get("case_norm", False))
        dev_gold = {k: gold[k] for k in dev_ids}
        W = cfg["infer"]["window"]
        print(f"\n== {run}  (window {W})")
        for stride in (64, 128, 256):
            logits = {d["id"]: doc_logits(model, d["ids"], W, stride, device) for d in docs}
            for name, dec, merge, prop in POST:
                if model.crf is not None and dec == "argmax":
                    continue
                preds = {}
                for d in docs:
                    ents = tags_to_spans(d["text"], d["spans"], decode(logits[d["id"]], tags, dec, crf=model.crf), tags)
                    if merge:
                        ents = merge_spans(d["text"], ents, merge)
                    if prop:
                        ents = propagate(d["text"], ents, prop)
                    preds[d["id"]] = ents
                s = strict_scores(dev_gold, preds, labels)
                print(f"  stride {stride:3d}  {name:28s} macro-F1 {s['macro_f1']:.4f}  P {s['macro_p']:.4f}  "
                      f"R {s['macro_r']:.4f}  span {s['span_only_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
