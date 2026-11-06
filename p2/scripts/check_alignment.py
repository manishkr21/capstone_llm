"""Sanity check of the whole offset pipeline WITHOUT a model.

Gold spans -> BIO tags on tokens -> decoded back to character spans -> strict macro-F1.
The result is the best score any tagger can reach with this tokenization; anything
below 1.0 comes from entity boundaries that fall inside a token.

    python scripts/check_alignment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lner import P1_ROOT  # noqa: E402
from lner.data import bio_tags, load_labels, load_train, tags_to_spans  # noqa: E402
from lner.engine import prepare_docs  # noqa: E402
from lner.metrics import strict_scores  # noqa: E402
from legal_lm.data.tokenizer import LegalTokenizer  # noqa: E402

DATA = P1_ROOT / "data/hf/Phase_2_LNER"
tok = LegalTokenizer.from_config({"type": "bpe", "path": str(P1_ROOT / "data/tokenizer/legal_bpe.json")})
labels = load_labels(DATA)
tags = bio_tags(labels)
tag2id = {t: i for i, t in enumerate(tags)}
texts, gold = load_train(DATA)
for presplit in (False, True):
    docs = prepare_docs(tok, texts, gold, presplit, tag2id)
    pred = {d["id"]: tags_to_spans(d["text"], d["spans"], d["tags"], tags) for d in docs}
    s = strict_scores(gold, pred, labels)
    n_tok = sum(len(d["ids"]) for d in docs)
    print(f"presplit={presplit}: oracle macro-F1 {s['macro_f1']}  micro-F1 {s['micro_f1']}  tokens {n_tok:,}")
    worst = sorted(s["per_label"].items(), key=lambda kv: kv[1]["f1"])[:4]
    print("   lowest:", ", ".join(f"{l} {v['f1']}" for l, v in worst))
