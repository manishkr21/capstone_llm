"""Phase-6 data: splits, sentence-level token cache and long-document condensation.

Splits (no public dev set in Phase 6):
  train : Train/train.json, exact-duplicate documents dropped (whitespace-normalised MD5), minus val
  val   : N_VAL documents drawn once with a fixed seed from the de-duplicated training data
  test  : Test/test.json (100 documents, no summaries)

A document is cached as sentence-split token ids (cache/<split>.npz: flat tokens, sentence offsets,
document offsets) so every condensation strategy below works on whole sentences.

Long documents (median ~3.6k tokens) do not fit the model's source budget, so the source is condensed:
  head     the first ``budget`` tokens (plain truncation)
  spread   evenly spaced sentences across the whole document, in order (coverage)
  salient  the opening of the case (``head_frac`` of the budget) + the sentences most similar to the
           document's own TF-IDF centroid (importance), restored to document order
"""
from __future__ import annotations

import hashlib
import json
import random
import re

import numpy as np

from summ import P1_ROOT, P6_ROOT

DATA = P1_ROOT / "data" / "hf" / "Phase_6_Summ"
CACHE = P6_ROOT / "cache"
N_VAL = 300
SENT_RE = re.compile(r"(?<=[.?!;])\s+(?=[A-Z0-9(\"'])")


def _h(text: str) -> str:
    return hashlib.md5(" ".join(text.split()).encode()).hexdigest()


def load_splits():
    raw = json.load(open(DATA / "Train" / "train.json"))
    seen, uniq = set(), []
    for r in raw:
        k = _h(r["document"])
        if k not in seen:
            seen.add(k)
            uniq.append({"id": str(r["id"]), "document": r["document"], "summary": r["summary"]})
    idx = set(random.Random(0).sample(range(len(uniq)), N_VAL))
    test = [{"id": str(r["id"]), "document": r["document"]} for r in json.load(open(DATA / "Test" / "test.json"))]
    return {"train": [d for i, d in enumerate(uniq) if i not in idx],
            "val": [d for i, d in enumerate(uniq) if i in idx], "test": test}


def split_sentences(text: str):
    return [s for s in SENT_RE.split(" ".join(text.split())) if s]


def build_cache(tok, splits=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = splits or load_splits()
    for name, docs in splits.items():
        path = CACHE / f"{name}.npz"
        if path.exists():
            continue
        toks, sent_offs, doc_offs, summ_toks, summ_offs = [], [0], [0], [], [0]
        for d in docs:
            sents = split_sentences(d["document"])
            enc = tok.encode_batch([(" " if i else "") + s for i, s in enumerate(sents)])
            for e in enc:
                toks.extend(e)
                sent_offs.append(len(toks))
            doc_offs.append(len(sent_offs) - 1)
            if "summary" in d:
                summ_toks.extend(tok.encode(" ".join(d["summary"].split())))
            summ_offs.append(len(summ_toks))
        np.savez(path, toks=np.array(toks, np.int32), sent_offs=np.array(sent_offs, np.int64),
                 doc_offs=np.array(doc_offs, np.int64), summ=np.array(summ_toks, np.int32),
                 summ_offs=np.array(summ_offs, np.int64), ids=np.array([d["id"] for d in docs]))
        print(f"cached {name}: {len(docs)} docs, {len(toks)} tokens", flush=True)


class Docs:
    """Token view of one split: ``sents(i)`` -> list of sentence token arrays, ``summary(i)``."""

    def __init__(self, split: str):
        z = np.load(CACHE / f"{split}.npz")
        self.toks, self.so, self.do = z["toks"], z["sent_offs"], z["doc_offs"]
        self.summ, self.mo, self.ids = z["summ"], z["summ_offs"], [str(x) for x in z["ids"]]

    def __len__(self):
        return len(self.ids)

    def sents(self, i):
        a, b = self.do[i], self.do[i + 1]
        return [self.toks[self.so[j]:self.so[j + 1]] for j in range(a, b)]

    def n_tokens(self, i):
        return int(self.so[self.do[i + 1]] - self.so[self.do[i]])

    def summary(self, i):
        return self.summ[self.mo[i]:self.mo[i + 1]]


def _fit_order(sents, order, budget):
    """Take sentences in priority ``order`` while they fit; return them in document order."""
    keep, used = [], 0
    for j in order:
        n = len(sents[j])
        if used + n <= budget:
            keep.append(j)
            used += n
    if not keep:  # a single huge sentence: truncate it
        return np.asarray(sents[order[0]][:budget])
    return np.concatenate([sents[j] for j in sorted(keep)])


def condense(sents, budget: int, strategy: str = "salient", head_frac: float = 0.15):
    """Token ids (<= budget) representing the document."""
    total = sum(len(s) for s in sents)
    if total <= budget:
        return np.concatenate(sents) if sents else np.zeros(0, np.int32)
    if strategy == "head":
        return np.concatenate(sents)[:budget]
    n = len(sents)
    if strategy == "spread":
        # sentence priority by a low-discrepancy sweep: 0, n/2, n/4, 3n/4, ... -> even coverage at any budget
        order, seen = [], set()
        k = 1
        while len(order) < n:
            for m in range(k):
                j = min(n - 1, int((2 * m + 1) * n / (2 * k)))
                if j not in seen:
                    seen.add(j)
                    order.append(j)
            k *= 2
        order = [0] + [j for j in order if j != 0]
        return _fit_order(sents, order, budget)
    if strategy == "salient":
        head, used, j = [], 0, 0
        while j < n and used + len(sents[j]) <= head_frac * budget:
            head.append(j)
            used += len(sents[j])
            j += 1
        # bag-of-token-ids TF with log-idf from within the document (sentence frequency)
        vocab = {}
        rows = []
        for s in sents:
            u, c = np.unique(s, return_counts=True)
            rows.append((u, c))
            for t in u:
                vocab[t] = vocab.get(t, 0) + 1
        idf = {t: np.log((1 + n) / (1 + df)) + 1 for t, df in vocab.items()}
        cent = {}
        vecs = []
        for u, c in rows:
            w = {t: (1 + np.log(x)) * idf[t] for t, x in zip(u.tolist(), c.tolist())}
            norm = np.sqrt(sum(v * v for v in w.values())) or 1.0
            w = {t: v / norm for t, v in w.items()}
            vecs.append(w)
            for t, v in w.items():
                cent[t] = cent.get(t, 0.0) + v
        score = np.array([sum(v * cent.get(t, 0.0) for t, v in w.items()) for w in vecs])
        score += 0.15 * score.max() * (np.arange(n) >= n - max(1, n // 10))  # the operative order at the end
        rest = [j for j in np.argsort(-score) if j not in set(head)]
        return _fit_order(sents, head + rest, budget)
    raise ValueError(strategy)
