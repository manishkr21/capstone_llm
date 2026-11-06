"""Splits, token cache and chunk selection for legal statute identification (100 statute ids, multi-label).

Splits (``load_splits``):
  train : Train/train.json with exact-duplicate texts merged (whitespace-normalised hash; the labels of
          duplicates are unioned -- 80 duplicate texts carry different label sets)
  val   : Train/dev.json minus documents whose text also occurs in train or in Test/test.json
  test  : Test/test.json (ids and text)

Token cache: every document is tokenized once with the Phase-1 BPE (shared by A1-A4) into
cache/<split>.npz (flat uint16 ids + offsets + multi-hot labels (n, 100) + ids).

Chunk selection: p4's strategies (first | last | first_last | spread) plus ``evidence``: statute names are
masked in this corpus as "<SECTION>" / "<ACT>", so for documents longer than n_chunks x chunk_len the chunks are
centred on those placeholders (the facts the statute applies to), and any remaining slots are filled with
evenly spread chunks.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Dict, List

import numpy as np

from lsi import P1_ROOT, P5_ROOT
from cj.data import batch_chunks  # noqa: F401  (re-exported)
from cj.data import select_chunks as _select_chunks

DATA = P1_ROOT / "data" / "hf" / "Phase_5_LSI"
CACHE = P5_ROOT / "cache"
N_LABELS = 100


def text_hash(t: str) -> str:
    return hashlib.md5(" ".join(t.split()).encode()).hexdigest()


def load_splits() -> Dict[str, List[dict]]:
    train = json.load(open(DATA / "Train" / "train.json"))
    dev = json.load(open(DATA / "Train" / "dev.json"))
    test = [{"id": r["id"], "text": r["text"]} for r in json.load(open(DATA / "Test" / "test.json"))]
    test_h = {text_hash(r["text"]) for r in test}
    merged: "OrderedDict[str, dict]" = OrderedDict()
    for r in train:
        h = text_hash(r["text"])
        if h in merged:
            merged[h]["labels"] = sorted(set(merged[h]["labels"]) | set(r["labels"]))
        else:
            merged[h] = {"id": r["id"], "text": r["text"], "labels": sorted(set(r["labels"]))}
    val = [{"id": r["id"], "text": r["text"], "labels": sorted(set(r["labels"]))} for r in dev
           if text_hash(r["text"]) not in test_h and text_hash(r["text"]) not in merged]
    return {"train": list(merged.values()), "val": val, "test": test}


def multi_hot(label_lists) -> np.ndarray:
    y = np.zeros((len(label_lists), N_LABELS), dtype=np.uint8)
    for i, ls in enumerate(label_lists):
        y[i, list(ls)] = 1
    return y


def build_cache(tok, splits=None, batch: int = 256):
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = splits or load_splits()
    for name, docs in splits.items():
        path = CACHE / f"{name}.npz"
        if path.exists():
            continue
        ids, offs = [], [0]
        for i in range(0, len(docs), batch):
            for e in tok.encode_batch([" " + " ".join(d["text"].split()) for d in docs[i:i + batch]]):
                ids.append(np.asarray(e, dtype=np.uint16))
                offs.append(offs[-1] + len(e))
        np.savez(path, ids=np.concatenate(ids), offs=np.asarray(offs, dtype=np.int64),
                 labels=multi_hot([d.get("labels", []) for d in docs]),
                 doc_ids=np.asarray([str(d["id"]) for d in docs]))
        print(f"  cached {name}: {len(docs):,} docs, {offs[-1]:,} tokens", flush=True)


class TokenDocs:
    def __init__(self, split: str):
        z = np.load(CACHE / f"{split}.npz", allow_pickle=False)
        self.ids, self.offs, self.labels, self.doc_ids = z["ids"], z["offs"], z["labels"], z["doc_ids"]

    def __len__(self):
        return len(self.offs) - 1

    def tokens(self, i):
        return self.ids[self.offs[i]:self.offs[i + 1]]

    @classmethod
    def concat(cls, a: "TokenDocs", b: "TokenDocs") -> "TokenDocs":
        out = cls.__new__(cls)
        out.ids = np.concatenate([a.ids, b.ids])
        out.offs = np.concatenate([a.offs, b.offs[1:] + a.offs[-1]])
        out.labels = np.concatenate([a.labels, b.labels])
        out.doc_ids = np.concatenate([a.doc_ids, b.doc_ids])
        return out


# token ids of "ECTION" and "CT" in " <SECTION>" / " <ACT>" (legal BPE): anchors for the evidence strategy
ANCHORS = (19274, 5472)


def select_chunks(tokens, strategy, n_chunks, chunk_len):
    n_tot = max(1, -(-len(tokens) // chunk_len))
    if strategy != "evidence" or n_tot <= n_chunks:
        return _select_chunks(tokens, "spread" if strategy == "evidence" else strategy, n_chunks, chunk_len)
    pos = np.flatnonzero(np.isin(tokens, ANCHORS))
    starts = []
    for p in pos:  # greedy: a new window for every placeholder not yet covered, placeholder at 1/4 of the window
        if any(s <= p < s + chunk_len for s in starts):
            continue
        starts.append(int(min(max(0, p - chunk_len // 4), len(tokens) - chunk_len)))
        if len(starts) == n_chunks:
            break
    if len(starts) < n_chunks:  # fill with evenly spread chunks that do not overlap the evidence windows
        for s in [int(round(x)) * chunk_len for x in np.linspace(0, n_tot - 1, n_chunks * 2)]:
            if len(starts) == n_chunks:
                break
            if all(abs(s - t) >= chunk_len for t in starts):
                starts.append(min(s, len(tokens) - chunk_len))
    return [tokens[s:s + chunk_len] for s in sorted(starts)]
