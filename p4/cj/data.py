"""Splits, token cache and chunk selection for court judgment prediction.

Splits (``load_splits``):
  train : Train/train.json with exact-duplicate texts removed (31,677 unique judgments, no label conflicts)
  val   : Train/dev.json MINUS every document whose text also appears in Test/test.json.  All 1,517 test
          documents occur verbatim, with labels, in dev.json, so model selection uses only the other
          1,988 dev documents.
  test  : Test/test.json, ids and text only -- its ``label`` field is dropped on load and never read.

Token cache: every document is tokenized once with the Phase-1 BPE (shared by A1-A4) into
cache/<split>.npz (flat uint16 ids + offsets + labels + ids).

Chunk selection (``select_chunks``): a document is cut into chunks of ``chunk_len`` tokens and
``n_chunks`` of them are used:
  last   : the last n chunks (the end of a judgment holds the reasoning closest to the outcome)
  first  : the first n chunks
  first_last : half from the start, half from the end
  spread : n chunks evenly spaced over the whole document
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from cj import P1_ROOT, P4_ROOT

DATA = P1_ROOT / "data" / "hf" / "Phase_4_CJPE"
CACHE = P4_ROOT / "cache"


def _h(t: str) -> str:
    return hashlib.md5(t.encode()).hexdigest()


def load_splits() -> Dict[str, List[dict]]:
    train = json.load(open(DATA / "Train" / "train.json"))
    dev = json.load(open(DATA / "Train" / "dev.json"))
    test = json.load(open(DATA / "Test" / "test.json"))
    test = [{"id": r["id"], "text": r["text"]} for r in test]  # the test label is never kept
    test_h = {_h(r["text"]) for r in test}
    seen, tr = set(), []
    for r in train:
        h = _h(r["text"])
        if h not in seen and h not in test_h:
            seen.add(h)
            tr.append({"id": r["id"], "text": r["text"], "label": int(r["label"])})
    val = [{"id": r["id"], "text": r["text"], "label": int(r["label"])} for r in dev
           if _h(r["text"]) not in test_h and _h(r["text"]) not in seen]
    return {"train": tr, "val": val, "test": test}


def build_cache(tok, splits=None, batch: int = 256):
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = splits or load_splits()
    for name, docs in splits.items():
        path = CACHE / f"{name}.npz"
        if path.exists():
            continue
        ids, offs = [], [0]
        for i in range(0, len(docs), batch):
            for e in tok.encode_batch([" " + d["text"] for d in docs[i:i + batch]]):
                ids.append(np.asarray(e, dtype=np.uint16))
                offs.append(offs[-1] + len(e))
        np.savez(path, ids=np.concatenate(ids), offs=np.asarray(offs, dtype=np.int64),
                 labels=np.asarray([d.get("label", -1) for d in docs], dtype=np.int64),
                 doc_ids=np.asarray([d["id"] for d in docs]))
        print(f"  cached {name}: {len(docs):,} docs, {offs[-1]:,} tokens", flush=True)


class TokenDocs:
    def __init__(self, split: str):
        z = np.load(CACHE / f"{split}.npz", allow_pickle=False)
        self.ids, self.offs, self.labels, self.doc_ids = z["ids"], z["offs"], z["labels"], z["doc_ids"]

    def __len__(self):
        return len(self.offs) - 1

    def tokens(self, i):
        return self.ids[self.offs[i]:self.offs[i + 1]]


def select_chunks(tokens: np.ndarray, strategy: str, n_chunks: int, chunk_len: int) -> List[np.ndarray]:
    n_tot = max(1, -(-len(tokens) // chunk_len))
    starts = [i * chunk_len for i in range(n_tot)]
    if n_tot > n_chunks:
        if strategy == "last":
            starts = starts[-n_chunks:]
        elif strategy == "first":
            starts = starts[:n_chunks]
        elif strategy == "first_last":
            k = n_chunks // 2
            starts = starts[:k] + starts[-(n_chunks - k):]
        elif strategy == "spread":
            starts = [starts[int(round(j))] for j in np.linspace(0, n_tot - 1, n_chunks)]
        else:
            raise ValueError(strategy)
    if strategy == "last" and n_tot > 1:  # align the final chunk with the end of the document
        end = len(tokens)
        return [tokens[max(0, end - (len(starts) - j) * chunk_len):end - (len(starts) - j - 1) * chunk_len]
                for j in range(len(starts))]
    return [tokens[s:s + chunk_len] for s in starts]


def batch_chunks(docs_chunks: List[List[np.ndarray]], pad_id: int):
    """(C, L) ids, (C, L) mask, and doc index per chunk, for a list of documents' chunks."""
    flat = [(d, c) for d, chunks in enumerate(docs_chunks) for c in chunks]
    L = max(len(c) for _, c in flat)
    ids = torch.full((len(flat), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(flat), L), dtype=torch.bool)
    for r, (_, c) in enumerate(flat):
        ids[r, :len(c)] = torch.from_numpy(c.astype(np.int64))
        mask[r, :len(c)] = True
    return ids, mask, torch.tensor([d for d, _ in flat])
