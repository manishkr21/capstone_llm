"""Documents -> token windows with BIO tags, and back to character spans.

Character alignment
-------------------
The Phase-1 tokenizer is byte-level BPE.  Its offsets point into the original string,
but a token usually carries its leading space (" Court" -> (k, k+6)).  We strip that
whitespace so a token's span starts on its first visible character.

The GPT-2 style pre-tokenizer keeps punctuation runs together (").", "],"), and gold
entities often end in the middle of such a run.  With ``presplit=True`` we cut the text
into words / digit runs / single punctuation characters first and run BPE on each piece.
Words stay intact, so the model still sees familiar tokens.  On the training folds this
raises the share of entities whose boundaries fall on token boundaries from 98.2% to
99.3% (`scripts/check_alignment.py`), which is the ceiling for strict span F1.
"""
from __future__ import annotations

import bisect
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

PIECE = re.compile(r"\s*[A-Za-z]+|\s*\d+|\s*[^\w\s]|\s*\w+|\s+")


CAPS_WORD = re.compile(r"\b[A-Z]{4,}\b|\b(?:OF|THE|AND|IN|VS|FOR|ON|AT|TO|BY)\b(?=\s+[A-Z])")


def normalize_case(text: str) -> str:
    """ALL-CAPS words (4+ letters) -> Titlecase, short ALL-CAPS function words -> lowercase.
    'STATE OF HARYANA' -> 'State of Haryana', which BPE splits like ordinary prose.
    Only ASCII letters change, so the string length and every character offset stay the same;
    abbreviations of 2-3 letters (IPC, AIR, SC, LTD) are left alone."""
    def fix(m):
        w = m.group()
        return w[0] + w[1:].lower() if len(w) >= 4 else w.lower()
    out = CAPS_WORD.sub(fix, text)
    assert len(out) == len(text)
    return out


def load_labels(data_dir) -> List[str]:
    return [l.strip() for l in open(Path(data_dir) / "Train" / "labels.txt") if l.strip()]


def bio_tags(labels: List[str]) -> List[str]:
    return ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]


def load_train(data_dir) -> Tuple[Dict[str, str], Dict[str, list]]:
    texts, gold = {}, {}
    for fold in ("fold1", "fold2"):
        texts.update(json.load(open(Path(data_dir) / "Train" / f"{fold}-text.json")))
        gold.update(json.load(open(Path(data_dir) / "Train" / f"{fold}-gold.json")))
    return texts, gold


def load_test(data_dir) -> Dict[str, str]:
    return json.load(open(Path(data_dir) / "Test" / "fold3-text.json"))


def split_dev(doc_ids: List[str], n_dev: int, seed: int) -> Tuple[List[str], List[str]]:
    ids = sorted(doc_ids)
    random.Random(seed).shuffle(ids)
    return ids[n_dev:], ids[:n_dev]


def tokenize(tk, text: str, presplit: bool) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Token ids and whitespace-stripped character spans (start, end) of every token."""
    if not text:
        return [], []
    if presplit:
        pieces = [(m.start(), m.end()) for m in PIECE.finditer(text)]
        ids, offs = [], []
        for (a, _), enc in zip(pieces, tk.encode_batch([text[a:b] for a, b in pieces])):
            ids += enc.ids
            offs += [(a + s, a + e) for s, e in enc.offsets]
    else:
        enc = tk.encode(text)
        ids, offs = enc.ids, enc.offsets
    spans = []
    for s, e in offs:
        while s < e and text[s].isspace():
            s += 1
        spans.append((s, e))
    return ids, spans


def encode_tags(spans, entities, tag2id) -> List[int]:
    """BIO tag per token. A token belongs to an entity if its visible characters overlap it."""
    tags = [0] * len(spans)
    starts = [s for s, _ in spans]
    for s, e, label, *_ in entities:
        i = max(0, bisect.bisect_right(starts, s) - 1)
        first = True
        while i < len(spans) and spans[i][0] < e:
            ts, te = spans[i]
            if te > ts and te > s:
                tags[i] = tag2id[("B-" if first else "I-") + label]
                first = False
            i += 1
    return tags


def windows(n: int, size: int, stride: int) -> List[Tuple[int, int]]:
    """Full-length windows covering [0, n); the last one is shifted back to end at n,
    so every window has the same length and no padding is needed."""
    if n <= size:
        return [(0, n)] if n else []
    out = [(s, s + size) for s in range(0, n - size, stride)]
    out.append((n - size, n))
    return out


def tags_to_spans(text: str, spans, tag_ids, id2tag) -> List[list]:
    """Decode BIO tags into [start, end, label]. An I- tag whose type differs from the open
    entity (or with no open entity) starts a new entity, so no prediction is thrown away."""
    ents, cur = [], None  # cur = [start, end, label]
    for (s, e), t in zip(spans, tag_ids):
        tag = id2tag[t]
        if e <= s:  # whitespace-only token: does not open or close anything
            continue
        if tag == "O":
            if cur:
                ents.append(cur)
            cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != label:
            if cur:
                ents.append(cur)
            cur = [s, e, label]
        else:
            cur[1] = e
    if cur:
        ents.append(cur)
    out = []
    for s, e, label in ents:
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append([s, e, label, text[s:e]])
    return out
