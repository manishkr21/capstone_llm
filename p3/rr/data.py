"""Documents, cross-validation folds, and packing sentences into backbone windows.

A document is {"id", "domain" (CL|IT), "text": [sentences], "labels": [labels] (train only)}.

Packing (``context``):
  window : consecutive sentences are packed into windows of at most ``window`` tokens, so
           every sentence is encoded together with its neighbours (context-aware);
  none   : every sentence is its own window (sentence-only baseline).
A sentence keeps at most ``max_sent_tokens`` tokens.  Each sentence is encoded as
" " + text so that, inside a window, the BPE sees the same word boundaries as in running text.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import torch

DOMAINS = ("CL", "IT")
_WS = re.compile(r"\s+")


def load_labels(data_dir) -> List[str]:
    vocab = json.load(open(Path(data_dir) / "Train" / "label_vocab.json"))
    return [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def _read(path, domain):
    d = json.load(open(path))
    if isinstance(d, dict):  # test files: {id: [sentences]}
        return [{"id": k, "domain": domain, "text": v} for k, v in d.items()]
    return [dict(x, domain=domain) for x in d]


def load_split(data_dir, split: str) -> List[dict]:
    """split: train | dev | test (both domains)."""
    data_dir = Path(data_dir)
    sub = "Test" if split == "test" else "Train"
    return [doc for dom in DOMAINS for doc in _read(data_dir / sub / f"{dom}_{split}.json", dom)]


def cv_folds(docs: List[dict], n_folds: int, seed: int) -> List[List[str]]:
    """Document-level folds, stratified by domain: each fold gets ~1/n of the CL and of the IT docs."""
    folds = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for dom in DOMAINS:
        ids = sorted(d["id"] for d in docs if d["domain"] == dom)
        rng.shuffle(ids)
        for i, doc_id in enumerate(ids):
            folds[i % n_folds].append(doc_id)
    return folds


def clean_sentence(s: str) -> str:
    return _WS.sub(" ", s).strip()


# Lexical cues for the most-confused roles (used when model.cues is true).  Each is a 0/1 flag per sentence.
CUES = [
    ("case_v", r"\b(v|vs)\.?\s+[A-Z]|\bversus\b"),                                   # case citation "X v. Y"
    ("law_report", r"\b(AIR|SCC|SCR|ITR|SCALE|JT|Bom\s?LR|All\s?LJ|DLT|CompLJ|Comp\s?Cas|taxmann)\b"),
    ("cite_year", r"[\(\[](19|20)\d\d[\)\]]"),                                      # (2005) / [2010]
    ("relied", r"(?i)\brel(y|ied|iance)\b|\bfollowed\b|\bplaced reliance\b|\bsquarely\b"),
    ("distinguished", r"(?i)distinguish|not applicable|inapplicable|does not apply|not relevant|no application"),
    ("overruled", r"(?i)overrul"),
    ("appellant", r"(?i)\b(appellant|petitioner|assessee|applicant|informant|complainant)s?\b"),
    ("respondent", r"(?i)\b(respondent|revenue|department|opposite part(y|ies))s?\b"),
    ("counsel", r"(?i)\b(learned )?(counsel|advocate|senior counsel|a\.?s\.?g|solicitor)\b"),
    ("argued", r"(?i)\b(contend|submi[ts]|argu|urg(e|ed)|plead)"),
    ("lower_body", r"(?i)\b(tribunal|ITAT|CIT\s*\(A\)|commissioner|assessing officer|trial court|"
                   r"single judge|commission|CCI|DG|director general|adjudicating authority)\b"),
    ("impugned", r"(?i)\bimpugned\b|\border dated\b|\bpassed by\b|\bheld that\b"),
    ("we", r"(?i)\b(we|our|us)\b|\bin (my|our) (view|opinion)\b"),
    ("disposal", r"(?i)\b(dismissed|allowed|set aside|disposed of|quashed|remanded|no order as to costs)\b"),
    ("statute", r"(?i)\bsection\b|\bsec\.|\bs\.\s?\d|\bact\b|\brule\b|\barticle\b|\bregulation\b"),
    ("question", r"(?i)\bquestion\b|\bwhether\b|\bissue\b|\bpoint for (determination|consideration)\b"),
    ("dissent", r"(?i)\bregret\b|unable to agree|\bdissent|with (great )?respect,? i\b"),
    ("facts", r"(?i)\b(filed|on \d|dated|\d{1,2}\.\d{1,2}\.(19|20)\d\d|19\d\d|20\d\d)\b"),
]
_CUE_RE = [re.compile(p) for _, p in CUES]


def cue_features(sentence: str) -> List[float]:
    return [1.0 if r.search(sentence) else 0.0 for r in _CUE_RE]


def encode_doc(tok, doc: dict, max_sent_tokens: int, label2id: Dict[str, int]) -> dict:
    sents = [tok.encode(" " + clean_sentence(s))[:max_sent_tokens] or [tok.eos_id] for s in doc["text"]]
    out = {"id": doc["id"], "domain": DOMAINS.index(doc["domain"]), "sents": sents,
           "feats": torch.tensor([cue_features(s) for s in doc["text"]])}
    if "labels" in doc:
        out["y"] = torch.tensor([label2id[l] for l in doc["labels"]])
    return out


def pack(sents: List[List[int]], context: str, window: int):
    """Group sentence indices into windows. Returns a list of lists of sentence indices."""
    if context == "none":
        return [[i] for i in range(len(sents))]
    groups, cur, used = [], [], 0
    for i, s in enumerate(sents):
        if cur and used + len(s) > window:
            groups.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += len(s)
    if cur:
        groups.append(cur)
    return groups


def window_batch(sents: List[List[int]], groups, pad_id: int, n_prompt: int = 0):
    """Token ids (W, L) for a list of windows and, per token, the index (into the flat list of
    sentences of these windows, in order) of the sentence it belongs to (-1 for pad/prompt)."""
    rows, owners, k = [], [], 0
    for g in groups:
        ids, own = [pad_id] * n_prompt, [-1] * n_prompt
        for i in g:
            ids += sents[i]
            own += [k] * len(sents[i])
            k += 1
        rows.append(ids)
        owners.append(own)
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    own = torch.full((len(rows), L), -1, dtype=torch.long)
    for r, (a, b) in enumerate(zip(rows, owners)):
        ids[r, :len(a)] = torch.tensor(a)
        own[r, :len(b)] = torch.tensor(b)
    return ids, own
