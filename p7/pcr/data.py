"""Phase-7 data: query cases, split-specific candidate pools, relevance labels.

  load(split) -> queries {id: {"text": str, "rel": set}}, pool {id: str}
Every query also occurs in its own pool (never labelled relevant to itself), so rankers drop it.
Text: sentences joined; the anonymisation tags <NAME>/<ORG>/<CITATION> are kept as-is.
"""
import json
import re

from . import P1_ROOT

DATA = P1_ROOT / "data" / "hf" / "Phase_7_PCR"
FILES = {"train": ("Train", "train"), "dev": ("Dev", "dev"), "test": ("Test", "test")}
K = 7


def load(split):
    d, f = FILES[split]
    q = json.load(open(DATA / d / f"{f}.json"))
    c = json.load(open(DATA / d / f"{f}_candidates.json"))
    queries = {k: {"text": " ".join(v["text"]), "rel": set(v.get("relevant_candidates", []))} for k, v in q.items()}
    pool = {k: " ".join(v["text"]) for k, v in c.items()}
    return queries, pool


_DATE = re.compile(r"DATE OF JUDGMENT:?\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")
_YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")


def year(text):
    """Judgment year: the DATE OF JUDGMENT header if present, else the latest plausible year in the
    first 3000 characters (headers / opening paragraphs)."""
    m = _DATE.search(text[:2000])
    if m:
        return int(m.group(3))
    ys = [int(y) for y in _YEAR.findall(text[:3000])]
    return max(ys) if ys else None


def micro_f1(ranked, queries, k=K):
    """ranked: {qid: [cid, ...]} -> micro P/R/F1 over the first k ids (queries without labels skipped)."""
    hit = ret = rel = 0
    for q, v in queries.items():
        if not v["rel"] and "rel" in v and q not in ranked:
            continue
        top = ranked[q][:k]
        hit += len(set(top) & v["rel"])
        ret += len(top)
        rel += len(v["rel"])
    p, r = hit / max(1, ret), hit / max(1, rel)
    return {"f1": 2 * p * r / max(1e-9, p + r), "p": p, "r": r}


def topk(scores, qids, cids, k=K, exclude_self=True):
    """scores: (n_q, n_c) array -> {qid: top-k cids}, the query's own id removed."""
    import numpy as np
    col = {c: j for j, c in enumerate(cids)}
    out = {}
    for i, q in enumerate(qids):
        s = scores[i].astype(float).copy()
        if exclude_self and q in col:
            s[col[q]] = -np.inf
        out[q] = [cids[j] for j in np.argsort(-s)[:k]]
    return out
