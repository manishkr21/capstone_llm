"""Macro precision / recall / F1 over sentence labels (labels present in gold or predictions,
as in sklearn's macro average), overall and per domain, plus per-label scores and a confusion table."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List


def macro_scores(gold: List[str], pred: List[str]) -> dict:
    labels = sorted(set(gold) | set(pred))
    tp, gc, pc = Counter(), Counter(gold), Counter(pred)
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
    per = {}
    for l in labels:
        p = tp[l] / pc[l] if pc[l] else 0.0
        r = tp[l] / gc[l] if gc[l] else 0.0
        per[l] = {"p": round(p, 4), "r": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
                  "gold": gc[l], "pred": pc[l]}
    n = max(1, len(labels))
    return {"macro_f1": sum(v["f1"] for v in per.values()) / n,
            "macro_p": sum(v["p"] for v in per.values()) / n,
            "macro_r": sum(v["r"] for v in per.values()) / n,
            "accuracy": sum(tp.values()) / max(1, len(gold)), "per_label": per}


def evaluate(docs: List[dict], preds: Dict[str, List[str]]) -> dict:
    out = {}
    for name, keep in (("all", lambda d: True), ("CL", lambda d: d["domain"] == "CL"), ("IT", lambda d: d["domain"] == "IT")):
        g, p = [], []
        for d in docs:
            if keep(d):
                g += d["labels"]
                p += preds[d["id"]]
        out[name] = macro_scores(g, p)
    return out


def confusion(docs, preds) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter] = {}
    for d in docs:
        for g, p in zip(d["labels"], preds[d["id"]]):
            table.setdefault(g, Counter())[p] += 1
    return {g: dict(c.most_common()) for g, c in table.items()}
