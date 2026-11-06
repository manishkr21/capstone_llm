"""Strict span metrics, as on Codabench: a prediction counts only if start, end AND label match."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List


def strict_scores(gold: Dict[str, list], pred: Dict[str, list], labels: List[str]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    span_tp = n_gold = n_pred = 0
    for doc in gold:
        g = {(s, e, l) for s, e, l, *_ in gold[doc]}
        p = {(s, e, l) for s, e, l, *_ in pred.get(doc, [])}
        for x in p & g:
            tp[x[2]] += 1
        for x in p - g:
            fp[x[2]] += 1
        for x in g - p:
            fn[x[2]] += 1
        span_tp += len({x[:2] for x in p} & {x[:2] for x in g})  # boundary right, label ignored
        n_gold += len(g)
        n_pred += len(p)
    per = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[l] = {"p": round(P, 4), "r": round(R, 4), "f1": round(F, 4), "gold": tp[l] + fn[l], "pred": tp[l] + fp[l]}
    n = len(labels)
    TP = sum(tp.values())
    micro_p = TP / n_pred if n_pred else 0.0
    micro_r = TP / n_gold if n_gold else 0.0
    return {
        "macro_f1": round(sum(v["f1"] for v in per.values()) / n, 4),
        "macro_p": round(sum(v["p"] for v in per.values()) / n, 4),
        "macro_r": round(sum(v["r"] for v in per.values()) / n, 4),
        "micro_f1": round(2 * micro_p * micro_r / (micro_p + micro_r), 4) if micro_p + micro_r else 0.0,
        "span_only_f1": round(2 * span_tp / (n_gold + n_pred), 4) if n_gold + n_pred else 0.0,
        "per_label": per,
    }
