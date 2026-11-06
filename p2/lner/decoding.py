"""Token tag decoding: plain argmax, Viterbi restricted to valid BIO sequences, or a CRF.

Valid transitions: anything -> O, anything -> B-X, and I-X only after B-X or I-X.
Viterbi picks the most probable *valid* sequence, which removes broken spans such as
"O I-PREC I-PREC" or "B-APP I-RESP" that argmax can produce at window or entity edges.
With a CRF head the transition scores are learned instead of fixed to 0 / -inf.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


def allowed_transitions(tags: List[str]):
    n = len(tags)
    ok = np.ones((n, n), dtype=bool)  # ok[prev, next]
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            ok[:, j] = False
            ok[tags.index("B-" + t[2:]), j] = True
            ok[j, j] = True
    start_ok = np.array([not t.startswith("I-") for t in tags])
    return ok, start_ok


def viterbi(scores: np.ndarray, trans: np.ndarray, start: np.ndarray, end: Optional[np.ndarray] = None) -> List[int]:
    """Best path through per-token scores (T, n) with additive transition (n, n) and start/end scores."""
    T, n = scores.shape
    if T == 0:
        return []
    score = scores[0] + start
    back = np.zeros((T, n), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + trans  # (prev, next)
        back[t] = cand.argmax(0)
        score = cand.max(0) + scores[t]
    if end is not None:
        score = score + end
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def decode(logits: torch.Tensor, tags: List[str], method: str, crf=None) -> List[int]:
    if crf is not None:
        f = lambda t: t.detach().float().cpu().numpy()
        return viterbi(f(logits), f(crf._trans()), f(crf._start()), f(crf.end))
    if method == "argmax":
        return logits.argmax(-1).tolist()
    ok, start_ok = allowed_transitions(tags)
    return viterbi(torch.log_softmax(logits.float(), -1).cpu().numpy(),
                   np.where(ok, 0.0, -1e9), np.where(start_ok, 0.0, -1e9))
