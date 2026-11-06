"""Linear-chain CRF (Lafferty et al., 2001) on top of the per-token emission scores.

score(y) = start[y_0] + sum_t emit[t, y_t] + sum_t trans[y_{t-1}, y_t] + end[y_T-1]
loss     = log Z - score(gold),  log Z by the forward algorithm.

Transitions that break BIO (e.g. O -> I-X) are fixed to -1e4, so the CRF learns only the
valid ones. Decoding is Viterbi with the learned transitions.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from lner.decoding import allowed_transitions

NEG = -1e4


class CRF(nn.Module):
    def __init__(self, tags: List[str]):
        super().__init__()
        n = len(tags)
        ok, start_ok = allowed_transitions(tags)
        self.register_buffer("mask", torch.tensor(ok))
        self.register_buffer("start_mask", torch.tensor(start_ok))
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def _trans(self):
        return self.trans.masked_fill(~self.mask, NEG)

    def _start(self):
        return self.start.masked_fill(~self.start_mask, NEG)

    def nll(self, emit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood. emit (B, T, n) float32, y (B, T)."""
        trans, start = self._trans(), self._start()
        B, T, _ = emit.shape
        gold = start[y[:, 0]] + emit.gather(2, y[..., None]).squeeze(-1).sum(1)
        gold = gold + trans[y[:, :-1], y[:, 1:]].sum(1) + self.end[y[:, -1]]
        alpha = start + emit[:, 0]
        for t in range(1, T):
            alpha = torch.logsumexp(alpha[:, :, None] + trans[None], dim=1) + emit[:, t]
        log_z = torch.logsumexp(alpha + self.end, dim=1)
        return (log_z - gold).mean() / T  # per token, comparable to cross-entropy
