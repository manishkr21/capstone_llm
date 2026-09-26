"""The common interface every architecture implements.

Training / evaluation
---------------------
``logits = model(idx, prefix_len)``

* ``idx`` is a (B, L) block of token ids with L = block_size + 1.
* The first ``prefix_len`` tokens are *context*; the model returns logits that
  predict ``idx[:, prefix_len:]`` (shape (B, L - prefix_len, vocab)).
* Decoder-only models train with prefix_len = 1 (plain next-token prediction).
  The encoder-decoder and prefix-LM sample a random prefix length each step.
* Evaluation uses the *same* prefix length for all four models (e.g. 768 tokens
  of context, 257 target tokens), so their validation losses are comparable and
  directly measure the Phase-1 task: continue a legal prompt.

Generation
----------
``logits, state = model.prefill(prompt)`` reads the whole prompt once, then
``logits, state = model.decode_step(next_token, state)`` adds one token at a time
using cached keys/values.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn as nn

from legal_lm.models.config import ModelConfig


class LegalLM(nn.Module):
    arch_name = "base"
    description = ""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    # ---- training
    def sample_prefix_len(self, L: int, rng: random.Random) -> int:
        return 1

    def forward(self, idx: torch.Tensor, prefix_len: int) -> torch.Tensor:
        raise NotImplementedError

    # ---- generation
    def prefill(self, prompt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def decode_step(self, token: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def max_prompt_len(self, max_new_tokens: int) -> int:
        return self.cfg.max_seq_len - max_new_tokens

    # ---- bookkeeping
    def embedding_params(self) -> int:
        return self.tok_emb.weight.numel()

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())  # tied weights counted once
        emb = self.embedding_params()
        return {"total": total, "embedding": emb, "non_embedding": total - emb}
