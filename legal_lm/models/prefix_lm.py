"""A3 - Hybrid encoder/decoder in ONE stack: a Prefix-LM with ALiBi (UniLM / GLM style).

A single Transformer stack whose attention mask changes the role of each token:

    prompt tokens      -> attend bidirectionally to the whole prompt (encoder-like)
    continuation tokens-> attend to the prompt + earlier continuation (decoder-like)

        keys:    p0 p1 p2 | c0 c1 c2
    p0..p2       1  1  1  | 0  0  0      bidirectional block
    c0           1  1  1  | 1  0  0      causal block
    c1           1  1  1  | 1  1  0
    c2           1  1  1  | 1  1  1

Positions: no position embeddings at all.  ALiBi adds a penalty -slope_h * |i - j|
to attention scores, which lets the model generalise to longer inputs than it saw.

Objective: with probability ``causal_prob`` a normal causal example (prefix length 1,
every token supervised); otherwise a random prefix of 64..0.75L tokens is read
bidirectionally and only the rest is supervised.

Differences from A1/A2: encoder-style bidirectional attention without a separate
encoder or cross-attention (parameters shared between "reading" and "writing"),
relative ALiBi positions instead of learned or sinusoidal ones.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from legal_lm.models.base import LegalLM
from legal_lm.models.layers import Attention, FeedForward, alibi_slopes, init_weights, new_caches


class PrefixBlock(nn.Module):
    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, dropout=p)
        self.mlp = FeedForward(d, d_ff, p, activation="gelu")
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, cache=None):
        x = x + self.drop(self.attn(self.ln1(x), mask=mask, cache=cache))
        return x + self.mlp(self.ln2(x))


class PrefixLM(LegalLM):
    arch_name = "prefixlm"
    description = "Prefix-LM hybrid: bidirectional prompt + causal continuation in one stack, ALiBi"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([PrefixBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                     for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        self.register_buffer("slopes", alibi_slopes(cfg.n_heads), persistent=False)
        init_weights(self, cfg.n_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def attention_bias(self, n_query, n_key, prefix_len, past, device):
        """Additive (1, H, Tq, Tk) mask: ALiBi distance penalty, -inf where not allowed."""
        qpos = torch.arange(past, past + n_query, device=device)[:, None]
        kpos = torch.arange(n_key, device=device)[None, :]
        allowed = (kpos < prefix_len) | (kpos <= qpos)
        bias = -self.slopes[:, None, None] * (qpos - kpos).abs().float()[None]
        return bias.masked_fill(~allowed[None], float("-inf"))[None]

    def _run(self, ids, prefix_len, past=0, caches=None):
        x = self.drop(self.tok_emb(ids))
        mask = self.attention_bias(ids.size(1), past + ids.size(1), prefix_len, past, ids.device)
        for i, block in enumerate(self.blocks):
            x = block(x, mask, None if caches is None else caches[i])
        return self.ln_f(x)

    def sample_prefix_len(self, L, rng: random.Random):
        if rng.random() < self.cfg.causal_prob:
            return 1
        lo, hi = min(64, L // 4), max(min(64, L // 4) + 1, int(0.75 * L))
        return rng.randint(lo, hi)

    def forward(self, idx, prefix_len):
        h = self._run(idx[:, :-1], prefix_len)
        return self.lm_head(h[:, prefix_len - 1:])

    @torch.no_grad()
    def prefill(self, prompt):
        caches = new_caches(len(self.blocks))
        P = prompt.size(1)
        h = self._run(prompt, prefix_len=P, past=0, caches=caches)
        return self.lm_head(h[:, -1]), {"caches": caches, "pos": P, "prefix_len": P}

    @torch.no_grad()
    def decode_step(self, token, state):
        h = self._run(token, state["prefix_len"], state["pos"], state["caches"])
        state["pos"] += 1
        return self.lm_head(h[:, -1]), state
