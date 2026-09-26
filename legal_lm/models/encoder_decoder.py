"""A2 - Encoder-decoder Transformer (Vaswani et al. 2017 / T5 / BART family).

    prompt  -> [ENCODER]  N_enc x [ bidirectional self-attention, ReLU MLP ]  -> memory
    target  -> [DECODER]  N_dec x [ causal self-attention,
                                    cross-attention to memory, ReLU MLP ] -> logits

Objective: prefix-to-continuation.  Each training block is cut at a random point p:
the encoder reads tokens [0, p) and the decoder predicts tokens [p, L).  The decoder
input starts with the last prompt token, so no extra BOS token is needed.
This mirrors the test task exactly: read a legal prompt, write its continuation.

Differences from A1: separate encoder and decoder stacks, cross-attention,
bidirectional prompt encoding, fixed sinusoidal positions, ReLU MLP.
Loss is computed only on the continuation part (about 60% of tokens per block).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from legal_lm.models.base import LegalLM
from legal_lm.models.layers import Attention, FeedForward, init_weights, new_caches, sinusoidal_table


class EncoderBlock(nn.Module):
    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, dropout=p)
        self.mlp = FeedForward(d, d_ff, p, activation="relu")
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))  # no mask: every prompt token sees all others
        return x + self.mlp(self.ln2(x))


class DecoderBlock(nn.Module):
    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, h, dropout=p)
        self.cross_attn = Attention(d, h, dropout=p)
        self.mlp = FeedForward(d, d_ff, p, activation="relu")
        self.drop = nn.Dropout(p)

    def forward(self, x, memory, is_causal, cache=None):
        self_cache = None if cache is None else cache.setdefault("self", {})
        cross_cache = None if cache is None else cache.setdefault("cross", {})
        x = x + self.drop(self.self_attn(self.ln1(x), is_causal=is_causal, cache=self_cache))
        x = x + self.drop(self.cross_attn(self.ln2(x), memory=memory, cache=cross_cache))
        return x + self.mlp(self.ln3(x))


class EncoderDecoderLM(LegalLM):
    arch_name = "encdec"
    description = "Encoder-decoder: bidirectional prompt encoder + cross-attending decoder, sinusoidal positions"

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.padded_vocab, d)  # shared by encoder, decoder and head
        self.register_buffer("pos_table", sinusoidal_table(2 * cfg.max_seq_len, d), persistent=False)
        self.emb_scale = math.sqrt(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([EncoderBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                      for _ in range(cfg.n_enc_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                      for _ in range(cfg.n_dec_layers)])
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.padded_vocab, bias=False)
        init_weights(self, cfg.n_enc_layers + cfg.n_dec_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def _embed(self, ids, pos=0):
        T = ids.size(1)
        return self.drop(self.tok_emb(ids) * self.emb_scale + self.pos_table[pos:pos + T].to(ids.device))

    def encode(self, src):
        x = self._embed(src)
        for block in self.encoder:
            x = block(x)
        return self.enc_norm(x)

    def decode(self, tgt, memory, pos=0, caches=None):
        x = self._embed(tgt, pos)
        causal = pos == 0 and tgt.size(1) > 1
        for i, block in enumerate(self.decoder):
            x = block(x, memory, is_causal=causal, cache=None if caches is None else caches[i])
        return self.dec_norm(x)

    def sample_prefix_len(self, L, rng: random.Random):
        lo, hi = max(1, L // 4), max(L // 4 + 1, L - 64)
        return rng.randint(lo, hi)

    def forward(self, idx, prefix_len):
        memory = self.encode(idx[:, :prefix_len])
        h = self.decode(idx[:, prefix_len - 1:-1], memory)
        return self.lm_head(h)

    @torch.no_grad()
    def prefill(self, prompt):
        memory = self.encode(prompt)
        caches = new_caches(len(self.decoder))
        h = self.decode(prompt[:, -1:], memory, 0, caches)
        return self.lm_head(h[:, -1]), {"memory": memory, "caches": caches, "pos": 1}

    @torch.no_grad()
    def decode_step(self, token, state):
        h = self.decode(token, state["memory"], state["pos"], state["caches"])
        state["pos"] += 1
        return self.lm_head(h[:, -1]), state
