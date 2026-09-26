"""A1 - Decoder-only Transformer (GPT-2 style). The baseline.

    tokens -> token embedding + learned absolute position embedding
           -> N x [ LayerNorm -> causal multi-head self-attention -> +residual
                    LayerNorm -> GELU MLP                          -> +residual ]
           -> LayerNorm -> linear head (tied to the token embedding) -> next-token logits

Objective: causal language modelling, every token predicts the next one.
Positions: a learned table, so the model cannot read beyond ``max_seq_len`` tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from legal_lm.models.base import LegalLM
from legal_lm.models.layers import Attention, FeedForward, init_weights, new_caches


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout=dropout, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout, activation="gelu")
        self.drop = nn.Dropout(dropout)

    def forward(self, x, is_causal=True, cache=None):
        x = x + self.drop(self.attn(self.ln1(x), is_causal=is_causal, cache=cache))
        return x + self.mlp(self.ln2(x))


class DecoderOnlyGPT(LegalLM):
    arch_name = "gpt"
    description = "Decoder-only GPT: learned positions, LayerNorm, GELU, full causal MHA"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        init_weights(self, cfg.n_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def embedding_params(self):
        return self.tok_emb.weight.numel() + self.pos_emb.weight.numel()

    def _run(self, ids, pos=0, caches=None):
        T = ids.size(1)
        positions = torch.arange(pos, pos + T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(positions))
        causal = pos == 0 and T > 1  # a single new token may see every cached key
        for i, block in enumerate(self.blocks):
            x = block(x, is_causal=causal, cache=None if caches is None else caches[i])
        return self.ln_f(x)

    def forward(self, idx, prefix_len):
        h = self._run(idx[:, :-1])
        return self.lm_head(h[:, prefix_len - 1:])  # only the positions we score

    @torch.no_grad()
    def prefill(self, prompt):
        caches = new_caches(len(self.blocks))
        h = self._run(prompt, 0, caches)
        return self.lm_head(h[:, -1]), {"caches": caches, "pos": prompt.size(1)}

    @torch.no_grad()
    def decode_step(self, token, state):
        h = self._run(token, state["pos"], state["caches"])
        state["pos"] += 1
        return self.lm_head(h[:, -1]), state
