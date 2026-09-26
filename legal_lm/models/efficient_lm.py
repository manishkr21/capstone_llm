"""A4 - Efficient / lightweight decoder (LLaMA + Mistral + Gemma-2 ideas).

    tokens -> token embedding (no position table)
           -> N x [ RMSNorm -> attention (RoPE, grouped-query, local OR global) -> +res
                    RMSNorm -> SwiGLU MLP                                        -> +res ]
           -> RMSNorm -> tied linear head

Efficiency ideas, each with a clear cost argument:
  * Grouped-query attention (GQA): e.g. 12 query heads share 4 key/value heads,
    so the K/V projections and the generation KV-cache are 3x smaller.
  * Sliding-window attention: most layers only look at the last ``window`` tokens
    (cost O(T * window) instead of O(T^2)); every ``global_every``-th layer is
    global so long-range legal references (parties, sections) are still reachable.
    During generation local layers keep a rolling cache of ``window - 1`` keys.
  * RMSNorm and bias-free linear layers: fewer parameters and memory reads.
  * RoPE relative positions and a SwiGLU gated MLP (better quality per parameter).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from legal_lm.models.base import LegalLM
from legal_lm.models.layers import Attention, RMSNorm, Rotary, SwiGLU, causal_mask, init_weights, new_caches


class EfficientBlock(nn.Module):
    def __init__(self, cfg, is_global: bool):
        super().__init__()
        self.is_global = is_global
        self.window = None if is_global else cfg.window
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, dropout=cfg.dropout, bias=False)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, pos, local_mask, cache=None):
        causal = pos == 0 and x.size(1) > 1
        mask = local_mask if (self.window is not None and causal) else None
        keep = self.window - 1 if self.window else None
        h = self.attn(self.norm1(x), mask=mask, is_causal=causal, rope=rope, pos=pos,
                      cache=cache, cache_keep=keep)
        x = x + self.drop(h)
        return x + self.mlp(self.norm2(x))


class EfficientLM(LegalLM):
    arch_name = "efficient"
    description = "Efficient decoder: RoPE, RMSNorm, SwiGLU, grouped-query + sliding-window attention"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = Rotary(cfg.d_model // cfg.n_heads, 2 * cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList([
            EfficientBlock(cfg, is_global=(i % cfg.global_every == cfg.global_every - 1))
            for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        init_weights(self, cfg.n_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def _run(self, ids, pos=0, caches=None):
        x = self.drop(self.tok_emb(ids))
        T = ids.size(1)
        local_mask = causal_mask(T, ids.device, self.cfg.window) if (pos == 0 and T > self.cfg.window) else None
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, pos, local_mask, None if caches is None else caches[i])
        return self.norm(x)

    def forward(self, idx, prefix_len):
        h = self._run(idx[:, :-1])
        return self.lm_head(h[:, prefix_len - 1:])

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
