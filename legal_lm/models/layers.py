"""Building blocks shared by the four architectures.

Attention uses torch's fused ``scaled_dot_product_attention``.  It supports
  * grouped-query attention (fewer key/value heads than query heads),
  * rotary position embeddings,
  * boolean masks (causal, sliding-window, prefix) or additive float masks (ALiBi),
  * a key/value cache for fast token-by-token generation,
  * cross-attention to an encoder memory.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- norms
class RMSNorm(nn.Module):
    """Root-mean-square norm: LayerNorm without mean-centering or bias (cheaper)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


# --------------------------------------------------------------------------- feed-forward
class FeedForward(nn.Module):
    """Classic two-layer MLP with GELU or ReLU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu", bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=bias)
        self.proj = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class SwiGLU(nn.Module):
    """Gated MLP used by LLaMA/Mistral: down( silu(gate(x)) * up(x) )."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


# --------------------------------------------------------------------------- positions
def sinusoidal_table(n_pos: int, dim: int) -> torch.Tensor:
    """Fixed sin/cos position table from 'Attention Is All You Need'."""
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(n_pos, dim)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class Rotary(nn.Module):
    """Rotary position embedding (RoPE): rotates query/key pairs by a position-dependent
    angle, so attention scores depend on the *relative* distance between tokens."""

    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_len)

    def _build(self, n: int) -> None:
        freqs = torch.outer(torch.arange(n, dtype=torch.float32, device=self.inv_freq.device), self.inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:  # x: (B, H, T, hd)
        T = x.size(2)
        if offset + T > self.cos.size(0):
            self._build(2 * (offset + T))
        cos = self.cos[offset:offset + T].to(x.dtype)[None, None]
        sin = self.sin[offset:offset + T].to(x.dtype)[None, None]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Head-specific slopes from the ALiBi paper (Press et al., 2022)."""

    def power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = power_of_two(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = power_of_two(closest) + power_of_two(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)


# --------------------------------------------------------------------------- masks
def causal_mask(T: int, device, window: Optional[int] = None) -> torch.Tensor:
    """Boolean (T, T) mask, True = may attend. Optional sliding window of `window` keys."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    allowed = j <= i
    if window is not None and window < T:
        allowed &= j > i - window
    return allowed


# --------------------------------------------------------------------------- attention
class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.n_kv = n_kv_heads or n_heads
        assert n_heads % self.n_kv == 0, "n_heads must be a multiple of n_kv_heads"
        self.hd = d_model // n_heads
        self.q = nn.Linear(d_model, n_heads * self.hd, bias=bias)
        self.k = nn.Linear(d_model, self.n_kv * self.hd, bias=bias)
        self.v = nn.Linear(d_model, self.n_kv * self.hd, bias=bias)
        self.o = nn.Linear(n_heads * self.hd, d_model, bias=bias)
        self.o.is_residual_proj = True
        self.dropout = dropout

    def _heads(self, t: torch.Tensor, n: int) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, n, self.hd).transpose(1, 2)  # (B, n, T, hd)

    def forward(self, x, memory=None, mask=None, is_causal: bool = False,
                rope: Optional[Rotary] = None, pos: int = 0,
                cache: Optional[dict] = None, cache_keep: Optional[int] = None):
        """
        x       : (B, T, d) queries (and keys/values for self-attention)
        memory  : (B, S, d) encoder output for cross-attention, else None
        mask    : bool or additive float mask broadcastable to (B, H, T, S)
        pos     : absolute position of x[:, 0] (for RoPE during generation)
        cache   : dict filled with past keys/values (generation only)
        cache_keep : keep only the last N keys in the cache (sliding window)
        """
        B, T, _ = x.shape
        q = self._heads(self.q(x), self.n_heads)
        if memory is not None and cache is not None and "k" in cache:
            k, v = cache["k"], cache["v"]  # encoder memory projections computed once
        else:
            src = x if memory is None else memory
            k = self._heads(self.k(src), self.n_kv)
            v = self._heads(self.v(src), self.n_kv)
            if rope is not None:
                k = rope(k, pos)
            if memory is None and cache is not None and "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            if cache is not None:  # store before GQA expansion: the cache stays small
                keep = cache_keep if (cache_keep and memory is None) else None
                cache["k"] = k[:, :, -keep:] if keep else k
                cache["v"] = v[:, :, -keep:] if keep else v
        if rope is not None:
            q = rope(q, pos)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.to(q.dtype)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and mask is None,
        )
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


# --------------------------------------------------------------------------- init
def init_weights(module: nn.Module, n_residual_layers: int) -> None:
    """GPT-2 style init; residual output projections are scaled by 1/sqrt(2*L)."""
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "is_residual_proj", False) or name.endswith("proj"):
                std = 0.02 / math.sqrt(2 * n_residual_layers)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


def new_caches(n: int) -> List[dict]:
    return [{} for _ in range(n)]
