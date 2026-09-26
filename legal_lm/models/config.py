"""One dataclass holds the hyper-parameters of every architecture.
Each architecture reads only the fields it needs (see the comments)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class ModelConfig:
    arch: str = "gpt"              # gpt | encdec | prefixlm | efficient
    vocab_size: int = 32000        # set from the tokenizer at build time
    max_seq_len: int = 1024        # = train.block_size
    d_model: int = 768
    n_layers: int = 8              # gpt, prefixlm, efficient
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.0
    tie_embeddings: bool = True
    # A2 encoder-decoder
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    # A3 prefix-LM: probability of a plain causal (prefix length 1) training example
    causal_prob: float = 0.5
    # A4 efficient: grouped-query attention, sliding windows, rotary positions
    n_kv_heads: Optional[int] = None
    window: int = 256
    global_every: int = 3
    rope_theta: float = 10000.0

    @property
    def padded_vocab(self) -> int:
        """Round the embedding table up to a multiple of 64 (faster matmuls on GPU)."""
        return ((self.vocab_size + 63) // 64) * 64

    @classmethod
    def from_dict(cls, d: dict, **overrides) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        merged = {**d, **overrides}
        return cls(**{k: v for k, v in merged.items() if k in known})
