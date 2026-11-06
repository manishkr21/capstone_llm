"""Turn a Phase-1 language model into an encoder that returns one hidden vector per token.

    A1 gpt       : token + learned positions -> causal blocks (or bidirectional, see below)
    A2 encdec    : the bidirectional ENCODER only; the decoder and LM head are dropped
    A3 prefixlm  : the whole window is the "prefix", so attention is fully bidirectional
    A4 efficient : RoPE + GQA blocks; local layers keep their sliding window

``attention: causal`` keeps the Phase-1 masks of A1/A4 (a token sees only its left
context).  ``attention: bidirectional`` removes the causal mask (for A4 the local layers
see +-window tokens), which the fine-tuning has to adapt to.  A2/A3 are always bidirectional.

``keep_layers`` keeps a subset of the Phase-1 blocks (used to build a smaller
knowledge-distillation student that starts from Phase-1 weights).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from lner import P1_ROOT  # noqa: F401  (puts legal_lm on sys.path)
from lner.crf import CRF
from legal_lm.data.tokenizer import LegalTokenizer
from legal_lm.models import ModelConfig, build_model


def load_phase1(ckpt_path):
    ckpt = torch.load(P1_ROOT / ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tok_cfg = dict(cfg["tokenizer"])
    if tok_cfg["type"] != "gpt2":
        tok_cfg["path"] = str(P1_ROOT / tok_cfg["path"])
    tok = LegalTokenizer.from_config(tok_cfg)
    mcfg = ModelConfig.from_dict(cfg["model"], vocab_size=tok.vocab_size, max_seq_len=cfg["train"]["block_size"])
    lm = build_model(mcfg)
    lm.load_state_dict(ckpt["model"])
    return lm, tok


class Backbone(nn.Module):
    def __init__(self, lm, attention: str = "causal", keep_layers: Optional[List[int]] = None,
                 encdec_output: str = "encoder"):
        super().__init__()
        self.arch = lm.cfg.arch
        self.cfg = lm.cfg
        self.lm = lm
        # A2: "encoder" tags from the encoder output; "decoder" runs the decoder over the same
        # window, cross-attending to the encoder (BART-style token classification)
        self.encdec_output = encdec_output
        if self.arch == "encdec" and encdec_output == "encoder":
            del lm.decoder, lm.dec_norm
        lm.lm_head = nn.Identity()  # tied to tok_emb, so no weights are lost
        self.bidirectional = self.arch == "prefixlm" or attention == "bidirectional" or (
            self.arch == "encdec" and encdec_output == "encoder")
        if keep_layers is not None:
            blocks = self.blocks
            kept = nn.ModuleList([blocks[i] for i in keep_layers])
            if self.arch == "encdec":
                lm.encoder = kept
            else:
                lm.blocks = kept
        self.d_model = lm.cfg.d_model

    @property
    def blocks(self) -> nn.ModuleList:
        return self.lm.encoder if self.arch == "encdec" else self.lm.blocks

    @property
    def all_blocks(self) -> list:
        """Every block that processes the window (A2 in decoder mode: encoder + decoder)."""
        if self.arch == "encdec" and self.encdec_output == "decoder":
            return list(self.lm.encoder) + list(self.lm.decoder)
        return list(self.blocks)

    @property
    def max_len(self) -> int:
        return self.cfg.max_seq_len if self.arch == "gpt" else 10 ** 9

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        lm, T = self.lm, ids.size(1)
        if self.arch == "encdec":
            memory = lm.encode(ids)
            if self.encdec_output == "encoder":
                return memory
            x = lm._embed(ids)
            for block in lm.decoder:
                x = block(x, memory, is_causal=not self.bidirectional)
            return lm.dec_norm(x)
        if self.arch == "prefixlm":
            return lm._run(ids, prefix_len=T)
        if self.arch == "gpt":
            x = lm.drop(lm.tok_emb(ids) + lm.pos_emb(torch.arange(T, device=ids.device)))
            for block in lm.blocks:
                x = block(x, is_causal=not self.bidirectional)
            return lm.ln_f(x)
        # efficient (A4): replicate EfficientBlock.forward with our own masks
        x = lm.drop(lm.tok_emb(ids))
        i = torch.arange(T, device=ids.device)
        dist = i[:, None] - i[None, :]
        w = self.cfg.window
        local = (dist.abs() < w) if self.bidirectional else ((dist >= 0) & (dist < w))
        for block in lm.blocks:
            mask = None if block.is_global else local
            causal = (not self.bidirectional) and mask is None
            x = x + block.drop(block.attn(block.norm1(x), mask=mask, is_causal=causal, rope=lm.rope, pos=0))
            x = x + block.mlp(block.norm2(x))
        return lm.norm(x)


class Tagger(nn.Module):
    """Backbone + token-classification head (linear, small MLP, or linear + CRF)."""

    def __init__(self, backbone: Backbone, tags: List[str], head: str = "linear", dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        d = backbone.d_model
        n_tags = len(tags)
        self.drop = nn.Dropout(dropout)
        self.crf = CRF(tags) if head == "crf" else None
        if head == "mlp":
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_tags))
        else:
            self.head = nn.Linear(d, n_tags)

    def forward(self, ids):
        return self.head(self.drop(self.backbone(ids)))


def build_tagger(mcfg: dict, tags: List[str]):
    lm, tok = load_phase1(mcfg["phase1_ckpt"])
    bb = Backbone(lm, mcfg.get("attention", "causal"), mcfg.get("keep_layers"),
                  mcfg.get("encdec_output", "encoder"))
    return Tagger(bb, tags, mcfg.get("head", "linear"), mcfg.get("dropout", 0.1)), tok
