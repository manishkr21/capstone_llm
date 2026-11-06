"""Hierarchical judgment classifier.

    chunks (n per document, <= chunk_len tokens) -> Phase-1 backbone -> masked mean pool per chunk
    -> projection + chunk-position embedding -> document layer -> 2-way logits

document layer (``model.doc_layer``):
    gru_attn : BiGRU over the chunk sequence, then attention pooling (learned scoring vector)
    attn     : attention pooling over the projected chunk vectors
    mean     : mean of the chunk vectors
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cj import P1_ROOT  # noqa: F401
from lner.backbones import Backbone, load_phase1
from lner.peft import apply_method
from rr.peft_extra import EXTRA, AdaLoRAController, apply_extra


class ChunkEncoder(nn.Module):
    """``pool``: mean of the chunk's token states, or ``attn`` -- a learned scoring vector weights the tokens
    (lets a few outcome-bearing tokens dominate the chunk vector)."""

    def __init__(self, backbone: Backbone, pool: str = "mean"):
        super().__init__()
        self.backbone, self.pool = backbone, pool
        if pool == "attn":
            self.pool_score = nn.Linear(backbone.d_model, 1)

    def forward(self, ids, mask):
        h = self.backbone(ids).float()                       # (C, L, d)
        if self.pool == "attn":
            a = self.pool_score(h).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(-1)
            return (a.unsqueeze(-1) * h).sum(1)
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)      # (C, d)


class CJModel(nn.Module):
    def __init__(self, encoder: ChunkEncoder, mcfg: dict):
        super().__init__()
        d, h = encoder.backbone.d_model, mcfg.get("hidden", 256)
        self.encoder = encoder
        self.doc_layer = mcfg.get("doc_layer", "gru_attn")
        self.proj = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(mcfg.get("dropout", 0.1)))
        self.pos = nn.Embedding(mcfg.get("max_chunks", 64), h)
        if self.doc_layer == "gru_attn":
            self.gru = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        self.score = nn.Linear(h, 1)
        self.out = nn.Sequential(nn.Dropout(mcfg.get("dropout", 0.1)), nn.Linear(h, 2))

    def head_modules(self):
        mods = [self.proj, self.pos, self.score, self.out]
        if getattr(self.encoder, "pool", "mean") == "attn":
            mods.append(self.encoder.pool_score)
        return mods + ([self.gru] if self.doc_layer == "gru_attn" else [])

    def doc_logits(self, chunk_vecs, doc_idx, n_docs):
        """chunk_vecs (C, d) in document order; doc_idx (C,) -> logits (n_docs, 2)."""
        x = self.proj(chunk_vecs)
        counts = torch.bincount(doc_idx, minlength=n_docs)
        K = int(counts.max())
        pos = torch.cat([torch.arange(int(c), device=x.device) for c in counts])
        x = x + self.pos(pos.clamp(max=self.pos.num_embeddings - 1))
        pad = torch.zeros(n_docs, K, x.size(-1), device=x.device)
        valid = torch.zeros(n_docs, K, dtype=torch.bool, device=x.device)
        pad[doc_idx, pos] = x
        valid[doc_idx, pos] = True
        if self.doc_layer == "mean":
            doc = (pad * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        else:
            if self.doc_layer == "gru_attn":
                pad = self.gru(pad)[0]
            a = self.score(pad).squeeze(-1).masked_fill(~valid, float("-inf")).softmax(-1)
            doc = (a.unsqueeze(-1) * pad).sum(1)
        return self.out(doc)


class GradualUnfreeze:
    """Layer freezing / unfreezing (ULMFiT-style): start with only the top ``start_blocks`` blocks (and the
    final norm) trainable, unfreeze one more block every ``every`` optimizer steps, top to bottom."""

    def __init__(self, bb, pcfg):
        self.blocks = list(bb.all_blocks)
        self.every, self.n = pcfg.get("unfreeze_every", 200), pcfg.get("start_blocks", 1)
        for name, p in bb.lm.named_parameters():
            p.requires_grad = any(k in name for k in ("norm.", "ln_f", "dec_norm")) and "blocks" not in name
        self._apply()

    def _apply(self):
        for b in self.blocks[len(self.blocks) - self.n:]:
            for p in b.parameters():
                p.requires_grad = True

    def step(self, step):
        if step and step % self.every == 0 and self.n < len(self.blocks):
            self.n += 1
            self._apply()

    def penalty(self):
        return 0.0


def build_model(cfg: dict):
    mcfg, pcfg = cfg["model"], cfg["peft"]
    lm, tok = load_phase1(mcfg["phase1_ckpt"])
    bb = Backbone(lm, mcfg.get("attention", "causal"), mcfg.get("keep_layers"), mcfg.get("encdec_output", "encoder"))
    model = CJModel(ChunkEncoder(bb, mcfg.get("chunk_pool", "mean")), mcfg)
    ctrl = None
    if pcfg["method"] in EXTRA:
        apply_extra(model, pcfg)
    elif pcfg["method"] == "gradual":
        ctrl = GradualUnfreeze(bb, pcfg)
    else:
        shim = nn.Module()
        shim.backbone, shim.head = bb, nn.ModuleList(model.head_modules())
        apply_method(shim, pcfg)
    for m in model.head_modules():
        for p in m.parameters():
            p.requires_grad = True
    if pcfg["method"] == "gradual":  # count what is trainable by the end of the schedule
        train = sum(p.numel() for p in model.parameters())
    else:
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, tok, ctrl, {"total_params": total, "trainable_params": train,
                              "trainable_pct": round(100 * train / total, 3)}


def make_controller(model, ctrl, pcfg, total_steps):
    if ctrl is not None:
        return ctrl
    return AdaLoRAController(model, pcfg, total_steps) if pcfg["method"] == "adalora" else None


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers() if n.endswith("rank_mask")}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
