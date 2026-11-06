"""Hierarchical multi-label statute classifier.

    chunks (<= n per document, chunk_len tokens) -> Phase-1 backbone -> pooled chunk vector
    -> projection + chunk-position embedding -> document layer -> 100 statute logits

document layer (``model.doc_layer``):
    gru_attn  : BiGRU over chunks, then one attention pooling shared by all labels
    label_attn: BiGRU over chunks, then per-label attention (each statute attends to the chunks that
                mention it -- statutes are often cited in one place of a long judgment)
    max       : element-wise max over chunk vectors (a statute cited in any chunk counts)
    mean      : mean of the chunk vectors
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lsi import P1_ROOT  # noqa: F401
from lsi.data import N_LABELS
from cj.model import ChunkEncoder, GradualUnfreeze
from lner.backbones import Backbone, load_phase1
from lner.peft import apply_method
from rr.peft_extra import EXTRA, AdaLoRAController, apply_extra


class LSIModel(nn.Module):
    def __init__(self, encoder: ChunkEncoder, mcfg: dict, n_labels: int = N_LABELS):
        super().__init__()
        d, h = encoder.backbone.d_model, mcfg.get("hidden", 384)
        self.encoder, self.n_labels = encoder, n_labels
        self.doc_layer = mcfg.get("doc_layer", "label_attn")
        self.proj = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(mcfg.get("dropout", 0.1)))
        self.pos = nn.Embedding(mcfg.get("max_chunks", 64), h)
        if self.doc_layer in ("gru_attn", "label_attn"):
            self.gru = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        if self.doc_layer == "label_attn":
            self.label_q = nn.Parameter(torch.randn(n_labels, h) * 0.02)   # one query per statute
            self.label_w = nn.Parameter(torch.randn(n_labels, h) * 0.02)   # one output vector per statute
            self.label_b = nn.Parameter(torch.zeros(n_labels))
        else:
            self.score = nn.Linear(h, 1)
            self.out = nn.Sequential(nn.Dropout(mcfg.get("dropout", 0.1)), nn.Linear(h, n_labels))
        self.drop = nn.Dropout(mcfg.get("dropout", 0.1))

    def head_modules(self):
        mods = [self.proj, self.pos]
        if self.doc_layer == "label_attn":
            mods.append(_ParamHolder(self))
        else:
            mods += [self.score, self.out]
        if getattr(self.encoder, "pool", "mean") == "attn":
            mods.append(self.encoder.pool_score)
        return mods + ([self.gru] if hasattr(self, "gru") else [])

    def doc_logits(self, chunk_vecs, doc_idx, n_docs):
        x = self.proj(chunk_vecs)
        counts = torch.bincount(doc_idx, minlength=n_docs)
        K = int(counts.max())
        pos = torch.cat([torch.arange(int(c), device=x.device) for c in counts])
        x = x + self.pos(pos.clamp(max=self.pos.num_embeddings - 1))
        pad = torch.zeros(n_docs, K, x.size(-1), device=x.device)
        valid = torch.zeros(n_docs, K, dtype=torch.bool, device=x.device)
        pad[doc_idx, pos] = x
        valid[doc_idx, pos] = True
        if hasattr(self, "gru"):
            pad = self.gru(pad)[0]
        if self.doc_layer == "label_attn":
            s = torch.einsum("bkh,lh->blk", pad, self.label_q).masked_fill(~valid[:, None, :], float("-inf"))
            ctx = torch.einsum("blk,bkh->blh", s.softmax(-1), pad)            # (B, L, h)
            return (self.drop(ctx) * self.label_w).sum(-1) + self.label_b
        if self.doc_layer == "max":
            doc = pad.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(1).values
        elif self.doc_layer == "mean":
            doc = (pad * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        else:
            a = self.score(pad).squeeze(-1).masked_fill(~valid, float("-inf")).softmax(-1)
            doc = (a.unsqueeze(-1) * pad).sum(1)
        return self.out(doc)


class _ParamHolder(nn.Module):
    """Exposes the label-attention parameters as a module so the optimiser / freezing code sees them."""

    def __init__(self, m):
        super().__init__()
        self._m = [m]

    def parameters(self, recurse=True):
        m = self._m[0]
        return iter([m.label_q, m.label_w, m.label_b])


def build_model(cfg: dict):
    mcfg, pcfg = cfg["model"], cfg["peft"]
    lm, tok = load_phase1(mcfg["phase1_ckpt"])
    bb = Backbone(lm, mcfg.get("attention", "causal"), mcfg.get("keep_layers"), mcfg.get("encdec_output", "encoder"))
    model = LSIModel(ChunkEncoder(bb, mcfg.get("chunk_pool", "mean")), mcfg)
    ctrl = None
    model.n_prompt = 0
    if pcfg["method"] in EXTRA:
        model.n_prompt = apply_extra(model, pcfg)  # prompt tuning: number of learned positions to prepend
    elif pcfg["method"] == "gradual":
        ctrl = GradualUnfreeze(bb, pcfg)
    else:
        shim = nn.Module()
        shim.backbone, shim.head = bb, nn.ModuleList([m for m in model.head_modules() if not isinstance(m, _ParamHolder)])
        apply_method(shim, pcfg)
    for m in model.head_modules():
        for p in m.parameters():
            p.requires_grad = True
    trainable = (sum(p.numel() for p in model.parameters()) if pcfg["method"] == "gradual"
                 else sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = sum(p.numel() for p in model.parameters())
    return model, tok, ctrl, {"total_params": total, "trainable_params": trainable,
                              "trainable_pct": round(100 * trainable / total, 3)}


def make_controller(model, ctrl, pcfg, total_steps):
    if ctrl is not None:
        return ctrl
    return AdaLoRAController(model, pcfg, total_steps) if pcfg["method"] == "adalora" else None


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers() if n.endswith("rank_mask")}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
