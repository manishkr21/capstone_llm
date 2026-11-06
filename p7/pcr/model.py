"""Citation-window bi-encoder on a Phase-1 backbone.

    query  : one vector per citation window (+-60 words around each <CITATION> of the query case)
    case   : one vector per 512-token chunk of the candidate case
    s(w,c) = max over the chunks of c of cos(w, chunk)            (which part of the prior case is cited)
    s(q,c) = window aggregation over the windows of q             (max | lse | mean)

A window and a chunk go through the same backbone + pooling + linear projection. Trained with a multi-positive
InfoNCE loss over the batch's candidates (gold, lexical hard negatives, the other queries' candidates).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cj.model import ChunkEncoder
from lner.backbones import Backbone, load_phase1
from lner.peft import apply_method
from rr.peft_extra import EXTRA, apply_extra

PAD = 1


class Retriever(nn.Module):
    def __init__(self, encoder: ChunkEncoder, mcfg: dict):
        super().__init__()
        d = encoder.backbone.d_model
        self.encoder = encoder
        self.head = nn.Linear(d, mcfg.get("dim", 256))
        self.win_agg = mcfg.get("win_agg", "lse")
        self.win_temp = mcfg.get("win_temp", 0.05)
        self.chunk_agg = mcfg.get("chunk_agg", "max")
        self.n_prompt = 0

    def head_modules(self):
        mods = [self.head]
        if getattr(self.encoder, "pool", "mean") == "attn":
            mods.append(self.encoder.pool_score)
        return mods

    def embed(self, seqs, device, max_tokens=65536, normalize=True):
        """seqs: list of int arrays -> (n, dim) unit vectors (raw head outputs if not ``normalize``: the
        cross-encoder's scores); batches sorted by length (padding is not masked inside the backbone, so
        similar lengths keep its effect small)."""
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        out = [None] * len(seqs)
        i = 0
        while i < len(order):
            L = len(seqs[order[i]]) + self.n_prompt
            n = max(1, max_tokens // L)
            idx = order[i:i + n]
            ids = torch.full((len(idx), L), PAD, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.bool)
            for r, j in enumerate(idx):
                s = torch.as_tensor(seqs[j], dtype=torch.long)
                ids[r, self.n_prompt:self.n_prompt + len(s)] = s
                mask[r, self.n_prompt:self.n_prompt + len(s)] = True
            v = self.head(self.encoder(ids.to(device), mask.to(device))).float()
            v = F.normalize(v, dim=-1) if normalize else v
            for r, j in enumerate(idx):
                out[j] = v[r]
            i += n
        return torch.stack(out)

    def score(self, W, w_owner, n_q, C, c_owner, n_c):
        """W (nw, d) windows of n_q queries, C (nc, d) chunks of n_c cases -> (n_q, n_c) scores."""
        sim = W @ C.T                                                     # (nw, nc_chunks)
        idx = c_owner.unsqueeze(0).expand_as(sim)
        if self.chunk_agg == "max":
            per = sim.new_full((sim.size(0), n_c), -1e4).scatter_reduce(1, idx, sim, "amax")
        else:
            cnt = torch.bincount(c_owner, minlength=n_c).clamp(min=1)
            per = sim.new_zeros((sim.size(0), n_c)).scatter_add(1, idx, sim) / cnt
        widx = w_owner.unsqueeze(1).expand_as(per)
        if self.win_agg == "max":
            return per.new_full((n_q, n_c), -1e4).scatter_reduce(0, widx, per, "amax")
        if self.win_agg == "mean":
            cnt = torch.bincount(w_owner, minlength=n_q).clamp(min=1)
            return per.new_zeros((n_q, n_c)).scatter_add(0, widx, per) / cnt[:, None]
        t = self.win_temp                                                 # lse: soft max over windows
        m = per.new_full((n_q, n_c), -1e4).scatter_reduce(0, widx, per, "amax")
        e = torch.exp((per - m[w_owner]) / t)
        return m + t * torch.log(per.new_zeros((n_q, n_c)).scatter_add(0, widx, e))


def build_model(cfg: dict):
    mcfg, pcfg = cfg["model"], cfg["peft"]
    lm, tok = load_phase1(mcfg["phase1_ckpt"])
    bb = Backbone(lm, mcfg.get("attention", "causal"), mcfg.get("keep_layers"), mcfg.get("encdec_output", "decoder"))
    model = Retriever(ChunkEncoder(bb, mcfg.get("chunk_pool", "mean")), mcfg)
    if pcfg["method"] in EXTRA:
        model.n_prompt = apply_extra(model, pcfg)
    else:
        shim = nn.Module()
        shim.backbone, shim.head = bb, nn.ModuleList(model.head_modules())
        apply_method(shim, pcfg)
    for m in model.head_modules():
        for p in m.parameters():
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tok, {"total_params": total, "trainable_params": trainable,
                        "trainable_pct": round(100 * trainable / total, 3)}


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def info_nce(S, pos, self_mask, tau):
    """Multi-positive InfoNCE: every (query, gold case) pair against all of that query's non-gold candidates.
    S, pos, self_mask: (n_q, n_c); self_mask marks a query's own case (dropped from its row)."""
    logits = (S / tau).masked_fill(self_mask, -1e4)
    neg = logits.masked_fill(pos, -1e4)                                   # (n_q, n_c) negatives only
    lse_neg = torch.logsumexp(neg, 1, keepdim=True)
    lp = logits - torch.logaddexp(logits, lse_neg)                        # log p(pos | pos + negatives)
    return -(lp * pos).sum() / pos.sum().clamp(min=1)
