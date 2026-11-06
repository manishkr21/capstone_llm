"""Hierarchical rhetorical-role model.

    sentence encoder : Phase-1 backbone over packed windows -> mean of each sentence's token states
    document layer   : [sentence vector ; relative-position embedding ; domain embedding]
                       -> BiLSTM over the sentences of the document (``doc_layer: bilstm``)
                       or nothing (``doc_layer: none``, sentence-only head)
    output           : linear emissions + linear-chain CRF over the sentence sequence
                       (``crf: true``; Viterbi decoding) or independent softmax.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from rr import P1_ROOT  # noqa: F401
from rr.data import pack, window_batch
from lner.backbones import Backbone, load_phase1
from lner.peft import apply_method
from rr.peft_extra import EXTRA, AdaLoRAController, apply_extra


class CRF(nn.Module):
    """Linear-chain CRF over sentences, all transitions allowed."""

    def __init__(self, n: int):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def nll(self, emit, y):
        """emit (T, n) float32, y (T,). Negative log-likelihood per sentence."""
        T = emit.shape[0]
        gold = self.start[y[0]] + emit.gather(1, y[:, None]).sum() + self.trans[y[:-1], y[1:]].sum() + self.end[y[-1]]
        alpha = self.start + emit[0]
        for t in range(1, T):
            alpha = torch.logsumexp(alpha[:, None] + self.trans, 0) + emit[t]
        return (torch.logsumexp(alpha + self.end, 0) - gold) / T

    @torch.no_grad()
    def viterbi(self, emit) -> List[int]:
        score, back = self.start + emit[0], []
        for t in range(1, emit.shape[0]):
            s = score[:, None] + self.trans
            back.append(s.argmax(0))
            score = s.max(0).values + emit[t]
        best = [int((score + self.end).argmax())]
        for b in reversed(back):
            best.append(int(b[best[-1]]))
        return best[::-1]


class SentenceEncoder(nn.Module):
    def __init__(self, backbone: Backbone, pad_id: int):
        super().__init__()
        self.backbone, self.pad_id, self.n_prompt = backbone, pad_id, 0

    def forward(self, sents, groups, max_windows: int = 16):
        """Vectors (n_sents_in_groups, d) for the sentences in ``groups`` (in order)."""
        outs = []
        dev = next(self.backbone.parameters()).device
        for i in range(0, len(groups), max_windows):
            ids, own = window_batch(sents, groups[i:i + max_windows], self.pad_id, self.n_prompt)
            ids, own = ids.to(dev), own.to(dev)
            h = self.backbone(ids).float()  # (W, L, d)
            n = int(own.max()) + 1
            flat, o = h.reshape(-1, h.size(-1)), own.reshape(-1)
            keep = o >= 0
            s = torch.zeros(n, h.size(-1), device=dev).index_add_(0, o[keep], flat[keep])
            c = torch.zeros(n, device=dev).index_add_(0, o[keep], torch.ones_like(o[keep], dtype=torch.float))
            outs.append(s / c[:, None])
        return torch.cat(outs)


class RRModel(nn.Module):
    def __init__(self, encoder: SentenceEncoder, n_labels: int, mcfg: dict):
        super().__init__()
        d, h = encoder.backbone.d_model, mcfg.get("hidden", 256)
        self.encoder = encoder
        self.doc_layer = mcfg.get("doc_layer", "bilstm")
        self.n_pos = mcfg.get("pos_buckets", 20)
        self.proj = nn.Sequential(nn.Dropout(mcfg.get("dropout", 0.1)), nn.Linear(d, h), nn.GELU())
        self.pos = nn.Embedding(self.n_pos, h)
        self.dom = nn.Embedding(2, h)
        self.cues = mcfg.get("cues", False)  # lexical cue flags (rr/data.py CUES) added to the sentence vector
        if self.cues:
            from rr.data import CUES
            self.cue = nn.Linear(len(CUES), h)
        if self.doc_layer == "bilstm":
            self.lstm = nn.LSTM(h, h // 2, num_layers=mcfg.get("lstm_layers", 1), bidirectional=True, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(mcfg.get("dropout", 0.1)), nn.Linear(h, n_labels))
        self.crf = CRF(n_labels) if mcfg.get("crf", True) else None

    def head_modules(self):
        mods = [self.proj, self.pos, self.dom, self.out] + ([self.cue] if self.cues else [])
        return mods + ([self.lstm] if self.doc_layer == "bilstm" else []) + ([self.crf] if self.crf else [])

    def emissions(self, sent_vecs, positions, n_total, domain, feats=None):
        """sent_vecs (T, d) for consecutive sentences starting at ``positions[0]`` of a doc of n_total."""
        bucket = (positions.float() / max(1, n_total) * self.n_pos).long().clamp(max=self.n_pos - 1)
        x = self.proj(sent_vecs) + self.pos(bucket) + self.dom(torch.full_like(bucket, domain))
        if self.cues:
            x = x + self.cue(feats.to(x.device, x.dtype))
        if self.doc_layer == "bilstm":
            x = self.lstm(x[None])[0][0]
        return self.out(x).float()

    def forward_doc(self, doc, start, end, context, window, max_windows=16):
        """Emissions for sentences [start, end) of an encoded doc."""
        sents = doc["sents"][start:end]
        groups = pack(sents, context, window)
        vecs = self.encoder(sents, groups, max_windows)
        pos = torch.arange(start, end, device=vecs.device)
        return self.emissions(vecs, pos, len(doc["sents"]), doc["domain"],
                              doc["feats"][start:end] if self.cues else None)


def build_model(cfg: dict, n_labels: int):
    mcfg, pcfg = cfg["model"], cfg["peft"]
    lm, tok = load_phase1(mcfg["phase1_ckpt"])
    bb = Backbone(lm, mcfg.get("attention", "causal"), mcfg.get("keep_layers"), mcfg.get("encdec_output", "encoder"))
    model = RRModel(SentenceEncoder(bb, tok.eos_id), n_labels, mcfg)

    if pcfg["method"] in EXTRA:
        model.encoder.n_prompt = apply_extra(model, pcfg)
    else:  # Phase-2 methods expect an object with .backbone and .head
        shim = nn.Module()
        shim.backbone, shim.head = bb, nn.ModuleList(model.head_modules())
        apply_method(shim, pcfg)
    for m in model.head_modules():
        for p in m.parameters():
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters()) + sum(b.numel() for n, b in model.named_buffers() if "packed" in n) * 2
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {"total_params": total, "trainable_params": train, "trainable_pct": round(100 * train / total, 3)}
    return model, tok, info


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers() if n.endswith("rank_mask")}  # AdaLoRA pruning state
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def make_controller(model, pcfg, total_steps):
    return AdaLoRAController(model, pcfg, total_steps) if pcfg["method"] == "adalora" else None
