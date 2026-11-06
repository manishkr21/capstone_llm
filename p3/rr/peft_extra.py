"""Adaptation methods added for Phase 3 (the rest come from ../p2/lner/peft.py).

method    what is trained / changed
------    -------------------------
rslora    LoRA with scale alpha / sqrt(r) instead of alpha / r          (Kalajdzievski, 2023)
qlora     frozen base weights stored as 4-bit NormalFloat (blockwise absmax), de-quantized
          on the fly, plus LoRA in higher precision                     (Dettmers et al., 2023)
adalora   SVD-form update P diag(lambda) Q with orthogonality regulariser and importance-
          based pruning of lambda to a global rank budget               (Zhang et al., 2023)
prefix    learned key/value prefix of P vectors in every self-attention layer, produced by a
          shared MLP reparameterisation                                  (Li & Liang, 2021)
prompt    P learned input embeddings prepended to every window           (Lester et al., 2021)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lner.peft import LoRALinear, _wrap_linears

# ----------------------------------------------------------------------------- rsLoRA
class RSLoRALinear(LoRALinear):
    def __init__(self, base, r, alpha, dropout):
        super().__init__(base, r, alpha, dropout)
        self.scale = alpha / math.sqrt(r)


# ----------------------------------------------------------------------------- QLoRA
# the 16 NormalFloat-4 levels: quantiles of N(0,1) rescaled to [-1, 1], with an exact zero
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141100883, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


class NF4Linear(nn.Module):
    """Frozen linear layer with 4-bit NF4 weights: blocks of ``block`` values share one absmax
    scale; two 4-bit codes are packed per byte."""

    def __init__(self, base: nn.Linear, block: int = 64):
        super().__init__()
        w = base.weight.detach().float()
        self.shape, self.block = tuple(w.shape), block
        flat = w.reshape(-1)
        pad = (-flat.numel()) % block
        flat = F.pad(flat, (0, pad)).view(-1, block)
        absmax = flat.abs().amax(1, keepdim=True).clamp_min(1e-8)
        codes = (flat / absmax).unsqueeze(-1).sub(NF4).abs().argmin(-1).to(torch.uint8).view(-1)
        self.register_buffer("packed", codes[0::2] | (codes[1::2] << 4))
        self.register_buffer("absmax", absmax.squeeze(1).to(torch.bfloat16))
        self.register_buffer("levels", NF4.clone())
        self.n = w.numel()
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = base.in_features, base.out_features

    def weight_dq(self, dtype):
        codes = torch.stack([self.packed & 15, self.packed >> 4], 1).view(-1).long()
        w = self.levels[codes].view(-1, self.block) * self.absmax.float()[:, None]
        return w.view(-1)[: self.n].view(self.shape).to(dtype)

    def forward(self, x):
        return F.linear(x, self.weight_dq(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def _quantize_rest(module: nn.Module, block: int):
    """Replace every remaining plain nn.Linear (any depth) by its frozen NF4 version."""
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            setattr(module, name, NF4Linear(child, block))
        else:
            _quantize_rest(child, block)


class QLoRALinear(LoRALinear):
    def __init__(self, base, r, alpha, dropout, block=64):
        super().__init__(base, r, alpha, dropout)
        self.base = NF4Linear(base, block)


# ----------------------------------------------------------------------------- AdaLoRA
class SVDLinear(nn.Module):
    """W x + scale * P (lambda * mask) Q x ;  P: out x r, Q: r x in, lambda: r."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.lam = nn.Parameter(torch.zeros(r))
        self.register_buffer("rank_mask", torch.ones(r))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        u = F.linear(self.drop(x), self.Q) * (self.lam * self.rank_mask)
        return self.base(x) + self.scale * F.linear(u, self.P)

    def orth_penalty(self):
        r = self.P.shape[1]
        eye = torch.eye(r, device=self.P.device)
        return ((self.P.T @ self.P - eye) ** 2).sum() + ((self.Q @ self.Q.T - eye) ** 2).sum()


class AdaLoRAController:
    """Sensitivity-based importance with EMA smoothing (beta1) and uncertainty (beta2); the
    global budget of kept singular values falls cubically from init_r to target_r per module
    between t_init and total - t_final steps; pruning every delta_t steps."""

    def __init__(self, model, pcfg: dict, total_steps: int):
        self.mods = [m for m in model.modules() if isinstance(m, SVDLinear)]
        self.b0 = sum(m.lam.numel() for m in self.mods)
        self.bT = int(self.b0 * pcfg.get("target_r", 8) / pcfg.get("r", 12))
        self.t_init = int(pcfg.get("t_init_frac", 0.1) * total_steps)
        self.t_final = int(pcfg.get("t_final_frac", 0.3) * total_steps)
        self.total, self.delta = total_steps, pcfg.get("delta_t", 10)
        self.b1, self.b2 = pcfg.get("beta1", 0.85), pcfg.get("beta2", 0.85)
        self.gamma = pcfg.get("orth_reg", 0.1)
        self.ipt, self.unc = {}, {}

    def penalty(self):
        return self.gamma * sum(m.orth_penalty() for m in self.mods) / max(1, len(self.mods))

    def budget(self, step):
        if step < self.t_init:
            return self.b0
        if step >= self.total - self.t_final:
            return self.bT
        frac = 1 - (step - self.t_init) / max(1, self.total - self.t_final - self.t_init)
        return int(self.bT + (self.b0 - self.bT) * frac ** 3)

    @torch.no_grad()
    def _score(self, p):
        s = (p * p.grad).abs() if p.grad is not None else torch.zeros_like(p)
        key = id(p)
        if key not in self.ipt:
            self.ipt[key], self.unc[key] = torch.zeros_like(p), torch.zeros_like(p)
        self.ipt[key].mul_(self.b1).add_((1 - self.b1) * s)
        self.unc[key].mul_(self.b2).add_((1 - self.b2) * (s - self.ipt[key]).abs())
        return self.ipt[key] * self.unc[key]

    @torch.no_grad()
    def step(self, step):
        """Call after backward, before optimizer.step."""
        scores = []
        for m in self.mods:
            s = self._score(m.lam) + self._score(m.P).mean(0) + self._score(m.Q).mean(1)
            scores.append(s)
        if step < self.t_init or step % self.delta:
            return
        b = self.budget(step)
        allv = torch.cat(scores)
        if b >= allv.numel():
            return
        thr = allv.kthvalue(allv.numel() - b).values
        for m, s in zip(self.mods, scores):
            m.rank_mask.copy_((s > thr).float())

    def ranks(self):
        return [int(m.rank_mask.sum()) for m in self.mods]


# ----------------------------------------------------------------------------- prefix tuning
class PrefixBank(nn.Module):
    """P prefix positions -> (per layer) key and value vectors, via a shared 2-layer MLP."""

    def __init__(self, n_layers, n_kv, hd, d_model, n_prefix, hidden):
        super().__init__()
        self.shape = (n_layers, 2, n_kv, n_prefix, hd)
        self.emb = nn.Parameter(torch.randn(n_prefix, d_model) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, n_layers * 2 * n_kv * hd))
        nn.init.zeros_(self.mlp[2].weight)  # start with zero keys/values: tiny effect at init
        nn.init.zeros_(self.mlp[2].bias)
        self.n_layers, self.n_kv, self.hd, self.n_prefix = n_layers, n_kv, hd, n_prefix

    def forward(self):
        out = self.mlp(self.emb)  # (P, L*2*n_kv*hd)
        return out.view(self.n_prefix, self.n_layers, 2, self.n_kv, self.hd).permute(1, 2, 3, 0, 4)


class PrefixAttention(nn.Module):
    """Self-attention with learned prefix keys/values (no positions: RoPE is not applied to them)."""

    def __init__(self, attn, bank: PrefixBank, layer: int):
        super().__init__()
        self.attn, self.layer = attn, layer
        self._bank = [bank]  # not a submodule: the bank is owned by the backbone

    def forward(self, x, memory=None, mask=None, is_causal=False, rope=None, pos=0, cache=None, cache_keep=None):
        a = self.attn
        B, T, _ = x.shape
        kv = self._bank[0]()[self.layer]  # (2, n_kv, P, hd)
        P = kv.shape[2]
        q = a._heads(a.q(x), a.n_heads)
        k = a._heads(a.k(x), a.n_kv)
        v = a._heads(a.v(x), a.n_kv)
        if rope is not None:
            k, q = rope(k, pos), rope(q, pos)
        k = torch.cat([kv[0].to(k.dtype).unsqueeze(0).expand(B, -1, -1, -1), k], 2)
        v = torch.cat([kv[1].to(v.dtype).unsqueeze(0).expand(B, -1, -1, -1), v], 2)
        if a.n_kv != a.n_heads:
            rep = a.n_heads // a.n_kv
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        if mask is None:
            if is_causal:
                i = torch.arange(T, device=x.device)
                mask = torch.cat([torch.ones(T, P, dtype=torch.bool, device=x.device), i[None, :] <= i[:, None]], 1)
        elif mask.dtype == torch.bool:
            mask = torch.cat([torch.ones(*mask.shape[:-1], P, dtype=torch.bool, device=x.device), mask], -1)
        else:  # additive (ALiBi) bias
            mask = torch.cat([torch.zeros(*mask.shape[:-1], P, dtype=mask.dtype, device=x.device), mask], -1).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=a.dropout if self.training else 0.0)
        return a.o(y.transpose(1, 2).reshape(B, T, -1))


def _self_attn_modules(bb):
    """(block, attribute name) of every self-attention module the window passes through."""
    out = []
    for block in bb.all_blocks:
        name = "attn" if hasattr(block, "attn") else "self_attn"
        out.append((block, name))
    return out


# ----------------------------------------------------------------------------- prompt tuning
class PromptEmbedding(nn.Module):
    """Wraps the token embedding: positions 0..P-1 of every window get the learned prompt."""

    def __init__(self, emb: nn.Embedding, n_prompt: int, seed: int = 0):
        super().__init__()
        self.emb, self.n = emb, n_prompt
        g = torch.Generator().manual_seed(seed)
        init = emb.weight[torch.randint(1000, emb.num_embeddings, (n_prompt,), generator=g)].detach().clone()
        self.prompt = nn.Parameter(init)  # initialised from real vocabulary embeddings

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        if ids.dim() == 2 and ids.size(1) >= self.n:
            x = torch.cat([self.prompt.to(x.dtype).unsqueeze(0).expand(ids.size(0), -1, -1), x[:, self.n:]], 1)
        return x


# ----------------------------------------------------------------------------- dispatch
EXTRA = ("rslora", "qlora", "adalora", "prefix", "prompt")


def apply_extra(model, pcfg: dict) -> int:
    """Inject one of EXTRA into model.encoder.backbone. Returns the number of prompt positions."""
    method, bb = pcfg["method"], model.encoder.backbone
    for p in bb.parameters():
        p.requires_grad = False
    targets = set(pcfg.get("targets", ["q", "k", "v", "o", "fc", "proj", "gate", "up"]))
    r, alpha, drop = pcfg.get("r", 16), pcfg.get("alpha", 32), pcfg.get("lora_dropout", 0.05)
    if method in ("rslora", "qlora"):
        make = (lambda lin: RSLoRALinear(lin, r, alpha, drop)) if method == "rslora" else \
               (lambda lin: QLoRALinear(lin, r, alpha, drop, pcfg.get("block", 64)))
        for block in bb.all_blocks:
            _wrap_linears(block, targets, make)
        if method == "qlora":  # QLoRA quantises every linear layer of the frozen base, not only the targets
            for block in bb.all_blocks:
                _quantize_rest(block, pcfg.get("block", 64))
        return 0
    if method == "adalora":
        for block in bb.all_blocks:
            _wrap_linears(block, targets, lambda lin: SVDLinear(lin, pcfg.get("r", 12), alpha, drop))
        return 0
    if method == "prefix":
        mods = _self_attn_modules(bb)
        a0 = getattr(*mods[0])
        bb.prefix_bank = PrefixBank(len(mods), a0.n_kv, a0.hd, bb.d_model, pcfg.get("n_prefix", 16),
                                    pcfg.get("prefix_hidden", 512))
        for i, (block, name) in enumerate(mods):
            setattr(block, name, PrefixAttention(getattr(block, name), bb.prefix_bank, i))
        return 0
    if method == "prompt":
        lm = bb.lm
        lm.tok_emb = PromptEmbedding(lm.tok_emb, pcfg.get("n_prompt", 20), pcfg.get("seed", 0))
        return pcfg.get("n_prompt", 20)
    raise ValueError(method)
