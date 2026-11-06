"""Fine-tuning / parameter-efficient adaptation methods, written from the papers.

method        what is trained
------        ---------------
full          every backbone weight + head
partial       the top ``n_train_layers`` blocks + final norm + head (layer-wise FT)
bitfit        only bias terms (and norm gains) + head            (Ben Zaken et al., 2022)
lora          W x + (alpha/r) B A x, B=0 at init                  (Hu et al., 2022)
lorma         (I + (alpha/r) B A) W0 x, computed as h + s*B(A h)  (Bihany, Patel & Modi, 2025; LoRMA+)
adapter       Houlsby bottleneck adapters after attention and MLP (Houlsby et al., 2019)

In every case the token-classification head is trained.  LoRA/LoRMA wrap the linear
layers whose attribute name is in ``targets`` (q,k,v,o = attention; fc,proj,gate,up = MLP).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA+ : W = (I + s*BA) W0 with B: (out x r), A: (r x out).
    Re-ordered as h0 = W0 x ; h = h0 + s * B (A h0), so the cost matches LoRA."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        d = base.out_features
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h0 = F.linear(x, self.base.weight)
        h = h0 + self.scale * F.linear(F.linear(self.drop(h0), self.A), self.B)
        return h if self.base.bias is None else h + self.base.bias


class LoRMAPiLinear(nn.Module):
    """LoRMA_pi : W = pi(BA) W0, where pi rotates row i of BA cyclically by i (rank inflation).
    Init: B[:, 0] = 1 (rest random), A[0, 0] = 1 (rest 0) -> every row of BA is e_0, and after
    the rotation pi(BA) = I, so training starts from the Phase-1 weights."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        d = base.out_features
        self.B = nn.Parameter(torch.randn(d, r) * 0.02)
        self.A = nn.Parameter(torch.zeros(r, d))
        with torch.no_grad():
            self.B[:, 0] = 1.0
            self.A[0, 0] = 1.0
        i = torch.arange(d)
        self.register_buffer("rot", (i[None, :] - i[:, None]) % d, persistent=False)  # rot[i, j] = (j - i) mod d

    def forward(self, x):
        h0 = F.linear(x, self.base.weight)
        m = (self.B @ self.A).gather(1, self.rot)  # pi(BA): row i rotated right by i
        h = F.linear(h0, m.to(h0.dtype))
        return h if self.base.bias is None else h + self.base.bias


class IA3Linear(nn.Module):
    """(IA)^3 (Liu et al., 2022): learned per-feature scaling, initialised to 1.
    side='out' scales the output (keys, values); side='in' scales the input (the MLP's
    intermediate activation feeding the down-projection)."""

    def __init__(self, base: nn.Linear, side: str):
        super().__init__()
        self.base = base
        self.side = side
        self.scale = nn.Parameter(torch.ones(base.out_features if side == "out" else base.in_features))

    def forward(self, x):
        return self.base(x) * self.scale if self.side == "out" else self.base(x * self.scale)


class Adapter(nn.Module):
    """Houlsby adapter applied to the output of a sub-layer: y + up(gelu(down(y)))."""

    def __init__(self, inner: nn.Module, d: int, bottleneck: int):
        super().__init__()
        self.inner = inner
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)  # starts as the identity
        nn.init.zeros_(self.up.bias)

    def forward(self, *args, **kwargs):
        y = self.inner(*args, **kwargs)
        return y + self.up(F.gelu(self.down(y)))


def _wrap_linears(module: nn.Module, targets, make):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, make(child))
        else:
            _wrap_linears(child, targets, make)


def _is_norm(name: str) -> bool:
    return any(k in name for k in ("ln", "norm"))


def apply_method(tagger, pcfg: dict) -> dict:
    """Freeze/inject according to ``pcfg['method']``. Returns a summary of trainable params."""
    method = pcfg["method"]
    bb = tagger.backbone
    for p in bb.parameters():
        p.requires_grad = method == "full"
    targets = set(pcfg.get("targets", ["q", "k", "v", "o", "fc", "proj", "gate", "up"]))
    r, alpha, drop = pcfg.get("r", 16), pcfg.get("alpha", 32), pcfg.get("lora_dropout", 0.05)

    if method == "partial":
        n = pcfg["n_train_layers"]
        for block in bb.all_blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for name, p in bb.lm.named_parameters():
            if _is_norm(name) and not name.split(".")[0] in ("blocks", "encoder"):
                p.requires_grad = True  # final norm
    elif method == "bitfit":
        for name, p in bb.named_parameters():
            if name.endswith("bias") or (pcfg.get("train_norms", True) and _is_norm(name)):
                p.requires_grad = True
    elif method in ("lora", "lorma", "lorma_pi"):
        cls = {"lora": LoRALinear, "lorma": LoRMALinear, "lorma_pi": LoRMAPiLinear}[method]
        for block in bb.all_blocks:
            _wrap_linears(block, targets, lambda lin: cls(lin, r, alpha, drop))
    elif method == "ia3":
        for block in bb.all_blocks:
            _wrap_linears(block, {"k", "v"}, lambda lin: IA3Linear(lin, "out"))
            _wrap_linears(block, {"proj"}, lambda lin: IA3Linear(lin, "in"))
    elif method == "adapter":
        k = pcfg.get("bottleneck", 64)
        for block in bb.all_blocks:
            attn_name = "attn" if hasattr(block, "attn") else "self_attn"
            setattr(block, attn_name, Adapter(getattr(block, attn_name), bb.d_model, k))
            block.mlp = Adapter(block.mlp, bb.d_model, k)
        for name, p in bb.named_parameters():
            if _is_norm(name):
                p.requires_grad = True  # Houlsby also trains the layer norms
    elif method != "full":
        raise ValueError(f"unknown method {method!r}")

    for p in tagger.head.parameters():
        p.requires_grad = True
    # newly created adapter / low-rank parameters default to requires_grad=True already
    total = sum(p.numel() for p in tagger.parameters())
    train = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": train, "trainable_pct": round(100 * train / total, 3)}
