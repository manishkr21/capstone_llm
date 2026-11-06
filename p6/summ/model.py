"""Summarizer: one wrapper that turns any Phase-1 language model into a document -> summary generator.

Layouts (``P`` = prompt-tuning slots, ``S`` = condensed source, ``sep`` = " Summary:" marker)

  A1 / A3 / A4 (one stack)   [P prompt][left pad][source S][sep] summary <eos>
      A1: causal, learned positions (table interpolated from 1024 to the needed length)
      A3: the whole context (P + S + sep) is a bidirectional prefix, the summary is causal (ALiBi)
      A4: causal; local layers keep their 256-token sliding window, global layers see everything
  A2 (encoder-decoder)       encoder: K chunks of the document, encoded independently
                             (fusion-in-decoder: the decoder cross-attends to all K*L memory slots)
                             decoder: sep summary <eos>

Left padding keeps the source end and the separator at the same positions for every document, so a
batch shares one KV cache during generation; padded keys are masked out.  Every attention call gets an
explicit mask, so the PEFT wrappers (LoRA, LoRMA, adapters, NF4) work unchanged; prefix tuning is a
learned key/value prefix pre-loaded into each self-attention cache (works for training and decoding).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from summ import P1_ROOT  # noqa: F401  (puts legal_lm / lner / rr on sys.path)
from legal_lm.data.tokenizer import LegalTokenizer
from legal_lm.models import ModelConfig, build_model as build_lm
from legal_lm.models.layers import new_caches
from lner.peft import Adapter, LoRALinear, LoRMALinear, _wrap_linears
from rr.peft_extra import PrefixBank, QLoRALinear, SVDLinear, AdaLoRAController, _quantize_rest

NEG = float("-inf")


def load_phase1(ckpt_path):
    ckpt = torch.load(P1_ROOT / ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tok_cfg = dict(cfg["tokenizer"])
    tok_cfg["path"] = str(P1_ROOT / tok_cfg["path"])
    tok = LegalTokenizer.from_config(tok_cfg)
    mcfg = ModelConfig.from_dict(cfg["model"], vocab_size=tok.vocab_size, max_seq_len=cfg["train"]["block_size"])
    lm = build_lm(mcfg)
    lm.load_state_dict(ckpt["model"])
    return lm, tok


def _attn(m):
    return m.inner if isinstance(m, Adapter) else m


class Summarizer(nn.Module):
    def __init__(self, lm, tok, mcfg: dict):
        super().__init__()
        self.lm, self.arch, self.cfg = lm, lm.cfg.arch, lm.cfg
        self.src_len = mcfg["src_len"]                  # A2: tokens per chunk
        self.n_chunks = mcfg.get("n_chunks", 1)        # A2 only
        self.tgt_len = mcfg["tgt_len"]
        self.eos = tok.eos_id
        self.pad = tok.tk.token_to_id("<|pad|>") or tok.eos_id
        self.sep = tok.encode(" Summary:")
        self.n_prompt, self.prompt, self.prefix_bank = 0, None, None
        self.dropout = mcfg.get("dropout", 0.1)
        for m in lm.modules():  # Phase-1 trained without dropout; a little helps on 7k examples
            if isinstance(m, nn.Dropout):
                m.p = self.dropout
            if hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = self.dropout
        keep = mcfg.get("keep_layers")
        if keep is not None:  # smaller student for knowledge distillation (Phase-1 weights of the kept blocks)
            if self.arch == "encdec":
                lm.encoder = nn.ModuleList([lm.encoder[i] for i in keep["enc"]])
                lm.decoder = nn.ModuleList([lm.decoder[i] for i in keep["dec"]])
            else:
                lm.blocks = nn.ModuleList([lm.blocks[i] for i in keep])
        if self.arch == "gpt":  # position interpolation (Chen et al., 2023) to fit source + summary
            need = mcfg.get("max_prompt", 32) + self.src_len + len(self.sep) + self.tgt_len + 2
            old = lm.pos_emb.weight.data
            if need > old.size(0):
                new = F.interpolate(old.T[None], size=need, mode="linear", align_corners=True)[0].T
                lm.pos_emb = nn.Embedding(need, old.size(1))
                lm.pos_emb.weight.data.copy_(new)
        if self.arch == "encdec":
            need = max(self.src_len + mcfg.get("max_prompt", 32), self.tgt_len + len(self.sep) + 2)
            if need > lm.pos_table.size(0):
                from legal_lm.models.layers import sinusoidal_table
                lm.pos_table = sinusoidal_table(need, self.cfg.d_model)

    # ------------------------------------------------------------------ structure
    @property
    def all_blocks(self):
        return list(self.lm.encoder) + list(self.lm.decoder) if self.arch == "encdec" else list(self.lm.blocks)

    def self_attn_slots(self):
        """(block, attribute) of every self-attention, in the order the prefix bank indexes them."""
        return [(b, "self_attn" if hasattr(b, "self_attn") else "attn") for b in self.all_blocks]

    @property
    def d_model(self):
        return self.cfg.d_model

    # ------------------------------------------------------------------ batches
    def _ctx_len(self, S):
        return self.n_prompt + S + len(self.sep)

    def make_batch(self, srcs, tgts=None, device="cuda"):
        """srcs: list of token arrays (A2: list of lists of chunk arrays); tgts: list of arrays or None."""
        B, P = len(srcs), self.n_prompt
        b = {}
        if tgts is not None:
            ys = [list(t[: self.tgt_len]) + [self.eos] for t in tgts]
            Ty = max(len(y) for y in ys)
            lab = torch.full((B, Ty), -100, dtype=torch.long)
            for i, y in enumerate(ys):
                lab[i, : len(y)] = torch.tensor(y)
            b["labels"] = lab.to(device)
        else:
            ys, Ty = None, 1
        if self.arch == "encdec":
            K = max(1, max(len(c) for c in srcs))
            L = P + max(1, max((len(x) for c in srcs for x in c), default=1))
            enc = torch.full((B, K, L), self.pad, dtype=torch.long)
            valid = torch.zeros(B, K, L, dtype=torch.bool)
            for i, chunks in enumerate(srcs):
                for k, x in enumerate(chunks):
                    enc[i, k, P:P + len(x)] = torch.as_tensor(x, dtype=torch.long)
                    valid[i, k, : P + len(x)] = True
            b["enc"], b["enc_valid"] = enc.to(device), valid.to(device)
            dec = torch.full((B, len(self.sep) + Ty - 1), self.pad, dtype=torch.long)
            dec[:, : len(self.sep)] = torch.tensor(self.sep)
            if ys is not None:
                for i, y in enumerate(ys):
                    dec[i, len(self.sep): len(self.sep) + len(y) - 1] = torch.tensor(y[:-1])
            b["ids"], b["ctx"] = dec.to(device), len(self.sep)
            return b
        S = max(1, max(len(x) for x in srcs))
        C = self._ctx_len(S)
        ids = torch.full((B, C + Ty - 1), self.pad, dtype=torch.long)
        valid = torch.ones(B, C + Ty - 1, dtype=torch.bool)
        for i, x in enumerate(srcs):
            ids[i, P + S - len(x): P + S] = torch.as_tensor(x, dtype=torch.long)
            valid[i, P: P + S - len(x)] = False
            if ys is not None:
                ids[i, C: C + len(ys[i]) - 1] = torch.tensor(ys[i][:-1])
        ids[:, P + S: C] = torch.tensor(self.sep)
        b["ids"], b["valid"], b["ctx"] = ids.to(device), valid.to(device), C
        return b

    # ------------------------------------------------------------------ pieces
    def _embed(self, ids, pos0=0, prompt=False):
        lm = self.lm
        te = lm.tok_emb(ids)
        if prompt and self.prompt is not None:
            te = torch.cat([self.prompt.to(te.dtype).expand(ids.size(0), -1, -1), te[:, self.n_prompt:]], 1)
        T = ids.size(1)
        if self.arch == "encdec":
            return lm.drop(te * lm.emb_scale + lm.pos_table[pos0:pos0 + T].to(te.device, te.dtype))
        if self.arch == "gpt":
            te = te + lm.pos_emb(torch.arange(pos0, pos0 + T, device=ids.device))
        return lm.drop(te)

    def _prefix_caches(self, B, caches):
        """Pre-load the learned prefix keys/values into each self-attention cache."""
        if self.prefix_bank is None:
            return caches
        kv = self.prefix_bank()  # (n_layers, 2, n_kv, P, hd)
        for i, c in enumerate(caches):
            c["k"] = kv[i, 0].unsqueeze(0).expand(B, -1, -1, -1)
            c["v"] = kv[i, 1].unsqueeze(0).expand(B, -1, -1, -1)
        return caches

    def _n_pre(self):
        return 0 if self.prefix_bank is None else self.prefix_bank.n_prefix

    def _dec_masks(self, valid, q0, Tq, C):
        """Masks for the one-stack models. valid: (B, q0+Tq) key validity. Returns {kind: mask}."""
        dev = valid.device
        Lk = valid.size(1)
        qp = torch.arange(q0, q0 + Tq, device=dev)[:, None]
        kp = torch.arange(Lk, device=dev)[None, :]
        allowed = kp <= qp
        if self.arch == "prefixlm":
            allowed = allowed | (kp < C)
        out = {}
        kinds = ["global", "local"] if self.arch == "efficient" else ["global"]
        for kind in kinds:
            a = allowed & ((qp - kp) < self.cfg.window) if kind == "local" else allowed
            a = (a[None] & valid[:, None, :]) | (kp == qp)[None]          # (B, Tq, Lk)
            a = a[:, None]                                                 # (B, 1, Tq, Lk)
            if self.arch == "prefixlm":
                bias = -self.lm.slopes[:, None, None] * (qp - kp).abs().float()[None]   # (H, Tq, Lk)
                a = bias[None].masked_fill(~a, NEG)                        # (B, H, Tq, Lk)
            if self._n_pre():
                P = self._n_pre()
                pad = torch.ones if a.dtype == torch.bool else torch.zeros
                a = torch.cat([pad(*a.shape[:-1], P, dtype=a.dtype, device=dev), a], -1)
            out[kind] = a
        return out

    def _run_stack(self, x, masks, pos0, caches):
        lm = self.lm
        for i, blk in enumerate(lm.blocks):
            c = None if caches is None else caches[i]
            if self.arch == "efficient":
                m = masks["global"] if blk.is_global else masks["local"]
                x = x + blk.drop(blk.attn(blk.norm1(x), mask=m, rope=lm.rope, pos=pos0, cache=c))
                x = x + blk.mlp(blk.norm2(x))
            else:
                x = x + blk.drop(blk.attn(blk.ln1(x), mask=masks["global"], cache=c))
                x = x + blk.mlp(blk.ln2(x))
        return lm.norm(x) if self.arch == "efficient" else lm.ln_f(x)

    def _encode(self, enc, enc_valid):
        B, K, L = enc.shape
        ids, valid = enc.view(B * K, L), enc_valid.view(B * K, L)
        x = self._embed(ids, 0, prompt=True)
        eye = torch.eye(L, dtype=torch.bool, device=ids.device)
        m = (valid[:, None, None, :] | eye[None, None])                   # (BK, 1, L, L)
        P = self._n_pre()
        if P:
            m = torch.cat([torch.ones(B * K, 1, L, P, dtype=torch.bool, device=ids.device), m], -1)
        caches = self._prefix_caches(B * K, new_caches(len(self.lm.encoder)))
        for i, blk in enumerate(self.lm.encoder):
            x = x + blk.drop(blk.attn(blk.ln1(x), mask=m, cache=caches[i] if P else None))
            x = x + blk.mlp(blk.ln2(x))
        return self.lm.enc_norm(x).view(B, K * L, -1), enc_valid.view(B, K * L)

    def _decode(self, x, memory, mem_valid, pos0, caches):
        lm = self.lm
        Tq = x.size(1)
        n_dec = len(lm.decoder)
        Lk = pos0 + Tq
        qp = torch.arange(pos0, Lk, device=x.device)[:, None]
        sm = (torch.arange(Lk, device=x.device)[None] <= qp)[None, None]
        P = self._n_pre()
        if P:
            sm = torch.cat([torch.ones(1, 1, Tq, P, dtype=torch.bool, device=x.device), sm], -1)
        cm = mem_valid[:, None, None, :]
        for i, blk in enumerate(lm.decoder):
            c = caches[i]
            sc, cc = c.setdefault("self", {}), c.setdefault("cross", {})
            x = x + blk.drop(blk.self_attn(blk.ln1(x), mask=sm, cache=sc))
            x = x + blk.drop(blk.cross_attn(blk.ln2(x), memory=memory, mask=cm, cache=cc))
            x = x + blk.mlp(blk.ln3(x))
        return lm.dec_norm(x)

    def _dec_caches(self, B):
        caches = new_caches(len(self.lm.decoder))
        if self.prefix_bank is not None:
            kv = self.prefix_bank()
            ne = len(self.lm.encoder)
            for i, c in enumerate(caches):
                c["self"] = {"k": kv[ne + i, 0].unsqueeze(0).expand(B, -1, -1, -1),
                             "v": kv[ne + i, 1].unsqueeze(0).expand(B, -1, -1, -1)}
        return caches

    # ------------------------------------------------------------------ training
    def target_logits(self, b):
        """Logits (B, Ty, V) for the summary tokens (aligned with b['labels'])."""
        if self.arch == "encdec":
            memory, mv = self._encode(b["enc"], b["enc_valid"])
            ids = b["ids"]
            h = self._decode(self._embed(ids), memory, mv, 0, self._dec_caches(ids.size(0)))
            return self.lm.lm_head(h[:, b["ctx"] - 1:])
        ids, C = b["ids"], b["ctx"]
        masks = self._dec_masks(b["valid"], 0, ids.size(1), C)
        caches = self._prefix_caches(ids.size(0), new_caches(len(self.lm.blocks))) if self._n_pre() else None
        h = self._run_stack(self._embed(ids, 0, prompt=True), masks, 0, caches)
        return self.lm.lm_head(h[:, C - 1:])

    def loss(self, b, label_smoothing=0.0):
        logits = self.target_logits(b)
        V = self.cfg.vocab_size
        return F.cross_entropy(logits[..., :V].float().reshape(-1, V), b["labels"].reshape(-1),
                               ignore_index=-100, label_smoothing=label_smoothing), logits

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def start(self, b):
        """Read the source once. Returns (state, next-token logits (B, V))."""
        V = self.cfg.vocab_size
        if self.arch == "encdec":
            B = b["enc"].size(0)
            memory, mv = self._encode(b["enc"], b["enc_valid"])
            caches = self._dec_caches(B)
            ids = b["ids"][:, : b["ctx"]]
            h = self._decode(self._embed(ids), memory, mv, 0, caches)
            st = {"caches": caches, "memory": memory, "mv": mv, "pos": ids.size(1)}
        else:
            ids, C = b["ids"][:, : b["ctx"]], b["ctx"]
            valid = b["valid"][:, :C]
            caches = self._prefix_caches(ids.size(0), new_caches(len(self.lm.blocks)))
            h = self._run_stack(self._embed(ids, 0, prompt=True), self._dec_masks(valid, 0, C, C), 0, caches)
            st = {"caches": caches, "valid": valid, "C": C, "pos": C}
        return st, self.lm.lm_head(h[:, -1])[:, :V].float()

    @torch.no_grad()
    def step(self, st, nxt):
        """Feed one token per row; returns next-token logits."""
        pos = st["pos"]
        x = self._embed(nxt[:, None], pos)
        if self.arch == "encdec":
            h = self._decode(x, st["memory"], st["mv"], pos, st["caches"])
        else:
            st["valid"] = torch.cat([st["valid"], torch.ones_like(st["valid"][:, :1])], 1)
            h = self._run_stack(x, self._dec_masks(st["valid"], pos, 1, st["C"]), pos, st["caches"])
        st["pos"] = pos + 1
        return self.lm.lm_head(h[:, -1])[:, : self.cfg.vocab_size].float()

    def generate(self, b, g: dict):
        return decode([self], [b], g)


@torch.no_grad()
def decode(models, batches, g: dict, weights=None):
    """Greedy / sampled / beam decoding with repetition controls.  With several models (same tokenizer)
    the next-token log-probabilities are averaged at every step (token-level ensemble), each model keeping
    its own source layout and KV cache.  Returns a list of token lists."""
    m0 = models[0]
    eos, V = m0.eos, m0.cfg.vocab_size
    weights = weights or [1.0 / len(models)] * len(models)
    started = [m.start(b) for m, b in zip(models, batches)]
    states = [s for s, _ in started]

    def mix(logit_list):
        if len(logit_list) == 1:
            return logit_list[0]
        return sum(w * torch.log_softmax(l, -1) for w, l in zip(weights, logit_list))

    logits = mix([l for _, l in started])
    B, dev = logits.size(0), logits.device
    nb = g.get("beams", 1)
    if nb > 1:  # every document gets nb beams; caches/masks are expanded and re-ordered with them
        rep = lambda t: t.repeat_interleave(nb, 0)
        states = [_map_tensors(s, rep) for s in states]
        logits = rep(logits)
    N = B * nb
    out = torch.zeros(N, 0, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    scores = torch.zeros(B, nb, device=dev)
    scores[:, 1:] = NEG  # at the first step only beam 0 is real
    scores = scores.view(-1)
    n_ng, rp = g.get("no_repeat_ngram", 4), g.get("repetition_penalty", 1.0)
    for step in range(g["max_new_tokens"]):
        if step < g.get("min_new_tokens", 0):
            logits[:, eos] = NEG
        if rp != 1.0 and out.size(1):
            recent = out[:, -g.get("repetition_window", 64):]
            s = logits.gather(1, recent)
            logits.scatter_(1, recent, torch.where(s < 0, s * rp, s / rp))
        if n_ng and out.size(1) >= n_ng:
            win = out.unfold(1, n_ng, 1)                                  # (N, W, n)
            hit = (win[:, :, :-1] == out[:, None, -(n_ng - 1):]).all(-1)  # (N, W)
            ban = torch.where(hit, win[:, :, -1], torch.full_like(win[:, :, -1], V))  # V = "no ban" column
            tmp = torch.zeros(N, V + 1, device=dev, dtype=torch.bool)
            tmp.scatter_(1, ban, True)
            logits[tmp[:, :V]] = NEG
        if nb > 1:
            logp = torch.log_softmax(logits, -1)
            logp[done] = NEG
            logp[done, eos] = 0.0  # finished beams only extend with eos, at no cost
            cand = (scores[:, None] + logp).view(B, nb * V)
            vals, flat = cand.topk(nb, -1)
            src = (torch.arange(B, device=dev)[:, None] * nb + flat // V).view(-1)
            nxt = (flat % V).view(-1)
            scores = vals.view(-1)
            out, done = out[src], done[src]
            states = [_map_tensors(s, lambda t: t[src]) for s in states]
        elif g.get("temperature", 0) > 0:
            probs = torch.softmax(logits / g["temperature"], -1)
            nxt = torch.multinomial(probs, 1)[:, 0]
        else:
            nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        out = torch.cat([out, nxt[:, None]], 1)
        done |= nxt == eos
        if done.all():
            break
        logits = mix([m.step(s, nxt) for m, s in zip(models, states)])
    rows = out.tolist()
    lens = [(r.index(eos) if eos in r else len(r)) for r in rows]
    if nb > 1:  # length-normalised beam score
        lp = g.get("length_penalty", 1.0)
        norm = scores.cpu() / torch.tensor([max(1, n) for n in lens], dtype=torch.float) ** lp
        pick = norm.view(B, nb).argmax(1) + torch.arange(B) * nb
        return [rows[i][: lens[i]] for i in pick.tolist()]
    return [r[:n] for r, n in zip(rows, lens)]


def _map_tensors(obj, fn):
    """Apply fn to every tensor inside nested lists/dicts (KV caches), in place for dicts."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = _map_tensors(obj[k], fn)
        return obj
    if isinstance(obj, list):
        return [_map_tensors(x, fn) for x in obj]
    return obj


# ---------------------------------------------------------------------- adaptation methods
def apply_peft(model: Summarizer, pcfg: dict) -> dict:
    """method: full | partial | lora | lorma | qlora | adalora | adapter (Houlsby) | pfeiffer | prefix | prompt"""
    method = pcfg["method"]
    for p in model.parameters():
        p.requires_grad = method == "full"
    targets = set(pcfg.get("targets", ["q", "k", "v", "o", "fc", "proj", "gate", "up"]))
    r, alpha, drop = pcfg.get("r", 16), pcfg.get("alpha", 32), pcfg.get("lora_dropout", 0.05)
    blocks = model.all_blocks
    if method == "partial":
        n = pcfg["n_train_layers"]
        tops = (list(model.lm.encoder)[-n:] + list(model.lm.decoder)[-n:]) if model.arch == "encdec" else blocks[-n:]
        for blk in tops:
            for p in blk.parameters():
                p.requires_grad = True
        for name, p in model.lm.named_parameters():
            if name.split(".")[0] in ("ln_f", "norm", "enc_norm", "dec_norm"):
                p.requires_grad = True
    elif method in ("lora", "lorma"):
        cls = LoRALinear if method == "lora" else LoRMALinear
        for blk in blocks:
            _wrap_linears(blk, targets, lambda lin: cls(lin, r, alpha, drop))
    elif method == "qlora":
        for blk in blocks:
            _wrap_linears(blk, targets, lambda lin: QLoRALinear(lin, r, alpha, drop, pcfg.get("block", 64)))
            _quantize_rest(blk, pcfg.get("block", 64))
    elif method == "adalora":
        for blk in blocks:
            _wrap_linears(blk, targets, lambda lin: SVDLinear(lin, r, alpha, drop))
    elif method in ("adapter", "pfeiffer"):
        k = pcfg.get("bottleneck", 128)
        for blk in blocks:
            if method == "adapter":  # Houlsby: after attention and after the MLP
                name = "self_attn" if hasattr(blk, "self_attn") else "attn"
                setattr(blk, name, Adapter(getattr(blk, name), model.d_model, k))
            blk.mlp = Adapter(blk.mlp, model.d_model, k)  # Pfeiffer: after the MLP only
        for name, p in model.lm.named_parameters():
            if any(t in name for t in ("ln", "norm")):
                p.requires_grad = True
    elif method == "prefix":
        slots = model.self_attn_slots()
        a0 = _attn(getattr(*slots[0]))
        model.prefix_bank = PrefixBank(len(slots), a0.n_kv, a0.hd, model.d_model, pcfg.get("n_prefix", 32),
                                       pcfg.get("prefix_hidden", 512))
    elif method == "prompt":
        P = pcfg.get("n_prompt", 32)
        g = torch.Generator().manual_seed(0)
        w = model.lm.tok_emb.weight
        model.prompt = nn.Parameter(w[torch.randint(1000, model.cfg.vocab_size, (P,), generator=g)].detach().clone()[None])
        model.n_prompt = P
    elif method != "full":
        raise ValueError(method)
    if model.arch == "gpt" and pcfg.get("train_pos", True):
        model.lm.pos_emb.weight.requires_grad = True  # the interpolated position table must adapt
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": train, "trainable_pct": round(100 * train / total, 3)}


def build_summarizer(cfg: dict):
    lm, tok = load_phase1(cfg["model"]["phase1_ckpt"])
    model = Summarizer(lm, tok, cfg["model"])
    info = apply_peft(model, cfg["peft"])
    return model, tok, info


def make_controller(model, pcfg, total_steps):
    return AdaLoRAController(model, pcfg, total_steps) if pcfg["method"] == "adalora" else None


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers() if n.endswith("rank_mask")}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
