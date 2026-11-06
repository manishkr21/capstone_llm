"""Training batches, sliding-window inference and run loading shared by the scripts."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from lner.backbones import build_tagger
from lner.data import encode_tags, normalize_case, tags_to_spans, tokenize, windows
from lner.decoding import decode
from lner.peft import apply_method


def prepare_docs(tok, texts: Dict[str, str], gold, presplit: bool, tag2id, case_norm: bool = False) -> List[dict]:
    """Tokenize every document. With case_norm the model reads normalize_case(text); offsets are
    identical, so spans are still cut from the original text."""
    docs = []
    for doc_id, text in texts.items():
        ids, spans = tokenize(tok.tk, normalize_case(text) if case_norm else text, presplit)
        d = {"id": doc_id, "text": text, "ids": ids, "spans": spans}
        if gold is not None:
            d["tags"] = encode_tags(spans, gold[doc_id], tag2id)
        docs.append(d)
    return docs


def train_windows(docs, size: int, stride: int, rng: random.Random):
    """All windows of every document, with a random start shift so each epoch sees
    entities at different positions inside the window."""
    out = []
    for d in docs:
        n = len(d["ids"])
        shift = rng.randrange(stride) if n > size + stride else 0
        for s, e in windows(n - shift, size, stride):
            out.append((d, s + shift, e + shift))
    rng.shuffle(out)
    return out


def batch_tensors(chunk, device):
    ids = torch.tensor([d["ids"][s:e] for d, s, e in chunk], device=device)
    tags = torch.tensor([d["tags"][s:e] for d, s, e in chunk], device=device)
    return ids, tags


@torch.no_grad()
def doc_logits(model, ids: List[int], size: int, stride: int, device, batch_size: int = 32,
               dtype=torch.bfloat16) -> torch.Tensor:
    """(T, n_tags) logits for one document from overlapping windows.
    Each token keeps the logits of the window where it has the most context:
    the most central window (bidirectional) or the one with the longest left context (causal)."""
    n = len(ids)
    bidir = model.backbone.bidirectional
    wins = windows(n, size, stride)
    out = best = None
    for b in range(0, len(wins), batch_size):
        chunk = wins[b:b + batch_size]
        x = torch.tensor([ids[s:e] for s, e in chunk], device=device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(x).float()
        if out is None:
            out = torch.zeros(n, logits.size(-1), device=device)
            best = torch.full((n,), -1, device=device)
        for (s, e), lg in zip(chunk, logits):
            pos = torch.arange(e - s, device=device)
            # context on each side; at a document edge no window can offer more
            left = pos + (s == 0) * n
            right = (e - s - 1 - pos) + (e == n) * n
            score = torch.minimum(left, right) if bidir else left
            take = score > best[s:e]
            out[s:e][take] = lg[take]
            best[s:e] = torch.where(take, score, best[s:e])
    return out


def predict_docs(model, docs, tags: List[str], icfg: dict, device) -> Dict[str, list]:
    model.eval()
    preds = {}
    for d in docs:
        if not d["ids"]:
            preds[d["id"]] = []
            continue
        lg = doc_logits(model, d["ids"], icfg["window"], icfg["stride"], device)
        tag_ids = decode(lg, tags, icfg.get("decode", "viterbi"), crf=model.crf)
        ents = tags_to_spans(d["text"], d["spans"], tag_ids, tags)
        if icfg.get("merge_gap"):
            ents = merge_spans(d["text"], ents, icfg["merge_gap"])
        if icfg.get("propagate"):
            ents = propagate(d["text"], ents, icfg["propagate"])
        preds[d["id"]] = ents
    return preds


def merge_spans(text: str, ents: List[list], gap_chars: str) -> List[list]:
    """Join consecutive spans with the same label when only `gap_chars` lie between them
    (e.g. 'OPEATIVE' + ' ' + 'SOCIETY LTD')."""
    out = []
    for e in ents:
        if out and out[-1][2] == e[2] and all(c in gap_chars for c in text[out[-1][1]:e[0]]):
            out[-1] = [out[-1][0], e[1], e[2], text[out[-1][0]:e[1]]]
        else:
            out.append(list(e))
    return out


def propagate(text: str, ents: List[list], min_len: int) -> List[list]:
    """Document-level consistency: a string predicted as an entity (>= min_len chars) is also
    tagged, with its majority label, wherever else it occurs as a whole word untagged."""
    votes: Dict[str, Counter] = {}
    for s, e, l, *_ in ents:
        if e - s >= min_len:
            votes.setdefault(text[s:e], Counter())[l] += 1
    taken = sorted((s, e) for s, e, *_ in ents)
    covered = bytearray(len(text))
    for s, e in taken:
        covered[s:e] = b"\x01" * (e - s)
    added = []
    for string, c in votes.items():
        label = c.most_common(1)[0][0]
        i = text.find(string)
        while i != -1:
            j = i + len(string)
            whole = (i == 0 or not text[i - 1].isalnum()) and (j == len(text) or not text[j].isalnum())
            if whole and not any(covered[i:j]):
                added.append([i, j, label, string])
                covered[i:j] = b"\x01" * (j - i)
            i = text.find(string, i + 1)
    return sorted(ents + added, key=lambda x: x[0])


def build_run_model(cfg: dict, tags: List[str]):
    tagger, tok = build_tagger(cfg["model"], tags)
    info = apply_method(tagger, cfg["peft"])
    return tagger, tok, info


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def load_run(run_dir, tags: List[str], device, ckpt: str = "best.pt"):
    """Phase-1 weights + this run's injected modules + its saved trainable parameters."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model, tok, _ = build_run_model(cfg, tags)
    state = torch.load(run_dir / ckpt, map_location="cpu")
    missing = set(state) - set(model.state_dict())
    if missing:
        raise RuntimeError(f"checkpoint keys not in model: {sorted(missing)[:5]}")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), tok, cfg
