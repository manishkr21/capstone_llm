"""Turning a trained model into prediction.json.

Decoding controls (all in the `generation:` config section):
  temperature        0 = greedy; higher = more random
  top_k / top_p      keep only the k most likely tokens / the smallest set with mass p
  repetition_penalty divide the logits of recently used tokens (>1 discourages loops)
  repetition_window  how many recent tokens the penalty looks at
  ban_eos            never stop early: the task needs exactly 128 words
  target_words       stop once every sequence has this many complete words

Batching trick: prompts are sorted by token length and each batch is left-truncated
to its shortest member, so no padding (and no padding masks) is ever needed.
"""
from __future__ import annotations

import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from legal_lm.data.tokenizer import LegalTokenizer
from legal_lm.models import LegalLM
from legal_lm.utils import save_json


def sample_next(logits: torch.Tensor, history: torch.Tensor, g: dict, banned: List[int]) -> torch.Tensor:
    logits = logits.float().clone()
    if banned:
        logits[:, banned] = float("-inf")
    rp = g.get("repetition_penalty", 1.0)
    if rp != 1.0:
        recent = history[:, -g.get("repetition_window", 256):]
        scores = logits.gather(1, recent)
        scores = torch.where(scores < 0, scores * rp, scores / rp)
        logits.scatter_(1, recent, scores)
    temperature = g.get("temperature", 1.0)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    top_k = g.get("top_k") or 0
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    top_p = g.get("top_p", 1.0)
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p  # keep the token that crosses p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)


@torch.no_grad()
def generate_batch(model: LegalLM, prompt: torch.Tensor, tok: LegalTokenizer, g: dict) -> List[str]:
    banned = [tok.eos_id] if g.get("ban_eos", True) else []
    banned += list(range(tok.vocab_size, model.cfg.padded_vocab))  # padding rows of the vocab
    need = g["target_words"] + 1  # one extra so the last kept word is complete
    logits, state = model.prefill(prompt)
    history, new = prompt, []
    for step in range(g["max_new_tokens"]):
        nxt = sample_next(logits, history, g, banned)
        new.append(nxt)
        history = torch.cat([history, nxt], dim=1)
        if (step + 1) % 16 == 0:
            texts = [tok.decode(row) for row in torch.cat(new, 1).tolist()]
            if all(len(t.split()) >= need for t in texts):
                break
        logits, state = model.decode_step(nxt, state)
    return [tok.decode(row) for row in torch.cat(new, 1).tolist()]


def postprocess(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words])


def generate_predictions(model: LegalLM, tok: LegalTokenizer, prompts: Dict[str, dict], g: dict,
                         device) -> Dict[str, dict]:
    model.eval()
    ids = list(prompts)
    encoded = tok.encode_batch([prompts[i]["text"] for i in ids])
    order = sorted(range(len(ids)), key=lambda k: len(encoded[k]))
    limit = model.max_prompt_len(g["max_new_tokens"])
    torch.manual_seed(g.get("seed", 0))
    out, short, start = {}, 0, time.time()
    for b in range(0, len(order), g["batch_size"]):
        chunk = order[b:b + g["batch_size"]]
        keep = max(1, min(limit, min(len(encoded[k]) for k in chunk)))
        batch = torch.tensor([encoded[k][-keep:] for k in chunk], dtype=torch.long, device=device)
        for k, text in zip(chunk, generate_batch(model, batch, tok, g)):
            text = postprocess(text, g["target_words"])
            short += len(text.split()) < g["target_words"]
            out[ids[k]] = {"text": text}
        print(f"  generated {min(b + len(chunk), len(ids))}/{len(ids)}"
              f"  ({time.time() - start:.0f}s, prompt kept {keep} tokens)", flush=True)
    if short:
        print(f"  WARNING: {short} continuations have fewer than {g['target_words']} words; "
              f"raise generation.max_new_tokens")
    return {i: out[i] for i in ids}  # original id order


def write_submission(predictions: Dict[str, dict], out_dir) -> Path:
    """prediction.json + submission_<date>_<time>.zip with the JSON at the root of the archive.
    The timestamp keeps every submission distinct, so re-running never overwrites an earlier zip."""
    out_dir = Path(out_dir)
    pred_path = out_dir / "prediction.json"
    save_json(predictions, pred_path)
    zip_path = out_dir / f"submission_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    return zip_path
