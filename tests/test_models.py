"""Sanity tests for the four architectures (run: python tests/test_models.py, or pytest).

1. Shapes: model(idx, p) returns logits for exactly the L - p target tokens.
2. Causality: changing a *future* token never changes earlier predictions.
3. Cache correctness: prefill + decode_step reproduce the full forward pass, which
   proves the KV-cache, RoPE offsets, ALiBi offsets and sliding windows are right.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legal_lm.models import ARCHITECTURES, ModelConfig, build_model  # noqa: E402

TINY = dict(vocab_size=101, max_seq_len=48, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=64,
            n_enc_layers=2, n_dec_layers=2, window=8, global_every=2)


def make(arch):
    torch.manual_seed(0)
    return build_model(ModelConfig(arch=arch, **TINY)).eval()


def test_shapes_and_prefix_sampling():
    rng = random.Random(0)
    for arch in ARCHITECTURES:
        m = make(arch)
        L = TINY["max_seq_len"] + 1
        idx = torch.randint(0, 101, (2, L))
        for _ in range(5):
            p = m.sample_prefix_len(L, rng)
            assert 1 <= p < L, (arch, p)
            assert m(idx, p).shape == (2, L - p, m.cfg.padded_vocab), arch


def test_causality():
    for arch in ARCHITECTURES:
        m = make(arch)
        L, p = TINY["max_seq_len"] + 1, 20
        a = torch.randint(0, 101, (1, L))
        b = a.clone()
        b[0, 40] = (b[0, 40] + 1) % 101  # change a future token
        la, lb = m(a, p), m(b, p)
        # logits for target positions < 40 (index t-p) must be identical
        assert torch.allclose(la[:, : 40 - p], lb[:, : 40 - p], atol=1e-5), arch


def test_cache_matches_full_forward():
    for arch in ARCHITECTURES:
        m = make(arch)
        P, n_new = 20, 12
        seq = torch.randint(0, 101, (2, P + n_new + 1))
        full = m(seq[:, : P + n_new + 1], P)  # predicts seq[P:], given prompt of length P
        logits, state = m.prefill(seq[:, :P])
        steps = [logits]
        for t in range(n_new):
            logits, state = m.decode_step(seq[:, P + t: P + t + 1], state)
            steps.append(logits)
        cached = torch.stack(steps, 1)
        assert torch.allclose(full[:, : n_new + 1], cached, atol=1e-4), \
            f"{arch}: max diff {(full[:, :n_new + 1] - cached).abs().max():.2e}"


if __name__ == "__main__":
    for fn in (test_shapes_and_prefix_sampling, test_causality, test_cache_matches_full_forward):
        fn()
        print(f"PASS {fn.__name__}")
