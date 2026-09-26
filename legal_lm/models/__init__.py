"""Model registry: ``build_model(ModelConfig)`` returns one of the four architectures."""
from __future__ import annotations

from legal_lm.models.base import LegalLM
from legal_lm.models.config import ModelConfig
from legal_lm.models.decoder_gpt import DecoderOnlyGPT
from legal_lm.models.efficient_lm import EfficientLM
from legal_lm.models.encoder_decoder import EncoderDecoderLM
from legal_lm.models.prefix_lm import PrefixLM

ARCHITECTURES = {
    "gpt": DecoderOnlyGPT,        # A1 decoder-only baseline
    "encdec": EncoderDecoderLM,   # A2 encoder-decoder
    "prefixlm": PrefixLM,         # A3 hybrid prefix-LM
    "efficient": EfficientLM,     # A4 efficient / lightweight
}


def build_model(cfg: ModelConfig) -> LegalLM:
    if cfg.arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {cfg.arch!r}; choose from {list(ARCHITECTURES)}")
    return ARCHITECTURES[cfg.arch](cfg)


__all__ = ["ARCHITECTURES", "LegalLM", "ModelConfig", "build_model"]
