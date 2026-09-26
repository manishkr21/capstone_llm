"""legal_lm: a small, modular toolkit for CS-781 Capstone Phase 1.

Phase 1 asks for a legal language model trained from scratch (50M-300M params)
that continues a 512-word legal prompt with 128 words.  The package is split into:

    legal_lm.config     YAML config loading and command-line overrides
    legal_lm.data       corpus streaming, tokenizer training, token-bin datasets
    legal_lm.models     the four architectures behind one common interface
    legal_lm.trainer    one training loop shared by all four architectures
    legal_lm.generation decoding strategies and prediction.json writing
    legal_lm.metrics    local BERTScore / ROUGE-L / BLEU-4 and format checks
"""

__version__ = "0.1.0"
