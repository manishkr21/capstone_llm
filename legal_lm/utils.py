"""Small helpers shared by every stage of the pipeline."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_WS = re.compile(r"\s+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(preference: str = "auto") -> torch.device:
    """Return cuda > mps > cpu unless a specific device is requested."""
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_text(text: str, collapse_whitespace: bool = True) -> str:
    """Light normalisation. The test prompts are word sequences, so we collapse
    all whitespace (newlines, tabs, repeated spaces) into single spaces."""
    text = text.replace("\x00", " ")
    return _WS.sub(" ", text).strip() if collapse_whitespace else text.strip()


def first_n_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def save_json(obj: Any, path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_torch_save(obj: Any, path) -> None:
    """Write to a temp file first so a Colab disconnect never leaves a broken checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))
