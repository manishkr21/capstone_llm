"""Configuration = plain nested dicts loaded from YAML.

* A YAML file may start with ``inherit: base.yaml`` to extend another file.
* Several files can be stacked:  --config configs/gpt.yaml configs/smoke.yaml
* Any value can be overridden:    --set train.max_steps=100 model.dropout=0.1
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable, Optional

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_one(path) -> dict:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("inherit", None)
    if parent:
        data = deep_merge(_load_one(path.parent / parent), data)
    return data


def apply_overrides(cfg: dict, overrides: Optional[Iterable[str]]) -> dict:
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(paths: Iterable, overrides: Optional[Iterable[str]] = None) -> dict:
    cfg: dict = {}
    for p in paths:
        cfg = deep_merge(cfg, _load_one(p))
    return apply_overrides(cfg, overrides)


def add_config_args(parser: argparse.ArgumentParser, default: str = "configs/base.yaml") -> None:
    parser.add_argument("--config", nargs="+", default=[default],
                        help="one or more YAML files, later files override earlier ones")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.max_steps=200")


def config_from_args(args) -> dict:
    return load_config(args.config, args.set)
