"""YAML configs with ``_base_`` inheritance and ``--set key.sub=value``
overrides, so every variant is written as its difference from PointWM."""
from __future__ import annotations

import copy
import os

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def apply_overrides(cfg: dict, items) -> dict:
    """items: ["optim.epochs=10", "model.masks=[causal]", ...]; values are YAML."""
    cfg = copy.deepcopy(cfg)
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg
