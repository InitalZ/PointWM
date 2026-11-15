#!/usr/bin/env python
"""Trainable parameters of a pretraining config (EMA copy excluded) and of
the downstream models (Table S3).

    python tools/count_params.py configs/pretrain/pointwm.yaml configs/pretrain/leak.yaml
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointwm.models.downstream import ClsModel, SegModel
from pointwm.models.pointwm import PointWM
from pointwm.utils.config import load_config


def count(m, prefix=None):
    return sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and (prefix is None or n.startswith(prefix)))


def main():
    for path in sys.argv[1:]:
        mc = load_config(path)["model"]
        m = PointWM(**mc)
        parts = {k: count(m, k) for k in ("tokenizer.", "encoder.", "predictor.", "idm.", "prior.")}
        print(f"{path}: trainable {count(m) / 1e6:.2f}M  " + "  ".join(f"{k[:-1]} {v / 1e6:.3f}M" for k, v in parts.items()))
        dims = dict(dim=mc.get("dim", 384), depth=mc.get("depth", 6), heads=mc.get("heads", 6))
    seg, cls = SegModel(**dims), ClsModel(**dims)
    print(f"downstream backbone (tokenizer + encoder) {(count(seg, 'tokenizer.') + count(seg, 'encoder.')) / 1e6:.2f}M, "
          f"+ seg head {count(seg) / 1e6:.2f}M, + MSR head {count(cls) / 1e6:.2f}M")


if __name__ == "__main__":
    main()
