from __future__ import annotations

import json
import os
import time

import torch


def atomic_save(obj, path: str) -> None:
    """torch.save via a temp file + os.replace, so a crash never leaves a
    truncated checkpoint at ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


class JsonlLogger:
    """One {"step", "t", **scalars} object per line in <out_dir>/log.jsonl."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self._f = open(os.path.join(out_dir, "log.jsonl"), "a", buffering=1, encoding="utf-8")

    def log(self, step: int, **scalars):
        rec = {"step": int(step), "t": round(time.time(), 1)}
        rec.update({k: float(v) for k, v in scalars.items()})
        self._f.write(json.dumps(rec) + "\n")

    def close(self):
        self._f.close()
